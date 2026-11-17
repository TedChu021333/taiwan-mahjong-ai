"""出牌當下模擬挑牌（6 秒場用，`python -m hint --search`）。

每手有約 4 秒可以想時：
1. 候選：網路打牌分數前 CANDIDATES 名，加上網路＋護欄原本要打的那張（預設）。
2. 補完（complete_state）：把看不到的牌補成完整局面——三家暗手（已宣告的一定聽牌、且不聽他打過與宣告後
   別人打過的牌；沒宣告的依聽牌率決定要不要組成聽牌手牌）、內容沒讀到的副露、牌牆順序、之後槓牌加的金牌。
3. 每補完一次，所有候選都用同一副牌、同一個種子，讓網路四家（不開護欄、取最高分）接著打到這局結束，
   記下自己扣抽成後的輸贏。共用同一副牌能扣掉大部分運氣。
4. 只有某張候選比預設平均多贏超過 SIGNIFICANCE 倍標準誤（同一副牌逐次相減）才換。

10/28 的驗證（training/net_probe，已刪）：網路前 3 名各模擬 200 次、比第一名好超過 1 倍標準誤才換，
每個打牌決定 +0.037±0.020 台；次數只有 50～100 時增益約減半。所以要整台電腦一起算：每個子程序一條
GPU 批次執行緒＋THREADS 條模擬執行緒（Python 的 GIL 讓單一程序每秒只能模擬約 35 次）。
"""
from __future__ import annotations

import itertools
import json
import math
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

from agent.opponent_model import (
    UNDECLARED_TENPAI_SHARE,
    Calibration,
    default_calibration,
    discard_count,
    sample_tenpai_hand,
    unseen_counts,
)
from game.rules import DEAD_WALL_SIZE, Action, ActionType, GameState, Meld, Phase, legal_actions
from game.simulator import after_rake, payments, play_game

CANDIDATES = 3
"""候選張數（含預設）：預設＋網路打牌分數最高的其他牌（10/28 驗證：網路前 3 名比前 2 名好）。"""
SIGNIFICANCE = 1.0
"""比預設平均多贏超過幾倍標準誤才換（10/28：1 倍最好；0 倍在次數少時是負的）。"""
MIN_SAMPLES = 40
"""至少補完幾次才考慮換牌。"""
THREADS = 48
"""每個子程序幾條模擬執行緒（等 GPU 時讓別條先跑）。"""
MAX_STEPS = 2_000
GATHER_SECONDS = 0.002
"""GPU 推論程序收到第一批請求後再等多久湊別的模擬程序的請求。"""


# ---------------------------------------------------------------- 補完局面

def _sample_meld(pool: list[int], known: tuple[int, ...], rng: Random) -> tuple[int, ...] | None:
    """從 pool 補一組含 known 的面子（刻子或順子）；回傳要從 pool 拿走的牌。"""
    options: list[tuple[int, ...]] = []
    for tile in range(34):  # 刻子
        need = [tile] * 3
        for seen in known:
            if seen in need:
                need.remove(seen)
        if len(need) == 3 - len(known) and all(pool[t] >= need.count(t) for t in set(need)):
            options.append(tuple(need))
    for start in range(27):  # 順子
        if start % 9 > 6:
            continue
        need = [start, start + 1, start + 2]
        if any(seen not in need for seen in known) or len(set(known)) != len(known):
            continue
        for seen in known:
            need.remove(seen)
        if all(pool[t] >= 1 for t in need):
            options.append(tuple(need))
    if not options:
        return None
    return rng.choice(options)


def _fill_melds(players, pool: list[int], rng: Random) -> None:
    """內容沒讀到的副露（kind 不是 chow/pung/kong）補成真的面子，用掉的牌從 pool 扣。"""
    for other in players:
        filled = []
        for meld in other.melds:
            if meld.kind in ("chow", "pung", "kong"):
                filled.append(meld)
                continue
            known = tuple(meld.tiles[:2])
            need = _sample_meld(pool, known, rng) or _sample_meld(pool, (), rng)
            if need is None:  # 牌不夠組（辨識出的局面有誤）：只放看得到的牌
                filled.append(Meld("pung", tuple(meld.tiles) or (0, 0, 0), meld.from_player))
                continue
            for tile in need:
                pool[tile] -= 1
            tiles = tuple(sorted(known + need)) if known else tuple(sorted(need))
            kind = "pung" if len(set(tiles)) == 1 else "chow"
            filled.append(Meld(kind, tiles, meld.from_player))
        other.melds = filled


def _random_hand(pool: list[int], size: int, rng: Random) -> list[int]:
    tiles = [tile for tile, count in enumerate(pool) for _ in range(count)]
    hand = [0] * 34
    for tile in rng.sample(tiles, min(size, len(tiles))):
        hand[tile] += 1
    return hand


def complete_state(state: GameState, player: int, rng: Random,
                   calibration: Calibration | None = None,
                   undeclared_share: float = UNDECLARED_TENPAI_SHARE) -> GameState:
    """把 player 看不到的部分補成完整、可以接著打的局面（不改動 state）。"""
    calibration = calibration or default_calibration()
    full = deepcopy(state)
    pool = unseen_counts(full, player)
    _fill_melds(full.players, pool, rng)
    order = sorted((seat for seat in range(4) if seat != player),
                   key=lambda seat: not full.declared[seat])  # 有限制的先抽
    for seat in order:
        other = full.players[seat]
        size = max(1, 16 - 3 * len(other.melds))
        declared = full.declared[seat]
        tenpai = declared or rng.random() < calibration.undeclared_tenpai_probability(
            discard_count(full, seat), len(other.melds), undeclared_share)
        hand = None
        if tenpai:
            forbidden = set(other.discards) | set(full.declared_safe[seat]) if declared else set()
            sampled = sample_tenpai_hand(pool, len(other.melds), rng, forbidden)
            if sampled is not None:
                hand = sampled[0]
        else:  # 沒聽牌：照模擬器量出的分布抽向聽數（1 向聽以上）
            weights = shanten_weights(discard_count(full, seat), len(other.melds))[1:]
            level = rng.choices(range(1, len(weights) + 1), weights=weights)[0] if sum(weights) else 3
            hand = sample_hand_with_shanten(pool, len(other.melds), level, rng)
        if hand is None:
            hand = _random_hand(pool, size, rng)
        for tile, count in enumerate(hand):
            pool[tile] -= count
        other.hand = hand
    rest = [tile for tile, count in enumerate(pool) for _ in range(count)]
    rng.shuffle(rest)
    drawable = min(state.drawable, len(rest))
    reserve = rest[drawable:drawable + DEAD_WALL_SIZE]
    full.wall = reserve + rest[:drawable]  # 從尾巴摸
    full.reserve = len(reserve)
    golds = [tile for tile in range(34) if tile not in full.gold_tiles]
    rng.shuffle(golds)
    full.gold_pool = golds
    if not full.dealer_known:  # 不知道莊家時照實戰 AI 的作法：不加莊家台
        full.dealer_streak = 0
    full.min_discards = 0
    return full


# ---------------------------------------------------------------- 模擬

def discard_candidates(logits: np.ndarray, lookup: dict[int, Action], default: Action,
                       count: int = CANDIDATES) -> list[int]:
    """預設那張排第一個，接著是網路打牌分數最高的其他牌，共 count 張。"""
    discards = [(float(logits[index]), action.tile) for index, action in lookup.items()
                if action.kind == ActionType.DISCARD]
    discards.sort(reverse=True)
    tiles = [default.tile] if default.kind == ActionType.DISCARD else []
    for _, tile in discards:
        if tile not in tiles and len(tiles) < count:
            tiles.append(tile)
    return tiles


def rollout(start: GameState, player: int, tiles: list[int], seed: int, policy) -> list[float]:
    """同一副補完的牌，每張候選各打一次到局末：自己扣抽成後的輸贏（台）。"""
    values = []
    for tile in tiles:
        state = deepcopy(start)
        result = play_game(Random(seed), max_steps=MAX_STEPS, player_policy=policy, start=_after_discard(state, player, tile))
        values.append(after_rake(payments(result)[player]))
    return values


def _after_discard(state: GameState, player: int, tile: int) -> GameState:
    from game.rules import apply_action

    apply_action(state, Action(ActionType.DISCARD, tile=tile), player)
    return state


@dataclass
class Choice:
    tile: int
    note: str | None
    samples: int


def choose(tiles: list[int], values: list[list[float]], significance: float = SIGNIFICANCE,
           min_samples: int = MIN_SAMPLES) -> Choice:
    """tiles[0] 是預設；別張比它平均多贏超過 significance 倍標準誤才換（取平均多最多的）。"""
    n = len(values)
    if n < max(2, min_samples) or len(tiles) < 2:
        return Choice(tiles[0], None, n)
    data = np.asarray(values, dtype=np.float64)
    best, best_gain, best_se = 0, 0.0, 0.0
    for column in range(1, len(tiles)):
        diff = data[:, column] - data[:, 0]
        gain, se = diff.mean(), diff.std(ddof=1) / math.sqrt(n)
        if gain > significance * se and gain > best_gain:
            best, best_gain, best_se = column, gain, se
    if best == 0:
        return Choice(tiles[0], None, n)
    from game.tiles import tile_name

    note = (f"模擬：打{tile_name(tiles[best])}比{tile_name(tiles[0])}每局多 {best_gain:+.2f}±{best_se:.2f} 台"
            f"（{n} 副牌）")
    return Choice(tiles[best], note, n)


# ---------------------------------------------------------------- 多程序：一個 GPU 推論程序＋多個模擬程序
# Windows 上每個程序各自佔一份 GPU 時，程序之間要不斷切換：1 個程序 4 秒約 56 副牌、10 個程序加起來也只有 60 副。
# 所以只有一個程序用 GPU；模擬程序把自己各條執行緒的請求湊成一批、經 Pipe 送過去，拿回分數最高的動作。


class Deadline(Exception):
    """時間到了，正在模擬的這一局放棄。"""


def _inference_server(model_path: str, device: str, connections) -> None:
    from multiprocessing.connection import wait

    import torch

    from training.model import load

    torch.set_num_threads(2)
    model = load(model_path, device).eval()
    run = _graph_runner(model) if device == "cuda" else _plain_runner(model, device)
    alive = list(connections)
    while alive:
        ready = wait(alive)
        # 多等一下讓別的模擬程序的請求也到：GPU 一次算 48 個或 500 個都約 3 毫秒
        gather_until = time.perf_counter() + GATHER_SECONDS
        batch = []
        while ready:
            for conn in ready:
                try:
                    batch.append((conn, conn.recv()))
                except (EOFError, OSError):
                    alive.remove(conn)
            waiting = [conn for conn in alive if all(conn is not got for got, _ in batch)]
            left = gather_until - time.perf_counter()
            ready = wait(waiting, timeout=left) if waiting and left > 0 else []
        if not batch:
            continue
        obs = np.concatenate([item[1][0] for item in batch])
        masks = np.concatenate([item[1][1] for item in batch])
        logits = run(obs)
        choice = np.where(masks, logits, -1e9).argmax(1).astype(np.int16)
        start = 0
        for conn, (part, _) in batch:
            conn.send(choice[start:start + len(part)])
            start += len(part)


GRAPH_SIZES = (32, 64, 128, 256, 512, 1024)


def _plain_runner(model, device: str):
    import torch

    def run(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(obs).to(device).float())[0].cpu().numpy()
    return run


def _graph_runner(model):
    """半精度＋CUDA Graph：整個網路錄成一張圖重播，省掉逐層啟動的開銷（其他程序搶 CPU 時特別明顯）。
    一批 128 個 2.8 → 1.3 毫秒；選出的動作和單精度一樣。批次補到 GRAPH_SIZES 裡最近的大小。"""
    import torch

    from game.tiles import NUM_TILE_TYPES
    from training.encoding import CHANNELS

    model = model.half()
    graphs: dict[int, tuple] = {}

    def graph(size: int):
        if size not in graphs:
            static_in = torch.zeros(size, CHANNELS, NUM_TILE_TYPES, device="cuda", dtype=torch.half)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(stream):
                for _ in range(3):
                    model(static_in)
            torch.cuda.current_stream().wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(captured):
                static_out = model(static_in)[0]
            graphs[size] = (captured, static_in, static_out)
        return graphs[size]

    for size in GRAPH_SIZES:  # 啟動時就錄好，不在搜尋途中花時間
        graph(size)

    def run(obs: np.ndarray) -> np.ndarray:
        parts = []
        for start in range(0, len(obs), GRAPH_SIZES[-1]):
            chunk = obs[start:start + GRAPH_SIZES[-1]]
            size = next(size for size in GRAPH_SIZES if size >= len(chunk))
            captured, static_in, static_out = graph(size)
            static_in[:len(chunk)].copy_(torch.from_numpy(chunk).to("cuda"))
            static_in[len(chunk):].zero_()
            captured.replay()
            parts.append(static_out[:len(chunk)].float().cpu().numpy())
        return np.concatenate(parts)
    return run


class PipeBatcher:
    """這個模擬程序裡各條執行緒的請求湊成一批，送給 GPU 推論程序。"""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.requests: queue.Queue = queue.Queue()
        self.deadline = float("inf")
        threading.Thread(target=self._loop, daemon=True).start()

    def ask(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if time.time() > self.deadline:
            raise Deadline
        slot = [obs, mask, threading.Event(), None]
        self.requests.put(slot)
        slot[2].wait()
        return slot[3]

    def _loop(self) -> None:
        while True:
            batch = [self.requests.get()]
            deadline = time.perf_counter() + 0.001
            while len(batch) < THREADS:
                try:
                    batch.append(self.requests.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            self.conn.send((np.stack([item[0] for item in batch]).astype(np.float16),
                            np.stack([item[1] for item in batch])))
            choice = self.conn.recv()
            for row, item in enumerate(batch):
                item[3] = int(choice[row])
                item[2].set()


class BatchedNetwork:
    """模擬裡四家共用的網路（不開護欄）：摸牌、換三張照規則式，其他照網路最高分。"""

    def __init__(self, batcher) -> None:
        self.batcher = batcher

    def __call__(self, state: GameState, player: int, rng: Random) -> Action | None:
        from agent.rule_agent import choose_rule_action_for_player
        from training.actions import legal_mask
        from training.encoding import encode

        actions = legal_actions(state, player)
        if not actions:
            return None
        mask, lookup = legal_mask(actions)
        if not lookup:
            return choose_rule_action_for_player(state, player, rng)
        if len(lookup) == 1:
            return next(iter(lookup.values()))
        return lookup[self.batcher.ask(encode(state, player), mask)]


def search_once(policy: BatchedNetwork, state: GameState, player: int, tiles: list[int], seed: int,
                deadline: float, undeclared_share: float = UNDECLARED_TENPAI_SHARE,
                limit: int | None = None) -> list[list[float]]:
    """用 THREADS 條執行緒一直補完、模擬，直到 deadline（time.time()）或做滿 limit 副牌。
    時間到時正在打的那幾局直接放棄（不會拖過 deadline）。"""
    calibration = default_calibration()
    results: list[list[float]] = []
    counter = itertools.count()
    lock = threading.Lock()
    policy.batcher.deadline = deadline

    def worker() -> None:
        while time.time() < deadline:
            with lock:
                index = next(counter)
            if limit is not None and index >= limit:
                return
            run_seed = seed * 100_003 + index
            try:
                full = complete_state(state, player, Random(run_seed), calibration, undeclared_share)
                values = rollout(full, player, tiles, run_seed, policy)
            except Deadline:
                return
            except Exception:  # 補完出不合法的局面（辨識有誤）：這次不算
                continue
            with lock:
                results.append(values)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    policy.batcher.deadline = float("inf")
    with lock:
        return list(results)


def _search_worker(conn, tasks, results) -> None:
    policy = BatchedNetwork(PipeBatcher(conn))
    while True:
        task = tasks.get()
        if task is None:
            return
        job, args = task
        try:
            results.put((job, search_once(policy, *args)))
        except Exception:  # 不能讓子程序死掉
            results.put((job, []))


def _sample_position() -> GameState:
    """暖機用：規則式四家打到第一次輪到座位 0 打牌的局面。"""
    from agent.rule_agent import choose_rule_action_for_player

    snapshot: list[GameState] = []

    def grab(state, player, rng):
        if player == 0 and state.phase == Phase.DISCARD and not snapshot:
            snapshot.append(deepcopy(state))
        return choose_rule_action_for_player(state, player, rng)

    play_game(Random(0), player_policy=grab)
    return snapshot[0]


class Searcher:
    """實戰用：開局前啟動一個 GPU 推論程序與 workers 個模擬程序，之後每手呼叫 search。"""

    def __init__(self, model_path: str, workers: int, device: str | None = None,
                 undeclared_share: float = UNDECLARED_TENPAI_SHARE) -> None:
        import multiprocessing

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        context = multiprocessing.get_context("spawn")
        pipes = [context.Pipe() for _ in range(workers)]
        self.server = context.Process(target=_inference_server, daemon=True,
                                      args=(model_path, device, [server for server, _ in pipes]))
        self.server.start()
        self.results = context.Queue()
        self.tasks = [context.Queue() for _ in range(workers)]
        self.workers = [context.Process(target=_search_worker, args=(client, tasks, self.results), daemon=True)
                        for (_, client), tasks in zip(pipes, self.tasks)]
        for process in self.workers:
            process.start()
        self.undeclared_share = undeclared_share
        self.job = 0
        state = _sample_position()  # 暖機：載入網路、初始化 CUDA
        self.search(state, 0, [tile for tile in range(34) if state.players[0].hand[tile]][:2], 60.0, limit=2)

    def search(self, state: GameState, player: int, tiles: list[int], seconds: float,
               limit: int | None = None) -> Choice:
        """tiles[0] 是預設；最多想 seconds 秒（limit：每個模擬程序最多幾副牌）。"""
        if len(tiles) < 2 or seconds <= 0:
            return Choice(tiles[0], None, 0)
        deadline = time.time() + seconds
        self.job += 1
        for index, tasks in enumerate(self.tasks):
            tasks.put((self.job, (state, player, tiles, self.job * 1000 + index, deadline,
                                  self.undeclared_share, limit)))
        values: list[list[float]] = []
        received = 0
        while received < len(self.tasks):
            try:
                job, part = self.results.get(timeout=max(0.05, deadline - time.time() + 0.5))
            except queue.Empty:
                break
            if job == self.job:
                values += part
                received += 1
        return choose(tiles, values)

    def close(self) -> None:
        for tasks in self.tasks:
            tasks.put(None)
        for process in self.workers + [self.server]:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()


# ---------------------------------------------------------------- 六秒版的一手

def decide(policy, searcher: "Searcher | None", state: GameState, player: int, rng: Random,
           seconds: float) -> tuple[Action | None, str | None]:
    """先照網路＋護欄（policy，guard=True）得到預設；輪到自己打牌時再模擬比較網路前幾名。
    模擬換了牌之後照樣過護欄的放槍檢查與按聽規則（policy._guarded）。回傳 (動作, 記錄用說明)。"""
    from training.actions import legal_mask

    default = policy(state, player, rng)
    note = policy.last_note
    if searcher is None or default is None or default.kind != ActionType.DISCARD \
            or state.phase != Phase.DISCARD or state.declared[player] or player != state.current_player:
        return default, note  # 吃碰、按聽、槓、胡照原本的網路＋護欄
    actions = legal_actions(state, player)
    mask, lookup = legal_mask(actions)
    if len({action.tile for action in lookup.values() if action.kind == ActionType.DISCARD}) < 2:
        return default, note
    tiles = discard_candidates(policy._scores(state, player, mask), lookup, default)
    choice = searcher.search(state, player, tiles, seconds)
    if choice.tile == default.tile:
        return default, note
    action = policy._guarded(state, player, Action(ActionType.DISCARD, tile=choice.tile), actions, rng)
    notes = [choice.note, policy.last_note]
    policy.last_note = "；".join(text for text in notes if text) or None
    return action, policy.last_note


# ---------------------------------------------------------------- 對手向聽數的分布（補完時用）
# 沒宣告的對手原本是從看不到的牌隨機抽 16 張，幾乎都是 4～5 向聽；真的對手打了幾巡通常是 1～2 向聽。
# 模擬因此低估別人胡牌、偏向進攻（11/17 驗證時看到：真的三家 1、0、2 向聽，補完的 5、0、4）。
# 改成照模擬器量出的「打了幾張、幾組副露 → 幾向聽」抽向聽數，再組出那個向聽數的手牌。

SHANTEN_TABLE = Path(__file__).with_name("shanten_table.json")
MAX_TABLE_DISCARDS = 20
MAX_TABLE_SHANTEN = 6


def _table_sample(args: tuple[int, int]) -> dict[str, list[int]]:
    from agent.rule_agent import choose_rule_action_for_player
    from agent.shanten import shanten

    first, games = args
    table: dict[str, list[int]] = {}

    def policy(state, player, rng):
        if state.phase == Phase.DISCARD and player == state.current_player:
            for seat in range(4):
                other = state.players[seat]
                if seat == player or state.declared[seat] or sum(other.hand) != 16 - 3 * len(other.melds):
                    continue
                key = f"{min(len(other.discards), MAX_TABLE_DISCARDS)},{min(len(other.melds), 4)}"
                level = min(max(shanten(other.hand, len(other.melds)), 0), MAX_TABLE_SHANTEN)
                table.setdefault(key, [0] * (MAX_TABLE_SHANTEN + 1))[level] += 1
        return choose_rule_action_for_player(state, player, rng)

    for seed in range(first, first + games):
        rng = Random(seed)
        play_game(rng, player_policy=policy, swap=True, dealer=rng.randrange(4))
    return table


def build_shanten_table(games: int, workers: int) -> dict[str, list[int]]:
    from multiprocessing import Pool

    chunk = 50
    with Pool(workers) as pool:
        parts = pool.map(_table_sample, [(first, chunk) for first in range(0, games, chunk)])
    table: dict[str, list[int]] = {}
    for part in parts:
        for key, counts in part.items():
            merged = table.setdefault(key, [0] * (MAX_TABLE_SHANTEN + 1))
            for level, count in enumerate(counts):
                merged[level] += count
    return table


_table_cache: dict | None = None


def shanten_weights(discards: int, melds: int) -> list[int]:
    """打了 discards 張、melds 組副露、沒宣告的對手是 0～6 向聽的次數（樣本少時合併相鄰張數）。"""
    global _table_cache
    if _table_cache is None:
        _table_cache = json.loads(SHANTEN_TABLE.read_text(encoding="utf-8")) if SHANTEN_TABLE.exists() else {}
    discards, melds = min(discards, MAX_TABLE_DISCARDS), min(melds, 4)
    total = [0] * (MAX_TABLE_SHANTEN + 1)
    for spread in range(0, MAX_TABLE_DISCARDS + 1):
        for d in {discards - spread, discards + spread}:
            for level, count in enumerate(_table_cache.get(f"{d},{melds}", ())):
                total[level] += count
        if sum(total) >= 200:
            return total
    return total if sum(total) else [0, 1, 2, 3, 2, 1, 0]


def sample_hand_with_shanten(pool: list[int], open_melds: int, level: int, rng: Random) -> list[int] | None:
    """從 pool 組一副 level 向聽的暗手（先組聽牌，再把 level 張換成隨機的牌）；組不出來回 None。不改動 pool。"""
    from agent.shanten import shanten

    best, best_gap = None, None
    for _ in range(8):
        sampled = sample_tenpai_hand(pool, open_melds, rng)
        if sampled is None:
            return None
        hand = list(sampled[0])
        left = [count - hand[tile] for tile, count in enumerate(pool)]
        for _ in range(level):
            held = [tile for tile, count in enumerate(hand) for _ in range(count)]
            out = rng.choice(held)
            spare = [tile for tile, count in enumerate(left) if tile != out for _ in range(count)]
            if not spare:
                break
            into = rng.choice(spare)
            hand[out] -= 1
            left[out] += 1
            hand[into] += 1
            left[into] -= 1
        gap = abs(shanten(hand, open_melds) - level)
        if gap == 0:
            return hand
        if best_gap is None or gap < best_gap:
            best, best_gap = hand, gap
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="重建對手向聽數分布表（agent/shanten_table.json）")
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    started = time.time()
    built = build_shanten_table(args.games, args.workers)
    SHANTEN_TABLE.write_text(json.dumps(built, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"{args.games} 局、{sum(sum(v) for v in built.values())} 筆（{time.time() - started:.0f} 秒）")
