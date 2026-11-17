"""產生監督學習資料：讓老師 AI 在模擬器四家互打，記下每個要做判斷的決定。

每筆：輸入平面（training.encoding，含模仿實戰辨識缺陷的雜訊）、合法動作遮罩、老師選的動作、
這位玩家這局扣抽成後的輸贏（台，價值頭的目標）。只有兩個以上合法動作時才記（只能打摸進的牌、
只能摸牌不需要判斷）。換三張與摸牌不交給網路。

輔助任務的標籤（11/17，training.model.AUX_GLOBAL）：當下三家（相對座位 1～3）有沒有聽牌、聽哪幾張
（opp_tenpai、opp_waits；張數不對、算不出來的 opp_valid=False），以及這局結果自己有沒有胡、有沒有放槍。

老師可以是 rule、ev，或 nnguard:<權重>（網路＋實戰護欄：把護欄的判斷學進網路）、nn:<權重>。
--rule-share：每局每個座位有這個機率換成規則式 AI（不記錄），讓老師見到不同的對手。

    python -m training.teacher_data --games 20000 --workers 2 --teacher rule --out training/data/rule
    python -m training.teacher_data --games 60000 --workers 14 --teacher nnguard:models/rl3_0960.pt \
        --rule-share 0.25 --out training/data/guard
資料分成多個 .npz 檔（每個工作程序每 CHUNK_GAMES 局一檔），np.savez_compressed 壓縮。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from random import Random

import numpy as np

from functools import lru_cache

from agent.shanten import shanten
from game.rules import legal_actions
from game.simulator import after_rake, payments, play_game
from game.win import winning_tiles
from training.actions import ACTION_COUNT, action_index, handled_by_network, legal_mask
from training.encoding import CHANNELS, LIVE_NOISE, Noise, encode

CHUNK_GAMES = 500


def teacher_policy(name: str):
    if name.startswith(("nn:", "nnguard:")):
        from eval.arena import make_policy  # 每個程序載入一次、一條執行緒
        return make_policy(name)
    if name == "rule":
        from agent.rule_agent import choose_rule_action_for_player
        return choose_rule_action_for_player
    if name == "ev":
        from agent.ev_agent import choose_ev_action_for_player
        return choose_ev_action_for_player
    raise ValueError(f"未知的老師：{name}")


@lru_cache(maxsize=200_000)
def _waits(hand: tuple[int, ...], n_open: int) -> tuple[int, ...] | None:
    """聽哪幾張（沒聽牌是空的）；張數不對（不是等著摸牌的 16 張型）回 None。"""
    if sum(hand) != 16 - 3 * n_open:
        return None
    if shanten(hand, n_open) > 0:
        return ()
    return tuple(winning_tiles(list(hand), n_open))


def opponent_labels(state, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三家（下家、對家、上家）的 (有沒有聽牌 [3], 聽哪幾張 [3, 34], 標籤能不能用 [3])。"""
    tenpai = np.zeros(3, dtype=bool)
    waits = np.zeros((3, 34), dtype=bool)
    valid = np.zeros(3, dtype=bool)
    for rel in range(1, 4):
        other = state.players[(player + rel) % 4]
        result = _waits(tuple(other.hand[:34]), other.open_melds)
        if result is None:
            continue
        valid[rel - 1] = True
        tenpai[rel - 1] = bool(result)
        waits[rel - 1, list(result)] = True
    return tenpai, waits, valid


@dataclass
class Recorder:
    """包住老師的 policy，記下每個要判斷的決定（之後補上這局的輸贏）。
    seats 之外的座位由 other（規則式）打、不記錄。"""
    policy: object
    noise: Noise
    rng: Random
    other: object = None
    seats: frozenset = frozenset(range(4))
    obs: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    players: list = field(default_factory=list)
    tenpai: list = field(default_factory=list)
    waits: list = field(default_factory=list)
    valid: list = field(default_factory=list)

    def __call__(self, state, player, rng):
        if player not in self.seats:
            return self.other(state, player, rng)
        action = self.policy(state, player, rng)
        if action is None or not handled_by_network(action):
            return action
        mask, _ = legal_mask(legal_actions(state, player))
        if mask.sum() >= 2:
            self.obs.append(encode(state, player, self.noise, self.rng).astype(np.float16))
            self.masks.append(mask)
            self.actions.append(action_index(action))
            self.players.append(player)
            tenpai, waits, valid = opponent_labels(state, player)
            self.tenpai.append(tenpai)
            self.waits.append(waits)
            self.valid.append(valid)
        return action


def play_chunk(args: tuple) -> tuple[str, int, int]:
    teacher, seed, games, out, noise, rule_share = (*args, 0.0)[:6]
    path = Path(f"{out}_{seed:05d}.npz")
    if path.exists():  # 中斷重跑時跳過已完成的檔
        return str(path), 0, 0
    from agent.rule_agent import choose_rule_action_for_player

    recorder = Recorder(teacher_policy(teacher), noise, Random(seed + 1), choose_rule_action_for_player)
    seat_rng = Random(seed + 2)
    values, wins, deal_ins = [], [], []
    for index in range(games):
        start = len(recorder.actions)
        rng = Random(seed * 1_000_003 + index)
        seats = frozenset(seat for seat in range(4) if seat_rng.random() >= rule_share)
        recorder.seats = seats or frozenset({seat_rng.randrange(4)})
        result = play_game(rng, player_policy=recorder, swap=True, dealer=rng.randrange(4))
        net = [after_rake(amount) for amount in payments(result)]
        for player in recorder.players[start:]:
            values.append(net[player])
            wins.append(result.winner == player)
            deal_ins.append(result.winner is not None and result.win_by_discard and result.discarder == player)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.stack(recorder.obs) if recorder.obs else np.zeros((0, CHANNELS, 34), np.float16),
        mask=np.stack(recorder.masks) if recorder.masks else np.zeros((0, ACTION_COUNT), bool),
        action=np.asarray(recorder.actions, dtype=np.int16),
        value=np.asarray(values, dtype=np.float32),
        opp_tenpai=np.asarray(recorder.tenpai, dtype=bool).reshape(-1, 3),
        opp_waits=np.asarray(recorder.waits, dtype=bool).reshape(-1, 3, 34),
        opp_valid=np.asarray(recorder.valid, dtype=bool).reshape(-1, 3),
        win=np.asarray(wins, dtype=bool),
        deal_in=np.asarray(deal_ins, dtype=bool))
    return str(path), games, len(recorder.actions)


def main() -> None:
    try:  # 不讓電腦睡眠（見 training.selfplay.keep_awake；這裡不載入 PyTorch）
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="產生監督學習資料")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--teacher", default="rule", help="rule、ev、nn:<權重>、nnguard:<權重>")
    parser.add_argument("--rule-share", type=float, default=0.0, help="每個座位換成規則式 AI（不記錄）的機率")
    parser.add_argument("--out", default="training/data/rule")
    parser.add_argument("--seed", type=int, default=0, help="第一個檔的編號（續產生時接著編）")
    parser.add_argument("--no-noise", action="store_true", help="不模仿實戰辨識缺陷")
    args = parser.parse_args()
    noise = Noise() if args.no_noise else LIVE_NOISE
    chunks = [(args.teacher, args.seed + index, min(CHUNK_GAMES, args.games - index * CHUNK_GAMES),
               args.out, noise, args.rule_share)
              for index in range((args.games + CHUNK_GAMES - 1) // CHUNK_GAMES)]
    started, total_games, total_samples = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for path, games, samples in pool.imap_unordered(play_chunk, chunks):
            total_games += games
            total_samples += samples
            print(f"{path}：{games} 局 {samples} 筆（累計 {total_games} 局 {total_samples} 筆，"
                  f"{time.time() - started:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
