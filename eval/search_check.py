"""驗證六秒版（出牌當下模擬挑牌，agent.net_search）在模擬器裡是不是真的打得比較好。

    .venv-train/Scripts/python -m eval.search_check --positions 1000 --seconds 3.5 --workers 12

1. 收集局面：座位 0 是 nnguard（網路＋護欄）、三家規則式（和 arena 一樣），每局隨機抽一個座位 0 要打牌的局面。
2. 搜尋：每個局面照實戰的條件跑一次 net_search.decide（同樣的秒數與程序數），記下有沒有換牌。
   模擬只用座位 0 看得到的資訊（對手暗手、牌牆都是重新補的）。
3. 驗證（只看有換牌的局面）：用真正的局面——三家真的暗手——只把牌牆重洗，換牌前（預設）與換牌後各打
   --replays 次到局末（同一個種子配對；之後座位 0 照網路＋護欄、三家規則式），比較座位 0 扣抽成後的輸贏。
   這和搜尋用的世界不同（真的對手是規則式、手牌是真的），所以不會因為「用同一個模型驗證自己」而偏高。

結果：換牌的比例、每次換牌多贏幾台、平均每個打牌決定多贏幾台（都附 95% 信賴區間）。
中間結果存在 --cache（預設 result/search_check.pkl），中斷重跑會從還沒做完的步驟接著做。
"""
from __future__ import annotations

import argparse
import math
import pickle
import time
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from random import Random

from game.rules import Action, ActionType, GameState, Phase, apply_action, legal_actions
from game.simulator import after_rake, payments, play_game

MODEL = "models/guard_plain.pt"
_policies: dict = {}


def _policy(model: str):
    if model not in _policies:
        from eval.arena import make_policy

        _policies[model] = make_policy(f"nnguard:{model}")
    return _policies[model]


def _mixed(model: str):
    from agent.rule_agent import choose_rule_action_for_player

    network = _policy(model)

    def policy(state, player, rng):
        if player == 0:
            return network(state, player, rng)
        return choose_rule_action_for_player(state, player, rng)
    return policy


def eligible(state: GameState, player: int = 0) -> bool:
    """座位 player 要打牌、還沒按聽、至少有兩種牌可以打。"""
    if state.phase != Phase.DISCARD or state.current_player != player or state.declared[player]:
        return False
    tiles = {action.tile for action in legal_actions(state, player) if action.kind == ActionType.DISCARD}
    return len(tiles) >= 2


def collect_one(args: tuple[str, int]) -> GameState | None:
    """打一局，隨機回傳一個座位 0 要打牌的局面（打之前；含三家真的暗手與牌牆）。"""
    model, seed = args
    rng = Random(seed)
    network = _mixed(model)
    found: list[GameState] = []

    def policy(state, player, game_rng):
        if player == 0 and eligible(state):
            found.append(deepcopy(state))
        return network(state, player, game_rng)

    play_game(rng, player_policy=policy, swap=True, dealer=rng.randrange(4))
    return Random(seed + 7).choice(found) if found else None


def reshuffled(state: GameState, seed: int) -> GameState:
    """同一個局面，只把牌牆重洗（張數、留牌張數不變）。"""
    copy = deepcopy(state)
    Random(seed).shuffle(copy.wall)
    return copy


def replay(args: tuple[str, GameState, Action, int]) -> float:
    """在重洗牌牆的局面打出 action，之後照 nnguard／規則式打到局末：座位 0 扣抽成後的輸贏。"""
    model, state, action, seed = args
    start = reshuffled(state, seed)
    apply_action(start, action, 0)
    result = play_game(Random(seed), player_policy=_mixed(model), start=start)
    return after_rake(payments(result)[0])


def interval(values: list[float]) -> tuple[float, float]:
    """平均與 95% 信賴區間半寬。"""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0), float("nan")
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=3.5, help="每手模擬幾秒（6 秒場約 3.5）")
    parser.add_argument("--workers", type=int, default=12, help="模擬程序數（搜尋）")
    parser.add_argument("--cpu-workers", type=int, default=14, help="收集局面與驗證用的程序數")
    parser.add_argument("--replays", type=int, default=40, help="驗證時每個動作重洗牌牆打幾次")
    parser.add_argument("--cache", type=Path, default=Path("result/search_check.pkl"))
    args = parser.parse_args()

    cache = pickle.loads(args.cache.read_bytes()) if args.cache.exists() else {}

    if "positions" not in cache:
        started = time.time()
        with Pool(args.cpu_workers) as pool:
            found = pool.map(collect_one, [(args.model, seed) for seed in range(args.positions)], chunksize=4)
        cache["positions"] = [state for state in found if state is not None]
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_bytes(pickle.dumps(cache))
        print(f"收集 {len(cache['positions'])} 個局面（{time.time() - started:.0f} 秒）", flush=True)
    positions: list[GameState] = cache["positions"]

    decisions = cache.setdefault("decisions", [])
    if len(decisions) < len(positions):
        from agent.net_search import Searcher, decide

        policy = _policy(args.model)
        searcher = Searcher(args.model, args.workers)
        started = time.time()
        try:
            for index in range(len(decisions), len(positions)):
                state = positions[index]
                rng = Random(index)
                default = policy(deepcopy(state), 0, Random(index))
                action, note = decide(policy, searcher, deepcopy(state), 0, rng, args.seconds)
                decisions.append((default, action, note))
                if (index + 1) % 50 == 0 or index + 1 == len(positions):
                    args.cache.write_bytes(pickle.dumps(cache))
                    switched = sum(1 for d, a, _ in decisions if d != a)
                    print(f"搜尋 {index + 1}/{len(positions)}：換牌 {switched}（{time.time() - started:.0f} 秒）",
                          flush=True)
        finally:
            searcher.close()

    switched = [index for index, (default, action, _) in enumerate(decisions)
                if default != action and default is not None and action is not None]
    gains = cache.setdefault("gains", {})
    todo = [index for index in switched if index not in gains]
    if todo:
        started = time.time()
        jobs = []
        for index in todo:
            default, action, _ = decisions[index]
            for k in range(args.replays):
                seed = index * 1000 + k
                jobs.append((args.model, positions[index], default, seed))
                jobs.append((args.model, positions[index], action, seed))
        with Pool(args.cpu_workers) as pool:
            values = pool.map(replay, jobs, chunksize=8)
        for offset, index in enumerate(todo):
            part = values[offset * 2 * args.replays:(offset + 1) * 2 * args.replays]
            diffs = [part[2 * k + 1] - part[2 * k] for k in range(args.replays)]
            gains[index] = sum(diffs) / len(diffs)
        args.cache.write_bytes(pickle.dumps(cache))
        print(f"驗證 {len(todo)} 個換牌的局面（{time.time() - started:.0f} 秒）", flush=True)

    per_switch = [gains[index] for index in switched]
    mean, half = interval(per_switch)
    per_decision = [gains.get(index, 0.0) for index in range(len(decisions))]
    total, total_half = interval(per_decision)
    print(f"局面 {len(decisions)}，換牌 {len(switched)}（{len(switched) / max(1, len(decisions)):.1%}）")
    print(f"每次換牌多贏 {mean:+.3f} ± {half:.3f} 台")
    print(f"平均每個打牌決定 {total:+.4f} ± {total_half:.4f} 台（每局約 9 次打牌 → 約 {9 * total:+.2f} 台）")


if __name__ == "__main__":
    main()
