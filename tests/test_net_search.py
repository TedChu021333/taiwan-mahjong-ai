"""六秒版：出牌當下補完看不到的牌、網路模擬到局末挑牌（agent.net_search）。"""
from copy import deepcopy
from random import Random

import numpy as np

import agent.net_search as ns
from agent.rule_agent import choose_rule_action_for_player
from agent.shanten import shanten
from game.rules import Action, ActionType, Meld, Phase, legal_actions
from game.simulator import play_game
from game.tiles import parse
from perception.state_parser import ObservedPlayer, ObservedTable, parse_observed_table


def position(seed: int, turn: int = 5):
    """規則式四家打到座位 0 第 turn 次要打牌的局面。"""
    snaps = []

    def policy(state, player, rng):
        if player == 0 and state.phase == Phase.DISCARD and state.current_player == 0 and not state.declared[0]:
            snaps.append(deepcopy(state))
        return choose_rule_action_for_player(state, player, rng)

    play_game(Random(seed), player_policy=policy, swap=True)
    return snaps[min(turn, len(snaps) - 1)]


def tile_counts(state):
    counts = [0] * 34
    for player in state.players:
        for tile, count in enumerate(player.hand):
            counts[tile] += count
        for meld in player.melds:
            for tile in meld.tiles:
                counts[tile] += 1
        for tile in player.discards:
            counts[tile] += 1
    for tile in state.wall:
        counts[tile] += 1
    return counts


def test_complete_state_keeps_what_the_player_sees_and_fills_the_rest():
    for seed in range(6):
        state = position(seed)
        before = deepcopy(state)
        full = ns.complete_state(state, 0, Random(seed))
        assert state.players[1].hand == before.players[1].hand  # 不改動原本的局面
        assert full.players[0].hand == state.players[0].hand
        assert max(tile_counts(full)) <= 4
        for seat, player in enumerate(full.players):
            assert sum(player.hand) == (17 if seat == 0 else 16) - 3 * len(player.melds)
            assert player.discards == state.players[seat].discards
        assert full.drawable == state.drawable and full.reserve > 0
        assert not set(full.gold_pool) & set(full.gold_tiles)


def test_complete_state_fills_melds_read_from_the_screen_without_contents():
    me = ObservedPlayer(hand=tuple(parse("123456789m12345p1s")[:17 - 3]), unknown_melds=1,
                        exposed=tuple(parse("55z")))  # 自己碰了發（畫面上只知道兩張）
    others = [ObservedPlayer(discards=tuple(parse("19m")), unknown_melds=1),
              ObservedPlayer(discards=tuple(parse("1z")), melds=(Meld("chow", tuple(parse("789p")), 3),)),
              ObservedPlayer(discards=tuple(parse("9s")))]
    table = ObservedTable(players=(me, *others), current_player=0, phase=Phase.DISCARD, wall_remaining=40)
    state = parse_observed_table(table)
    full = ns.complete_state(state, 0, Random(3))
    assert full.players[0].melds[0].kind == "pung" and full.players[0].melds[0].tiles == tuple(parse("555z"))
    assert full.players[1].melds[0].kind in ("pung", "chow") and len(full.players[1].melds[0].tiles) == 3
    assert full.players[2].melds[0].tiles == tuple(parse("789p"))
    assert max(tile_counts(full)) <= 4


def test_hands_with_a_given_shanten_come_from_the_pool():
    pool = [4] * 34
    rng = Random(0)
    hits = 0
    for level in (1, 2, 3):
        for _ in range(10):
            hand = ns.sample_hand_with_shanten(pool, 0, level, rng)
            assert sum(hand) == 16 and all(count <= 4 for count in hand)
            hits += shanten(hand, 0) == level
    assert pool == [4] * 34 and hits >= 25


def test_shanten_weights_get_lower_as_the_hand_goes_on():
    early, late = ns.shanten_weights(0, 0), ns.shanten_weights(10, 0)

    def mean(weights):
        return sum(level * count for level, count in enumerate(weights)) / sum(weights)

    assert mean(late) < mean(early)


def test_choose_switches_only_when_clearly_better():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 5, 300)
    same = [[b, b + rng.normal(0, 1), b - 0.1] for b in base]
    assert ns.choose([1, 2, 3], same).tile == 1
    better = [[b, b + 1.0 + rng.normal(0, 1), b] for b in base]
    choice = ns.choose([1, 2, 3], better)
    assert choice.tile == 2 and "多" in choice.note
    assert ns.choose([1, 2], [[0.0, 5.0]] * 10).tile == 1  # 次數太少不換


def test_candidates_put_the_default_first():
    logits = np.full(108, -1e9)
    lookup = {}
    for tile, score in ((3, 1.0), (5, 3.0), (7, 2.0), (9, 0.5)):
        logits[tile] = score
        lookup[tile] = Action(ActionType.DISCARD, tile=tile)
    assert ns.discard_candidates(logits, lookup, Action(ActionType.DISCARD, tile=9)) == [9, 5, 7]
    assert ns.discard_candidates(logits, lookup, Action(ActionType.DISCARD, tile=5)) == [5, 7, 3]


class FirstLegal:
    """假的批次推論：回傳第一個合法動作（測試不需要 GPU）。"""
    deadline = float("inf")

    def ask(self, obs, mask):
        if ns.time.time() > self.deadline:
            raise ns.Deadline
        return int(np.nonzero(mask)[0][0])


def test_search_once_plays_every_candidate_on_the_same_deal():
    state = position(1)
    tiles = [tile for tile in range(34) if state.players[0].hand[tile]][:3]
    values = ns.search_once(ns.BatchedNetwork(FirstLegal()), state, 0, tiles, 1, ns.time.time() + 30, limit=4)
    assert len(values) == 4 and all(len(row) == 3 for row in values)
    # 時間到就停，不會拖過截止時間
    started = ns.time.time()
    ns.search_once(ns.BatchedNetwork(FirstLegal()), state, 0, tiles, 2, started + 0.3)
    assert ns.time.time() - started < 2.0


class FakeSearcher:
    def __init__(self, pick):
        self.pick = pick
        self.calls = []

    def search(self, state, player, tiles, seconds):
        self.calls.append(tiles)
        return ns.Choice(self.pick(tiles), "模擬：測試", 100)


class FakePolicy:
    """網路＋護欄的替身：預設打分數最高的；_guarded 照收到的牌打。"""

    def __init__(self):
        self.last_note = None

    def _scores(self, state, player, mask):
        return np.where(mask, np.arange(108, dtype=float), -1e9)

    def __call__(self, state, player, rng):
        actions = legal_actions(state, player)
        discards = [action for action in actions if action.kind == ActionType.DISCARD]
        return max(discards, key=lambda action: action.tile) if discards else actions[0]

    def _guarded(self, state, player, choice, actions, rng):
        self.last_note = None
        return choice


def test_decide_keeps_the_default_unless_the_search_switches():
    state = position(2)
    policy = FakePolicy()
    default = policy(state, 0, Random(0))
    assert ns.decide(policy, None, state, 0, Random(0), 3.5)[0] == default
    same = FakeSearcher(lambda tiles: tiles[0])
    assert ns.decide(policy, same, state, 0, Random(0), 3.5)[0] == default
    assert same.calls and same.calls[0][0] == default.tile
    other = FakeSearcher(lambda tiles: tiles[1])
    action, note = ns.decide(policy, other, state, 0, Random(0), 3.5)
    assert action.kind == ActionType.DISCARD and action.tile == other.calls[0][1] and "模擬" in note
