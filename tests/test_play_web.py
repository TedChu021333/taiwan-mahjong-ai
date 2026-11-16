import json
from random import Random

from agent.rule_agent import choose_rule_action_for_player as rule_bot
from game.rules import ActionType
from game.simulator import play_game, play_match
from play.game import HUMAN
from play.web import HandStepper, Session


def run(stepper):
    while stepper.result is None:
        stepper.step()
    return stepper.result


def test_stepping_plays_the_same_hand_as_the_simulator():
    """網頁版一次走一步，同一顆種子要和 play_game 打出一模一樣的結果。"""
    for seed in range(12):
        expected = play_game(Random(seed), player_policy=rule_bot, dealer=seed % 4, swap=True, dealer_streak=seed % 3)
        result = run(HandStepper(Random(seed), rule_bot, dealer=seed % 4, swap=True, dealer_streak=seed % 3, human=None))
        assert (result.winner, result.steps, result.win_tile, result.discarder, result.win_by_discard) == \
            (expected.winner, expected.steps, expected.win_tile, expected.discarder, expected.win_by_discard)
        assert (result.score.tai if result.score else None) == (expected.score.tai if expected.score else None)
        assert result.discards_by_player == expected.discards_by_player
        assert result.state.players[0].hand == expected.state.players[0].hand


def test_session_round_matches_play_match_totals():
    session = Session(rule_bot, seed=5, watch=True)
    kinds = []
    while True:
        out = session.step()
        json.dumps(out)  # 回傳值都要能轉成 JSON 給網頁
        kinds.append(out["type"])
        if out["type"] == "round_end":
            break
    assert kinds[0] == "deal" and "hand_end" in kinds and "ask" not in kinds
    assert sum(out["totals"]) == 0 and any(out["totals"])


def test_human_is_asked_and_answers():
    session = Session(rule_bot, seed=3)
    asked = set()
    for _ in range(5000):
        out = session.step()
        if out["type"] == "ask":
            view = out["view"]
            assert view["players"][HUMAN]["hand"] is not None and view["players"][1]["hand"] is None
            kinds = {a["kind"] for a in out["actions"]}
            asked |= kinds
            if kinds == {str(ActionType.SWAP)}:
                session.answer(swap=view["players"][HUMAN]["hand"][:2])
            else:
                wins = [a for a in out["actions"] if a["kind"] == str(ActionType.WIN)]
                discards = [a for a in out["actions"] if a["kind"] == str(ActionType.DISCARD)]
                passes = [a for a in out["actions"] if a["kind"] == str(ActionType.PASS)]
                session.answer(index=(wins or discards or passes or out["actions"])[0]["index"])
        elif out["type"] == "hand_end":
            assert all(p["hand"] is not None for p in out["view"]["players"])  # 結算亮四家的牌
            assert sum(out["summary"]["pay"]) == 0
        elif out["type"] == "round_end":
            break
    else:
        raise AssertionError("一圈沒打完")
    assert str(ActionType.SWAP) in asked and str(ActionType.DISCARD) in asked


def test_simple_mode_is_one_hand_without_gold_or_swap():
    session = Session(rule_bot, seed=1, mode="simple", watch=True)
    out = session.step()
    assert out["view"]["gold"] == [] and out["view"]["phase"] != "swap"
    ends = 0
    while out["type"] != "round_end":
        out = session.step()
        ends += out["type"] == "hand_end"
    assert ends == 1


def test_wrong_answers_are_rejected():
    import pytest

    session = Session(rule_bot, seed=3)
    out = session.step()
    while out["type"] != "ask":
        out = session.step()
    with pytest.raises(ValueError):
        session.answer(swap=[0, 0, 0, 0, 0])


def test_opponent_concealed_kong_is_hidden():
    from game.rules import Meld
    from play.web import _meld_view

    meld = Meld("kong", (5, 5, 5, 5), None)
    assert _meld_view(meld, hide_concealed=True)["tiles"] == [-1, -1, -1, -1]
    assert _meld_view(meld, hide_concealed=False)["tiles"] == [5, 5, 5, 5]
