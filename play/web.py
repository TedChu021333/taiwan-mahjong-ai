"""網頁版人機對戰的牌局驅動（11/16 使用者要求：別人不用安裝就能和 AI 打）。

瀏覽器裡用 Pyodide 跑這份 Python（規則、計分、AI 都是同一套程式），沒有執行緒，不能像 Tk 版那樣
「牌局執行緒卡著等使用者點」。所以這裡把 game.simulator.play_game 的迴圈拆成一次走一步：
`Session.step()` 走一步、回傳畫面要的資料（JSON 能表示的 dict）；輪到自己要決定時回傳 type="ask"，
網頁呼叫 `answer()` 之後再 `step()`。同一顆亂數種子、同樣的策略，走出來的牌局和 play_game 一模一樣（有測試）。

兩種玩法：full＝和 `python -m play` 一樣（換三張、金牌、連莊、打一圈）；simple＝單局、沒有換三張和金牌。
"""
from __future__ import annotations

from random import Random

from game.rules import Action, ActionType, GameState, Phase, apply_action, initial_state, legal_actions
from game.simulator import (MAX_HANDS, GameResult, PlayerPolicy, _flower_win_result, _win_events, payments,
                            resolve_choices, win_score)
from play.game import HUMAN, MONEY_PER_TAI, SEAT_NAMES, WIND_NAMES, describe_action, score_lines

MODES = ("full", "simple")


class HandStepper:
    """一局，一次走一步。human=None 時四家都是電腦。"""

    def __init__(self, rng: Random, bot: PlayerPolicy, dealer: int = 0, swap: bool = True, dealer_streak: int = 0,
                 gold: bool = True, human: int | None = HUMAN, max_steps: int = 10_000) -> None:
        self.rng = rng
        self.bot = bot
        self.human = human
        self.max_steps = max_steps
        self.state: GameState = initial_state(rng, dealer=dealer, swap=swap, dealer_streak=dealer_streak)
        if not gold:
            self.state.gold_tiles, self.state.gold_pool = [], []
        self.steps = 0
        self.discard_count = 0
        self.discards_by_player = [0, 0, 0, 0]
        self.result: GameResult | None = None
        self.pending: list[Action] | None = None
        """等使用者決定的動作。"""
        self._answer: Action | None = None
        self._choices: dict[int, Action | None] = {}
        """這一步已經決定好的人（使用者還沒回答時，不必重問電腦）。"""
        self._passed: tuple | None = None
        """已經說「不要」的那張牌：沒人吃碰時模擬器一次只收下第一家的「過」，沒輪到的人下一步會再被問。"""

    def answer(self, action: Action) -> None:
        if self.pending is None:
            raise ValueError("現在不是你要決定的時候")
        if action.kind == ActionType.SWAP:
            me = self.state.players[self.human]
            wanted = [0] * 34
            for tile in action.tiles:
                wanted[tile] += 1
            if len(action.tiles) > 3 or any(wanted[t] > me.hand[t] for t in range(34)):
                raise ValueError("換三張：最多選 3 張自己手上的牌")
        elif action not in self.pending:
            raise ValueError("不能這樣做")
        self._answer = action

    def _human_choice(self) -> tuple[bool, Action | None]:
        """回傳 (決定好了沒, 動作)。只有一種選擇時自動做（摸牌、只能過、聽牌後代打）。"""
        state = self.state
        actions = legal_actions(state, self.human)
        if not actions:
            return True, None
        if state.phase != Phase.SWAP and len(actions) == 1:
            return True, actions[0]
        key = None
        if state.phase == Phase.RESPONSE:
            key = (state.discard_player, state.last_discard, state.robbing_kong,
                   tuple(len(p.discards) for p in state.players), tuple(len(p.melds) for p in state.players))
            if key == self._passed:
                return True, next(a for a in actions if a.kind == ActionType.PASS)
        if self._answer is None:
            self.pending = actions
            return False, None
        choice, self._answer, self.pending = self._answer, None, None
        self._passed = key if key is not None and choice.kind == ActionType.PASS else None
        return True, choice

    def _finish(self, **fields) -> None:
        self.result = GameResult(self.state, self.steps, discard_count=self.discard_count,
                                 discards_by_player=tuple(self.discards_by_player), **fields)

    def step(self) -> tuple[int, Action] | None:
        """走一步。回傳 (誰, 做了什麼)；等使用者決定（self.pending）或這局結束（self.result）時回傳 None。"""
        state = self.state
        if self.result is not None:
            return None
        if state.flower_win is not None:
            self.result = _flower_win_result(state, self.steps, [], self.discard_count, self.discards_by_player)
            return None
        if state.phase == Phase.ENDED or self.steps >= self.max_steps:
            state.phase = Phase.ENDED
            self._finish()
            return None
        players = state.response_players if state.phase == Phase.RESPONSE else [state.current_player]
        if self.human in players and self.human not in self._choices:  # 先問人，按鈕馬上出現
            decided, choice = self._human_choice()
            if not decided:
                return None
            self._choices[self.human] = choice
        for player in players:
            if player not in self._choices:
                self._choices[player] = self.bot(state, player, self.rng)
        selected = resolve_choices([(player, self._choices[player]) for player in players])
        self._choices = {}
        if selected is None:
            state.phase = Phase.ENDED
            self._finish()
            return None
        self.steps += 1
        player, action = selected
        win = action.kind == ActionType.WIN
        discarder = state.discard_player if win and state.phase == Phase.RESPONSE else None
        if win:
            events = _win_events(state, self_draw=discarder is None)
            win_tile = action.tile if discarder is not None else state.last_drawn
        if action.kind in (ActionType.DISCARD, ActionType.DECLARE):
            self.discard_count += 1
            self.discards_by_player[player] += 1
        apply_action(state, action, player)
        if win:
            self._finish(winner=player, win_tile=win_tile, win_by_discard=discarder is not None,
                         score=win_score(state, player, discarder, win_tile, events), discarder=discarder)
        return player, action


def _meld_view(meld, hide_concealed: bool) -> dict:
    concealed = meld.kind == "kong" and meld.from_player is None
    tiles = list(meld.tiles)
    if concealed and hide_concealed:
        tiles = [-1] * len(tiles)  # 別家的暗槓整組蓋著（遊戲裡也看不到是哪張）
    return {"kind": meld.kind, "tiles": tiles, "concealed": concealed}


def just_drew(state: GameState, seat: int) -> bool:
    return (state.phase == Phase.DISCARD and state.current_player == seat and not state.claimed
            and state.last_drawn is not None and state.players[seat].hand[state.last_drawn] > 0)


def hand_order(state: GameState, seat: int) -> list[int]:
    """手牌排序，剛摸進的那張放最右邊。"""
    tiles = [t for t in range(34) for _ in range(state.players[seat].hand[t])]
    if just_drew(state, seat):
        tiles.remove(state.last_drawn)
        tiles.append(state.last_drawn)
    return tiles


def action_view(index: int, action: Action, state: GameState) -> dict:
    return {"index": index, "kind": str(action.kind), "tile": action.tile, "tiles": list(action.tiles),
            "self_draw": state.phase == Phase.DISCARD}


class Session:
    """網頁的一圈（或一局）。所有回傳值都是 JSON 能表示的 dict。"""

    def __init__(self, bot: PlayerPolicy, seed: int | None = None, mode: str = "full", watch: bool = False) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 要是 {MODES}")
        self.bot = bot
        self.rng = Random(seed)
        self.mode = mode
        self.human = None if watch else HUMAN
        self.dealer, self.streak, self.passes, self.number = 0, 0, 0, 0
        self.totals = [0, 0, 0, 0]
        self.hand: HandStepper | None = None
        self.over = False

    # ---- 畫面資料 ----
    def view(self, reveal: bool = False) -> dict:
        state = self.hand.state
        players = []
        for seat, player in enumerate(state.players):
            mine = seat == self.human or self.human is None or reveal
            players.append({
                "name": SEAT_NAMES[seat],
                "wind": WIND_NAMES[state.seat_winds[seat]],
                "hand": hand_order(state, seat) if mine else None,
                "count": sum(player.hand),
                "drew": just_drew(state, seat),
                "melds": [_meld_view(meld, hide_concealed=not mine) for meld in player.melds],
                "flowers": list(player.flowers),
                "discards": list(player.discards),
                "declared": bool(state.declared[seat]),
                "total": self.totals[seat],
            })
        if state.phase == Phase.RESPONSE:
            turn = None
        else:
            turn = state.current_player
        return {
            "mode": self.mode, "number": self.number, "dealer": state.dealer, "streak": state.dealer_streak,
            "wall": state.drawable, "gold": list(state.gold_tiles), "phase": str(state.phase),
            "turn": turn, "last_discard": state.last_discard if state.phase == Phase.RESPONSE else None,
            "robbing_kong": bool(state.robbing_kong) and state.phase == Phase.RESPONSE,
            "discard_player": state.discard_player if state.phase == Phase.RESPONSE else None,
            "players": players, "money_per_tai": MONEY_PER_TAI,
        }

    def _start_hand(self) -> None:
        self.number += 1
        full = self.mode == "full"
        self.hand = HandStepper(self.rng, self.bot, dealer=self.dealer, swap=full, dealer_streak=self.streak,
                                gold=full, human=self.human)

    def _hand_end(self) -> dict:
        result = self.hand.result
        pay = payments(result)
        for seat, amount in enumerate(pay):
            self.totals[seat] += amount
        dealer, streak = self.dealer, self.streak
        if self.mode == "simple":
            self.over = True
        elif result.winner is None or result.winner == self.dealer:
            self.streak += 1
        else:
            self.dealer, self.streak, self.passes = (self.dealer + 1) % 4, 0, self.passes + 1
        if self.passes >= 4 or self.number >= MAX_HANDS:
            self.over = True
        if result.winner is None:
            outcome = "流局"
        elif result.win_by_discard:
            outcome = f"{SEAT_NAMES[result.winner]} 胡牌　{SEAT_NAMES[result.discarder]} 放槍"
        else:
            outcome = f"{SEAT_NAMES[result.winner]} 自摸"
        summary = {
            "number": self.number, "dealer": dealer, "streak": streak, "winner": result.winner,
            "discarder": result.discarder, "win_tile": result.win_tile, "outcome": outcome,
            "tai": result.score.tai if result.score is not None else None, "lines": score_lines(result),
            "pay": pay, "totals": list(self.totals), "over": self.over,
        }
        view = self.view(reveal=True)
        self.hand = None
        return {"type": "hand_end", "view": view, "summary": summary}

    # ---- 網頁呼叫的 ----
    def step(self) -> dict:
        if self.over and self.hand is None:
            return {"type": "round_end", "totals": list(self.totals)}
        if self.hand is None:
            self._start_hand()
            return {"type": "deal", "view": self.view(), "message": None}
        hand = self.hand
        moved = hand.step()
        if hand.result is not None:
            last = None
            if moved is not None:
                last = {"player": moved[0], "kind": str(moved[1].kind), "tile": moved[1].tile}
            ended = self._hand_end()
            ended["last"] = last
            return ended
        if moved is None:
            actions = [action_view(i, action, hand.state) for i, action in enumerate(hand.pending)]
            return {"type": "ask", "view": self.view(), "actions": actions}
        player, action = moved
        return {"type": "move", "view": self.view(), "player": player, "kind": str(action.kind),
                "tile": action.tile, "message": describe_action(player, action)}

    def answer(self, index: int | None = None, swap: list[int] | None = None) -> None:
        """index＝step() 回傳的 actions 裡的編號；換三張時給 swap（要換掉的牌，0～3 張）。"""
        hand = self.hand
        if hand is None or hand.pending is None:
            raise ValueError("現在不是你要決定的時候")
        if hand.pending[0].kind == ActionType.SWAP:
            hand.answer(Action(ActionType.SWAP, tiles=tuple(int(t) for t in (swap or []))))
        else:
            hand.answer(hand.pending[int(index)])
