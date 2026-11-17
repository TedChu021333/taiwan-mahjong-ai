"""輔助任務（三家聽牌、聽哪幾張、自己胡／放槍）與「把護欄學進網路」的資料、網路、訓練。"""
from random import Random

import numpy as np
import pytest

from game.rules import GameState, Meld, Phase, PlayerState
from game.tiles import parse, to_counts
from training.encoding import Noise
from training.teacher_data import opponent_labels, play_chunk


def P(value):
    return parse(value)[0]


def test_opponent_labels_mark_tenpai_and_waits_by_relative_seat():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("123456789m1234p11z"))
    players[1].hand = to_counts(parse("123456789m123p4599s"))   # 下家：聽三條、六條
    players[2].hand = to_counts(parse("199m19p19s1234567z15m"))  # 對家：沒聽
    players[3].hand = to_counts(parse("123m456p"))              # 上家：張數不對（只有 6 張）
    state = GameState([0] * 60, players, phase=Phase.DISCARD)
    tenpai, waits, valid = opponent_labels(state, 0)
    assert list(valid) == [True, True, False]
    assert list(tenpai) == [True, False, False]
    assert set(np.nonzero(waits[0])[0]) == {P("3s"), P("6s")} and not waits[1].any()
    # 從下家看：自己（座位 0）變成上家（相對座位 3）
    players[0].hand = to_counts(parse("123456789m123p4599s"))
    tenpai, waits, valid = opponent_labels(state, 1)
    assert tenpai[2] and set(np.nonzero(waits[2])[0]) == {P("3s"), P("6s")}


def test_opponent_labels_count_melds():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = to_counts(parse("123456789m5p"))          # 兩組副露 + 10 張：單吊五筒
    players[1].melds = [Meld("pung", tuple(parse("111z")), 0), Meld("pung", tuple(parse("222z")), 0)]
    state = GameState([0] * 60, players, phase=Phase.DISCARD)
    tenpai, waits, valid = opponent_labels(state, 0)
    assert valid[0] and tenpai[0] and set(np.nonzero(waits[0])[0]) == {P("5p")}


def test_rule_teacher_chunk_records_aux_labels_and_skips_rule_seats(tmp_path):
    out = tmp_path / "data"
    path, games, samples = play_chunk(("rule", 0, 3, str(out), Noise(), 0.5))
    data = np.load(path)
    assert games == 3 and samples == len(data["action"]) > 0
    n = len(data["action"])
    assert data["opp_tenpai"].shape == (n, 3) and data["opp_waits"].shape == (n, 3, 34)
    assert data["win"].shape == (n,) and data["deal_in"].shape == (n,)
    assert not (data["win"] & data["deal_in"]).any()
    # 有聽牌標籤的那家一定聽某幾張
    tenpai = data["opp_tenpai"]
    assert (data["opp_waits"].any(2) == tenpai).all()
    # 已經產生過的檔不重做
    assert play_chunk(("rule", 0, 3, str(out), Noise(), 0.5))[1] == 0


def test_aux_heads_and_loading_an_old_network_with_them(tmp_path):
    torch = pytest.importorskip("torch")
    from training.model import AUX_GLOBAL, MahjongNet, load, save

    old = MahjongNet(width=16, blocks=1)
    path = tmp_path / "old.pt"
    save(old, path)
    model = load(path, aux=True)
    x = torch.zeros(2, *next(old.stem.parameters()).shape[1:2], 34)
    logits, value, aux = model.forward_all(x)
    assert aux["waits"].shape == (2, 3, 34) and aux["global"].shape == (2, len(AUX_GLOBAL))
    # 主體與策略頭照搬舊網路；forward 仍只回傳兩個值（自我對打、提示模式不用改）
    with torch.no_grad():
        assert torch.allclose(model(x)[0], old.eval()(x)[0])
    assert len(model(x)) == 2
    assert load(path).config.get("aux", False) is False


def test_numpy_export_ignores_aux_heads(tmp_path):
    torch = pytest.importorskip("torch")
    from training.model import MahjongNet, save
    from training.numpy_model import NumpyNet, export

    model = MahjongNet(width=16, blocks=1, aux=True).eval()
    pt, npz = tmp_path / "net.pt", tmp_path / "net.npz"
    save(model, pt)
    export(pt, npz)
    planes = np.random.default_rng(0).random((60, 34), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(planes[None]))[0][0].numpy()
    assert np.allclose(NumpyNet.from_file(npz).logits(planes), expected, atol=1e-4)


def test_supervised_training_with_aux_from_an_existing_network(tmp_path, capsys):
    pytest.importorskip("torch")
    from training.model import MahjongNet, load, save
    from training.supervised import train

    files = [play_chunk(("rule", seed, 2, str(tmp_path / "d"), Noise(), 0.0))[0] for seed in range(3)]
    init = tmp_path / "init.pt"
    save(MahjongNet(width=16, blocks=1), init)
    out = tmp_path / "out.pt"
    train([__import__("pathlib").Path(f) for f in files], out, epochs=1, batch_size=64, lr=1e-3,
          width=16, blocks=1, device="cpu", init=init, aux_weight=0.5)
    printed = capsys.readouterr().out
    assert "waits" in printed and "deal_in" in printed
    assert load(out).config["aux"] is True
