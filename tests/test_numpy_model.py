from pathlib import Path
from random import Random

import numpy as np
import pytest

from agent.nn_agent import NetworkPolicy
from game.simulator import play_game
from training.encoding import CHANNELS
from training.numpy_model import NumpyNet, fold

torch = pytest.importorskip("torch")
MODEL = Path("models/rl3_0960.pt")


def test_numpy_network_matches_pytorch():
    """網頁版沒有 PyTorch：numpy 版（BatchNorm 併進卷積）的輸出要和原本的網路一樣。"""
    from training.model import MahjongNet

    torch.manual_seed(0)
    model = MahjongNet(width=16, blocks=2).eval()
    for module in model.modules():  # 讓 BatchNorm 的統計量不是初始值
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 2.0)
    planes = np.random.default_rng(0).random((CHANNELS, 34), dtype=np.float32)
    with torch.no_grad():
        logits, value = model(torch.from_numpy(planes[None]))
    mine, my_value = NumpyNet(fold(model.state_dict())).forward(planes)
    assert np.allclose(mine, logits[0].numpy(), atol=1e-4)
    assert abs(my_value - float(value[0])) < 1e-4


@pytest.mark.skipif(not MODEL.exists(), reason="沒有模型檔")
def test_numpy_policy_plays_the_same_game_as_the_pytorch_policy():
    from training.model import load

    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    results = []
    for model in (load(MODEL), NumpyNet(fold(checkpoint["state"]))):
        policy = NetworkPolicy(model)
        result = play_game(Random(7), player_policy=policy, swap=True)
        results.append((result.winner, result.steps, result.score.tai if result.score else None))
    assert results[0] == results[1]
