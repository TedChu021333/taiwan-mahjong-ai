"""網路的 numpy 版：只做推論、不需要 PyTorch，網頁版（Pyodide，瀏覽器裡的 Python）用。

權重由 `python -m training.numpy_model models/rl3_0960.pt web/model.npz` 匯出（這一步要 PyTorch）：
BatchNorm 併進前面的卷積，存成 float32；輸出和 training.model.MahjongNet 相同（測試比對到 1e-3）。
"""
from __future__ import annotations

import sys

import numpy as np

BN_EPS = 1e-5


def _conv3(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1D 卷積（寬 3、補零 1）：x [C, 34]、weight [3, O, C]（先照位置排，三次矩陣乘法）。"""
    padded = np.zeros((x.shape[0], x.shape[1] + 2), dtype=x.dtype)
    padded[:, 1:-1] = x
    length = x.shape[1]
    out = weight[0] @ padded[:, 0:length]
    out += weight[1] @ padded[:, 1:length + 1]
    out += weight[2] @ padded[:, 2:length + 2]
    out += bias[:, None]
    return out


class NumpyNet:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.w = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in weights.items()}
        self.blocks = int(sum(1 for name in self.w if name.endswith(".conv1.w")))

    @classmethod
    def from_file(cls, path) -> "NumpyNet":
        with np.load(path) as data:
            return cls({name: data[name] for name in data.files})

    def forward(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        """planes [CHANNELS, 34] → (動作分數 [ACTION_COUNT], 預期輸贏（單位 VALUE_SCALE 台）)。"""
        w = self.w
        h = np.maximum(_conv3(planes.astype(np.float32), w["stem.w"], w["stem.b"]), 0)
        for i in range(self.blocks):
            y = np.maximum(_conv3(h, w[f"body.{i}.conv1.w"], w[f"body.{i}.conv1.b"]), 0)
            y = _conv3(y, w[f"body.{i}.conv2.w"], w[f"body.{i}.conv2.b"])
            h = np.maximum(h + y, 0)
        per_tile = (w["tile_head.w"] @ h + w["tile_head.b"][:, None]).reshape(-1)
        g = np.maximum(w["global.w"] @ h.reshape(-1) + w["global.b"], 0)
        other = w["other.w"] @ g + w["other.b"]
        value = float(w["value.w"] @ g) + float(w["value.b"][0])
        return np.concatenate([per_tile, other]), value

    def logits(self, planes: np.ndarray) -> np.ndarray:
        return self.forward(planes)[0]


def fold(state: dict) -> dict[str, np.ndarray]:
    """PyTorch 的 state_dict → numpy 權重（BatchNorm 併進卷積；卷積權重改成 [3, O, C]）。"""
    def array(name: str) -> np.ndarray:
        return state[name].detach().cpu().numpy().astype(np.float64)

    def conv_bn(conv: str, bn: str) -> tuple[np.ndarray, np.ndarray]:
        scale = array(f"{bn}.weight") / np.sqrt(array(f"{bn}.running_var") + BN_EPS)
        weight = array(f"{conv}.weight") * scale[:, None, None]
        bias = array(f"{bn}.bias") - array(f"{bn}.running_mean") * scale
        return weight.transpose(2, 0, 1).astype(np.float32), bias.astype(np.float32)

    out: dict[str, np.ndarray] = {}
    out["stem.w"], out["stem.b"] = conv_bn("stem.0", "stem.1")
    block = 0
    while f"body.{block}.conv1.weight" in state:
        for n in (1, 2):
            weight, bias = conv_bn(f"body.{block}.conv{n}", f"body.{block}.bn{n}")
            out[f"body.{block}.conv{n}.w"], out[f"body.{block}.conv{n}.b"] = weight, bias
        block += 1
    out["tile_head.w"] = array("tile_head.weight")[:, :, 0].astype(np.float32)
    out["tile_head.b"] = array("tile_head.bias").astype(np.float32)
    out["global.w"] = array("global_head.1.weight").astype(np.float32)
    out["global.b"] = array("global_head.1.bias").astype(np.float32)
    out["other.w"] = array("other_actions.weight").astype(np.float32)
    out["other.b"] = array("other_actions.bias").astype(np.float32)
    out["value.w"] = array("value.weight")[0].astype(np.float32)
    out["value.b"] = array("value.bias").astype(np.float32)
    return out


def export(checkpoint_path, out_path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    np.savez(out_path, **fold(checkpoint["state"]))


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
    print(f"已匯出 {sys.argv[2]}")
