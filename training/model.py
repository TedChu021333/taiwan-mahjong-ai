"""學習型 AI 的網路：1D 殘差卷積網路沿 34 種牌掃描（相鄰的牌組成順子、搭子），
輸出每個動作的分數（策略頭，不合法的動作由遮罩擋掉）與這局預期輸贏（價值頭，台）。

需要 PyTorch（訓練用的虛擬環境 .venv-train，見 training/README）。
"""
from __future__ import annotations

import torch
from torch import nn

from training.actions import ACTION_COUNT
from training.encoding import CHANNELS

VALUE_SCALE = 10.0
"""價值頭輸出 × 10 = 台；輸贏多在 ±30 台內，縮小後比較好訓練。"""
AUX_GLOBAL = ("tenpai_1", "tenpai_2", "tenpai_3", "win", "deal_in")
"""輔助任務（aux=True，11/17）：三家（下家、對家、上家）現在有沒有聽牌、自己這局會不會胡、會不會放槍。
另有每家聽哪幾張（wait_head，3 × 34）。標籤來自模擬器的完整局面（training.teacher_data），
只在訓練時當額外的學習目標，讓共用的網路主體學會看對手；做決定時不用。"""


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class MahjongNet(nn.Module):
    def __init__(self, width: int = 128, blocks: int = 10, aux: bool = False) -> None:
        super().__init__()
        self.config = {"width": width, "blocks": blocks, "aux": aux}
        self.stem = nn.Sequential(nn.Conv1d(CHANNELS, width, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        # 打牌、宣告、槓都是「對某張牌」的動作：每種牌各出 3 個分數；碰吃過胡看整體
        self.tile_head = nn.Conv1d(width, 3, 1)
        self.global_head = nn.Sequential(nn.Flatten(), nn.Linear(width * 34, 256), nn.ReLU())
        self.other_actions = nn.Linear(256, ACTION_COUNT - 3 * 34)
        self.value = nn.Linear(256, 1)
        if aux:
            self.wait_head = nn.Conv1d(width, 3, 1)
            self.aux_global = nn.Linear(256, len(AUX_GLOBAL))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (動作分數 [B, ACTION_COUNT], 預期輸贏 [B]，單位 VALUE_SCALE 台)。"""
        logits, value, _ = self.forward_all(x, with_aux=False)
        return logits, value

    def forward_all(self, x: torch.Tensor, with_aux: bool = True):
        """多回傳輔助任務的 logits：{"waits": [B, 3, 34], "global": [B, len(AUX_GLOBAL)]}（沒有輔助頭時 None）。"""
        h = self.body(self.stem(x))
        per_tile = self.tile_head(h).flatten(1)  # [B, 3*34]：打牌、宣告、槓依序
        g = self.global_head(h)
        logits = torch.cat([per_tile, self.other_actions(g)], dim=1)
        aux = None
        if with_aux and self.config.get("aux"):
            aux = {"waits": self.wait_head(h), "global": self.aux_global(g)}
        return logits, self.value(g).squeeze(1), aux


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def save(model: MahjongNet, path, **extra) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), **extra}, path)


def load(path, device: str = "cpu", aux: bool | None = None) -> MahjongNet:
    """aux 給 True／False 時改成有／沒有輔助頭（從舊網路接著訓練輔助任務；輔助頭從頭學）。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    if aux is not None:
        config["aux"] = aux
    model = MahjongNet(**config)
    state = checkpoint["state"]
    if aux is not None:
        own = model.state_dict()
        state = {key: value for key, value in state.items() if key in own}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state)
    return model.to(device).eval()
