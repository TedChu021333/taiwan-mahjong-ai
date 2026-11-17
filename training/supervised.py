"""監督學習：網路模仿老師 AI 的選擇（策略頭），同時學預測這局輸贏（價值頭）。

    .venv-train/Scripts/python -m training.supervised --data training/data/rule --out models/sl_rule.pt

資料檔逐批讀入（全部載入要十幾 GB），每次混 BUFFER_FILES 個檔打散；最後 VALIDATION_FILES 個檔
不訓練，用來看「和老師選一樣」的比例（top-1 一致率）與價值誤差。

把護欄學進網路（11/17）：老師是 nnguard（training.teacher_data --teacher nnguard:...），從原本的網路接著訓練：

    .venv-train/Scripts/python -m training.supervised --data training/data/guard/guard \
        --init models/rl3_0960.pt --lr 3e-4 --epochs 2 --aux-weight 0.5 --out models/guard_aux.pt

--aux-weight > 0 時網路多輔助頭（training.model.AUX_GLOBAL），同時學三家聽牌、聽哪幾張、自己會不會胡或放槍。
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.model import VALUE_SCALE, MahjongNet, load, masked_logits, save

BUFFER_FILES = 6
VALIDATION_FILES = 2
VALUE_WEIGHT = 0.5


KEYS = ("obs", "mask", "action", "value")
AUX_KEYS = ("opp_tenpai", "opp_waits", "opp_valid", "win", "deal_in")


def load_files(paths: list[Path], aux: bool = False) -> dict[str, np.ndarray]:
    parts = [np.load(path) for path in paths]
    return {key: np.concatenate([part[key] for part in parts]) for key in KEYS + (AUX_KEYS if aux else ())}


def batches(data: dict[str, np.ndarray], size: int, shuffle: bool, device: str):
    """每批 (obs, mask, action, value, aux)；aux 是輔助標籤的 dict（資料沒有時 None）。"""
    order = np.random.permutation(len(data["action"])) if shuffle else np.arange(len(data["action"]))
    for start in range(0, len(order), size):
        index = order[start:start + size]
        aux = None
        if "opp_tenpai" in data:
            aux = {key: torch.from_numpy(data[key][index]).to(device) for key in AUX_KEYS}
        yield (torch.from_numpy(data["obs"][index].astype(np.float32)).to(device),
               torch.from_numpy(data["mask"][index]).to(device),
               torch.from_numpy(data["action"][index].astype(np.int64)).to(device),
               torch.from_numpy(data["value"][index]).to(device) / VALUE_SCALE,
               aux)


def aux_losses(predicted: dict, labels: dict) -> dict[str, torch.Tensor]:
    """輔助任務的 BCE：三家聽哪幾張、三家有沒有聽牌（只算標籤能用的那幾家）、自己會不會胡、會不會放槍。"""
    bce = nn.functional.binary_cross_entropy_with_logits
    valid = labels["opp_valid"].float()
    waits = bce(predicted["waits"], labels["opp_waits"].float(), reduction="none").mean(2)
    tenpai = bce(predicted["global"][:, :3], labels["opp_tenpai"].float(), reduction="none")
    denominator = valid.sum().clamp(min=1)
    return {
        "waits": (waits * valid).sum() / denominator,
        "tenpai": (tenpai * valid).sum() / denominator,
        "win": bce(predicted["global"][:, 3], labels["win"].float()),
        "deal_in": bce(predicted["global"][:, 4], labels["deal_in"].float()),
    }


def aux_total(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    # 聽哪幾張是 34 格各自的 BCE、正例很少，乘 10 讓它和其他幾項差不多大
    return 10 * losses["waits"] + losses["tenpai"] + losses["win"] + losses["deal_in"]


def evaluate(model: MahjongNet, data: dict[str, np.ndarray], device: str) -> tuple[float, float, dict]:
    """(一致率, 價值均方誤差, 輔助任務各項平均 BCE)。"""
    model.eval()
    correct, total, value_error = 0, 0, 0.0
    aux_sum: dict[str, float] = {}
    with torch.no_grad():
        for obs, mask, action, value, aux in batches(data, 2048, False, device):
            logits, predicted, heads = model.forward_all(obs)
            correct += (masked_logits(logits, mask).argmax(1) == action).sum().item()
            value_error += ((predicted - value) ** 2).sum().item()
            total += len(action)
            if heads is not None and aux is not None:
                for key, loss in aux_losses(heads, aux).items():
                    aux_sum[key] = aux_sum.get(key, 0.0) + loss.item() * len(action)
    model.train()
    return correct / total, value_error / total, {key: value / total for key, value in aux_sum.items()}


def train(files: list[Path], out: Path, epochs: int, batch_size: int, lr: float,
          width: int, blocks: int, device: str, init: Path | None = None, aux_weight: float = 0.0) -> None:
    validation, training = files[-VALIDATION_FILES:], files[:-VALIDATION_FILES]
    aux = aux_weight > 0
    held_out = load_files(validation, aux)
    if init is not None:
        model = load(init, device, aux=aux).train()
    else:
        model = MahjongNet(width, blocks, aux=aux).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = sum(len(np.load(path)["action"]) for path in training) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs * steps_per_epoch + 10)
    policy_loss, value_loss = nn.CrossEntropyLoss(), nn.MSELoss()
    started = time.time()
    print(f"訓練 {len(training)} 檔、驗證 {len(validation)} 檔（{len(held_out['action'])} 筆），"
          f"每輪約 {steps_per_epoch} 步", flush=True)
    for epoch in range(epochs):
        random.shuffle(training)
        seen, running = 0, 0.0
        for group in range(0, len(training), BUFFER_FILES):
            data = load_files(training[group:group + BUFFER_FILES], aux)
            for obs, mask, action, value, labels in batches(data, batch_size, True, device):
                logits, predicted, heads = model.forward_all(obs, with_aux=aux)
                loss = policy_loss(masked_logits(logits, mask), action) \
                    + VALUE_WEIGHT * value_loss(predicted, value)
                if aux:
                    loss = loss + aux_weight * aux_total(aux_losses(heads, labels))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()
                running += loss.item() * len(action)
                seen += len(action)
        accuracy, value_mse, aux_bce = evaluate(model, held_out, device)
        extra = "".join(f"，{key} {value:.4f}" for key, value in aux_bce.items())
        print(f"第 {epoch + 1} 輪：訓練損失 {running / seen:.3f}，驗證一致率 {accuracy:.1%}，"
              f"價值均方誤差 {value_mse:.3f}{extra}（{time.time() - started:.0f} 秒）", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        save(model, out, epoch=epoch + 1, accuracy=accuracy, value_mse=value_mse)


def main() -> None:
    from training.selfplay import keep_awake

    keep_awake()
    parser = argparse.ArgumentParser(description="監督學習：模仿老師 AI")
    parser.add_argument("--data", default="training/data/rule", help="資料檔前綴")
    parser.add_argument("--out", type=Path, default=Path("models/sl_rule.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="只用前幾個檔（測試用）")
    parser.add_argument("--init", type=Path, help="從這個網路接著訓練（寬度、層數照它）")
    parser.add_argument("--aux-weight", type=float, default=0.0, help="輔助任務損失的權重（0 = 不加輔助頭）")
    args = parser.parse_args()
    prefix = Path(args.data)
    files = sorted(prefix.parent.glob(prefix.name + "_*.npz"))
    if args.limit:
        files = files[:args.limit]
    if len(files) <= VALIDATION_FILES:
        raise SystemExit(f"資料檔太少：{len(files)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"裝置 {device}", flush=True)
    train(files, args.out, args.epochs, args.batch, args.lr, args.width, args.blocks, device,
          args.init, args.aux_weight)


if __name__ == "__main__":
    main()
