"""畫面辨識的準確率（給報告、口試用的總表）。

    python -m eval.vision [--md docs/vision.md]

兩種資料：
1. 人工標註的截圖（`train/經典_*.png`，答案和 tests/ 裡的一樣）。這些截圖有些也被切成模板，直接讀會偏樂觀，所以：
   手牌用「留一驗證」（每張只用其他、不相鄰畫面的牌當模板）；牌河、副露把人工標註的模板（manual_*）拿掉再讀。
2. 實戰記錄檔（`result/log_*.txt`）：自己打的每一張牌都有答案（AI 的決定），看牌河追蹤之後有沒有讀到、讀對沒有。
   樣本數是幾千張，最能代表實際表現。

train/ 和 result/ 都不在 repo 裡（遊戲截圖、個人戰績），沒有的部分會略過。
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from game.tiles import parse, tile_name

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train"
RESULT = ROOT / "result"

HAND_LABELS = {  # 同 tests/test_hand_reader.py：暗手牌由左到右，含第 17 張
    "經典_a20": "1123789m4p567789s37z", "經典_a45": "112237899m4p567789s", "經典_a75": "115789m4p567789s",
    "經典_a124": "2337m7p12234688s775z", "經典_a133": "2337m7p12234688s", "經典_a200": "7p1223489s77z",
    "經典_a280": "12289s77z", "經典_a283": "12289s772z", "經典_b10": "34489m1789p2458s465z",
    "經典_b43": "3344899m57889p2458s", "經典_b72": "3344899m557889p458s2p", "經典_b99": "3344899m557889p458s4m",
    "經典_c67": "22344688m22345567p",
}
RIVER_LABELS = {  # 同 tests/test_river_reader.py
    "經典_牌河290": {"top": "3z1p6z2p3z1z2s9s5z", "me": "5z7m5z3m9p7p9p2z", "left": "9s4s5s3m",
                   "right": "2p9s6z1p1s6z1s9p"},
    "經典_牌河第二排": {"me": "676m6z44z3z9p5p8m4z2z1p", "right": "1s1p6p2m6z2p9m7z7p1p2p8m",
                     "top": "9p7p2z2s2z6m7z4p7p7p", "left": "7z6z9m3p9m5z9m3z9p7z"},
    "經典_牌河對家第二排": {"me": "3z8m1z1z2m7m1s8m4m1m", "right": "4p6z1m9p5s2m5m2s8m2z2m",
                       "top": "1p2z7m1z6z5m2z2m2s7m3s3s", "left": "1m4m1z2z1s6z1s9m2s3z"},
}
MELD_LABELS = {  # 同 tests/test_meld_reader.py（目視核對過的三家副露）
    "經典_副露_上家碰白_下家碰一筒.png": {"left": "777z", "right": "111p", "top": ""},
    "經典_副露_下家吃四五六筒.png": {"left": "", "right": "654p", "top": ""},
    "經典_副露_對家碰發.png": {"left": "", "right": "", "top": "666z"},
    "經典_副露_對家碰八筒最右.png": {"left": "", "right": "777z345s", "top": "888p333z"},
    "經典_副露_驗證_202741.png": {"top": "444m"}, "經典_副露_驗證_203038.png": {"top": "222p"},
    "經典_副露_驗證_203436.png": {"top": "789p111s"}, "經典_副露_驗證_203544.png": {"top": "123m111s"},
    "經典_副露_驗證_200831.png": {"top": "789m111s666z"}, "經典_副露_驗證_193456.png": {"top": "456p"},
}
TABLE_LABELS = {  # 同 tests/test_table_reader.py：(按鈕, 可吃碰的牌, 來源, 金牌)
    "經典_碰": ({"chow", "pung"}, "2m", "left", "1s9p7s"),
    "經典_對家碰": ({"chow", "pung"}, "2s", "top", "4z7p4p"),
    "經典_下家打牌": (None, "5z", "right", "1s9p7s"),
}
GOLD_LABELS = {"經典_金牌四種": "7m7s3m1p", "經典_金牌五種": "7m7s3m1p1m"}


@dataclass
class Row:
    item: str
    data: str
    correct: int = 0
    wrong: int = 0
    unread: int = 0
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.unread

    def rate(self, n: int) -> str:
        return f"{n / self.total:.1%}" if self.total else "—"


def _without_manual(source: Path, tmp: Path) -> Path:
    """複製模板資料夾、拿掉人工標註的（manual_*，正是從評估截圖切的）。"""
    target = tmp / source.name
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("manual_*"))
    return target


def hand_accuracy() -> Row | None:
    from perception.capture import read_image
    from perception.hand_reader import TileClassifier, crop_tile, locate_hand_tiles

    if not all((TRAIN / f"{name}.png").exists() for name in HAND_LABELS):
        return None
    data = {}
    for name, labels in HAND_LABELS.items():
        image = read_image(TRAIN / f"{name}.png")
        boxes = locate_hand_tiles(image)
        data[name] = [crop_tile(image, box) for box in boxes], parse(labels)

    def nearby(a: str, b: str) -> bool:
        return a[3] == b[3] and abs(int(a[4:]) - int(b[4:])) <= 15

    row = Row("自己的手牌", f"{len(HAND_LABELS)} 張影片畫面，留一驗證")
    located = sum(len(crops) == len(labels) for crops, labels in data.values())
    for name, (crops, labels) in data.items():
        classifier = TileClassifier([(tile, crop) for other, (other_crops, other_labels) in data.items()
                                     if not nearby(name, other) for tile, crop in zip(other_labels, other_crops)])
        for crop, tile in zip(crops, labels):
            if tile not in classifier.known_tiles:
                continue  # 其他畫面沒出現過這種牌，沒有模板可比
            predicted, _ = classifier.classify(crop)
            if predicted == tile:
                row.correct += 1
            elif predicted is None:
                row.unread += 1
            else:
                row.wrong += 1
    skipped = sum(len(labels) for _, labels in data.values()) - row.total
    row.note = (f"牌的位置 {located}/{len(data)} 張畫面全部找對；只用其他畫面的牌當模板（實戰的模板更多），"
                f"其他畫面沒出現過的 {skipped} 張不算")
    return row


def river_accuracy() -> Row | None:
    from perception.capture import read_image
    from perception.config import RIVER_TEMPLATE_DIR
    from perception.hand_reader import TileClassifier
    from perception.river_reader import RiverReader

    if not RIVER_TEMPLATE_DIR.exists() or not all((TRAIN / f"{n}.png").exists() for n in RIVER_LABELS):
        return None
    row = Row("牌河（四家打出的牌）", f"{len(RIVER_LABELS)} 張人工標註的牌桌截圖")
    missed = 0
    with tempfile.TemporaryDirectory() as tmp:
        reader = RiverReader(TileClassifier.from_directory(_without_manual(RIVER_TEMPLATE_DIR, Path(tmp)),
                                                           min_score=0.7))
        for name, labels in RIVER_LABELS.items():
            rivers = reader.read(read_image(TRAIN / f"{name}.png"))
            for seat, answer in labels.items():
                want = parse(answer)
                got = [tile.tile for tile in rivers[seat]]
                missed += max(0, len(want) - len(got))
                for truth, read in zip(want, got):
                    if read == truth:
                        row.correct += 1
                    elif read is None:
                        row.unread += 1
                    else:
                        row.wrong += 1
    row.unread += missed
    row.note = f"不用人工標註的模板；認不出含 {missed} 張沒框到"
    return row


def meld_accuracy() -> Row | None:
    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    if not all((TRAIN / name).exists() for name in MELD_LABELS):
        return None
    row = Row("對手副露（吃碰槓）", f"{len(MELD_LABELS)} 張目視核對的截圖")
    extra = 0
    with tempfile.TemporaryDirectory() as tmp:
        seat_dir = _without_manual(TRAIN / "meld_templates", Path(tmp)) if (TRAIN / "meld_templates").exists() \
            else Path(tmp) / "none"
        reader = MeldReader.from_directory(seat_dir=seat_dir)
        if reader is None:
            return None
        for name, labels in MELD_LABELS.items():
            read = reader.read(read_image(TRAIN / name))
            for seat, answer in labels.items():
                got = Counter(hit.tile for hit in read[seat])
                want = Counter(parse(answer)) if answer else Counter()
                row.correct += sum((got & want).values())
                row.unread += sum((want - got).values())
                extra += sum((got - want).values())
    row.wrong = extra  # 讀錯的牌一定同時少讀一張正確的；這裡把「多讀出來的」算讀錯
    row.unread -= min(row.unread, extra)
    row.note = "單張辨識（不含之後用牌河推算的補正）；不用人工標註的模板"
    return row


def table_accuracy() -> list[Row]:
    from perception.capture import read_image
    from perception.config import CLAIM_TEMPLATE_BOX, GOLD_MIN_SCORE, GOLD_TEMPLATE_BOX, TEMPLATE_DIR
    from perception.hand_reader import TileClassifier
    from perception.table_reader import read_buttons, read_claim_tile, read_gold_tiles

    if not TEMPLATE_DIR.exists() or not all((TRAIN / f"{n}.png").exists() for n in TABLE_LABELS):
        return []
    claim_reader = TileClassifier.from_directory(template_box=CLAIM_TEMPLATE_BOX)
    gold = TileClassifier.from_directory(template_box=GOLD_TEMPLATE_BOX, min_score=GOLD_MIN_SCORE)
    buttons = Row("吃碰槓胡按鈕", f"{len(TABLE_LABELS)} 張截圖")
    claim = Row("被打出、可吃碰的那張", f"{len(TABLE_LABELS)} 張截圖")
    gold_row = Row("金牌面板", f"{len(TABLE_LABELS) + len(GOLD_LABELS)} 張截圖")
    for name, (want_buttons, want_claim, source, gold_tiles) in TABLE_LABELS.items():
        image = read_image(TRAIN / f"{name}.png")
        got = read_buttons(image)
        ok = got == (None if want_buttons is None else frozenset(want_buttons))
        buttons.correct += ok
        buttons.wrong += not ok
        reading = read_claim_tile(image, claim_reader)
        ok = (reading.tile, reading.source) == (parse(want_claim)[0], source)
        claim.correct += ok
        claim.wrong += not ok
        _score_list(gold_row, read_gold_tiles(image, gold), parse(gold_tiles))
    shots = 0
    for name, tiles in GOLD_LABELS.items():
        if (TRAIN / f"{name}.png").exists():
            shots += 1
            _score_list(gold_row, read_gold_tiles(read_image(TRAIN / f"{name}.png"), gold), parse(tiles))
    gold_row.data = f"{len(TABLE_LABELS) + shots} 張截圖"
    buttons.note = "整排按鈕亮哪幾個完全對才算對（樣本少，僅供參考）"
    claim.note = "牌和誰打的都對才算對（樣本少，僅供參考）"
    gold_row.note = "以張計"
    return [buttons, claim, gold_row]


def _score_list(row: Row, got: list[int], want: list[int]) -> None:
    for i, truth in enumerate(want):
        if i >= len(got):
            row.unread += 1
        elif got[i] == truth:
            row.correct += 1
        else:
            row.wrong += 1


_NAMES = {tile_name(t): t for t in range(34)}


CLEANED = "20261020"
"""10/19 清掉混進牌河模板的錯圖（當天之前自己的牌只讀對 43～54%），之後才是現在的水準。"""


def live_river_accuracy(directory: Path = RESULT, since: str | None = None) -> Row | None:
    """實戰：自己打的每張牌（記錄檔的決定），之後 8 秒內牌河追蹤報的第一個自己的事件。"""
    logs = sorted(log for log in directory.glob("log_*.txt") if since is None or log.stem[4:] >= since)
    if not logs:
        return None
    decision = re.compile(r"\] 打 (\S+?)(?:，並按「聽」)?(?:（|　|$)")
    event = re.compile(r"牌河 me (discard|claimed|claimed_unseen) (\S+)")
    counts = Counter()
    days = set()
    for log in logs:
        lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                 if re.match(r"\d\d:\d\d:\d\d ", line)]
        for i, line in enumerate(lines):
            match = decision.search(line)
            if not match or "（手牌" not in line or match.group(1) not in _NAMES:
                continue
            days.add(log.stem[4:])
            start = datetime.strptime(line[:8], "%H:%M:%S")
            outcome = "沒看到"
            for later in lines[i + 1:i + 60]:
                seconds = (datetime.strptime(later[:8], "%H:%M:%S") - start).total_seconds()
                if seconds < 0 or seconds > 8 or "] 打 " in later or "結算截圖" in later:
                    break
                found = event.search(later)
                if found:
                    kind, tile = found.groups()
                    if kind != "discard":
                        outcome = "打出就被吃碰"
                    elif tile == match.group(1):
                        outcome = "讀對"
                    elif tile == "?":
                        outcome = "認不出"
                    else:
                        outcome = "讀錯"
                    break
            counts[outcome] += 1
    label = "牌河：自己打出的牌（實戰，全部）" if since is None else f"牌河：自己打出的牌（實戰，{since[4:6]}/{since[6:]} 起）"
    row = Row(label, f"{len(days)} 天實戰記錄",
              correct=counts["讀對"], wrong=counts["讀錯"], unread=counts["認不出"])
    row.extra = {"沒看到": counts["沒看到"], "打出就被吃碰": counts["打出就被吃碰"]}
    seen = row.total
    row.note = (f"另有 {counts['沒看到']} 張 8 秒內沒看到、{counts['打出就被吃碰']} 張打出就被吃碰"
                f"（多半是打出就被拿走或那局結束）；看到的 {seen} 張中讀對 {row.rate(row.correct)}")
    return row


def collect() -> list[Row]:
    rows: list[Row] = []
    for make in (hand_accuracy, river_accuracy, meld_accuracy):
        row = make()
        if row is not None:
            rows.append(row)
    rows += table_accuracy()
    for since in (None, CLEANED):
        live = live_river_accuracy(since=since)
        if live is not None:
            rows.append(live)
    return rows


def markdown(rows: list[Row]) -> str:
    lines = ["| 項目 | 資料 | 樣本 | 讀對 | 讀錯 | 認不出 | 說明 |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row.item} | {row.data} | {row.total} | {row.rate(row.correct)} | {row.rate(row.wrong)} "
                     f"| {row.rate(row.unread)} | {row.note} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="畫面辨識準確率總表")
    parser.add_argument("--md", type=Path, help="另存成 Markdown 表格")
    args = parser.parse_args()
    table = markdown(collect())
    print(table)
    if args.md:
        args.md.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
