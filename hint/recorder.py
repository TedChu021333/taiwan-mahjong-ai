"""錄下牌局給簡報、YouTube 用（`python -m hint --record`）。

浮動小視窗設成「螢幕擷取時看不到」（hint.overlay），不擋辨識，但 Windows 錄影、OBS 也錄不到它們
（11/12 使用者要錄影放簡報）。做法分兩步：
1. 錄影：ffmpeg 用 Windows 桌面複製（ddagrab）抓畫面、NVIDIA 硬體編碼（h264_nvenc），1080p 每秒 60 張，
   幾乎不吃 CPU，不影響辨識。同時把提示、放槍表、三家副露標籤「幾秒時顯示什麼」記下來，存成同名的 .ass 字幕檔。
2. 輸出：`python -m hint.recorder <影片> [--start 15:48 --end 17:36] [--out 檔案]` 把 .ass 照小視窗的位置、配色
   燒進影片（可只剪一段）。
11/16 第一版在 Python 裡每張畫好再送 ffmpeg：1080p 只跟得上每秒約 17 張，720p、CRF 26 又太糊（使用者：畫質有必要那麼差嗎）。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import ImageFont

from hint.overlay import (BACKGROUND, HEIGHT, IDLE_TEXT, MELD_COLOR, MELD_DEFAULTS, SCREEN_MARGIN, TABLE_DEFAULT,
                          TABLE_HEIGHT, TABLE_IDLE, TABLE_POSITION_FILE, TABLE_WIDTH, WIDTH, color_for, load_position,
                          wrap_parts)

FPS = 60
SCREEN = (1920, 1080)
CQ = 18
"""NVENC 固定品質（數字越小越好；18 接近看不出壓縮痕跡）。"""
FONT_PATH = "C:/Windows/Fonts/msjhbd.ttc"
FONT_NAME = "Microsoft JhengHei"
BIG, SMALL = 37, 20
"""小視窗字型 28pt、15pt 換成像素。"""
ASS_FONT_SCALE = 1.4
"""libass 的字級是整行高度（含行距），微軟正黑體的字面只佔約七成；放大 1.4 倍才和小視窗的字一樣大。"""
ACTIVE_ALPHA, IDLE_ALPHA = 0.92, 0.45
HOLD_SECONDS = 3.0
"""燒進影片時，每個建議至少停 3 秒才變回「等待中」（自動模式一出建議就點下去，吃碰、按聽的提示只亮 1～2 秒，
影片裡看不清楚；11/16 使用者看第 1 支剪輯覺得吃碰沒錄進去）。下一個建議來了照樣馬上換。"""


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Timeline:
    """三種小視窗的內容隨時間的變化：(秒, 值)，值 None 代表空的。"""
    advice: list = field(default_factory=lambda: [(0.0, None)])
    table: list = field(default_factory=lambda: [(0.0, None)])
    melds: list = field(default_factory=lambda: [(0.0, {})])

    @staticmethod
    def add(track: list, t: float, value) -> None:
        if track[-1][1] != value:
            track.append((t, value))

    def remap(self, convert: Callable[[float], float]) -> "Timeline":
        """把時間換算過（例如對到接起來的影片）；換算後時間一樣的只留最後一個。"""
        def mapped(track: list) -> list:
            out: list = []
            for t, value in track:
                t = convert(t)
                if out and out[-1][0] == t:
                    out[-1] = (t, value)
                elif not out or out[-1][1] != value:
                    out.append((t, value))
            return out or list(track[:1])

        return Timeline(mapped(self.advice), mapped(self.table), mapped(self.melds))

    def hold(self, seconds: float) -> "Timeline":
        """提示框的建議至少停 seconds 秒：接在建議後面的「空的」往後移（最多到下一個建議）。"""
        advice: list = []
        for i, (t, value) in enumerate(self.advice):
            if value is None and advice and advice[-1][1] is not None:
                limit = self.advice[i + 1][0] if i + 1 < len(self.advice) else float("inf")
                t = min(max(t, advice[-1][0] + seconds), limit)
            if advice and advice[-1][0] >= t:
                advice[-1] = (advice[-1][0], value)  # 被擠掉了：直接換成新的
                continue
            advice.append((t, value))
        return Timeline(advice, list(self.table), list(self.melds))


WATCH_INTERVAL = 1.0
"""每秒看一次 ffmpeg 還活著沒。"""
CLUSTER_MS = 1000
"""mkv 每秒寫一個 cluster（預設 5 秒）、每個封包立刻寫進檔案（-flush_packets；ffmpeg 的輸出緩衝 256 KB，
被砍掉時留在緩衝裡的就沒了：砍在 5.5 秒的測試只剩半個 cluster、一張畫面都讀不到）。"""
QUICK_FAILURE_SECONDS = 10.0
MAX_QUICK_FAILURES = 3
"""重新開始後 10 秒內又死掉算一次快速失敗，連續 3 次就放棄，免得一直重開。"""


class Recorder:
    """ffmpeg 在背景錄螢幕；中途死掉就自動開新的一段（11/16 15:14 錄了 41 秒就停，之後兩小時全沒錄到，
    當時 stderr 丟掉了、原因不明，ffmpeg 單獨錄又正常），結束時各段接成一個 mp4，字幕時間照各段實際長度對齊
    （缺的那段時間直接跳過）。ffmpeg 的訊息存成同名 .log。"""

    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic,
                 launcher: Callable[[list[str]], subprocess.Popen] | None = None,
                 notify: Callable[[str], None] = print,
                 duration: Callable[[Path], float] | None = None) -> None:
        self.path = path
        self.clock = clock
        self._launch = launcher or self._popen
        self.notify = notify
        self._duration = duration or video_duration
        self.timeline = Timeline()
        self.started: float | None = None
        self.seconds = 0.0
        self.interruptions = 0
        self.segments: list[tuple[Path, float]] = []
        """(各段的 mkv, 開始錄的時間（從開錄算起的秒數）)。"""
        self._process: subprocess.Popen | None = None
        self._log = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None
        self._quick_failures = 0

    @property
    def subtitle_path(self) -> Path:
        return self.path.with_suffix(".ass")

    @property
    def log_path(self) -> Path:
        return self.path.with_suffix(".log")

    def _now(self) -> float:
        return 0.0 if self.started is None else max(0.0, self.clock() - self.started)

    def show(self, text: str | None) -> None:
        with self._lock:
            Timeline.add(self.timeline.advice, self._now(), text)

    def show_table(self, text: str | None) -> None:
        with self._lock:
            Timeline.add(self.timeline.table, self._now(), text)

    def show_melds(self, texts: dict[str, str | None]) -> None:
        with self._lock:
            Timeline.add(self.timeline.melds, self._now(), {k: v for k, v in texts.items() if v})

    def _popen(self, command: list[str]) -> subprocess.Popen:
        if self._log is None:
            self._log = open(self.log_path, "a", encoding="utf-8")
        self._log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        self._log.flush()
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._log)

    def _segment_path(self) -> Path:
        n = len(self.segments) + 1
        return self.path.with_suffix(".mkv") if n == 1 else self.path.with_name(f"{self.path.stem}_{n}.mkv")

    def _launch_segment(self) -> None:
        path = self._segment_path()
        command = [ffmpeg(), "-hide_banner", "-nostats", "-y", "-f", "lavfi",
                   "-i", f"ddagrab=output_idx=0:framerate={FPS}",
                   "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(CQ), "-b:v", "0",
                   "-cluster_time_limit", str(CLUSTER_MS), "-flush_packets", "1", str(path)]
        self._process = self._launch(command)
        self.segments.append((path, self._now()))

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = self.clock()
        self._launch_segment()
        self._watcher = threading.Thread(target=self._watch, name="recorder-watch", daemon=True)
        self._watcher.start()

    def _watch(self) -> None:
        while not self._stopping.wait(WATCH_INTERVAL):
            self.check()

    def check(self) -> None:
        """ffmpeg 死了就記一筆、重新開一段（背景執行緒每秒呼叫一次）。"""
        with self._lock:
            process = self._process
            if process is None or self._stopping.is_set() or process.poll() is None:
                return
            now = self._now()
            self.interruptions += 1
            if now - self.segments[-1][1] < QUICK_FAILURE_SECONDS:
                self._quick_failures += 1
            else:
                self._quick_failures = 0
            self._process = None
            if self._quick_failures >= MAX_QUICK_FAILURES:
                self.notify(f"錄影在 {now / 60:.1f} 分時中斷（ffmpeg 結束碼 {process.returncode}），"
                            f"連續 {MAX_QUICK_FAILURES} 次剛開就死掉，不再重試；原因看 {self.log_path}")
                return
            self._launch_segment()
            self.notify(f"錄影在 {now / 60:.1f} 分時中斷（ffmpeg 結束碼 {process.returncode}），已重新開始錄第 "
                        f"{len(self.segments)} 段；原因看 {self.log_path}")

    def _end_process(self) -> None:
        if self._process is None:
            return
        try:
            self._process.communicate(b"q", timeout=15)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            self._process.kill()
        self._process = None

    def video_time(self, durations: list[float]) -> Callable[[float], float]:
        """開錄算起的秒數 → 接起來的影片裡的秒數：落在各段之間的缺口就對到前一段的結尾。"""
        offsets, total = [], 0.0
        for length in durations:
            offsets.append(total)
            total += length

        def convert(t: float) -> float:
            index = max((i for i, (_, begin) in enumerate(self.segments) if begin <= t), default=0)
            return offsets[index] + min(max(0.0, t - self.segments[index][1]), durations[index])

        return convert

    def stop(self) -> None:
        """停止錄影：各段 mkv（中途當掉也讀得到）接成 mp4，寫出 .ass。"""
        self._stopping.set()
        with self._lock:
            self._end_process()
        if self._watcher is not None:
            self._watcher.join(timeout=5)
        wall = self._now()
        durations = []
        for i, (path, begin) in enumerate(self.segments):
            end = self.segments[i + 1][1] if i + 1 < len(self.segments) else wall
            durations.append(max(0.0, min(end - begin, self._duration(path) or 0.0)))
        convert = self.video_time(durations) if self.segments else (lambda t: t)
        self.seconds = sum(durations)
        usable = [(path, length) for (path, _), length in zip(self.segments, durations) if length > 0 and path.exists()]
        if usable and self._concat(usable):
            for path, _ in self.segments:
                path.unlink(missing_ok=True)
        if self._log is not None:
            self._log.close()
            self._log = None
        self.subtitle_path.write_text(build_ass(self.timeline.remap(convert), self.seconds), encoding="utf-8-sig")

    def _concat(self, segments: list[tuple[Path, float]]) -> bool:
        """各段（mkv, 秒數）不重新編碼接成一個 mp4；被砍掉的 mkv 沒有長度，concat 清單裡要寫明。"""
        if len(segments) == 1:
            command = [ffmpeg(), "-v", "error", "-y", "-i", str(segments[0][0])]
        else:
            listing = self.path.with_suffix(".txt")
            listing.write_text("".join(f"file '{path.resolve().as_posix()}'\nduration {length:.3f}\n"
                                       for path, length in segments), encoding="utf-8")
            command = [ffmpeg(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
        done = subprocess.run(command + ["-c", "copy", "-movflags", "+faststart", str(self.path)])
        if len(segments) > 1:
            self.path.with_suffix(".txt").unlink(missing_ok=True)
        return done.returncode == 0 and self.path.exists()


def video_duration(path: Path) -> float:
    """影片實際有幾秒（讀不到畫面算 0）：imageio-ffmpeg 沒附 ffprobe，而且 ffmpeg 被砍掉留下的 mkv 沒寫結尾、
    標頭沒有長度，所以整個檔案走一遍（-c copy 不解碼，兩小時約幾秒）、讀進度最後的 time=。"""
    if not path.exists():
        return 0.0
    done = subprocess.run([ffmpeg(), "-hide_banner", "-v", "error", "-stats", "-i", str(path), "-c", "copy",
                          "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    times = re.findall(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)", done.stderr)
    if not times:
        return 0.0
    hours, minutes, seconds = times[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# ---- 字幕（.ass）：照 hint.overlay 小視窗的位置、大小、配色 ----

def _ass_color(hex_color: str, alpha: float = 1.0) -> tuple[str, str]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"&H{b:02X}{g:02X}{r:02X}&", f"&H{round((1 - alpha) * 255):02X}&"


def _escape(text: str) -> str:
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", "\\N")


def _clock(seconds: float) -> str:
    cs = round(seconds * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _box(x: int, y: int, width: int, height: int, alpha: float) -> str:
    color, transparency = _ass_color(BACKGROUND, alpha)
    return (f"{{\\an7\\pos({x},{y})\\p1\\bord0\\shad0\\1c{color}\\1a{transparency}}}"
            f"m 0 0 l {width} 0 {width} {height} 0 {height}{{\\p0}}")


def _text(x: int, y: int, anchor: int, size: int, color: str, text: str) -> str:
    code, _ = _ass_color(color)
    return f"{{\\an{anchor}\\pos({x},{y})\\fs{round(size * ASS_FONT_SCALE)}\\b1\\1c{code}\\bord0\\shad0}}{_escape(text)}"


def _measure() -> Callable[[str], float]:
    try:
        return ImageFont.truetype(FONT_PATH, SMALL).getlength
    except OSError:
        return lambda text: len(text) * SMALL


def overlay_lines(advice: str | None, table: str | None, melds: dict[str, str],
                  positions: dict[str, tuple[int, int]]) -> list[tuple[int, str]]:
    """某一刻畫面上要畫的東西：(圖層, ASS 文字)，圖層 0 是半透明底框、1 是字。"""
    lines: list[tuple[int, str]] = []
    x, y = positions["advice"]
    text = advice or IDLE_TEXT
    lines.append((0, _box(x, y, WIDTH, HEIGHT, ACTIVE_ALPHA if advice else IDLE_ALPHA)))
    lines.append((1, _text(x + WIDTH // 2, y + HEIGHT // 2, 5, BIG, color_for(text) if advice else "#9e9e9e", text)))
    x, y = positions["table"]
    lines.append((0, _box(x, y, TABLE_WIDTH, TABLE_HEIGHT, ACTIVE_ALPHA if table else IDLE_ALPHA)))
    lines.append((1, _text(x + 10, y + 6, 7, SMALL, "#e0e0e0" if table else "#9e9e9e", table or TABLE_IDLE)))
    measure = _measure()
    for seat, label in melds.items():
        if not label or seat not in positions:
            continue
        x, y = positions[seat]
        # 和小標籤一樣：放不下就在段落之間換行
        rows = wrap_parts(label, measure, max(SCREEN[0] - x - SCREEN_MARGIN - 20, 120)).split("\n")
        width = int(max(measure(row) for row in rows)) + 20
        lines.append((0, _box(x, y, width, len(rows) * (SMALL + 8) + 8, ACTIVE_ALPHA)))
        lines.append((1, _text(x + 10, y + 4, 7, SMALL, MELD_COLOR, "\n".join(rows))))
    return lines


def default_positions() -> dict[str, tuple[int, int]]:
    """小視窗目前的位置（使用者拖過的會存檔）。"""
    positions = {"advice": load_position(), "table": load_position(TABLE_POSITION_FILE, TABLE_DEFAULT)}
    for seat, default in MELD_DEFAULTS.items():
        positions[seat] = load_position(Path(__file__).with_name(f"meld_position_{seat}.txt"), default)
    return positions


def build_ass(timeline: Timeline, end: float, positions: dict[str, tuple[int, int]] | None = None,
              hold: float = HOLD_SECONDS) -> str:
    positions = positions or default_positions()
    if hold > 0:
        timeline = timeline.hold(hold)
    changes = sorted({t for track in (timeline.advice, timeline.table, timeline.melds) for t, _ in track} | {end})
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {SCREEN[0]}\nPlayResY: {SCREEN[1]}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{SMALL},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,"
        "7,0,0,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    def at(track: list, t: float):
        value = track[0][1]
        for when, item in track:
            if when <= t:
                value = item
        return value

    events = []
    for start, stop in zip(changes, changes[1:]):
        if stop <= start:
            continue
        for layer, text in overlay_lines(at(timeline.advice, start), at(timeline.table, start),
                                         at(timeline.melds, start), positions):
            events.append(f"Dialogue: {layer},{_clock(start)},{_clock(stop)},Default,,0,0,0,,{text}")
    return header + "\n".join(events) + "\n"


# ---- 把字幕燒進影片（可只剪一段）----

def parse_time(text: str) -> float:
    """「15:48」「1:02:03」「95.5」→ 秒。"""
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def burn_command(video: Path, subtitles: Path, out: Path, start: float | None = None, end: float | None = None,
                 cq: int = CQ) -> list[str]:
    """-ss 放在 -i 前面會讓時間從 0 開始、字幕就對不上，所以疊字幕前用 setpts 把時間平移回原本的位置。"""
    command = [ffmpeg(), "-hide_banner", "-v", "error", "-y"]
    if start:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(video)]
    if end is not None:
        command += ["-t", f"{end - (start or 0):.3f}"]
    ass = str(subtitles).replace("\\", "/").replace(":", "\\:")
    filters = f"setpts=PTS+{start or 0:.3f}/TB,ass='{ass}':fontsdir='C\\:/Windows/Fonts',setpts=PTS-STARTPTS"
    command += ["-vf", filters, "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", str(cq),
                "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="把錄影時記下的提示、放槍表、副露標籤燒進影片（可只剪一段）")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", help="從第幾分幾秒開始，例如 15:48")
    parser.add_argument("--end", help="到第幾分幾秒，例如 17:36")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cq", type=int, default=CQ, help="畫質（越小越好、檔案越大）")
    args = parser.parse_args()
    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None
    out = args.out or args.video.with_name(f"{args.video.stem}_提示.mp4")
    subprocess.run(burn_command(args.video, args.video.with_suffix(".ass"), out, start, end, args.cq), check=True)
    print(f"已輸出 {out}")


if __name__ == "__main__":
    main()
