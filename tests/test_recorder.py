from pathlib import Path

from hint.recorder import Recorder, Timeline, build_ass, burn_command, overlay_lines, parse_time

POSITIONS = {"advice": (650, 140), "table": (20, 440), "left": (10, 150), "top": (1170, 20), "right": (1610, 150)}


class FakeProcess:
    def __init__(self) -> None:
        self.sent = None

    def communicate(self, data, timeout=None):
        self.sent = data


def test_recorder_keeps_a_timeline_of_what_the_windows_showed(tmp_path):
    now = [100.0]
    commands, process = [], FakeProcess()
    recorder = Recorder(tmp_path / "rec.mp4", clock=lambda: now[0],
                        launcher=lambda command: commands.append(command) or process)
    recorder.start()
    assert "ddagrab=output_idx=0:framerate=60" in commands[0] and "h264_nvenc" in commands[0]
    now[0] = 102.5
    recorder.show("打 五萬")
    recorder.show("打 五萬")  # 一樣的內容不重複記
    now[0] = 104.0
    recorder.show(None)
    recorder.show_melds({"left": "上家　碰 白", "top": None})
    now[0] = 110.0
    recorder.stop()
    assert process.sent == b"q"  # 讓 ffmpeg 正常收尾
    assert recorder.timeline.advice == [(0.0, None), (2.5, "打 五萬"), (4.0, None)]
    assert recorder.timeline.melds == [(0.0, {}), (4.0, {"left": "上家　碰 白"})]
    text = recorder.subtitle_path.read_text(encoding="utf-8-sig")
    assert "0:00:02.50,0:00:04.00" in text and "打 五萬" in text
    assert "上家　碰 白" in text and recorder.seconds == 10.0


def test_overlay_lines_follow_the_floating_windows():
    lines = overlay_lines("胡！", None, {"right": "下家　碰 一萬　吃 三四五筒　暗槓"}, POSITIONS)
    texts = [text for _, text in lines]
    assert any("\\pos(960,183)" in text and "胡！" in text for text in texts)  # 提示框置中
    assert any("&H5252FF&" in text for text in texts)  # 胡是紅色（ASS 色碼為 BGR）
    assert any("放槍機率：等輪到自己" in text for text in texts)  # 放槍表空著時顯示等待字樣
    meld = next(text for text in texts if "下家" in text)
    assert "\\N" in meld  # 下家在螢幕右邊、放不下就換行
    assert sorted({layer for layer, _ in lines}) == [0, 1]


def test_build_ass_splits_time_at_every_change():
    timeline = Timeline()
    Timeline.add(timeline.advice, 1.0, "打 五萬")
    Timeline.add(timeline.table, 2.0, "放槍機率\n八萬 12%")
    text = build_ass(timeline, 3.0, POSITIONS)
    starts = {line.split(",")[1] for line in text.splitlines() if line.startswith("Dialogue")}
    assert starts == {"0:00:00.00", "0:00:01.00", "0:00:02.00"}
    assert "放槍機率\\N八萬 12%" in text


def test_burn_command_shifts_time_so_subtitles_line_up():
    command = burn_command(Path("C:/r/a.mp4"), Path("C:/r/a.ass"), Path("out.mp4"), parse_time("15:48"),
                           parse_time("17:36"))
    assert command[command.index("-ss") + 1] == "948.000"
    assert command[command.index("-t") + 1] == "108.000"
    filters = command[command.index("-vf") + 1]
    assert filters.startswith("setpts=PTS+948.000/TB,ass='C\\:/r/a.ass'")
