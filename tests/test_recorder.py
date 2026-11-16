from pathlib import Path

from hint.recorder import Recorder, Timeline, build_ass, burn_command, overlay_lines, parse_time

POSITIONS = {"advice": (650, 140), "table": (20, 440), "left": (10, 150), "top": (1170, 20), "right": (1610, 150)}


class FakeProcess:
    def __init__(self) -> None:
        self.sent = None
        self.returncode = None

    def poll(self):
        return self.returncode

    def communicate(self, data, timeout=None):
        self.sent = data


def test_recorder_keeps_a_timeline_of_what_the_windows_showed(tmp_path):
    now = [100.0]
    commands, process = [], FakeProcess()
    recorder = Recorder(tmp_path / "rec.mp4", clock=lambda: now[0], duration=lambda path: 10.0,
                        launcher=lambda command: commands.append(command) or process)
    recorder.start()
    assert "ddagrab=output_idx=0:framerate=60" in commands[0] and "h264_nvenc" in commands[0]
    assert str(tmp_path / "rec.mkv") == commands[0][-1]
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
    assert recorder.interruptions == 0 and len(recorder.segments) == 1


def test_recorder_restarts_ffmpeg_when_it_dies_and_lines_the_subtitles_up(tmp_path):
    """11/16 15:14 ffmpeg 錄了 41 秒就死掉、之後兩小時都沒錄到：看門狗每秒檢查，死了就開新的一段，
    結束時各段接起來，字幕時間照各段實際長度對齊（缺口裡的提示對到前一段結尾）。"""
    now = [0.0]
    processes, notes = [], []
    lengths = {tmp_path / "rec.mkv": 41.0, tmp_path / "rec_2.mkv": 30.0}  # 第一段實際只錄到 41 秒
    recorder = Recorder(tmp_path / "rec.mp4", clock=lambda: now[0], notify=notes.append,
                        launcher=lambda command: processes.append(FakeProcess()) or processes[-1],
                        duration=lambda path: lengths.get(path, 0.0))
    recorder.start()
    now[0] = 20.0
    recorder.show("打 五萬")
    now[0] = 45.0
    processes[0].returncode = 1  # ffmpeg 死了（實際上影片只錄到 41 秒）
    now[0] = 60.0
    recorder.check()  # 看門狗發現
    assert len(processes) == 2 and recorder.segments[1] == (tmp_path / "rec_2.mkv", 60.0)
    assert recorder.interruptions == 1 and "第 2 段" in notes[0] and "rec.log" in notes[0]
    recorder.show("打 六萬")  # 缺口裡的提示 → 對到第一段結尾 41 秒
    now[0] = 70.0
    recorder.show("打 七萬")  # 第二段的 10 秒 → 影片 51 秒
    now[0] = 95.0
    recorder.stop()
    assert processes[1].sent == b"q" and recorder.seconds == 71.0
    assert recorder.timeline.remap(recorder.video_time([41.0, 30.0])).advice == [
        (0.0, None), (20.0, "打 五萬"), (41.0, "打 六萬"), (51.0, "打 七萬")]
    text = recorder.subtitle_path.read_text(encoding="utf-8-sig")
    assert "0:00:41.00,0:00:51.00" in text and "0:00:51.00,0:01:11.00" in text


def test_recorder_gives_up_after_three_quick_failures(tmp_path):
    now = [0.0]
    processes, notes = [], []
    recorder = Recorder(tmp_path / "rec.mp4", clock=lambda: now[0], notify=notes.append,
                        launcher=lambda command: processes.append(FakeProcess()) or processes[-1],
                        duration=lambda path: 0.0)
    recorder.start()
    for _ in range(3):
        now[0] += 2.0
        processes[-1].returncode = 1
        recorder.check()
    assert len(processes) == 3 and recorder.interruptions == 3 and "不再重試" in notes[-1]
    recorder.check()  # 放棄後不再開新的
    assert len(processes) == 3
    recorder.stop()
    assert recorder.subtitle_path.exists()


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


def test_burned_advice_stays_at_least_three_seconds():
    """自動模式一出「碰」就點下去，提示只亮 1 秒；影片裡至少停 3 秒，但下一個建議來了就換。"""
    timeline = Timeline()
    Timeline.add(timeline.advice, 10.0, "碰 九條")
    Timeline.add(timeline.advice, 11.0, None)
    Timeline.add(timeline.advice, 20.0, "吃，組成 三筒四筒五筒")
    Timeline.add(timeline.advice, 21.0, None)
    Timeline.add(timeline.advice, 21.5, "打 六萬，並按「聽」")
    Timeline.add(timeline.advice, 22.0, None)
    Timeline.add(timeline.advice, 23.0, "打 五萬")
    Timeline.add(timeline.advice, 30.0, None)
    assert timeline.hold(3.0).advice == [
        (0.0, None), (10.0, "碰 九條"), (13.0, None), (20.0, "吃，組成 三筒四筒五筒"),
        (21.5, "打 六萬，並按「聽」"), (23.0, "打 五萬"), (30.0, None)]
    text = build_ass(timeline, 31.0, POSITIONS)
    assert "0:00:10.00,0:00:13.00" in text and "0:00:21.50,0:00:23.00" in text
    assert "0:00:10.00,0:00:11.00" not in build_ass(timeline, 31.0, POSITIONS) \
        and "0:00:10.00,0:00:11.00" in build_ass(timeline, 31.0, POSITIONS, hold=0)


def test_burn_command_shifts_time_so_subtitles_line_up():
    command = burn_command(Path("C:/r/a.mp4"), Path("C:/r/a.ass"), Path("out.mp4"), parse_time("15:48"),
                           parse_time("17:36"))
    assert command[command.index("-ss") + 1] == "948.000"
    assert command[command.index("-t") + 1] == "108.000"
    filters = command[command.index("-vf") + 1]
    assert filters.startswith("setpts=PTS+948.000/TB,ass='C\\:/r/a.ass'")
