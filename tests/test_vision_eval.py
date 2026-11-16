from eval.vision import Row, live_river_accuracy, markdown

LOG = """\
10:00:00 [10:00:00] 打 五萬　（手牌 一萬二萬三萬五萬，金牌 東）
10:00:02 [10:00:02] 牌河 me discard 五萬
10:00:10 [10:00:10] 打 東，並按「聽」（按聽）　（手牌 東南西，金牌 東）
10:00:12 [10:00:12] 牌河 me discard 南
10:00:20 [10:00:20] 打 一條　（手牌 一條，金牌 東）
10:00:21 [10:00:21] 牌河 me claimed 一條
10:00:30 [10:00:30] 打 九筒　（手牌 九筒，金牌 東）
10:00:31 [10:00:31] 牌河 me discard ?
10:00:40 [10:00:40] 打 白　（手牌 白，金牌 東）
10:00:55 [10:00:55] 牌河 me discard 白
"""


def test_live_river_accuracy_matches_each_discard_with_what_the_river_saw(tmp_path):
    (tmp_path / "log_20261101.txt").write_text(LOG, encoding="utf-8")
    (tmp_path / "log_20261010.txt").write_text(LOG, encoding="utf-8")
    row = live_river_accuracy(tmp_path, since="20261020")  # 10/10 那天不算
    assert (row.correct, row.wrong, row.unread) == (1, 1, 1)
    assert row.extra == {"沒看到": 1, "打出就被吃碰": 1}  # 白：8 秒內沒看到
    assert "1 天" in row.data
    assert live_river_accuracy(tmp_path).total == 6


def test_markdown_table_has_one_line_per_row():
    text = markdown([Row("手牌", "13 張", correct=9, wrong=1), Row("空的", "沒有資料")])
    lines = text.splitlines()
    assert len(lines) == 4 and "90.0%" in lines[2] and "| — |" in lines[3]
