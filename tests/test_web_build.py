import subprocess
import sys
import zipfile
from pathlib import Path

from web import build

ROOT = Path(__file__).resolve().parent.parent


def test_browser_modules_do_not_need_pytorch():
    """網頁版在 Pyodide 裡跑，沒有 PyTorch、OpenCV、Tk：載入 AI 與牌局驅動時不能 import 它們。"""
    code = ("import sys\n"
            "for name in ('torch', 'cv2', 'tkinter', 'PIL'):\n"
            "    sys.modules[name] = None\n"  # 之後 import 這些會直接失敗
            "import agent.nn_agent, play.web, training.numpy_model\n"
            "print('ok')\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert done.stdout.strip() == "ok", done.stderr


def test_site_has_everything_the_page_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "SITE", tmp_path / "site")
    site = build.build()
    for name in ("index.html", "app.js", "worker.js", "style.css", "model.npz", "py.zip", "tiles/0.png", "tiles/33.png"):
        assert (site / name).exists(), name
    with zipfile.ZipFile(site / "py.zip") as archive:
        names = set(archive.namelist())
    for needed in ("game/rules.py", "agent/nn_agent.py", "agent/calibration.json", "play/web.py",
                   "training/numpy_model.py", "training/encoding.py"):
        assert needed in names, needed
    assert not any(name.startswith(("training/model.py", "hint/", "perception/")) for name in names)
