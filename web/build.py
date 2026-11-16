"""組出網頁版（`web/site/`，放到 GitHub Pages 的就是這個資料夾）。

    python -m web.build            # 組 site：網頁、牌面圖、模型權重、瀏覽器裡要跑的 Python 原始碼（py.zip）
    python -m web.build --tiles    # 另外從 train/templates 重新匯出牌面圖到 web/tiles（要有遊戲截圖模板）
    python -m web.build --model    # 另外從 models/rl3_0960.pt 重新匯出 web/model.npz（要 PyTorch）
    python -m web.build --serve    # 組完在 http://localhost:8000 開起來試

牌面圖（web/tiles）和權重（web/model.npz）都放進版控，GitHub Actions 組 site 時不需要 PyTorch 和遊戲截圖。
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SITE = WEB / "site"
STATIC = ("index.html", "style.css", "app.js", "worker.js")
PY_SOURCES = (
    "game/*.py", "agent/*.py", "agent/*.json", "play/__init__.py", "play/game.py", "play/web.py",
    "training/__init__.py", "training/actions.py", "training/encoding.py", "training/numpy_model.py",
)
"""瀏覽器裡（Pyodide）要用的 Python 檔：規則、AI、牌局驅動；不含 PyTorch、視窗、畫面辨識。"""
TILE_SIZE = (102, 173)


def export_tiles() -> None:
    from PIL import Image

    from play import art

    out = WEB / "tiles"
    out.mkdir(exist_ok=True)
    for tile in range(34):
        source = art.template(tile)
        if source is None:
            raise SystemExit(f"train/templates 沒有第 {tile} 種牌的模板")
        source.resize(TILE_SIZE, Image.LANCZOS).save(out / f"{tile}.png", optimize=True)
    print(f"已匯出 34 張牌面圖到 {out}")


def export_model() -> None:
    from training.numpy_model import export

    export(ROOT / "models" / "rl3_0960.pt", WEB / "model.npz")
    print(f"已匯出 {WEB / 'model.npz'}")


def build() -> Path:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    for name in STATIC:
        shutil.copy(WEB / name, SITE / name)
    shutil.copytree(WEB / "tiles", SITE / "tiles")
    shutil.copy(WEB / "model.npz", SITE / "model.npz")
    with zipfile.ZipFile(SITE / "py.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in PY_SOURCES:
            for path in sorted(ROOT.glob(pattern)):
                archive.write(path, path.relative_to(ROOT).as_posix())
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    print(f"已組好 {SITE}")
    return SITE


def serve(port: int = 8000) -> None:
    import functools
    import http.server

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    print(f"http://localhost:{port}（Ctrl+C 結束）")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="組出網頁版 web/site")
    parser.add_argument("--tiles", action="store_true", help="重新匯出牌面圖")
    parser.add_argument("--model", action="store_true", help="重新匯出模型權重")
    parser.add_argument("--serve", action="store_true", help="組完開本機伺服器")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.tiles:
        export_tiles()
    if args.model:
        export_model()
    build()
    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
