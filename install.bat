@echo off
chcp 65001 >nul
rem 一鍵安裝：建立 .venv-train 虛擬環境並安裝所有套件。雙擊執行即可。
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo 找不到 Python，請先到 https://www.python.org/downloads/ 安裝 Python 3.12，安裝時勾選 "Add python.exe to PATH"。
    pause
    exit /b 1
)

if not exist .venv-train (
    echo 建立虛擬環境 .venv-train ...
    python -m venv .venv-train
)

.venv-train\Scripts\python -m pip install --upgrade pip

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo 沒有偵測到 NVIDIA 顯示卡，安裝 CPU 版 PyTorch（可以玩、可以用提示，但訓練會很慢）。
    .venv-train\Scripts\pip install torch
) else (
    echo 偵測到 NVIDIA 顯示卡，安裝 CUDA 12.8 版 PyTorch。
    .venv-train\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
)

.venv-train\Scripts\pip install -r requirements.txt

echo.
echo 安裝完成。啟動：雙擊 start.bat，或執行 .venv-train\Scripts\python start.py
pause
