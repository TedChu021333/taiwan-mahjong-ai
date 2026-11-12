@echo off
chcp 65001 >nul
rem 雙擊開啟總選單。
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv-train\Scripts\python start.py %*
pause
