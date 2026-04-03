@echo off
cd /d %~dp0\..
if not exist .\.venv\Scripts\python.exe (
    echo Virtual environment not found at .\.venv
    exit /b 1
)
.\.venv\Scripts\python.exe backend\run_server.py
