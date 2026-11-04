@echo off
rem Double-click to run. First run creates a local venv and installs dependencies.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto run

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3 not found. Install it from https://www.python.org and tick "Add to PATH".
    pause
    exit /b 1
)
echo First run: setting up...
python -m venv .venv
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

:run
.venv\Scripts\python.exe screw_counter.py %*
if errorlevel 1 pause
