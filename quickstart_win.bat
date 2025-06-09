@echo off
REM Quickstart setup script for Tetris Trainer

REM 1. Check Python
python --version >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python 3.10+ is required but was not found in PATH.
    exit /B 1
)

REM 2. Create virtual environment if missing
IF NOT EXIST .env (
    python -m venv .env
)

call .env\Scripts\activate

REM 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. Prepare folder structure
IF NOT EXIST logs\tb (
    mkdir logs\tb
)
IF NOT EXIST logs\tb\.gitkeep (
    type NUL > logs\tb\.gitkeep
)

REM 5. Run tests
pytest -q

ECHO.
ECHO Setup complete. Train an agent with:
ECHO     python scripts\train.py
