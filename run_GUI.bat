@echo off
REM Change to the directory where this BAT file is located
cd /d "%~dp0"

REM Check if the 'venv' directory exists
if not exist "venv\" (
    echo Error: 'venv' directory not found.
	echo Please run the included insall_MGAS BAT file first!
    pause
    exit /b 1
)

echo.
echo Starting MGAS GUI...
echo.
echo =====================================================================
echo.
echo                "Remember, lower IS better!" - r557
echo.
echo =====================================================================

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the Python script and keep the terminal open
python gui.py

pause
