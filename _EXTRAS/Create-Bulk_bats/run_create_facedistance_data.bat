@echo off
REM Change to the directory where this BAT file is located
cd /d "%~dp0"

REM Run the Python script and keep the terminal open
python create_facedistance_data.py
pause
