@echo off
REM Change to the directory where this BAT file is located
cd /d "%~dp0"

REM Run the Python script and keep the terminal open
python bulk_facedistance_statistics_v2.py
pause
