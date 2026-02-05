@echo off
echo ============================================================
echo Starting Dashboard
echo ============================================================
echo.

cd /d "%~dp0"
python dashboard/app.py

pause
