@echo off
cd /d "%~dp0"
echo Creating virtual environment in .venv ...
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
if not exist .venv\Scripts\python.exe (echo Could not create .venv - is Python 3.11/3.12 installed? & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo Install failed. & pause & exit /b 1)
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg
where deno >nul 2>nul || echo NOTE: deno not found. yt-dlp may need it for YouTube: winget install DenoLand.Deno
echo.
echo Done. Run run-mock.bat to try the UI, or run.bat for real processing.
pause
