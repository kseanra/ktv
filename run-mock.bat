@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set KARAOKE_MOCK=1
set KARAOKE_DATA=%~dp0data-mock
echo MOCK MODE - fake songs, no YouTube or AI models used.
echo Open http://localhost:8000 and paste any YouTube link.
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
pause
