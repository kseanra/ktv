@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Open http://localhost:8000   (phones: http://YOUR-PC-IP:8000/?remote)
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
pause
