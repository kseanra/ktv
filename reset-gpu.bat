@echo off
setlocal
cd /d "%~dp0"
rem Undo setup-gpu.bat: remove the NVIDIA packages, put back the normal (CPU)
rem ONNX Runtime, and keep Whisper on the CPU (KARAOKE_DEVICE=cpu in run.bat).
rem Safe to run more than once.

if not exist .venv\Scripts\python.exe (
  echo .venv not found - run setup.bat first.
  pause
  exit /b 1
)

echo [1/4] Removing the NVIDIA packages ...
.venv\Scripts\python -m pip uninstall -y onnxruntime-gpu nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12

echo.
echo [2/4] Putting back the normal ONNX Runtime ...
.venv\Scripts\python -m pip install --force-reinstall --no-deps onnxruntime
if errorlevel 1 (
  echo Install failed - check the internet connection and run this again.
  pause
  exit /b 1
)

echo.
echo [3/4] Checking ONNX Runtime - should list no CUDA ...
.venv\Scripts\python -c "import onnxruntime as o; print('   onnxruntime', o.__version__, o.get_available_providers())"

echo.
echo [4/4] Keeping Whisper on the CPU in run.bat ...
findstr /c:"KARAOKE_DEVICE" run.bat >nul 2>nul
if not errorlevel 1 goto already
> run.bat echo @echo off
>> run.bat echo cd /d "%%~dp0"
>> run.bat echo call .venv\Scripts\activate.bat
>> run.bat echo set KARAOKE_DEVICE=cpu
>> run.bat echo echo Open http://localhost:8000   (phones: http://YOUR-PC-IP:8000/?remote)
>> run.bat echo python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
>> run.bat echo pause
echo    run.bat now sets KARAOKE_DEVICE=cpu
goto done
:already
echo    run.bat already sets KARAOKE_DEVICE - left as it is

:done
echo.
echo Reset done. Start the server with run.bat.
pause
