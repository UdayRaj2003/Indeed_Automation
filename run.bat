@echo off
setlocal
cd /d "%~dp0"

REM Activate virtual environment if available
if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
)

REM Run Indeed bot with project python or fallback to system python
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" main.py %*
) else (
    python main.py %*
)

set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%

