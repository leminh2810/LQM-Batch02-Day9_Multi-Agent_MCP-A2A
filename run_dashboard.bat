@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv\Scripts\python.exe
  echo Hay tao moi truong truoc, hoac chay: python agent_dashboard.py
  pause
  exit /b 1
)
echo Starting Legal Multi-Agent Dashboard...
echo Open: http://127.0.0.1:8088
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" agent_dashboard.py
