@echo off
if "%~1"=="" (
  echo Usage: demo.bat path\to\video.mp4   ^(or drag a video onto this file^)
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo.ps1" -Video "%~1"
pause
