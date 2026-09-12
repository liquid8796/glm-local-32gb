@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0clean-project.ps1" %*
exit /b %ERRORLEVEL%
