@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish-studio.ps1" %*
exit /b %ERRORLEVEL%
