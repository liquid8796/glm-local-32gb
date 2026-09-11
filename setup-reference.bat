@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-reference.ps1" %*
set "GLM_REFERENCE_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %GLM_REFERENCE_EXIT%
