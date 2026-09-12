@echo off
setlocal
call "%~dp0glm.bat" parity --storage safetensors %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Report: %~dp0reports\parity-latest.md
pause
exit /b %GLM_EXIT%
