@echo off
setlocal
call "%~dp0glm.bat" probe %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Report: %~dp0reports\backend-probe-latest.md
pause
exit /b %GLM_EXIT%
