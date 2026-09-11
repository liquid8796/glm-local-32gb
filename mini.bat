@echo off
setlocal
call "%~dp0glm.bat" mini %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Report: %~dp0reports\mini-latest.md
pause
exit /b %GLM_EXIT%
