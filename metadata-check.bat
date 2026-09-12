@echo off
setlocal
call "%~dp0glm.bat" metadata-check %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Report: %~dp0reports\metadata-latest.md
pause
exit /b %GLM_EXIT%
