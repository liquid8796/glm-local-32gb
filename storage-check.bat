@echo off
setlocal
call "%~dp0glm.bat" storage-check %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Report: %~dp0reports\storage-latest.md
pause
exit /b %GLM_EXIT%
