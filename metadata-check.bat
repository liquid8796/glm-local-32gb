@echo off
setlocal
call "%~dp0glm.bat" metadata-check %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
pause
exit /b %GLM_EXIT%
