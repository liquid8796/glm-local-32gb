@echo off
setlocal
call "%~dp0glm.bat" projection-check %*
set "GLM_EXIT=%ERRORLEVEL%"
pause
exit /b %GLM_EXIT%
