@echo off
setlocal
call "%~dp0glm.bat" runtime-plan %*
set "GLM_EXIT=%ERRORLEVEL%"
pause
exit /b %GLM_EXIT%
