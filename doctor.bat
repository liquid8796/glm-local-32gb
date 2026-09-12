@echo off
call "%~dp0glm.bat" doctor %*
set "GLM_EXIT=%ERRORLEVEL%"
echo.
echo Exit 2 means the model is not ready to run. Exit 1 means the check failed.
pause
exit /b %GLM_EXIT%
