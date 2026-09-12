@echo off
setlocal
call "%~dp0glm.bat" tokenizer-check %*
set "GLM_EXIT=%ERRORLEVEL%"
pause
exit /b %GLM_EXIT%
