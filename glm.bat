@echo off
setlocal
pushd "%~dp0"
set "GLM_PYTHON=python"
if exist ".venv-reference\Scripts\python.exe" set "GLM_PYTHON=.venv-reference\Scripts\python.exe"
"%GLM_PYTHON%" -m glm_local %*
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
