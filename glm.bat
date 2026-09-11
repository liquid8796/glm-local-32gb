@echo off
setlocal
pushd "%~dp0"
python -m glm_local %*
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
