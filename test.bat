@echo off
setlocal
pushd "%~dp0"
python -m unittest discover -s tests -v
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
