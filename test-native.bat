@echo off
setlocal
pushd "%~dp0"
set "GLM_TEST_NATIVE=1"
set "GLM_TEST_CUDA=1"
python -m unittest discover -s tests -v
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
