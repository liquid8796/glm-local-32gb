@echo off
setlocal
pushd "%~dp0"
set "GLM_TEST_NATIVE=1"
set "GLM_TEST_CUDA=1"
set "GLM_TEST_MINI=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
python -m unittest discover -s tests -v
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
