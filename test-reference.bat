@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-reference\Scripts\python.exe" (
    echo Run setup-reference.bat first.
    popd
    exit /b 1
)
set "GLM_TEST_OFFICIAL=1"
set "GLM_TEST_NATIVE=1"
set "GLM_TEST_CUDA=1"
set "GLM_TEST_MINI=1"
set "GLM_TEST_STORAGE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
.venv-reference\Scripts\python.exe -m glm_local.reference_env --verify
if errorlevel 1 (
    popd
    exit /b 1
)
.venv-reference\Scripts\python.exe -m unittest discover -s tests -v
set "GLM_EXIT=%ERRORLEVEL%"
popd
exit /b %GLM_EXIT%
