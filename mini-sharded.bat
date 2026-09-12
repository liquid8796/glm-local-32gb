@echo off
setlocal
pushd "%~dp0"
if not exist ".venv-reference\Scripts\python.exe" (
    echo Run setup-reference.bat first, or use glm.bat mini with the storage-validation extra installed.
    popd
    exit /b 1
)
.venv-reference\Scripts\python.exe -m glm_local mini --storage safetensors %*
set "GLM_EXIT=%ERRORLEVEL%"
popd
echo.
echo Report: %~dp0reports\mini-latest.md
pause
exit /b %GLM_EXIT%
