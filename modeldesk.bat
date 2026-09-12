@echo off
setlocal
if exist "%~dp0artifacts\ModelDesk\win-x64\ModelDesk.exe" (
    start "ModelDesk" "%~dp0artifacts\ModelDesk\win-x64\ModelDesk.exe"
    exit /b 0
)
dotnet run --project "%~dp0studio\src\ModelDesk.Desktop\ModelDesk.Desktop.csproj" --configuration Release -- %*
exit /b %ERRORLEVEL%
