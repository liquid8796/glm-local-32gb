@echo off
setlocal
if exist "%~dp0artifacts\ModelDesk\win-x64\modeldesk-cli.exe" goto published
dotnet run --project "%~dp0studio\src\ModelDesk.Cli\ModelDesk.Cli.csproj" --configuration Release -- %*
exit /b %ERRORLEVEL%
:published
"%~dp0artifacts\ModelDesk\win-x64\modeldesk-cli.exe" %*
exit /b %ERRORLEVEL%
