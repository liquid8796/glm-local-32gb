[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Discover an installed C++ compiler. Nothing is downloaded or installed.
$projectRoot = $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
    throw 'vswhere.exe was not found. Install Visual Studio C++ build tools first.'
}
$installation = @(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if ($LASTEXITCODE -ne 0 -or $installation.Count -ne 1 -or [string]::IsNullOrWhiteSpace($installation[0])) {
    throw 'No Visual Studio installation with x64 C++ build tools was found.'
}
$vsRoot = $installation[0].Trim()
$devCommand = Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
$versionFile = Join-Path $vsRoot 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
$toolVersion = (Get-Content -LiteralPath $versionFile -Raw).Trim()
$compiler = Join-Path $vsRoot "VC\Tools\MSVC\$toolVersion\bin\Hostx64\x64\cl.exe"
if (-not (Test-Path -LiteralPath $devCommand -PathType Leaf) -or -not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
    throw 'Visual Studio compiler or developer environment script is missing.'
}

function Invoke-ChildProcess {
    param([System.Diagnostics.ProcessStartInfo]$StartInfo)
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $StartInfo
    try {
        if (-not $process.Start()) { throw 'Could not start build process.' }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $stdoutTask.GetAwaiter().GetResult()
            Stderr = $stderrTask.GetAwaiter().GetResult()
        }
    }
    finally { $process.Dispose() }
}

# Read environment variables from a child cmd; never change this shell or the machine.
$envStart = New-Object System.Diagnostics.ProcessStartInfo
$envStart.FileName = $env:ComSpec
$envStart.Arguments = '/d /s /c ""{0}" -arch=x64 -host_arch=x64 >nul && set"' -f $devCommand
$envStart.WorkingDirectory = $projectRoot
$envResult = Invoke-ChildProcess $envStart
if ($envResult.ExitCode -ne 0) {
    throw "Visual Studio environment initialization failed: $($envResult.Stderr)"
}

$buildRoot = Join-Path $projectRoot 'build'
[void](New-Item -ItemType Directory -Path $buildRoot -Force)
$source = Join-Path $projectRoot 'native\fp8_cpu.c'
$dll = Join-Path $buildRoot 'fp8_cpu.dll'
$objectFile = Join-Path $buildRoot 'fp8_cpu.obj'
$importLibrary = Join-Path $buildRoot 'fp8_cpu.lib'
$compileStart = New-Object System.Diagnostics.ProcessStartInfo
$compileStart.FileName = $compiler
$compileStart.WorkingDirectory = $buildRoot
$compileStart.Arguments = '/nologo /TC /std:c11 /O2 /W4 /WX /fp:strict /LD /Fo"{0}" /Fe"{1}" "{2}" /link /INCREMENTAL:NO /IMPLIB:"{3}"' -f $objectFile, $dll, $source, $importLibrary
foreach ($line in ($envResult.Stdout -split "`r?`n")) {
    if ($line -match '^([^=][^=]*)=(.*)$') {
        $compileStart.EnvironmentVariables[$matches[1]] = $matches[2]
    }
}
Write-Output "Compiler: $compiler"
Write-Output 'Building x64 CPU tile DLL with strict FP32 arithmetic...'
$compileResult = Invoke-ChildProcess $compileStart
if ($compileResult.Stdout) { Write-Output $compileResult.Stdout.TrimEnd() }
if ($compileResult.Stderr) { Write-Output $compileResult.Stderr.TrimEnd() }
if ($compileResult.ExitCode -ne 0) { throw "C compilation failed with exit code $($compileResult.ExitCode)." }
if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) { throw 'Compiler succeeded but no DLL was produced.' }
Write-Output "Built: $dll"

$topkSource = Join-Path $projectRoot 'native\topk_cpu.cpp'
$topkDll = Join-Path $buildRoot 'topk_cpu.dll'
$topkObject = Join-Path $buildRoot 'topk_cpu.obj'
$topkLibrary = Join-Path $buildRoot 'topk_cpu.lib'
$compileStart.Arguments = '/nologo /TP /std:c++17 /EHsc /O2 /W4 /WX /fp:strict /LD /Fo"{0}" /Fe"{1}" "{2}" /link /INCREMENTAL:NO /IMPLIB:"{3}"' -f $topkObject, $topkDll, $topkSource, $topkLibrary
$topkResult = Invoke-ChildProcess $compileStart
if ($topkResult.Stdout) { Write-Output $topkResult.Stdout.TrimEnd() }
if ($topkResult.Stderr) { Write-Output $topkResult.Stderr.TrimEnd() }
if ($topkResult.ExitCode -ne 0) { throw 'Bounded top-k compatibility build failed.' }
if (-not (Test-Path -LiteralPath $topkDll -PathType Leaf)) { throw 'No top-k DLL was produced.' }
Write-Output "Built: $topkDll"

$nvfp4Source = Join-Path $projectRoot 'native\nvfp4_cpu.c'
$nvfp4Dll = Join-Path $buildRoot 'nvfp4_cpu.dll'
$nvfp4Object = Join-Path $buildRoot 'nvfp4_cpu.obj'
$nvfp4Library = Join-Path $buildRoot 'nvfp4_cpu.lib'
$compileStart.Arguments = '/nologo /TC /std:c11 /O2 /W4 /WX /fp:strict /LD /Fo"{0}" /Fe"{1}" "{2}" /link /INCREMENTAL:NO /IMPLIB:"{3}"' -f $nvfp4Object, $nvfp4Dll, $nvfp4Source, $nvfp4Library
$nvfp4Result = Invoke-ChildProcess $compileStart
if ($nvfp4Result.Stdout) { Write-Output $nvfp4Result.Stdout.TrimEnd() }
if ($nvfp4Result.Stderr) { Write-Output $nvfp4Result.Stderr.TrimEnd() }
if ($nvfp4Result.ExitCode -ne 0) { throw 'NVFP4 CPU kernel build failed.' }
if (-not (Test-Path -LiteralPath $nvfp4Dll -PathType Leaf)) { throw 'No NVFP4 CPU DLL was produced.' }
Write-Output "Built: $nvfp4Dll"
