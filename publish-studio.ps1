[CmdletBinding()]
param([switch]$SkipTests, [switch]$FrameworkDependent)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$studioProjectRoot = $PSScriptRoot
Push-Location $studioProjectRoot
try {
    & (Join-Path $studioProjectRoot 'build-studio.ps1') -Configuration Release -Test:(-not $SkipTests)
    $studioOutput = Join-Path $studioProjectRoot 'artifacts\ModelDesk\win-x64'
    [void](New-Item -ItemType Directory -Force -Path $studioOutput)
    $selfContained = if ($FrameworkDependent) { 'false' } else { 'true' }
    foreach ($project in @('ModelDesk.Desktop', 'ModelDesk.Cli')) {
        & dotnet publish "studio\src\$project\$project.csproj" --configuration Release --runtime win-x64 --self-contained $selfContained --output $studioOutput
        if ($LASTEXITCODE -ne 0) { throw "Publish failed for $project" }
    }
    # Distribute the existing Python files verbatim; no venv, models or run data.
    $coreOutput = Join-Path $studioOutput 'core'
    [void](New-Item -ItemType Directory -Force -Path $coreOutput)
    foreach ($folder in @('glm_local', 'native', 'config', 'docs', 'tests')) {
        $sourceRoot = Join-Path $studioProjectRoot $folder
        foreach ($file in (Get-ChildItem -LiteralPath $sourceRoot -File -Recurse)) {
            if ($file.FullName -match '[\\/]__pycache__[\\/]' -or $file.Extension -in @('.pyc', '.pyo')) { continue }
            $relative = $file.FullName.Substring($studioProjectRoot.Length + 1)
            $target = Join-Path $coreOutput $relative
            [void](New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target))
            Copy-Item -LiteralPath $file.FullName -Destination $target -Force
        }
    }
    foreach ($file in (Get-ChildItem -LiteralPath $studioProjectRoot -File)) {
        if ($file.Name -in @('README.md', 'LICENSE', 'pyproject.toml', 'requirements-reference.lock.txt') -or
            ($file.Extension -in @('.bat', '.ps1') -and $file.Name -notmatch 'studio|modeldesk')) {
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $coreOutput $file.Name) -Force
        }
    }
    $nativeOutput = Join-Path $coreOutput 'build'
    [void](New-Item -ItemType Directory -Force -Path $nativeOutput)
    foreach ($name in @('fp8_cpu.dll', 'topk_cpu.dll', 'nvfp4_cpu.dll')) {
        $source = Join-Path $studioProjectRoot "build\$name"
        if (Test-Path -LiteralPath $source -PathType Leaf) { Copy-Item -LiteralPath $source -Destination (Join-Path $nativeOutput $name) -Force }
    }
    Copy-Item -LiteralPath (Join-Path $studioProjectRoot 'studio\README.md') -Destination (Join-Path $studioOutput 'README.md') -Force
    $receipt = [ordered]@{
        application = 'ModelDesk'; version = '1.0.0'; framework = '.NET 10'; runtime = 'win-x64'
        selfContained = (-not $FrameworkDependent); publishedUtc = [DateTime]::UtcNow.ToString('o')
        pythonBundled = $false; pythonCoreCopiedWithoutChanges = $true; modelWeightsIncluded = $false
    }
    $receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $studioOutput 'publish-receipt.json') -Encoding UTF8
    Write-Output "Published ModelDesk: $studioOutput"
}
finally { Pop-Location }
