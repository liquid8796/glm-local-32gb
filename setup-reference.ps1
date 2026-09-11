[CmdletBinding()]
param([string]$Python = 'python')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$referenceProjectRoot = $PSScriptRoot
$referenceEnvironmentRoot = Join-Path $referenceProjectRoot '.venv-reference'
$referencePython = Join-Path $referenceEnvironmentRoot 'Scripts\python.exe'
$referenceLockPath = Join-Path $referenceProjectRoot 'config\reference-lock.json'
$referenceRequirementsPath = Join-Path $referenceProjectRoot 'requirements-reference.lock.txt'
$referenceLock = Get-Content -LiteralPath $referenceLockPath -Raw | ConvertFrom-Json

if ($referenceLock.schema_version -ne 1) { throw 'Unsupported reference lock schema.' }
if ($referenceLock.torch_index_url -ne 'https://download.pytorch.org/whl/cpu' -or
    $referenceLock.pypi_index_url -ne 'https://pypi.org/simple') {
    throw 'Reference dependencies require the pinned official package indexes.'
}

Push-Location -LiteralPath $referenceProjectRoot
try {
    if (-not (Test-Path -LiteralPath $referenceEnvironmentRoot)) {
        Write-Output 'Creating project-local .venv-reference...'
        & $Python -m venv $referenceEnvironmentRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the reference virtual environment.' }
    }
    if (-not (Test-Path -LiteralPath $referencePython -PathType Leaf)) {
        throw 'The existing .venv-reference has no Windows Python executable; it was left unchanged.'
    }

    $referenceCheck = @(& $referencePython -m glm_local.reference_env --verify)
    if ($LASTEXITCODE -eq 0) {
        Write-Output 'Pinned reference environment already verified; no downloads or installs needed.'
        Write-Output $referenceCheck
    }
    else {
        Write-Output 'Installing the pinned optional oracle into .venv-reference...'
        & $referencePython -m pip --isolated --disable-pip-version-check install --no-input --no-cache-dir `
            --index-url $referenceLock.torch_index_url "torch==$($referenceLock.packages.torch)"
        if ($LASTEXITCODE -ne 0) { throw 'CPU Torch installation failed.' }

        if (Test-Path -LiteralPath $referenceRequirementsPath -PathType Leaf) {
            & $referencePython -m pip --isolated --disable-pip-version-check install --no-input --no-cache-dir `
                --index-url $referenceLock.pypi_index_url -r $referenceRequirementsPath
        }
        else {
            $referenceTransformersRequirement = "transformers @ $($referenceLock.transformers.url)#sha256=$($referenceLock.transformers.archive_sha256)"
            & $referencePython -m pip --isolated --disable-pip-version-check install --no-input --no-cache-dir `
                --index-url $referenceLock.pypi_index_url "numpy==$($referenceLock.packages.numpy)" `
                $referenceTransformersRequirement
        }
        if ($LASTEXITCODE -ne 0) { throw 'NumPy or Transformers reference installation failed.' }
        & $referencePython -m glm_local.reference_env --verify
        if ($LASTEXITCODE -ne 0) { throw 'Installed reference environment did not pass provenance verification.' }
    }
}
finally {
    Pop-Location
}
