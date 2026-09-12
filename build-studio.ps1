[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [switch]$Test
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Push-Location $PSScriptRoot
try {
    & dotnet build ModelDesk.sln --configuration $Configuration
    if ($LASTEXITCODE -ne 0) { throw 'ModelDesk solution build failed.' }
    if ($Test) {
        & dotnet test studio\tests\ModelDesk.Tests\ModelDesk.Tests.csproj --configuration $Configuration --no-build --logger 'trx;LogFileName=studio-tests.trx' --results-directory reports\studio\tests
        if ($LASTEXITCODE -ne 0) { throw 'ModelDesk tests failed.' }
    }
}
finally { Pop-Location }
