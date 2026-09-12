[CmdletBinding()]
param([switch]$Apply)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$cleanupRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$rootPrefix = $cleanupRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$candidates = [Collections.Generic.Dictionary[string, IO.FileInfo]]::new([StringComparer]::OrdinalIgnoreCase)
$skippedGroups = [Collections.Generic.List[object]]::new()

function Assert-ProjectPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Path escaped the project: $full" }
    $current = $full
    while ($current -ne $cleanupRoot) {
        if (Test-Path -LiteralPath $current) {
            if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Cache path contains a link/reparse point: $current"
            }
        }
        $current = Split-Path -Parent $current
    }
    return $full
}

function Add-CacheFiles([string]$Relative) {
    $directory = Assert-ProjectPath (Join-Path $cleanupRoot $Relative)
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { return }
    $files = @(Get-ChildItem -LiteralPath $directory -File -Recurse -Force)
    # Keep an entire output folder if an application/compiler still has one of
    # its files open. Do not leave an active application with half its files gone.
    foreach ($file in $files) {
        try {
            [void](Assert-ProjectPath $file.FullName)
            $probe = [IO.File]::Open($file.FullName, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
            $probe.Dispose()
        }
        catch {
            $skippedGroups.Add([ordered]@{ path = $directory; reason = 'In use or not writable'; file = $file.FullName })
            return
        }
    }
    foreach ($file in $files) {
        $checked = Assert-ProjectPath $file.FullName
        $candidates[$checked] = $file
    }
}

# Only generated file sets are eligible. Directories, Git internals, source,
# virtual environments, model data and active metadata evidence are preserved.
foreach ($relative in @('glm_local\__pycache__', 'glm_local\architecture\__pycache__', 'tests\__pycache__',
                         'studio\.agent-build', 'artifacts\qa', 'reports\studio\benchbuild')) {
    Add-CacheFiles $relative
}
foreach ($group in @('studio\src', 'studio\tests', 'studio\benchmarks')) {
    $directory = Assert-ProjectPath (Join-Path $cleanupRoot $group)
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { continue }
    foreach ($project in (Get-ChildItem -LiteralPath $directory -Directory)) {
        $relative = $project.FullName.Substring($rootPrefix.Length)
        Add-CacheFiles (Join-Path $relative 'bin')
        Add-CacheFiles (Join-Path $relative 'obj')
    }
}
$nativeBuild = Assert-ProjectPath (Join-Path $cleanupRoot 'build')
if (Test-Path -LiteralPath $nativeBuild -PathType Container) {
    foreach ($file in (Get-ChildItem -LiteralPath $nativeBuild -File)) {
        if ($file.Extension -in @('.obj', '.exp', '.lib', '.pdb')) { $candidates[(Assert-ProjectPath $file.FullName)] = $file }
    }
}

$plannedBytes = [long]0
foreach ($file in $candidates.Values) { $plannedBytes += $file.Length }
$removedBytes = [long]0
$removed = 0
$failed = [Collections.Generic.List[object]]::new()
if ($Apply) {
    foreach ($entry in ($candidates.GetEnumerator() | Sort-Object Key)) {
        try {
            $checked = Assert-ProjectPath $entry.Key
            $file = Get-Item -LiteralPath $checked -Force
            if ($file.PSIsContainer) { throw 'Candidate changed into a directory.' }
            $bytes = $file.Length
            # File-only deletion deliberately avoids recursive removal and never forces locked files.
            Remove-Item -LiteralPath $checked -ErrorAction Stop
            $removedBytes += $bytes
            $removed++
        }
        catch { $failed.Add([ordered]@{ path = $entry.Key; error = $_.Exception.Message }) }
    }
}
$summary = [ordered]@{
    mode = $(if ($Apply) { 'applied' } else { 'preview' })
    project = $cleanupRoot
    plannedFiles = $candidates.Count
    plannedBytes = [long]$plannedBytes
    removedFiles = $removed
    removedBytes = $removedBytes
    failed = @($failed.ToArray())
    skippedGroups = @($skippedGroups.ToArray())
    preserved = @('Python/native source', 'native runtime DLLs', 'models', '.venv-reference', 'metadata evidence', 'Git history', 'published app', 'local vendor checkout')
}
$reportDirectory = Join-Path $cleanupRoot 'reports\studio'
[void](New-Item -ItemType Directory -Path $reportDirectory -Force)
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportDirectory 'cleanup-latest.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 5
if ($failed.Count) { throw 'Some cache files were in use or could not be removed; see reports/studio/cleanup-latest.json.' }
