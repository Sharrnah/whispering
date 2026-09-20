<#
.SYNOPSIS
Build Windows and Linux backend ZIPs and UI executables into one release folder.
.EXAMPLE
.\build-all.bat -Version 1.3.19.3
.EXAMPLE
.\build-all.bat -CheckOnly
#>
param(
    [string]$Version = '',
    [string]$UIRepository = 'G:\Projekte\Repositories\whispering-tiger-ui',
    [ValidateSet('cpu', 'cu128')][string]$Flavor = 'cu128',
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$backendRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $backendRoot 'venv\Scripts\python.exe'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $Arguments" }
}

function Require-Path {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required build input is missing: $Path" }
}

Write-Host 'Checking Windows tools and the Linux Docker engine...'
foreach ($command in @('git', 'go', 'gcc', 'fyne', 'docker', 'powershell.exe')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Missing build tool: $command. See documentation/BUILD_ALL.md."
    }
}
Require-Path $python
$uiRoot = (Resolve-Path -LiteralPath $UIRepository).Path
foreach ($name in @('FyneApp.toml', 'BuildTools\package-windows.ps1')) {
    Require-Path (Join-Path $uiRoot $name)
}
foreach ($name in @(
    'audioWhisper.spec', 'builder\python-lib\include', 'builder\python-lib\libs',
    'builder\build-linux.ps1', 'builder\windows-package.py', 'builder\linux-startup-check.py',
    '.cache\nltk\tokenizers\punkt_tab\english',
    'toolchain\ffmpeg\bin\ffmpeg.exe', 'toolchain\ffmpeg\bin\ffprobe.exe',
    'toolchain\tcc\tcc.exe', 'dist_files\Plugins', 'markers', 'websocket_clients',
    'dist_files\help.bat', 'dist_files\get-device-list.bat', 'LICENSE', 'ignorelist.txt'
)) { Require-Path (Join-Path $backendRoot $name) }
Invoke-Checked $python @('-c', "import sys, struct, PyInstaller; assert sys.platform == 'win32' and struct.calcsize('P') == 8, 'A 64-bit Windows Python environment is required'; print('Windows Python / PyInstaller:', sys.version.split()[0], PyInstaller.__version__)")
$dockerOS = & docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0 -or ($dockerOS -join '').Trim() -ne 'linux') {
    throw 'Start Docker Desktop or Rancher Desktop with a Linux Docker engine, then run this script again.'
}
Write-Host "Backend Python: $python"
Write-Host "UI source: $uiRoot"
Write-Host 'UI release metadata (unchanged by either build):'
Get-Content -LiteralPath (Join-Path $uiRoot 'FyneApp.toml') |
    Where-Object { $_ -match '^\s*(Version|Build)\s*=' } | ForEach-Object { Write-Host "  $_" }
if ($CheckOnly) {
    Write-Host 'Build prerequisites found. No build was started.'
    return
}
if (-not $Version) { $Version = Read-Host 'Backend release version (for example 1.3.19.3)' }
if ($Version -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw 'Enter a backend version containing only letters, digits, dots, underscores or hyphens.'
}

$buildId = $Version + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
$releaseRoot = Join-Path $backendRoot ('dist\releases\' + $buildId)
$windowsRoot = Join-Path $releaseRoot 'windows'
$linuxRoot = Join-Path $releaseRoot 'linux'
$workRoot = Join-Path $backendRoot ('build\build-all\' + $buildId)
$frozenRoot = Join-Path $workRoot 'dist'
$originalLocation = Get-Location
$originalUTF8 = $env:PYTHONUTF8
New-Item -ItemType Directory -Path $windowsRoot, $workRoot | Out-Null
Write-Host "Release output: $releaseRoot"
Write-Host 'Windows uses the existing venv; Linux builds its own environment in Docker.'
Write-Host "Linux flavor: $Flavor. Release update checks are enabled."
try {
    Set-Location -LiteralPath $backendRoot
    $env:PYTHONUTF8 = '1'
    Write-Host '[1/4] Building the Windows Python backend...'
    Invoke-Checked $python @('-m', 'PyInstaller', 'audioWhisper.spec', '-y', '--clean',
        '--distpath', $frozenRoot, '--workpath', (Join-Path $workRoot 'pyinstaller'))
    $backend = Join-Path $frozenRoot 'audioWhisper'
    Invoke-Checked $python @('builder\linux-startup-check.py', (Join-Path $backend 'audioWhisper.exe'),
        '--log', (Join-Path $windowsRoot 'frozen-startup.log'))

    Write-Host '[2/4] Packaging the Windows UI...'
    Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $uiRoot 'BuildTools\package-windows.ps1'))

    Write-Host '[3/4] Creating the Windows backend ZIP and checksums...'
    Invoke-Checked $python @('builder\windows-package.py', $backend,
        (Join-Path $uiRoot 'Whispering Tiger.exe'), $windowsRoot, '--version', $Version)

    Write-Host '[4/4] Building and packaging the Linux backend and UI...'
    Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $PSScriptRoot 'build-linux.ps1'), '-UIRepository', $uiRoot,
        '-Version', $Version, '-Flavor', $Flavor, '-EnableUpdates', '-ArtifactDirectory', $linuxRoot)

    @"
Both platform builds completed successfully.
Backend version: $Version
UI version: Version.Build from $uiRoot\FyneApp.toml

windows\ : backend ZIP, Whispering Tiger.exe, SHA256SUMS and startup log.
linux\   : backend ZIP, Linux UI executable, SHA256SUMS and release notes.

Extract each backend ZIP and place its UI executable beside the audioWhisper directory.
The ZIPs contain the Python runtime. Model weights are downloaded when selected.
Nothing was uploaded or published. Test on the target desktop before publishing.
Windows intermediate files: $workRoot
"@ | Set-Content -LiteralPath (Join-Path $releaseRoot 'BUILD-SUCCESS.txt') -Encoding UTF8
    Write-Host ''
    Write-Host "BUILD COMPLETE: $releaseRoot"
    Write-Host "Windows: $windowsRoot"
    Write-Host "Linux:   $linuxRoot"
} catch {
    Write-Host "BUILD FAILED. Completed artifacts and logs remain in: $releaseRoot"
    throw
} finally {
    Set-Location -LiteralPath $originalLocation.Path
    $env:PYTHONUTF8 = $originalUTF8
}
