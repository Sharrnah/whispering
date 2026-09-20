# Exercise the real orchestrator with fake external compilers, without downloads.
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('wt-build-all-test ' + [guid]::NewGuid().ToString('N'))
$ui = Join-Path $fixture 'UI source'
$python = Join-Path $fixture 'venv\Scripts\python.exe'
$originalLocation = (Get-Location).Path
$originalUTF8 = $env:PYTHONUTF8
function Assert-True($condition, [string]$message) {
    if (-not $condition) { throw $message }
}
function Get-Argument($arguments, [string]$name) {
    $index = [array]::IndexOf($arguments, $name)
    if ($index -lt 0) { throw "Missing argument: $name" }
    return $arguments[$index + 1]
}
function global:docker {
    Assert-True (($args -join ' ') -eq 'info --format {{.OSType}}') 'Unexpected Docker command'
    $global:LASTEXITCODE = 0
    'linux'
}
function global:powershell.exe {
    $script = Get-Argument $args '-File'
    $stage = if ($args -contains '-CheckOnly') { 'ui-preflight' }
        elseif ($script -like '*package-windows.ps1') { 'ui' } else { 'linux' }
    $global:wtBuild_calls.Add($stage)
    $global:LASTEXITCODE = if ($global:wtBuild_failure -eq $stage) { 23 } else { 0 }
    if ($global:LASTEXITCODE) { return }
    if ($stage -eq 'linux') {
        Assert-True ($args -contains '-EnableUpdates') 'Linux must be a release build'
        Assert-True ((Get-Argument $args '-Version') -eq '1.2.3') 'Backend version lost'
        Assert-True ((Get-Argument $args '-Flavor') -eq 'cpu') 'Linux flavor lost'
        Assert-True ((Get-Argument $args '-UIRepository') -eq $global:wtBuild_ui) 'UI path lost'
        $output = Get-Argument $args '-ArtifactDirectory'
        Assert-True ($output -like '*\dist\releases\*\linux') 'Linux output not in shared release folder'
        New-Item -ItemType Directory -Path $output | Out-Null
        Set-Content -LiteralPath (Join-Path $output 'linux.zip') -Value 'Linux release'
    }
}
try {
    foreach ($name in @(
        'venv/Scripts/python.exe', 'audioWhisper.spec', 'builder/python-lib/include/stub.h',
        'builder/python-lib/libs/stub.lib', 'builder/build-linux.ps1', 'builder/windows-package.py',
        'builder/linux-startup-check.py', '.cache/nltk/tokenizers/punkt_tab/english/stub',
        'toolchain/ffmpeg/bin/ffmpeg.exe', 'toolchain/ffmpeg/bin/ffprobe.exe', 'toolchain/tcc/tcc.exe',
        'dist_files/Plugins/placeholder', 'markers/stub', 'websocket_clients/stub',
        'dist_files/help.bat', 'dist_files/get-device-list.bat', 'LICENSE', 'ignorelist.txt',
        'UI source/BuildTools/package-windows.ps1', 'UI source/FyneApp.toml'
    )) {
        $path = Join-Path $fixture $name
        New-Item -ItemType Directory -Path (Split-Path $path -Parent) -Force | Out-Null
        Set-Content -LiteralPath $path -Value 'fixture'
    }
    Copy-Item -LiteralPath (Join-Path $repository 'builder/build-all.ps1') -Destination (Join-Path $fixture 'builder/build-all.ps1')
    # PowerShell command resolution permits functions named by executable path.
    # The real script still constructs and invokes its normal argument arrays.
    Set-Item -LiteralPath "Function:global:$python" -Value {
        $stage = if ($args[0] -eq '-c') { 'preflight' }
            elseif ($args -contains 'PyInstaller') { 'backend' }
            elseif ($args[0] -like '*linux-startup-check.py') { 'startup' }
            else { 'package' }
        $global:wtBuild_calls.Add($stage)
        $global:LASTEXITCODE = if ($global:wtBuild_failure -eq $stage) { 23 } else { 0 }
        if ($global:LASTEXITCODE) { return }
        if ($stage -eq 'backend') {
            Assert-True ($args -contains '--clean') 'Missing clean build'
            Assert-True ((Get-Argument $args '--distpath') -like '*\build\build-all\*\dist') 'Build overwrites normal dist'
        }
        if ($stage -eq 'package') {
            Assert-True ((Get-Argument $args '--version') -eq '1.2.3') 'Windows version lost'
            Set-Content -LiteralPath (Join-Path $args[3] 'windows.zip') -Value 'Windows release'
        }
    }
    $global:wtBuild_ui = $ui
    foreach ($failure in @('', 'ui-preflight', 'backend', 'startup', 'ui', 'package', 'linux')) {
        $global:wtBuild_failure = $failure
        $global:wtBuild_calls = New-Object 'System.Collections.Generic.List[string]'
        $before = @(Get-ChildItem -LiteralPath (Join-Path $fixture 'dist/releases') -Filter BUILD-SUCCESS.txt -Recurse -ErrorAction SilentlyContinue).Count
        $caught = ''
        try {
            & (Join-Path $fixture 'builder/build-all.ps1') -Version '1.2.3' -UIRepository $ui -Flavor cpu
        } catch { $caught = $_.Exception.Message }
        if ($failure) {
            Assert-True ($caught -like 'Command failed (23):*') "Original failure lost: $caught"
            Assert-True ($global:wtBuild_calls[-1] -eq $failure) 'Build continued after failure'
        } else {
            Assert-True (-not $caught) "Unexpected failure: $caught"
            Assert-True (($global:wtBuild_calls -join ',') -eq 'preflight,ui-preflight,backend,startup,ui,package,linux') 'Incorrect build order'
        }
        $after = @(Get-ChildItem -LiteralPath (Join-Path $fixture 'dist/releases') -Filter BUILD-SUCCESS.txt -Recurse).Count
        Assert-True (($after - $before) -eq [int](-not $failure)) 'Incorrect success marker'
        Assert-True ((Get-Location).Path -eq $originalLocation) 'Working directory not restored'
        Assert-True ($env:PYTHONUTF8 -eq $originalUTF8) 'Python environment not restored'
        Write-Host "PASS: build flow (failure stage: '$failure')"
    }
    $global:wtBuild_calls.Clear()
    & (Join-Path $fixture 'builder/build-all.ps1') -CheckOnly -UIRepository $ui
    Assert-True (($global:wtBuild_calls -join ',') -eq 'preflight,ui-preflight') 'CheckOnly started a build'
    $global:wtBuild_failure = ''
    $global:wtBuild_calls.Clear()
    $resumeId = '1.2.3-20260101-120000-001'
    $oldBackend = Join-Path $fixture "build\build-all\$resumeId\dist\audioWhisper"
    New-Item -ItemType Directory -Path (Join-Path $oldBackend '_internal') -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $oldBackend 'audioWhisper.exe') -Value 'existing backend'
    & (Join-Path $fixture 'builder/build-all.ps1') -ResumeBuild $resumeId -UIRepository $ui -Flavor cpu
    Assert-True (($global:wtBuild_calls -join ',') -eq 'preflight,ui-preflight,startup,ui,package,linux') 'Resume rebuilt the Python backend or skipped startup validation'
    Assert-True ((Get-Content -LiteralPath (Join-Path $oldBackend 'audioWhisper.exe')) -eq 'existing backend') 'Resume modified the original backend'
    foreach ($badResume in @('../outside', '1.2.3-20260101-120000-999')) {
        $caught = ''
        try { & (Join-Path $fixture 'builder/build-all.ps1') -ResumeBuild $badResume -UIRepository $ui }
        catch { $caught = $_.Exception.Message }
        Assert-True ([bool]$caught) "Invalid or missing resume build accepted: $badResume"
    }
    $caught = ''
    try { & (Join-Path $fixture 'builder/build-all.ps1') -Version '../bad' -UIRepository $ui }
    catch { $caught = $_.Exception.Message }
    Assert-True ($caught -like 'Enter a backend version*') 'Invalid version accepted'
    Write-Host 'PASS: preflight only, version validation, paths with spaces and release failure handling.'
} finally {
    Remove-Item -LiteralPath "Function:global:$python" -ErrorAction SilentlyContinue
    Remove-Item Function:\docker, Function:\powershell.exe
    Remove-Variable wtBuild_calls, wtBuild_failure, wtBuild_ui -Scope Global -ErrorAction SilentlyContinue
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempPrefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path $resolved -Leaf) -notmatch '^wt-build-all-test [0-9a-f]{32}$') {
        throw "Refusing to remove unexpected fixture: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
