# Run inside Docker with PowerShell; no model downloads or actual builds.
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
$originalScript = Join-Path $repository 'builder/build-linux.ps1'
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('wt-cleanup-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $fixtureRoot 'builder'), (Join-Path $fixtureRoot 'ui') -Force | Out-Null
Copy-Item -LiteralPath $originalScript -Destination (Join-Path $fixtureRoot 'builder/build-linux.ps1')

function Assert-True($condition, [string]$message) {
    if (-not $condition) { throw $message }
}

function global:docker {
    $command = @($args | ForEach-Object { [string]$_ })
    $global:wtCleanupTest_calls.Add(($command -join ' '))
    $global:LASTEXITCODE = 0
    switch ($command[0]) {
        'build' {
            if ($global:wtCleanupTest_failure -eq 'image') { $global:LASTEXITCODE = 17 }
            return
        }
        'run' {
            Assert-True ($command -contains '--rm') 'Build container must use --rm'
            $label = $command[[array]::IndexOf($command, '--label') + 1]
            $name = $command[[array]::IndexOf($command, '--name') + 1]
            $stage = ($name -split '-')[-1]
            $global:wtCleanupTest_containers[$name] = $label
            if ($global:wtCleanupTest_failure -eq $stage) {
                # Also simulate an interrupted run leaving a container behind.
                $global:LASTEXITCODE = 17
                return
            }
            if ($stage -eq 'package') {
                $mount = $command | Where-Object { $_ -like 'type=bind,*target=/artifacts' }
                $directory = $mount -replace '^type=bind,source=', '' -replace ',target=/artifacts$', ''
                Set-Content -LiteralPath (Join-Path $directory 'release.zip') -Value 'preserve this artifact'
            }
            $global:wtCleanupTest_containers.Remove($name)
            return
        }
        'ps' {
            $filter = $command[[array]::IndexOf($command, '--filter') + 1] -replace '^label=', ''
            Assert-True ($filter -like 'space.libs.whispering.linux-build-run=*') 'Container cleanup must be scoped to one run'
            foreach ($name in $global:wtCleanupTest_containers.Keys) {
                if ($global:wtCleanupTest_containers[$name] -eq $filter) { $name }
            }
            return
        }
        'rm' {
            $name = $command[-1]
            Assert-True ($name -ne 'other-running-build') 'Cleanup touched another build container'
            $global:wtCleanupTest_containers.Remove($name)
            return
        }
        'volume' {
            switch ($command[1]) {
                'create' {
                    $name = $command[-1]
                    $label = $command[[array]::IndexOf($command, '--label') + 1]
                    Assert-True ($label -like 'space.libs.whispering.linux-build-run=*') 'Temporary volume is not labeled'
                    $global:wtCleanupTest_volumes[$name] = $label
                    $name
                    return
                }
                'ls' {
                    $filter = $null
                    if ($command -contains '--filter') {
                        $filter = $command[[array]::IndexOf($command, '--filter') + 1] -replace '^label=', ''
                    }
                    foreach ($name in $global:wtCleanupTest_volumes.Keys) {
                        if ($null -eq $filter -or $global:wtCleanupTest_volumes[$name] -eq $filter) { $name }
                    }
                    return
                }
                'rm' {
                    $name = $command[-1]
                    Assert-True ($name -notin @('other-project-data', 'other-build-source')) 'Cleanup touched unrelated data'
                    Assert-True ($command -notcontains '-f') 'Volume removal must respect active users'
                    if ($global:wtCleanupTest_cleanupFailure) {
                        $global:LASTEXITCODE = 19
                        'simulated Docker cleanup failure'
                        return
                    }
                    $global:wtCleanupTest_volumes.Remove($name)
                    return
                }
            }
        }
    }
    throw "Unexpected Docker operation: $command"
}

$cases = @(
    @{ Name = 'success'; Failure = '' },
    @{ Name = 'custom artifact directory'; Failure = ''; CustomOutput = $true },
    @{ Name = 'image failure'; Failure = 'image' },
    @{ Name = 'staging failure'; Failure = 'stage' },
    @{ Name = 'backend failure'; Failure = 'backend' },
    @{ Name = 'UI failure'; Failure = 'ui' },
    @{ Name = 'packaging failure'; Failure = 'package' },
    @{ Name = 'debug retention'; Failure = 'backend'; Keep = $true },
    @{ Name = 'explicit cache cleanup'; Failure = ''; Caches = $true },
    @{ Name = 'cleanup failure preserves build failure'; Failure = 'backend'; CleanupFailure = $true }
)
foreach ($case in $cases) {
    $global:wtCleanupTest_failure = $case.Failure
    $global:wtCleanupTest_cleanupFailure = [bool]$case.CleanupFailure
    $global:wtCleanupTest_calls = New-Object 'System.Collections.Generic.List[string]'
    $global:wtCleanupTest_volumes = @{
        'other-project-data' = ''
        'other-build-source' = 'space.libs.whispering.linux-build-run=another-run'
        'wt-linux-pip-cache' = ''
        'wt-linux-go-mod' = ''
        'wt-linux-go-build' = ''
        'wt-linux-go-cache' = ''
    }
    $global:wtCleanupTest_containers = @{ 'other-running-build' = 'space.libs.whispering.linux-build-run=another-run' }
    $failureMessage = ''
    $outputOptions = @{}
    if ($case.CustomOutput) { $outputOptions.ArtifactDirectory = Join-Path $fixtureRoot 'combined release\linux' }
    try {
        & (Join-Path $fixtureRoot 'builder/build-linux.ps1') -UIRepository (Join-Path $fixtureRoot 'ui') `
            -KeepBuildData:([bool]$case.Keep) -CleanCaches:([bool]$case.Caches) @outputOptions
    } catch {
        $failureMessage = $_.Exception.Message
    }
    if ($case.Failure) {
        Assert-True ($failureMessage -like 'Docker command failed (17):*') "Original failure lost in $($case.Name): $failureMessage"
    } else {
        Assert-True (-not $failureMessage) "Unexpected failure in $($case.Name): $failureMessage"
    }
    Assert-True ($global:wtCleanupTest_containers.Count -eq 1) "A temporary container survived $($case.Name)"
    Assert-True ($global:wtCleanupTest_volumes.ContainsKey('other-project-data')) 'Other project volume lost'
    Assert-True ($global:wtCleanupTest_volumes.ContainsKey('other-build-source')) 'Concurrent build volume lost'
    $temporaryVolumes = @($global:wtCleanupTest_volumes.Keys | Where-Object { $_ -match '^wt-linux-(source|dist)-' })
    $expectedCount = if ($case.Keep -or $case.CleanupFailure) { 2 } else { 0 }
    Assert-True ($temporaryVolumes.Count -eq $expectedCount) "Unexpected retained volumes in $($case.Name): $temporaryVolumes"
    Assert-True ($global:wtCleanupTest_volumes.ContainsKey('wt-linux-pip-cache') -ne [bool]$case.Caches) 'Cache policy was not respected'
    if (-not $case.Failure) {
        $artifactSearch = if ($case.CustomOutput) { $outputOptions.ArtifactDirectory } else { Join-Path $fixtureRoot '.linux-build/runs' }
        $artifacts = @(Get-ChildItem -LiteralPath $artifactSearch -Filter release.zip -Recurse)
        Assert-True ($artifacts.Count -gt 0) 'Finished artifact was removed'
    }
    Write-Host "PASS: $($case.Name)"
}
$callsBefore = $global:wtCleanupTest_calls.Count
$caught = ''
try {
    & (Join-Path $fixtureRoot 'builder/build-linux.ps1') -UIRepository (Join-Path $fixtureRoot 'ui') `
        -ArtifactDirectory (Join-Path $fixtureRoot 'combined release\linux')
} catch { $caught = $_.Exception.Message }
Assert-True ($caught -like 'Artifact directory must be empty*') 'Existing release artifacts can be overwritten'
Assert-True ($global:wtCleanupTest_calls.Count -eq $callsBefore) 'Docker started before checking existing artifacts'
Write-Host 'PASS: existing release output is preserved before Docker starts'
Write-Host "PASS: all $($cases.Count) build cleanup scenarios"
