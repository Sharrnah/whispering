param(
    [string]$UIRepository = 'G:\Projekte\Repositories\whispering-tiger-ui',
    [ValidateSet('cpu', 'cu128')][string]$Flavor = 'cu128',
    [string]$Version = ('linux-preview-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [switch]$EnableUpdates,
    [string]$DownloadBaseUrl = '',
    [switch]$KeepBuildData,
    [switch]$CleanCaches,
    [string]$ArtifactDirectory = ''
)
$ErrorActionPreference = 'Stop'
$backendRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$uiRoot = (Resolve-Path -LiteralPath $UIRepository).Path
$buildId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$buildRoot = Join-Path $backendRoot ('.linux-build\runs\' + $buildId)
$artifactRoot = if ($ArtifactDirectory) {
    [IO.Path]::GetFullPath($ArtifactDirectory)
} else { Join-Path $buildRoot 'artifacts' }
if ((Test-Path -LiteralPath $artifactRoot) -and
    @(Get-ChildItem -LiteralPath $artifactRoot -Force).Count -gt 0) {
    throw "Artifact directory must be empty to avoid overwriting an earlier build: $artifactRoot"
}
$sourceVolume = 'wt-linux-source-' + $buildId
$distVolume = 'wt-linux-dist-' + $buildId
$runLabel = 'space.libs.whispering.linux-build-run=' + [guid]::NewGuid().ToString('N')
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

function Invoke-DockerChecked {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed ($LASTEXITCODE): $args" }
}

function Invoke-DockerCleanup {
    param([string[]]$DockerArguments)
    # Cleanup must not replace the original build error if Docker stops or a
    # resource cannot be removed. Native stderr can throw in Windows PowerShell.
    try {
        $output = & docker @DockerArguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Docker cleanup failed: docker $DockerArguments -- $output"
            return
        }
        $output
    } catch {
        Write-Warning "Docker cleanup failed: docker $DockerArguments -- $_"
    }
}

function Clear-LinuxBuildTemporaryData {
    # Labels belong to this invocation only, so concurrent builds and other
    # Docker projects are never stopped or pruned here.
    $containers = @(Invoke-DockerCleanup -DockerArguments @('ps', '-aq', '--filter', "label=$runLabel"))
    foreach ($container in $containers) {
        Invoke-DockerCleanup -DockerArguments @('rm', '-f', $container) | Out-Null
    }
    if ($KeepBuildData) {
        Write-Host "Debug build data retained: $sourceVolume, $distVolume"
    } else {
        $volumes = @(Invoke-DockerCleanup -DockerArguments @('volume', 'ls', '-q', '--filter', "label=$runLabel"))
        foreach ($volume in $volumes) {
            Invoke-DockerCleanup -DockerArguments @('volume', 'rm', $volume) | Out-Null
        }
    }
    if ($CleanCaches) {
        # Remove only this project's reusable caches. Docker refuses removal
        # when another container still uses a cache; never force it.
        $volumes = @(Invoke-DockerCleanup -DockerArguments @('volume', 'ls', '-q'))
        foreach ($cache in @('wt-linux-pip-cache', 'wt-linux-go-mod', 'wt-linux-go-build', 'wt-linux-go-cache')) {
            if ($volumes -contains $cache) {
                Invoke-DockerCleanup -DockerArguments @('volume', 'rm', $cache) | Out-Null
            }
        }
    }
}

try {
    Invoke-DockerChecked build -f (Join-Path $PSScriptRoot 'Dockerfile-linux64') -t whispering-tiger-builder:linux $PSScriptRoot
    Invoke-DockerChecked volume create --label $runLabel $sourceVolume | Out-Null
    Invoke-DockerChecked volume create --label $runLabel $distVolume | Out-Null
    Write-Host "Backend source: $backendRoot"
    Write-Host "UI source: $uiRoot"
    Invoke-DockerChecked run --rm --name "wt-linux-$buildId-stage" --label $runLabel `
        --mount "type=bind,source=$backendRoot,target=/backend-source,readonly" `
        --mount "type=bind,source=$uiRoot,target=/ui-source,readonly" `
        --mount "type=volume,source=$sourceVolume,target=/work" `
        -e "WT_BACKEND_REPOSITORY=$backendRoot" -e "WT_UI_REPOSITORY=$uiRoot" `
        whispering-tiger-builder:linux python /backend-source/builder/linux-stage.py /backend-source /ui-source /work

    Invoke-DockerChecked run --rm --init --name "wt-linux-$buildId-backend" --label $runLabel `
        --mount "type=volume,source=$sourceVolume,target=/work" `
        --mount "type=volume,source=$distVolume,target=/out" `
        --mount "type=bind,source=$artifactRoot,target=/artifacts" `
        --mount 'type=volume,source=wt-linux-pip-cache,target=/root/.cache/pip' `
        -e SRCDIR=/work/backend -e WT_BUILD_LOG_DIR=/artifacts -e "TORCH_FLAVOR=$Flavor" whispering-tiger-builder:linux

    $preview = if ($EnableUpdates) { 'false' } else { 'true' }
    Invoke-DockerChecked run --rm --init --name "wt-linux-$buildId-ui" --label $runLabel `
        --mount "type=volume,source=$sourceVolume,target=/work" `
        --mount 'type=volume,source=wt-linux-go-mod,target=/go/pkg/mod' `
        --mount 'type=volume,source=wt-linux-go-build,target=/root/.cache/go-build' `
        -w /work/ui -e "TORCH_FLAVOR=$Flavor" -e "WT_LINUX_PREVIEW=$preview" whispering-tiger-builder:linux `
        python /work/backend/builder/linux-ui-build.py

    $downloadOptions = @()
    if ($DownloadBaseUrl) { $downloadOptions = @('--download-base-url', $DownloadBaseUrl) }
    if ($EnableUpdates) { $downloadOptions += '--release' }
    Invoke-DockerChecked run --rm --name "wt-linux-$buildId-package" --label $runLabel `
        --mount "type=volume,source=$sourceVolume,target=/work,readonly" `
        --mount "type=bind,source=$artifactRoot,target=/artifacts" `
        --mount "type=volume,source=$distVolume,target=/dist,readonly" whispering-tiger-builder:linux `
        python /work/backend/builder/linux-package.py /dist/audioWhisper /work/ui/Build/whispering-tiger-linux-amd64 /artifacts --flavor $Flavor --version $Version @downloadOptions
    Write-Host "Linux artifacts: $artifactRoot"
} finally {
    Clear-LinuxBuildTemporaryData
}
