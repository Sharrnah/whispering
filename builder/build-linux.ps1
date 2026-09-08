param(
    [string]$UIRepository = 'G:\Projekte\Repositories\whispering-tiger-ui',
    [ValidateSet('cpu', 'cu128')][string]$Flavor = 'cu128',
    [string]$Version = ('linux-preview-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [switch]$EnableUpdates
)
$ErrorActionPreference = 'Stop'
$backendRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$uiRoot = (Resolve-Path -LiteralPath $UIRepository).Path
$buildRoot = Join-Path $backendRoot ('.linux-build\runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$artifactRoot = Join-Path $buildRoot 'artifacts'
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

function Invoke-DockerChecked {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed ($LASTEXITCODE): $args" }
}

Invoke-DockerChecked build -f (Join-Path $PSScriptRoot 'Dockerfile-linux64') -t whispering-tiger-builder:linux $PSScriptRoot
Invoke-DockerChecked run --rm `
    --mount "type=bind,source=$backendRoot,target=/backend-source,readonly" `
    --mount "type=bind,source=$uiRoot,target=/ui-source,readonly" `
    --mount "type=bind,source=$buildRoot,target=/work" `
    whispering-tiger-builder:linux python /backend-source/builder/linux-stage.py /backend-source /ui-source /work

$backendStage = Join-Path $buildRoot 'backend'
$uiStage = Join-Path $buildRoot 'ui'
$distVolume = 'wt-linux-dist-' + (Split-Path $buildRoot -Leaf)
Invoke-DockerChecked run --rm --init `
    --mount "type=bind,source=$backendStage,target=/src" `
    --mount "type=volume,source=$distVolume,target=/out" `
    --mount 'type=volume,source=wt-linux-pip-cache,target=/root/.cache/pip' `
    -e "TORCH_FLAVOR=$Flavor" whispering-tiger-builder:linux

$preview = if ($EnableUpdates) { 'false' } else { 'true' }
Invoke-DockerChecked run --rm --init `
    --mount "type=bind,source=$uiStage,target=/src" `
    --mount "type=bind,source=$backendStage,target=/build-tools,readonly" `
    --mount 'type=volume,source=wt-linux-go-mod,target=/go/pkg/mod' `
    --mount 'type=volume,source=wt-linux-go-build,target=/root/.cache/go-build' `
    -w /src -e "TORCH_FLAVOR=$Flavor" -e "WT_LINUX_PREVIEW=$preview" whispering-tiger-builder:linux `
    python /build-tools/builder/linux-ui-build.py

Invoke-DockerChecked run --rm `
    --mount "type=bind,source=$buildRoot,target=/work" `
    --mount "type=volume,source=$distVolume,target=/dist,readonly" whispering-tiger-builder:linux `
    python /work/backend/builder/linux-package.py /dist/audioWhisper /work/ui/Build/whispering-tiger-linux-amd64 /work/artifacts --flavor $Flavor --version $Version
Write-Host "Linux artifacts: $artifactRoot"
Write-Host "Unpacked backend retained in Docker volume: $distVolume"
