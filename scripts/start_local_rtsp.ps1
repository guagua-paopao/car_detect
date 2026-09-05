[CmdletBinding()]
param(
    [string]$Video = ".\demo\rtsp_standard\output\vcas_rtsp_demo_60s.mp4",
    [int]$RtspPort = 18554,
    [string]$StreamName = "vcas-perf",
    [string]$StatePath = ".\runtime\rtsp-test\state.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$videoPath = if ([IO.Path]::IsPathRooted($Video)) {
    [IO.Path]::GetFullPath($Video)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $Video))
}
if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) {
    throw "RTSP test video is missing: $videoPath"
}
if ($RtspPort -lt 1024 -or $RtspPort -gt 65535) {
    throw "RtspPort must be in [1024, 65535]"
}
if ($StreamName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "StreamName may only contain letters, numbers, dot, underscore, and dash"
}
$existingListener = Get-NetTCPConnection -State Listen -LocalPort $RtspPort `
    -ErrorAction SilentlyContinue
if ($existingListener) {
    throw "RTSP test port is already in use: $RtspPort"
}

$ffmpeg = (Get-Command ffmpeg.exe -CommandType Application -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe.exe -CommandType Application -ErrorAction Stop).Source
$mediaArchive = Join-Path $projectRoot `
    "demo\rtsp_standard\tools\mediamtx_v1.19.3_windows_amd64.zip"
if (-not (Test-Path -LiteralPath $mediaArchive -PathType Leaf)) {
    throw "Bundled MediaMTX archive is missing: $mediaArchive"
}

$runtimeRoot = Join-Path $projectRoot "runtime\rtsp-test"
$mediaRoot = Join-Path $runtimeRoot "mediamtx-1.19.3"
$mediaExe = Join-Path $mediaRoot "mediamtx.exe"
$logRoot = Join-Path $runtimeRoot "logs"
New-Item -ItemType Directory -Force -Path $runtimeRoot, $logRoot | Out-Null
if (-not (Test-Path -LiteralPath $mediaExe -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $mediaRoot | Out-Null
    Expand-Archive -LiteralPath $mediaArchive -DestinationPath $mediaRoot -Force
}

$resolvedStatePath = if ([IO.Path]::IsPathRooted($StatePath)) {
    [IO.Path]::GetFullPath($StatePath)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $StatePath))
}
$stateDirectory = Split-Path -Parent $resolvedStatePath
New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
$stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$mediaStdout = Join-Path $logRoot "mediamtx-$stamp.stdout.log"
$mediaStderr = Join-Path $logRoot "mediamtx-$stamp.stderr.log"
$publisherStdout = Join-Path $logRoot "publisher-$stamp.stdout.log"
$publisherStderr = Join-Path $logRoot "publisher-$stamp.stderr.log"
$rtspUri = "rtsp://127.0.0.1:$RtspPort/$StreamName"
$mediaProcess = $null
$publisherProcess = $null

try {
    $savedRtspAddress = $env:MTX_RTSPADDRESS
    $savedLogLevel = $env:MTX_LOGLEVEL
    $env:MTX_RTSPADDRESS = ":$RtspPort"
    $env:MTX_LOGLEVEL = "warn"
    try {
        $mediaProcess = Start-Process -FilePath $mediaExe -WorkingDirectory $mediaRoot `
            -WindowStyle Hidden -RedirectStandardOutput $mediaStdout `
            -RedirectStandardError $mediaStderr -PassThru
    }
    finally {
        $env:MTX_RTSPADDRESS = $savedRtspAddress
        $env:MTX_LOGLEVEL = $savedLogLevel
    }

    $deadline = (Get-Date).AddSeconds(20)
    do {
        if ($mediaProcess.HasExited) {
            throw "MediaMTX exited before opening the RTSP port"
        }
        $listener = Get-NetTCPConnection -State Listen -LocalPort $RtspPort `
            -ErrorAction SilentlyContinue
        if ($listener) { break }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $deadline)
    if (-not $listener) { throw "MediaMTX did not open port $RtspPort" }

    $publisherProcess = Start-Process -FilePath $ffmpeg -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList @(
            "-hide_banner", "-loglevel", "warning",
            "-re", "-stream_loop", "-1", "-i", $videoPath,
            "-map", "0:v:0", "-an", "-c:v", "copy",
            "-f", "rtsp", "-rtsp_transport", "tcp", $rtspUri
        ) -RedirectStandardOutput $publisherStdout `
          -RedirectStandardError $publisherStderr -PassThru

    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    do {
        if ($publisherProcess.HasExited) {
            throw "FFmpeg RTSP publisher exited before the stream became ready"
        }
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $ffprobe -v error -rtsp_transport tcp `
            -show_entries "stream=codec_name,width,height,r_frame_rate,avg_frame_rate" `
            -of json $rtspUri 2>$null | Out-Null
        $probeExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorActionPreference
        if ($probeExitCode -eq 0) { $ready = $true; break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    if (-not $ready) { throw "RTSP test stream did not become readable" }

    $state = [ordered]@{
        schema_version = "1.0"
        started_at = [DateTimeOffset]::UtcNow.ToString("o")
        rtsp_uri = $rtspUri
        video_path = $videoPath
        media_pid = $mediaProcess.Id
        media_exe = $mediaExe
        publisher_pid = $publisherProcess.Id
        publisher_exe = $ffmpeg
        logs = [ordered]@{
            mediamtx_stdout = $mediaStdout
            mediamtx_stderr = $mediaStderr
            publisher_stdout = $publisherStdout
            publisher_stderr = $publisherStderr
        }
    }
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resolvedStatePath -Encoding UTF8
    Write-Output "PASS: local RTSP test stream is ready at $rtspUri"
}
catch {
    if ($publisherProcess -and -not $publisherProcess.HasExited) {
        Stop-Process -Id $publisherProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($mediaProcess -and -not $mediaProcess.HasExited) {
        Stop-Process -Id $mediaProcess.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
