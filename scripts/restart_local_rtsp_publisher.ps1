[CmdletBinding()]
param([string]$StatePath = ".\runtime\rtsp-test\state.json")

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedStatePath = if ([IO.Path]::IsPathRooted($StatePath)) {
    [IO.Path]::GetFullPath($StatePath)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $StatePath))
}
if (-not (Test-Path -LiteralPath $resolvedStatePath -PathType Leaf)) {
    throw "RTSP test state is missing: $resolvedStatePath"
}

$state = Get-Content -Raw -LiteralPath $resolvedStatePath | ConvertFrom-Json
$publisherPid = [int]$state.publisher_pid
$publisherExe = [IO.Path]::GetFullPath([string]$state.publisher_exe)
$videoPath = [IO.Path]::GetFullPath([string]$state.video_path)
$rtspUri = [string]$state.rtsp_uri
if ($publisherPid -le 0 -or -not (Test-Path -LiteralPath $publisherExe -PathType Leaf) -or
    -not (Test-Path -LiteralPath $videoPath -PathType Leaf) -or
    -not $rtspUri.StartsWith("rtsp://127.0.0.1:", [StringComparison]::OrdinalIgnoreCase)) {
    throw "RTSP test state contains an invalid goal-owned publisher"
}

$existing = Get-CimInstance Win32_Process -Filter "ProcessId=$publisherPid" `
    -ErrorAction SilentlyContinue
if (-not $existing) {
    throw "Goal-owned RTSP publisher process is not running: PID $publisherPid"
}
$actualExe = [IO.Path]::GetFullPath([string]$existing.ExecutablePath)
if (-not $actualExe.Equals($publisherExe, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stop reused PID ${publisherPid}: executable mismatch"
}

$uri = [Uri]$rtspUri
$listener = Get-NetTCPConnection -State Listen -LocalPort $uri.Port `
    -ErrorAction SilentlyContinue
if (-not $listener) {
    throw "Goal-owned MediaMTX listener is not available on port $($uri.Port)"
}

$logRoot = Join-Path $projectRoot "runtime\rtsp-test\logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$publisherStdout = Join-Path $logRoot "publisher-restart-$stamp.stdout.log"
$publisherStderr = Join-Path $logRoot "publisher-restart-$stamp.stderr.log"
$replacement = $null
try {
    Stop-Process -Id $publisherPid -Force -ErrorAction Stop
    Wait-Process -Id $publisherPid -Timeout 5 -ErrorAction SilentlyContinue
    $replacement = Start-Process -FilePath $publisherExe -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList @(
            "-hide_banner", "-loglevel", "warning",
            "-re", "-stream_loop", "-1", "-i", $videoPath,
            "-map", "0:v:0", "-an", "-c:v", "copy",
            "-f", "rtsp", "-rtsp_transport", "tcp", $rtspUri
        ) -RedirectStandardOutput $publisherStdout `
          -RedirectStandardError $publisherStderr -PassThru

    $ffprobe = Join-Path (Split-Path -Parent $publisherExe) "ffprobe.exe"
    if (-not (Test-Path -LiteralPath $ffprobe -PathType Leaf)) {
        $ffprobe = (Get-Command ffprobe.exe -CommandType Application -ErrorAction Stop).Source
    }
    $deadline = (Get-Date).AddSeconds(30)
    $ready = $false
    do {
        if ($replacement.HasExited) {
            throw "Replacement FFmpeg publisher exited before the stream became ready"
        }
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $ffprobe -v error -rtsp_transport tcp -show_entries "stream=codec_name" `
            -of json $rtspUri 2>$null | Out-Null
        $probeExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($probeExitCode -eq 0) { $ready = $true; break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    if (-not $ready) { throw "Replacement RTSP publisher did not become readable" }

    $restart = [pscustomobject]@{
        restarted_at = [DateTimeOffset]::UtcNow.ToString("o")
        old_publisher_pid = $publisherPid
        new_publisher_pid = $replacement.Id
        stdout = $publisherStdout
        stderr = $publisherStderr
    }
    $history = @($state.publisher_restarts | Where-Object { $null -ne $_ }) +
        @($restart)
    $state.publisher_pid = $replacement.Id
    $state.logs.publisher_stdout = $publisherStdout
    $state.logs.publisher_stderr = $publisherStderr
    $state | Add-Member -NotePropertyName publisher_restarts `
        -NotePropertyValue $history -Force
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resolvedStatePath `
        -Encoding UTF8
    Write-Output "PASS: goal-owned RTSP publisher restarted old_pid=$publisherPid new_pid=$($replacement.Id)"
}
catch {
    if ($replacement -and -not $replacement.HasExited) {
        Stop-Process -Id $replacement.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
