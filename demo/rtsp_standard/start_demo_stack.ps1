[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$Video = "",
    [string]$BuildDir = ".\out\build\backend-model-validation",
    [int]$RtspPort = 8554
)

$ErrorActionPreference = "Stop"
$root = if ($ProjectRoot) {
    (Resolve-Path -LiteralPath $ProjectRoot).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
}
$videoPath = if ($Video) {
    [IO.Path]::GetFullPath($Video)
}
else {
    Join-Path $PSScriptRoot "output\vcas_rtsp_demo_60s.mp4"
}
if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) {
    throw "Demo video is missing: $videoPath"
}

$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
$ffmpeg = (Get-Command ffmpeg.exe -ErrorAction Stop).Source
$ffprobe = (Get-Command ffprobe.exe -ErrorAction Stop).Source
$docker = (Get-Command docker.exe -ErrorAction Stop).Source

& $docker version --format "{{.Server.Version}}" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker engine is not ready" }

$postgresName = "vcas-demo-postgres"
$redisName = "vcas-demo-redis"
$mediaName = "vcas-demo-mediamtx"

function Test-RedisPing {
    param([string]$HostName = "127.0.0.1", [int]$Port = 6379)
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $client.SendTimeout = 2000
        $client.ReceiveTimeout = 2000
        $client.Connect($HostName, $Port)
        $stream = $client.GetStream()
        $request = '*1' + "`r`n" + '$4' + "`r`n" + 'PING' + "`r`n"
        $bytes = [Text.Encoding]::ASCII.GetBytes($request)
        $stream.Write($bytes, 0, $bytes.Length)
        $buffer = New-Object byte[] 64
        $read = $stream.Read($buffer, 0, $buffer.Length)
        return [Text.Encoding]::ASCII.GetString($buffer, 0, $read) `
            -eq "+PONG`r`n"
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

foreach ($name in @($postgresName, $redisName, $mediaName)) {
    $existingName = & $docker ps -a `
        --filter "name=^/$name$" --format "{{.Names}}"
    if ($existingName -eq $name) {
        & $docker rm -f $name | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to remove previous demo container: $name"
        }
    }
}

$postgresPassword = [Guid]::NewGuid().ToString("N")
& $docker run --rm -d --name $postgresName `
    -e "POSTGRES_PASSWORD=$postgresPassword" `
    -e "POSTGRES_DB=vcas_demo" `
    -p "127.0.0.1:5432:5432" "postgres:16-alpine" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL demo container failed" }
$redisMode = "existing"
if (-not (Test-RedisPing)) {
    $redisMode = "container"
    & $docker run --rm -d --name $redisName `
        -p "127.0.0.1:6379:6379" "redis:7-alpine" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Redis demo container failed" }
}
& $docker run --rm -d --name $mediaName `
    -e "MTX_RTSPTRANSPORTS=tcp" `
    -p "127.0.0.1:${RtspPort}:8554" `
    "bluenviron/mediamtx:1.18.2" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "MediaMTX demo container failed" }

$deadline = (Get-Date).AddSeconds(60)
$postgresReady = $false
while ((Get-Date) -lt $deadline) {
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $docker exec $postgresName pg_isready -U postgres -d vcas_demo `
        2>$null | Out-Null
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($probeExitCode -eq 0) { $postgresReady = $true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $postgresReady) { throw "PostgreSQL did not become ready" }

$logRoot = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$rtspUri = "rtsp://127.0.0.1:$RtspPort/vcas-demo"
$publisher = Start-Process -FilePath $ffmpeg -ArgumentList @(
    "-hide_banner", "-loglevel", "warning",
    "-re", "-stream_loop", "-1", "-i", $videoPath,
    "-an", "-c:v", "copy",
    "-f", "rtsp", "-rtsp_transport", "tcp", $rtspUri
) -WorkingDirectory $root -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $logRoot "publisher.stdout.log") `
  -RedirectStandardError (Join-Path $logRoot "publisher.stderr.log") `
  -PassThru

$deadline = (Get-Date).AddSeconds(30)
$rtspReady = $false
while ((Get-Date) -lt $deadline) {
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & $ffprobe -v error -rtsp_transport tcp `
        -show_entries "stream=codec_name,width,height,r_frame_rate" `
        -of json $rtspUri 2>$null | Out-Null
    $probeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($probeExitCode -eq 0) { $rtspReady = $true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $rtspReady) { throw "RTSP publisher did not become ready" }

$adminToken = "vcas-demo-local-admin"
$env:YOLO11_CAMERA_ENTRY_URL = $rtspUri
$env:YOLO11_CAMERA_TASK_ADMIN_TOKEN = $adminToken
$env:YOLO11_DEMO_DISABLE_ANALYSIS = "0"
$env:YOLO11_DEMO_VEHICLE_ANALYSIS = "1"
$env:YOLO11_POSTGRES_DSN = (
    "host=127.0.0.1 port=5432 dbname=vcas_demo " +
    "user=postgres password=$postgresPassword"
)
$startArguments = @{
    Root = $root
    BuildDir = $BuildDir
    ReadyTimeoutSeconds = 90
}
& (Join-Path $root "scripts\start_demo.ps1") @startArguments

$headers = @{
    Authorization = "Bearer $adminToken"
    "Content-Type" = "application/json"
    "Idempotency-Key" = "vcas-rtsp-standard-demo-v1"
}
$body = @{
    camera_id = "vcas_rtsp_demo"
    name = "VCAS 60-second RTSP standard demo"
    camera_profile = "entry_camera_01"
    desired_state = "running"
    frame_interval_ms = 100
    output_mode = "both"
    jpeg_quality = 90
    max_width = 1280
    max_height = 720
    retention_days = 1
    max_saved_frames = 1000
    analysis = @{
        enabled = $true
        target_infer_fps = 12.0
        algorithm_profile = "vehicle-cascade-v1"
        algorithms = @("vehicle_detection", "vehicle_attribute")
    }
    callback_profile = ""
} | ConvertTo-Json -Depth 8
$baseUrl = "http://127.0.0.1:8087/api/v1"
$create = Invoke-RestMethod -Method Post -Uri "$baseUrl/cameras" `
    -Headers $headers -Body $body -TimeoutSec 15

$deadline = (Get-Date).AddSeconds(45)
$status = $null
while ((Get-Date) -lt $deadline) {
    $status = Invoke-RestMethod -Method Get `
        -Uri "$baseUrl/cameras/vcas_rtsp_demo/status" `
        -Headers @{Authorization="Bearer $adminToken"} -TimeoutSec 5
    if ($status.status -eq "running" -and $status.pipeline.thread_running) {
        break
    }
    if ($status.status -eq "failed") {
        throw "Camera pipeline failed: $($status.error_code)"
    }
    Start-Sleep -Milliseconds 500
}
if (-not $status -or $status.status -ne "running") {
    throw "Camera pipeline did not reach running state"
}

$modelStatus = Invoke-RestMethod -Method Get -Uri "$baseUrl/models/status" `
    -Headers @{Authorization="Bearer $adminToken"} -TimeoutSec 5
$state = [ordered]@{
    schema_version = "1.0"
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    rtsp_uri = $rtspUri
    publisher_pid = $publisher.Id
    redis_mode = $redisMode
    camera_id = "vcas_rtsp_demo"
    admin_token = $adminToken
    camera_admin_url = "http://127.0.0.1:8087/camera-admin"
    model_status = $modelStatus
    initial_camera_status = $status
}
$statePath = Join-Path $PSScriptRoot "output\demo_stack_state.json"
$state | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Output (
    "PASS: VCAS demo is running; RTSP=$rtspUri " +
    "camera=http://127.0.0.1:8087/camera-admin token=$adminToken"
)
