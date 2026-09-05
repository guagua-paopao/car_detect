[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8087/api/v1",
    [string]$CameraId = "vcas_rtsp_demo",
    [string]$AdminToken = "vcas-demo-local-admin",
    [int]$DurationSeconds = 60,
    [int]$SampleIntervalSeconds = 5
)

$ErrorActionPreference = "Stop"
$headers = @{ Authorization = "Bearer $AdminToken" }
$outputRoot = Join-Path $PSScriptRoot "output"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$samples = @()
$deadline = (Get-Date).AddSeconds($DurationSeconds)

while ((Get-Date) -lt $deadline) {
    $status = Invoke-RestMethod -Method Get `
        -Uri "$BaseUrl/cameras/$CameraId/status" `
        -Headers $headers -TimeoutSec 5
    $realtime = Invoke-RestMethod -Method Get `
        -Uri "$BaseUrl/cameras/$CameraId/vehicles/realtime" `
        -Headers $headers -TimeoutSec 5
    $samples += [ordered]@{
        sampled_at = [DateTimeOffset]::UtcNow.ToString("o")
        camera_status = $status.status
        thread_running = [bool]$status.pipeline.thread_running
        sampled_frames = $status.pipeline.sampled_frames
        source_sequence = $status.pipeline.last_source_sequence
        realtime_available = [bool]$realtime.available
        realtime_vehicle_count = @($realtime.items).Count
        realtime_vehicles = @($realtime.items)
    }
    Start-Sleep -Seconds $SampleIntervalSeconds
}

$history = Invoke-RestMethod -Method Get `
    -Uri "$BaseUrl/cameras/$CameraId/vehicle-events?limit=100&offset=0" `
    -Headers $headers -TimeoutSec 10
$modelStatus = Invoke-RestMethod -Method Get `
    -Uri "$BaseUrl/models/status" -Headers $headers -TimeoutSec 5
$framePath = Join-Path $outputRoot "latest_frame.jpg"
Invoke-WebRequest -Method Get -Uri "$BaseUrl/cameras/$CameraId/latest-frame" `
    -Headers $headers -OutFile $framePath -TimeoutSec 15

$result = [ordered]@{
    schema_version = "1.0"
    captured_at = [DateTimeOffset]::UtcNow.ToString("o")
    duration_seconds = $DurationSeconds
    camera_id = $CameraId
    models_all_ready = [bool]$modelStatus.all_ready
    model_status = $modelStatus
    samples = $samples
    maximum_realtime_vehicle_count = (
        $samples | Measure-Object -Property realtime_vehicle_count -Maximum
    ).Maximum
    history_event_count = @($history.items).Count
    history = $history
    latest_frame = $framePath
}
$resultPath = Join-Path $outputRoot "demo_results.json"
$result | ConvertTo-Json -Depth 20 |
    Set-Content -LiteralPath $resultPath -Encoding UTF8
Write-Output (
    "PASS: results=$resultPath frame=$framePath " +
    "max_realtime=$($result.maximum_realtime_vehicle_count) " +
    "events=$($result.history_event_count)"
)
