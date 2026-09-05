[CmdletBinding()]
param(
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [string]$BuildDir = ".\out\build\gpu-pipeline-round2-baseline",
    [int]$Iterations = 45000,
    [int]$WarmupFrames = 200,
    [int]$RestartAfterSeconds = 120,
    [double]$MaximumRssWindowGrowthMiB = 64.0,
    [double]$MaximumGpuMemoryWindowGrowthMiB = 64.0,
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round3\stability-30m-reconnect",
    [string]$StatePath = ".\runtime\rtsp-test\round3-state.json"
)

$ErrorActionPreference = "Stop"
if ($Iterations -lt 45000 -or $WarmupFrames -lt 200 -or
    $RestartAfterSeconds -lt 30) {
    throw "Stability gate requires >=45000 measured frames, >=200 warmup frames, and restart after >=30 seconds"
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildPath = if ([IO.Path]::IsPathRooted($BuildDir)) {
    [IO.Path]::GetFullPath($BuildDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDir))
}
$benchmarkExecutable = Join-Path $buildPath "vehicle_tensorrt_benchmark.exe"
if (-not (Test-Path -LiteralPath $benchmarkExecutable -PathType Leaf)) {
    throw "Benchmark executable is missing: $benchmarkExecutable"
}
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
$runner = Join-Path $PSScriptRoot "run_vehicle_rtsp_benchmark.ps1"
$restart = Join-Path $PSScriptRoot "restart_local_rtsp_publisher.ps1"
$stdout = Join-Path $evidencePath "stability-runner.stdout.log"
$stderr = Join-Path $evidencePath "stability-runner.stderr.log"
$started = [DateTimeOffset]::UtcNow
. (Join-Path $PSScriptRoot "process_tree_helpers.ps1")

$savedAttributes = $env:VCAS_BENCHMARK_ATTRIBUTES
$savedCompatibility = $env:VCAS_BENCHMARK_COMPATIBILITY
$savedBusinessChain = $env:VCAS_BENCHMARK_BUSINESS_CHAIN
$savedAsyncSnapshots = $env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS
$process = $null
$benchmarkProcessId = $null
try {
    $env:VCAS_BENCHMARK_ATTRIBUTES = "1"
    $env:VCAS_BENCHMARK_BUSINESS_CHAIN = "1"
    $env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS = "1"
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath powershell.exe -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner,
            "-RtspUri", $RtspUri, "-BuildDir", $BuildDir,
            "-Iterations", $Iterations, "-WarmupFrames", $WarmupFrames,
            "-Runs", 1, "-EvidenceDir", $evidencePath
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}
finally {
    $env:VCAS_BENCHMARK_ATTRIBUTES = $savedAttributes
    $env:VCAS_BENCHMARK_COMPATIBILITY = $savedCompatibility
    $env:VCAS_BENCHMARK_BUSINESS_CHAIN = $savedBusinessChain
    $env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS = $savedAsyncSnapshots
}

for ($attempt = 0; $attempt -lt 100 -and $null -eq $benchmarkProcessId; ++$attempt) {
    $process.Refresh()
    if ($process.HasExited) { break }
    $children = @(Get-NativeChildProcessId `
        -ParentProcessId $process.Id `
        -ExecutableName "vehicle_tensorrt_benchmark.exe")
    if ($children.Count -gt 0) {
        $benchmarkProcessId = [int]$children[0]
        break
    }
    Start-Sleep -Milliseconds 50
}

function Stop-TrackedBenchmarkTree {
    if ($null -eq $benchmarkProcessId) { return }
    $target = Get-Process -Id $benchmarkProcessId -ErrorAction SilentlyContinue
    if ($null -eq $target) { return }
    if (-not [string]::Equals(
            $target.Path, $benchmarkExecutable,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to stop an unverified benchmark process: $benchmarkProcessId"
    }
    $decoders = @(Get-NativeChildProcessId `
        -ParentProcessId $benchmarkProcessId -ExecutableName "ffmpeg.exe")
    foreach ($decoder in $decoders) {
        Stop-Process -Id $decoder -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $benchmarkProcessId -Force -ErrorAction SilentlyContinue
}

$restartDeadline = (Get-Date).AddSeconds($RestartAfterSeconds)
while ((Get-Date) -lt $restartDeadline) {
    if ($null -eq (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
        throw "Stability benchmark exited before the reconnect injection: $(Get-Content -Raw -LiteralPath $stderr)"
    }
    Start-Sleep -Seconds 5
}

$restartOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $restart `
    -StatePath $StatePath | Out-String
if ($LASTEXITCODE -ne 0) { throw "RTSP publisher restart failed: $restartOutput" }
$restartedAt = [DateTimeOffset]::UtcNow

while ($null -ne (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 5
}
$reportPath = Join-Path $evidencePath "benchmark.json"
$telemetryPath = Join-Path $evidencePath "telemetry-summary.json"
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $telemetryPath -PathType Leaf)) {
    Stop-TrackedBenchmarkTree
    throw "Stability benchmark did not produce complete reports: $(Get-Content -Raw -LiteralPath $stderr)"
}
$benchmark = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
$telemetry = Get-Content -Raw -LiteralPath $telemetryPath | ConvertFrom-Json
$run = $benchmark.runs[0]
$business = $run.business_chain
$writer = $business.snapshot_writer
$expectedSnapshots = [int][Math]::Ceiling($Iterations / 12.0)
if ($benchmark.iterations_per_run -ne $Iterations -or $run.samples -ne $Iterations -or
    -not $benchmark.attributes_enabled -or -not $benchmark.business_chain_enabled -or
    -not $benchmark.async_snapshots -or $run.attributes_total -le 0 -or
    $benchmark.capture.reconnect_count -lt 1 -or $benchmark.capture.open_count -lt 2 -or
    $benchmark.capture.state -ne "running" -or $run.cuda_graph_fallback_frames -ne 0 -or
    $run.attribute_cuda_graph_fallback_frames -ne 0 -or
    $business.frames -ne $Iterations -or
    $business.snapshots_encoded -ne $expectedSnapshots -or
    $business.snapshot_failures -ne 0 -or
    $business.stale_results_rejected -ne 0 -or
    $business.attribute_stale_rejected -ne 0 -or
    $business.attribute_version_rejected -ne 0 -or
    $writer.submitted_jobs -ne $expectedSnapshots -or
    $writer.completed_jobs -ne $expectedSnapshots -or
    $writer.failed_jobs -ne 0 -or $writer.queue_depth -ne 0 -or
    $writer.active_jobs -ne 0 -or $writer.maximum_queue_depth -gt 2 -or
    $telemetry.working_set_window_trend_mib.growth -gt $MaximumRssWindowGrowthMiB -or
    $telemetry.gpu_memory_window_trend_mib.growth -gt $MaximumGpuMemoryWindowGrowthMiB) {
    throw "Stability acceptance failed; inspect $reportPath"
}

$completed = [DateTimeOffset]::UtcNow
$summary = [ordered]@{
    schema_version = "1.0"
    started_at = $started.ToString("o")
    publisher_restarted_at = $restartedAt.ToString("o")
    completed_at = $completed.ToString("o")
    orchestrator_wall_seconds = ($completed - $started).TotalSeconds
    publisher_restart_output = $restartOutput.Trim()
    acceptance = [ordered]@{
        measured_frames = $run.samples
        attributes_total = $run.attributes_total
        sustained_fps = $run.sustained_fps
        total_mean_ms = $run.total_ms.mean
        total_p95_ms = $run.total_ms.p95
        total_p99_ms = $run.total_ms.p99
        cuda_graph_frames = $run.cuda_graph_frames
        cuda_graph_fallback_frames = $run.cuda_graph_fallback_frames
        attribute_cuda_graph_frames = $run.attribute_cuda_graph_frames
        attribute_cuda_graph_fallback_frames = $run.attribute_cuda_graph_fallback_frames
        business_frames = $business.frames
        serialized_result_bytes = $business.serialized_result_bytes
        tracks_created = $business.tracks_created
        tracks_confirmed = $business.tracks_confirmed
        tracks_exited = $business.tracks_exited
        stale_results_rejected = $business.stale_results_rejected
        attribute_stale_rejected = $business.attribute_stale_rejected
        attribute_version_rejected = $business.attribute_version_rejected
        snapshots_encoded = $business.snapshots_encoded
        snapshot_jpeg_bytes = $business.snapshot_jpeg_bytes
        snapshot_writer_submitted_jobs = $writer.submitted_jobs
        snapshot_writer_completed_jobs = $writer.completed_jobs
        snapshot_writer_failed_jobs = $writer.failed_jobs
        snapshot_writer_final_queue_depth = $writer.queue_depth
        snapshot_writer_maximum_queue_depth = $writer.maximum_queue_depth
        snapshot_writer_mean_enqueue_wait_ms = $writer.mean_enqueue_wait_ms
        snapshot_writer_maximum_enqueue_wait_ms = $writer.maximum_enqueue_wait_ms
        consumer_sequence_gaps = $run.consumer_sequence_gaps
        capture_open_count = $benchmark.capture.open_count
        capture_reconnect_count = $benchmark.capture.reconnect_count
        capture_state = $benchmark.capture.state
        rss_window_growth_mib = $telemetry.working_set_window_trend_mib.growth
        maximum_rss_window_growth_mib = $MaximumRssWindowGrowthMiB
        gpu_memory_window_growth_mib = $telemetry.gpu_memory_window_trend_mib.growth
        maximum_gpu_memory_window_growth_mib = $MaximumGpuMemoryWindowGrowthMiB
    }
    result = "PASS"
}
$summaryPath = Join-Path $evidencePath "stability-summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Output "PASS: 30-minute GPU stability report=$summaryPath"
