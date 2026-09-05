[CmdletBinding()]
param(
    [string]$Source = ".\demo\rtsp_standard\output\vcas_rtsp_demo_60s.mp4",
    [string]$BuildDir = ".\out\build\gpu-pipeline-round2-baseline",
    [int]$Iterations = 1000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round3\snapshot-dispatch-ab"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = if ($Source.StartsWith("rtsp://", [StringComparison]::OrdinalIgnoreCase)) {
    $Source
} elseif ([IO.Path]::IsPathRooted($Source)) {
    [IO.Path]::GetFullPath($Source)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $Source))
}
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
$runner = Join-Path $PSScriptRoot "run_vehicle_rtsp_benchmark.ps1"
$sync = [Collections.Generic.List[object]]::new()
$async = [Collections.Generic.List[object]]::new()

function Invoke-One([string]$Mode, [int]$Run) {
    $env:VCAS_BENCHMARK_ATTRIBUTES = "1"
    $env:VCAS_BENCHMARK_BUSINESS_CHAIN = "1"
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    if ($Mode -eq "async") {
        $env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS = "1"
    } else {
        Remove-Item Env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS -ErrorAction SilentlyContinue
    }
    $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
        -RtspUri $sourcePath -BuildDir $BuildDir -Iterations $Iterations `
        -WarmupFrames $WarmupFrames -Runs 1 -EvidenceDir $runDir | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Mode snapshot run $Run failed" }
    return [ordered]@{
        run = $Run
        benchmark = Get-Content -Raw -LiteralPath (Join-Path $runDir "benchmark.json") |
            ConvertFrom-Json
        telemetry = Get-Content -Raw -LiteralPath (Join-Path $runDir "telemetry-summary.json") |
            ConvertFrom-Json
    }
}

try {
    for ($run = 1; $run -le $Runs; ++$run) {
        if (($run % 2) -eq 1) {
            $sync.Add((Invoke-One "sync" $run))
            $async.Add((Invoke-One "async" $run))
        } else {
            $async.Add((Invoke-One "async" $run))
            $sync.Add((Invoke-One "sync" $run))
        }
    }
}
finally {
    Remove-Item Env:VCAS_BENCHMARK_ATTRIBUTES -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_BENCHMARK_BUSINESS_CHAIN -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return [double]$sorted[[Math]::Floor($sorted.Count / 2)]
}

function Measure-Mode([object[]]$Entries) {
    $mean = Get-Median ([double[]]@($Entries | ForEach-Object {
        $_.benchmark.runs[0].total_ms.mean
    }))
    $p95 = Get-Median ([double[]]@($Entries | ForEach-Object {
        $_.benchmark.runs[0].total_ms.p95
    }))
    $p99 = Get-Median ([double[]]@($Entries | ForEach-Object {
        $_.benchmark.runs[0].total_ms.p99
    }))
    $fps = Get-Median ([double[]]@($Entries | ForEach-Object {
        $_.benchmark.runs[0].sustained_fps
    }))
    return [ordered]@{
        total_mean_ms = $mean
        total_p95_ms = $p95
        total_p99_ms = $p99
        sustained_fps = $fps
        snapshot_call_mean_ms = Get-Median ([double[]]@($Entries | ForEach-Object {
            $_.benchmark.runs[0].snapshot_ms.mean
        }))
        maximum_queue_depth = Get-Median ([double[]]@($Entries | ForEach-Object {
            $value = $_.benchmark.runs[0].business_chain.snapshot_writer.maximum_queue_depth
            if ($null -eq $value) { 0.0 } else { [double]$value }
        }))
        mean_enqueue_wait_ms = Get-Median ([double[]]@($Entries | ForEach-Object {
            $value = $_.benchmark.runs[0].business_chain.snapshot_writer.mean_enqueue_wait_ms
            if ($null -eq $value) { 0.0 } else { [double]$value }
        }))
        failed_snapshot_jobs = Get-Median ([double[]]@($Entries | ForEach-Object {
            [double]$_.benchmark.runs[0].business_chain.snapshot_failures
        }))
        snapshots_encoded = Get-Median ([double[]]@($Entries | ForEach-Object {
            [double]$_.benchmark.runs[0].business_chain.snapshots_encoded
        }))
        serialized_result_bytes = Get-Median ([double[]]@($Entries | ForEach-Object {
            [double]$_.benchmark.runs[0].business_chain.serialized_result_bytes
        }))
        rss_growth_mib = Get-Median ([double[]]@($Entries | ForEach-Object {
            [double]$_.telemetry.working_set_window_trend_mib.growth
        }))
    }
}

$syncMedian = Measure-Mode @($sync)
$asyncMedian = Measure-Mode @($async)
$report = [ordered]@{
    schema_version = "1.0"
    source = $sourcePath
    workload = "optimized detection/attribute + tracking + JSON + periodic JPEG"
    warmup_frames_per_process = $WarmupFrames
    measured_frames_per_process = $Iterations
    runs = $Runs
    order = "alternating sync/async"
    synchronous = $sync
    bounded_async = $async
    median = [ordered]@{
        synchronous = $syncMedian
        bounded_async = $asyncMedian
        mean_change_percent = 100.0 * (
            $asyncMedian.total_mean_ms / $syncMedian.total_mean_ms - 1.0)
        p95_change_percent = 100.0 * (
            $asyncMedian.total_p95_ms / $syncMedian.total_p95_ms - 1.0)
        p99_change_percent = 100.0 * (
            $asyncMedian.total_p99_ms / $syncMedian.total_p99_ms - 1.0)
        fps_change_percent = 100.0 * (
            $asyncMedian.sustained_fps / $syncMedian.sustained_fps - 1.0)
    }
}
$reportPath = Join-Path $evidencePath "comparison.json"
$report | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: snapshot dispatch A/B report=$reportPath"
