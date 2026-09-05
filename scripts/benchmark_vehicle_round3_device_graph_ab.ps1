[CmdletBinding()]
param(
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [string]$BuildDir = ".\out\build\gpu-pipeline-round2-baseline",
    [int]$Iterations = 2000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$EvidenceDir =
        ".\reports\gpu-pipeline-optimization\round3\device-graph-ab",
    [switch]$AggregateOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
$runner = Join-Path $PSScriptRoot "run_vehicle_rtsp_benchmark.ps1"
$modes = @("host-no-graph", "device-no-graph", "device-graph")
$entries = @{}
foreach ($mode in $modes) {
    $entries[$mode] = [Collections.Generic.List[object]]::new()
}

function Set-Mode([string]$Mode) {
    $env:VCAS_BENCHMARK_ATTRIBUTES = "1"
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_DISABLE_ATTRIBUTE_I420_ROI -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_SHA256 -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_COMPARE_PRODUCTION_ATTRIBUTE -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_DISABLE_ATTRIBUTE_DEVICE_I420 -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_DISABLE_ATTRIBUTE_CUDA_GRAPH -ErrorAction SilentlyContinue
    if ($Mode -eq "host-no-graph") {
        $env:VCAS_DISABLE_ATTRIBUTE_DEVICE_I420 = "1"
        $env:VCAS_DISABLE_ATTRIBUTE_CUDA_GRAPH = "1"
    }
    elseif ($Mode -eq "device-no-graph") {
        $env:VCAS_DISABLE_ATTRIBUTE_CUDA_GRAPH = "1"
    }
}

function Read-Entry([string]$Mode, [int]$Run) {
    $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    return [ordered]@{
        run = $Run
        benchmark = Get-Content -Raw -LiteralPath (
            Join-Path $runDir "benchmark.json") | ConvertFrom-Json
        telemetry = Get-Content -Raw -LiteralPath (
            Join-Path $runDir "telemetry-summary.json") | ConvertFrom-Json
    }
}

function Invoke-One([string]$Mode, [int]$Run) {
    Set-Mode $Mode
    $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
        -RtspUri $RtspUri -BuildDir $BuildDir -Iterations $Iterations `
        -WarmupFrames $WarmupFrames -Runs 1 -EvidenceDir $runDir | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "$Mode run $Run failed" }
    return Read-Entry $Mode $Run
}

try {
    if ($AggregateOnly) {
        foreach ($mode in $modes) {
            for ($run = 1; $run -le $Runs; ++$run) {
                $entries[$mode].Add((Read-Entry $mode $run))
            }
        }
    }
    else {
        $orders = @(
            @("host-no-graph", "device-no-graph", "device-graph"),
            @("device-graph", "device-no-graph", "host-no-graph"),
            @("device-no-graph", "host-no-graph", "device-graph")
        )
        for ($run = 1; $run -le $Runs; ++$run) {
            foreach ($mode in $orders[($run - 1) % $orders.Count]) {
                $entries[$mode].Add((Invoke-One $mode $run))
            }
        }
    }
}
finally {
    foreach ($name in @(
        "VCAS_BENCHMARK_ATTRIBUTES",
        "VCAS_BENCHMARK_COMPATIBILITY",
        "VCAS_DISABLE_ATTRIBUTE_I420_ROI",
        "VCAS_DISABLE_ATTRIBUTE_DEVICE_I420",
        "VCAS_DISABLE_ATTRIBUTE_CUDA_GRAPH",
        "VCAS_ATTRIBUTE_ENGINE_PATH",
        "VCAS_ATTRIBUTE_ENGINE_SHA256",
        "VCAS_COMPARE_PRODUCTION_ATTRIBUTE")) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return [double]$sorted[[Math]::Floor($sorted.Count / 2)]
}
function Get-EntryMedian([object[]]$Values, [scriptblock]$Selector) {
    return Get-Median ([double[]]@($Values | ForEach-Object { & $Selector $_ }))
}
function Measure-Mode([object[]]$Values) {
    return [ordered]@{
        total_ms = [ordered]@{
            mean = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].total_ms.mean }
            p50 = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].total_ms.p50 }
            p95 = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].total_ms.p95 }
            p99 = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].total_ms.p99 }
        }
        fps = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].sustained_fps }
        attribute_ms = [ordered]@{
            preprocess = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_preprocess_ms.mean
            }
            h2d = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_h2d_ms.mean
            }
            inference = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_inference_ms.mean
            }
            total = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_total_ms.mean
            }
        }
        attribute_h2d_bytes = Get-EntryMedian $Values {
            param($e) $e.benchmark.runs[0].attribute_h2d_bytes_per_batch
        }
        device_frame_copy_ms = Get-EntryMedian $Values {
            param($e) $e.benchmark.runs[0].device_frame_copy_ms.mean
        }
        attribute_graph_hits = Get-EntryMedian $Values {
            param($e) $e.benchmark.runs[0].attribute_cuda_graph_frames
        }
        attribute_graph_fallbacks = Get-EntryMedian $Values {
            param($e) $e.benchmark.runs[0].attribute_cuda_graph_fallback_frames
        }
        capture_to_result_ms = [ordered]@{
            mean = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].capture_to_result_ms.mean
            }
            p95 = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].capture_to_result_ms.p95
            }
            p99 = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].capture_to_result_ms.p99
            }
        }
        sequence_gaps = Get-EntryMedian $Values {
            param($e) $e.benchmark.measured_consumer_sequence_gaps
        }
        resources = [ordered]@{
            cpu_percent = Get-EntryMedian $Values {
                param($e) $e.telemetry.process_cpu_percent.mean
            }
            rss_mib = Get-EntryMedian $Values {
                param($e) $e.telemetry.working_set_mib.mean
            }
            gpu_percent = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_utilization_percent.mean
            }
            vram_mib = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_memory_used_mib.mean
            }
        }
    }
}
function Change([double]$Before, [double]$After) {
    return 100.0 * ($After / $Before - 1.0)
}

$median = [ordered]@{}
foreach ($mode in $modes) {
    $median[$mode] = Measure-Mode @($entries[$mode])
}
$report = [ordered]@{
    schema_version = "1.0"
    source = $RtspUri
    iterations_per_run = $Iterations
    warmup_frames_per_run = $WarmupFrames
    runs_per_mode = $Runs
    modes = $entries
    median = $median
    changes_percent = [ordered]@{
        device_vs_host = [ordered]@{
            mean = Change $median["host-no-graph"].total_ms.mean `
                $median["device-no-graph"].total_ms.mean
            p95 = Change $median["host-no-graph"].total_ms.p95 `
                $median["device-no-graph"].total_ms.p95
        }
        graph_vs_no_graph = [ordered]@{
            mean = Change $median["device-no-graph"].total_ms.mean `
                $median["device-graph"].total_ms.mean
            p95 = Change $median["device-no-graph"].total_ms.p95 `
                $median["device-graph"].total_ms.p95
        }
        combined_vs_host = [ordered]@{
            mean = Change $median["host-no-graph"].total_ms.mean `
                $median["device-graph"].total_ms.mean
            p95 = Change $median["host-no-graph"].total_ms.p95 `
                $median["device-graph"].total_ms.p95
        }
    }
}
$reportPath = Join-Path $evidencePath "comparison.json"
$report | ConvertTo-Json -Depth 30 |
    Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: round3 device/graph A/B report=$reportPath"
