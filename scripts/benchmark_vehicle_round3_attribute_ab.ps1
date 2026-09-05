[CmdletBinding()]
param(
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [string]$BuildDir = ".\out\build\gpu-pipeline-round2-baseline",
    [int]$Iterations = 2000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$CandidateEngine =
        "reports/gpu-pipeline-optimization/round3/tensorrt-attribute/engines/dynamic-aux2.engine",
    [string]$CandidateSha256 =
        "fbf861d32f86e71bdb75204a74470541e409005c20ac9933071a03be1d454134",
    [string]$EvidenceDir =
        ".\reports\gpu-pipeline-optimization\round3\attribute-checkpoint-ab",
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
$modes = @("production-bgr", "production-i420", "aux2-i420")
$entries = @{
    "production-bgr" = [Collections.Generic.List[object]]::new()
    "production-i420" = [Collections.Generic.List[object]]::new()
    "aux2-i420" = [Collections.Generic.List[object]]::new()
}

function Set-ModeEnvironment([string]$Mode) {
    $env:VCAS_BENCHMARK_ATTRIBUTES = "1"
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_DISABLE_ATTRIBUTE_I420_ROI -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_SHA256 -ErrorAction SilentlyContinue
    if ($Mode -eq "production-bgr") {
        $env:VCAS_DISABLE_ATTRIBUTE_I420_ROI = "1"
    }
    elseif ($Mode -eq "aux2-i420") {
        $env:VCAS_ATTRIBUTE_ENGINE_PATH = $CandidateEngine
        $env:VCAS_ATTRIBUTE_ENGINE_SHA256 = $CandidateSha256
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
    Set-ModeEnvironment $Mode
    $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    $runnerOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $runner -RtspUri $RtspUri -BuildDir $BuildDir `
        -Iterations $Iterations -WarmupFrames $WarmupFrames -Runs 1 `
        -EvidenceDir $runDir | Out-String
    if ($LASTEXITCODE -ne 0) { throw "$Mode run $Run failed" }
    $entry = Read-Entry $Mode $Run
    $entry.runner_output = $runnerOutput.Trim()
    return $entry
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
            @("production-bgr", "production-i420", "aux2-i420"),
            @("aux2-i420", "production-i420", "production-bgr"),
            @("production-i420", "production-bgr", "aux2-i420")
        )
        for ($run = 1; $run -le $Runs; ++$run) {
            $order = $orders[($run - 1) % $orders.Count]
            foreach ($mode in $order) {
                $entries[$mode].Add((Invoke-One $mode $run))
            }
        }
    }
}
finally {
    Remove-Item Env:VCAS_BENCHMARK_ATTRIBUTES -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_DISABLE_ATTRIBUTE_I420_ROI -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:VCAS_ATTRIBUTE_ENGINE_SHA256 -ErrorAction SilentlyContinue
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
            max = Get-EntryMedian $Values { param($e) $e.benchmark.runs[0].total_ms.max }
        }
        sustained_fps = Get-EntryMedian $Values {
            param($e) $e.benchmark.runs[0].sustained_fps
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
        attribute = [ordered]@{
            crop_mean_ms = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_crop_ms.mean
            }
            preprocess_mean_ms = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_preprocess_ms.mean
            }
            h2d_mean_ms = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_h2d_ms.mean
            }
            inference_mean_ms = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_inference_ms.mean
            }
            total_mean_ms = Get-EntryMedian $Values {
                param($e) $e.benchmark.runs[0].attribute_total_ms.mean
            }
        }
        consumer_sequence_gaps = Get-EntryMedian $Values {
            param($e) $e.benchmark.measured_consumer_sequence_gaps
        }
        resources = [ordered]@{
            cpu_mean_percent = Get-EntryMedian $Values {
                param($e) $e.telemetry.process_cpu_percent.mean
            }
            rss_mean_mib = Get-EntryMedian $Values {
                param($e) $e.telemetry.working_set_mib.mean
            }
            gpu_mean_percent = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_utilization_percent.mean
            }
            vram_mean_mib = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_memory_used_mib.mean
            }
            power_mean_w = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_power_w.mean
            }
            temperature_mean_c = Get-EntryMedian $Values {
                param($e) $e.telemetry.gpu_temperature_c.mean
            }
        }
    }
}

function Change-Percent([double]$Before, [double]$After) {
    return 100.0 * ($After / $Before - 1.0)
}

$summary = [ordered]@{}
foreach ($mode in $modes) {
    $summary[$mode] = Measure-Mode @($entries[$mode])
}
$report = [ordered]@{
    schema_version = "1.0"
    source = $RtspUri
    scope = "RTSP decode + detection + real ROI + attributes"
    iterations_per_run = $Iterations
    warmup_frames_per_run = $WarmupFrames
    runs_per_mode = $Runs
    run_order = "three-position rotating order across production-bgr, production-i420, aux2-i420"
    candidate_engine = $CandidateEngine
    candidate_sha256 = $CandidateSha256
    modes = $entries
    median = $summary
    changes_percent = [ordered]@{
        i420_vs_bgr = [ordered]@{
            mean = Change-Percent $summary["production-bgr"].total_ms.mean `
                $summary["production-i420"].total_ms.mean
            p95 = Change-Percent $summary["production-bgr"].total_ms.p95 `
                $summary["production-i420"].total_ms.p95
        }
        aux2_vs_production_i420 = [ordered]@{
            mean = Change-Percent $summary["production-i420"].total_ms.mean `
                $summary["aux2-i420"].total_ms.mean
            p95 = Change-Percent $summary["production-i420"].total_ms.p95 `
                $summary["aux2-i420"].total_ms.p95
        }
        combined_vs_bgr = [ordered]@{
            mean = Change-Percent $summary["production-bgr"].total_ms.mean `
                $summary["aux2-i420"].total_ms.mean
            p95 = Change-Percent $summary["production-bgr"].total_ms.p95 `
                $summary["aux2-i420"].total_ms.p95
        }
    }
}
$reportPath = Join-Path $evidencePath "comparison.json"
$report | ConvertTo-Json -Depth 30 |
    Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: round3 attribute A/B report=$reportPath"
