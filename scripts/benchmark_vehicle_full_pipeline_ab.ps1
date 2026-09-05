[CmdletBinding()]
param(
    [Alias("Source")]
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [string]$BuildDir = ".\out\build\gpu-pipeline-round2-baseline",
    [int]$Iterations = 2000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round3\rtsp-final",
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
$baseline = [Collections.Generic.List[object]]::new()
$optimized = [Collections.Generic.List[object]]::new()
$sourceKind = if ($RtspUri.StartsWith("rtsp://", [StringComparison]::OrdinalIgnoreCase)) {
    "rtsp"
} else {
    "file"
}

function Invoke-One([string]$Mode, [int]$Run) {
    $env:VCAS_BENCHMARK_ATTRIBUTES = "1"
    $env:VCAS_BENCHMARK_BUSINESS_CHAIN = "1"
    $env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS = "1"
    if ($Mode -eq "compatibility") {
        $env:VCAS_BENCHMARK_COMPATIBILITY = "1"
    } else {
        Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
    }
    $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    $runnerOutput = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $runner `
        -RtspUri $RtspUri -BuildDir $BuildDir -Iterations $Iterations `
        -WarmupFrames $WarmupFrames -Runs 1 -EvidenceDir $runDir | Out-String
    if ($LASTEXITCODE -ne 0) { throw "$Mode run $Run failed" }
    return [ordered]@{
        run = $Run
        runner_output = $runnerOutput.Trim()
        benchmark = Get-Content -Raw -LiteralPath (Join-Path $runDir "benchmark.json") |
            ConvertFrom-Json
        telemetry = Get-Content -Raw -LiteralPath (Join-Path $runDir "telemetry-summary.json") |
            ConvertFrom-Json
    }
}

if ($AggregateOnly) {
    for ($run = 1; $run -le $Runs; ++$run) {
        foreach ($item in @(
            @{ mode = "compatibility"; destination = $baseline },
            @{ mode = "optimized"; destination = $optimized }
        )) {
            $runDir = Join-Path $evidencePath ("{0}-run-{1}" -f $item.mode, $run)
            $item.destination.Add([ordered]@{
                run = $run
                benchmark = Get-Content -Raw -LiteralPath (Join-Path $runDir "benchmark.json") |
                    ConvertFrom-Json
                telemetry = Get-Content -Raw -LiteralPath (Join-Path $runDir "telemetry-summary.json") |
                    ConvertFrom-Json
            })
        }
    }
} else {
    try {
        for ($run = 1; $run -le $Runs; ++$run) {
            if (($run % 2) -eq 1) {
                $baseline.Add((Invoke-One "compatibility" $run))
                $optimized.Add((Invoke-One "optimized" $run))
            } else {
                $optimized.Add((Invoke-One "optimized" $run))
                $baseline.Add((Invoke-One "compatibility" $run))
            }
        }
    }
    finally {
        Remove-Item Env:VCAS_BENCHMARK_ATTRIBUTES -ErrorAction SilentlyContinue
        Remove-Item Env:VCAS_BENCHMARK_COMPATIBILITY -ErrorAction SilentlyContinue
        Remove-Item Env:VCAS_BENCHMARK_BUSINESS_CHAIN -ErrorAction SilentlyContinue
        Remove-Item Env:VCAS_BENCHMARK_ASYNC_SNAPSHOTS -ErrorAction SilentlyContinue
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return [double]$sorted[[Math]::Floor($sorted.Count / 2)]
}

function Get-EntryMedian([object[]]$Entries, [scriptblock]$Selector) {
    return Get-Median ([double[]]@($Entries | ForEach-Object { & $Selector $_ }))
}

function Measure-Case([object[]]$Entries) {
    return [ordered]@{
        total_ms = [ordered]@{
            mean = Get-EntryMedian $Entries { param($entry) $entry.benchmark.runs[0].total_ms.mean }
            p50 = Get-EntryMedian $Entries { param($entry) $entry.benchmark.runs[0].total_ms.p50 }
            p95 = Get-EntryMedian $Entries { param($entry) $entry.benchmark.runs[0].total_ms.p95 }
            p99 = Get-EntryMedian $Entries { param($entry) $entry.benchmark.runs[0].total_ms.p99 }
        }
        sustained_fps = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].sustained_fps
        }
        rtsp_pull_decode_wait_ms = [ordered]@{
            mean = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].rtsp_pull_decode_wait_ms.mean
            }
            p95 = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].rtsp_pull_decode_wait_ms.p95
            }
        }
        input_prepare_mean_ms = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].input_prepare_ms.mean
        }
        host_staging_mean_ms = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].host_staging_ms.mean
        }
        h2d_bytes_per_frame = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].h2d_bytes_per_frame
        }
        d2h_bytes_per_frame = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].d2h_bytes_per_frame
        }
        cuda_graph_mean_ms = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].cuda_graph_ms.mean
        }
        cuda_graph_frames = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].cuda_graph_frames
        }
        cuda_graph_fallback_frames = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.runs[0].cuda_graph_fallback_frames
        }
        attribute = [ordered]@{
            crop_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].attribute_crop_ms.mean
            }
            preprocess_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].attribute_preprocess_ms.mean
            }
            inference_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].attribute_inference_ms.mean
            }
            postprocess_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].attribute_postprocess_ms.mean
            }
            total_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].attribute_total_ms.mean
            }
        }
        business = [ordered]@{
            tracking_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].tracking_ms.mean
            }
            result_output_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].result_output_ms.mean
            }
            snapshot_mean_ms = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].snapshot_ms.mean
            }
            frames = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].business_chain.frames
            }
            snapshots_encoded = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].business_chain.snapshots_encoded
            }
            serialized_result_bytes = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].business_chain.serialized_result_bytes
            }
            stale_results_rejected = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].business_chain.stale_results_rejected
            }
            attribute_stale_rejected = Get-EntryMedian $Entries {
                param($entry) $entry.benchmark.runs[0].business_chain.attribute_stale_rejected
            }
        }
        consumer_sequence_gaps = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.measured_consumer_sequence_gaps
        }
        capture_reconnect_count = Get-EntryMedian $Entries {
            param($entry) $entry.benchmark.capture.reconnect_count
        }
        resources = [ordered]@{
            process_cpu_mean_percent = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.process_cpu_percent.mean
            }
            rss_mean_mib = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.working_set_mib.mean
            }
            rss_window_growth_mib = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.working_set_window_trend_mib.growth
            }
            gpu_utilization_mean_percent = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.gpu_utilization_percent.mean
            }
            gpu_memory_mean_mib = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.gpu_memory_used_mib.mean
            }
            gpu_memory_peak_mib = Get-EntryMedian $Entries {
                param($entry) $entry.telemetry.gpu_memory_used_mib.max
            }
        }
    }
}

$baselineMean = Get-Median ([double[]]@($baseline | ForEach-Object {
    $_.benchmark.runs[0].total_ms.mean
}))
$optimizedMean = Get-Median ([double[]]@($optimized | ForEach-Object {
    $_.benchmark.runs[0].total_ms.mean
}))
$baselineP95 = Get-Median ([double[]]@($baseline | ForEach-Object {
    $_.benchmark.runs[0].total_ms.p95
}))
$optimizedP95 = Get-Median ([double[]]@($optimized | ForEach-Object {
    $_.benchmark.runs[0].total_ms.p95
}))
$baselineFps = Get-Median ([double[]]@($baseline | ForEach-Object {
    $_.benchmark.runs[0].sustained_fps
}))
$optimizedFps = Get-Median ([double[]]@($optimized | ForEach-Object {
    $_.benchmark.runs[0].sustained_fps
}))
$report = [ordered]@{
    schema_version = "1.0"
    source = $RtspUri
    source_kind = $sourceKind
    scope = if ($sourceKind -eq "rtsp") {
        "RTSP decode + detection + real detection crops + production attribute engine + tracking + JSON/JPEG results"
    } else {
        "saturation file decode + detection + real detection crops + production attribute engine + tracking + JSON/JPEG results"
    }
    run_order = "alternating compatibility/optimized; odd-numbered pairs start with compatibility"
    iterations_per_run = $Iterations
    warmup_frames_per_run = $WarmupFrames
    runs_requested = $Runs
    compatibility = $baseline
    optimized = $optimized
    compact_median = [ordered]@{
        compatibility = Measure-Case @($baseline)
        optimized = Measure-Case @($optimized)
    }
    median = [ordered]@{
        compatibility_mean_ms = $baselineMean
        optimized_mean_ms = $optimizedMean
        mean_change_percent = 100.0 * ($optimizedMean / $baselineMean - 1.0)
        compatibility_p95_ms = $baselineP95
        optimized_p95_ms = $optimizedP95
        p95_change_percent = 100.0 * ($optimizedP95 / $baselineP95 - 1.0)
        compatibility_fps = $baselineFps
        optimized_fps = $optimizedFps
        fps_change_percent = 100.0 * ($optimizedFps / $baselineFps - 1.0)
    }
}
$reportPath = Join-Path $evidencePath "comparison.json"
$report | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: full-pipeline A/B report=$reportPath"
