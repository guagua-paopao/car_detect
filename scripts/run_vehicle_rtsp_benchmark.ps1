[CmdletBinding()]
param(
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [string]$BuildDir = ".\out\build\backend-live-verify",
    [int]$Iterations = 2000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round3\rtsp-baseline"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildPath = if ([IO.Path]::IsPathRooted($BuildDir)) {
    [IO.Path]::GetFullPath($BuildDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDir))
}
$benchmarkExe = Join-Path $buildPath "vehicle_tensorrt_benchmark.exe"
if (-not (Test-Path -LiteralPath $benchmarkExe -PathType Leaf)) {
    throw "Benchmark executable is missing: $benchmarkExe"
}
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
$reportPath = Join-Path $evidencePath "benchmark.json"
$telemetryPath = Join-Path $evidencePath "telemetry.jsonl"
$telemetrySummaryPath = Join-Path $evidencePath "telemetry-summary.json"
$stdoutPath = Join-Path $evidencePath "benchmark.stdout.log"
$stderrPath = Join-Path $evidencePath "benchmark.stderr.log"
$telemetryStream = [IO.FileStream]::new(
    $telemetryPath,
    [IO.FileMode]::Create,
    [IO.FileAccess]::Write,
    [IO.FileShare]::ReadWrite)
$telemetryWriter = [IO.StreamWriter]::new(
    $telemetryStream,
    [Text.UTF8Encoding]::new($false))
$telemetryWriter.AutoFlush = $true

$originalPath = $env:Path
$env:Path = "D:\GPU13.3\bin;D:\TensorRT-10.16.1.11\lib;" +
    "D:\libs\opencv\build\x64\vc16\bin;$env:Path"
try {
    $process = Start-Process -FilePath $benchmarkExe -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList @(
            $projectRoot, $RtspUri, $Iterations, $WarmupFrames, $Runs, $reportPath
        ) -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath `
          -PassThru
    $samples = [Collections.Generic.List[object]]::new()
    $previousAt = [DateTimeOffset]::UtcNow
    $previousCpu = 0.0
    while (-not $process.HasExited) {
        $capturedAt = [DateTimeOffset]::UtcNow
        $process.Refresh()
        $cpuSeconds = if ($null -eq $process.CPU) { $previousCpu } else { [double]$process.CPU }
        $elapsed = ($capturedAt - $previousAt).TotalSeconds
        $cpuPercent = if ($elapsed -gt 0) {
            100.0 * ($cpuSeconds - $previousCpu) / $elapsed / [Environment]::ProcessorCount
        } else { 0.0 }
        $gpuText = & nvidia-smi.exe `
            --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu `
            --format=csv,noheader,nounits | Out-String
        if ($LASTEXITCODE -ne 0) { throw "nvidia-smi telemetry query failed" }
        $gpu = @($gpuText.Trim().Split(",") | ForEach-Object { $_.Trim() })
        $sample = [ordered]@{
            captured_at = $capturedAt.ToString("o")
            benchmark_pid = $process.Id
            process_cpu_percent = [Math]::Max(0.0, [Math]::Round($cpuPercent, 3))
            working_set_mib = [Math]::Round($process.WorkingSet64 / 1MB, 3)
            gpu_utilization_percent = [double]$gpu[0]
            gpu_memory_used_mib = [double]$gpu[1]
            gpu_power_w = [double]$gpu[2]
            gpu_temperature_c = [double]$gpu[3]
        }
        $samples.Add([pscustomobject]$sample)
        $telemetryWriter.WriteLine(($sample | ConvertTo-Json -Compress))
        $previousAt = $capturedAt
        $previousCpu = $cpuSeconds
        Start-Sleep -Seconds 1
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $completeReport = $false
        if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
            try {
                $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
                $completeReport = $report.runs.Count -eq $Runs -and
                    $report.iterations_per_run -eq $Iterations
            }
            catch {
                $completeReport = $false
            }
        }
        $exitCode = if ($completeReport) { 0 } else { -1 }
    }
    if ($exitCode -ne 0) {
        $stderr = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -Raw -LiteralPath $stderrPath
        } else { "" }
        throw "Vehicle RTSP benchmark failed with exit code ${exitCode}: $stderr"
    }
    if ($samples.Count -eq 0) { throw "Benchmark produced no telemetry samples" }

    function Measure-Values([double[]]$Values) {
        $sorted = @($Values | Sort-Object)
        $average = ($Values | Measure-Object -Average).Average
        return [ordered]@{
            mean = [Math]::Round([double]$average, 3)
            min = [Math]::Round([double]$sorted[0], 3)
            max = [Math]::Round([double]$sorted[-1], 3)
        }
    }
    function Measure-WindowDelta([double[]]$Values) {
        $window = [Math]::Min(60, [Math]::Max(1, [int][Math]::Floor($Values.Count / 4)))
        $firstMean = [double](($Values[0..($window - 1)] | Measure-Object -Average).Average)
        $lastStart = $Values.Count - $window
        $lastMean = [double](($Values[$lastStart..($Values.Count - 1)] | Measure-Object -Average).Average)
        return [ordered]@{
            window_samples = $window
            first_mean = [Math]::Round($firstMean, 3)
            last_mean = [Math]::Round($lastMean, 3)
            growth = [Math]::Round($lastMean - $firstMean, 3)
        }
    }
    $summary = [ordered]@{
        schema_version = "1.0"
        sample_count = $samples.Count
        process_cpu_percent = Measure-Values @($samples.process_cpu_percent)
        working_set_mib = Measure-Values @($samples.working_set_mib)
        working_set_window_trend_mib = Measure-WindowDelta @($samples.working_set_mib)
        gpu_utilization_percent = Measure-Values @($samples.gpu_utilization_percent)
        gpu_memory_used_mib = Measure-Values @($samples.gpu_memory_used_mib)
        gpu_memory_window_trend_mib = Measure-WindowDelta @($samples.gpu_memory_used_mib)
        gpu_power_w = Measure-Values @($samples.gpu_power_w)
        gpu_temperature_c = Measure-Values @($samples.gpu_temperature_c)
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $telemetrySummaryPath -Encoding UTF8
    Write-Output "PASS: RTSP benchmark report=$reportPath telemetry=$telemetrySummaryPath"
}
finally {
    if ($null -ne $telemetryWriter) {
        $telemetryWriter.Dispose()
    }
    $env:Path = $originalPath
}
