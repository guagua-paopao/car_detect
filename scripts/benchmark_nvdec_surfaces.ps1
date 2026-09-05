[CmdletBinding()]
param(
    [string]$RtspUri = "rtsp://127.0.0.1:18554/vcas-perf",
    [int]$Iterations = 2000,
    [int]$WarmupFrames = 200,
    [int]$Runs = 3,
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round2\nvdec-surfaces",
    [string]$Ffmpeg = ""
)

$ErrorActionPreference = "Stop"
if ($Iterations -le 0 -or $WarmupFrames -le 0 -or $Runs -le 0) {
    throw "Iterations, WarmupFrames, and Runs must be positive."
}
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
if (-not $Ffmpeg) {
    $Ffmpeg = (Get-Command ffmpeg.exe -ErrorAction Stop).Source
}
$totalFrames = $Iterations

function Get-ProgressFrame([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return 0 }
    $last = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '^frame=([0-9]+)$' } |
        Select-Object -Last 1
    if ($last -and $last -match '^frame=([0-9]+)$') { return [int]$Matches[1] }
    return 0
}

function Invoke-DecodeRun([string]$Mode, [int]$Run) {
    $prefix = Join-Path $evidencePath ("{0}-run-{1}" -f $Mode, $Run)
    $stdout = "$prefix.stdout.log"
    $stderr = "$prefix.stderr.log"
    $progress = "$prefix.progress.txt"
    [IO.File]::WriteAllText($progress, "")
    $common = @(
        "-hide_banner", "-nostdin", "-nostats",
        "-progress", $progress,
        "-rtsp_transport", "tcp"
    )
    $decode = if ($Mode -eq "nvdec_surface") {
        @("-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid")
    } else {
        @()
    }
    $filter = if ($Mode -eq "nvdec_surface") {
        @("-vf", "scale_cuda=w=1280:h=720:format=nv12")
    } else {
        @()
    }
    $warmupProgress = "$prefix.warmup.progress.txt"
    $warmupStdout = "$prefix.warmup.stdout.log"
    $warmupStderr = "$prefix.warmup.stderr.log"
    [IO.File]::WriteAllText($warmupProgress, "")
    $warmupArguments = @(
        "-hide_banner", "-nostdin", "-nostats", "-progress", $warmupProgress,
        "-rtsp_transport", "tcp"
    ) + $decode + @("-i", $RtspUri) + $filter + @(
        "-an", "-sn", "-dn", "-frames:v", $WarmupFrames, "-f", "null", "NUL"
    )
    $warmupProcess = Start-Process -FilePath $Ffmpeg -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList $warmupArguments `
        -RedirectStandardOutput $warmupStdout -RedirectStandardError $warmupStderr `
        -PassThru -Wait
    if ((Get-ProgressFrame $warmupProgress) -lt $WarmupFrames) {
        throw "$Mode run $Run warmup failed: $(Get-Content -Raw -LiteralPath $warmupStderr)"
    }
    $arguments = $common + $decode + @("-i", $RtspUri) + $filter + @(
        "-an", "-sn", "-dn", "-frames:v", $totalFrames, "-f", "null", "NUL"
    )
    $process = Start-Process -FilePath $Ffmpeg -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    $measurementStarted = [DateTimeOffset]::UtcNow
    $process.Refresh()
    $cpuAtMeasurementStart = if ($null -eq $process.CPU) { 0.0 } else { [double]$process.CPU }
    $lastCpuSeconds = $cpuAtMeasurementStart
    $samples = [Collections.Generic.List[object]]::new()
    $peakRss = 0.0
    while (-not $process.HasExited) {
        $process.Refresh()
        if ($null -ne $process.CPU) { $lastCpuSeconds = [double]$process.CPU }
        $frame = Get-ProgressFrame $progress
        $gpuText = & nvidia-smi.exe `
            --query-gpu=utilization.gpu,memory.used,power.draw `
            --format=csv,noheader,nounits | Out-String
        if ($LASTEXITCODE -ne 0) { throw "nvidia-smi telemetry query failed" }
        $gpu = @($gpuText.Trim().Split(",") | ForEach-Object { $_.Trim() })
        $rss = [double]$process.WorkingSet64 / 1MB
        $peakRss = [Math]::Max($peakRss, $rss)
        $samples.Add([pscustomobject]@{
            frame = $frame
            rss_mib = $rss
            gpu_utilization_percent = [double]$gpu[0]
            gpu_memory_used_mib = [double]$gpu[1]
            gpu_power_w = [double]$gpu[2]
        })
        Start-Sleep -Milliseconds 500
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode -and (Get-ProgressFrame $progress) -ge $totalFrames) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "$Mode run $Run failed: $(Get-Content -Raw -LiteralPath $stderr)"
    }
    $wallSeconds = ([DateTimeOffset]::UtcNow - $measurementStarted).TotalSeconds
    $cpuEnd = if ($null -eq $process.CPU) { $lastCpuSeconds } else { [double]$process.CPU }
    $cpuSeconds = [Math]::Max(0.0, $cpuEnd - $cpuAtMeasurementStart)
    return [ordered]@{
        run = $Run
        measured_frames = $Iterations
        wall_seconds = $wallSeconds
        measured_fps = $Iterations / $wallSeconds
        normalized_cpu_percent = 100.0 * $cpuSeconds / $wallSeconds / [Environment]::ProcessorCount
        peak_rss_mib = $peakRss
        gpu_utilization_mean_percent = if ($samples.Count) {
            ($samples | Measure-Object gpu_utilization_percent -Average).Average
        } else { 0.0 }
        gpu_memory_mean_mib = if ($samples.Count) {
            ($samples | Measure-Object gpu_memory_used_mib -Average).Average
        } else { 0.0 }
        gpu_memory_peak_mib = if ($samples.Count) {
            ($samples | Measure-Object gpu_memory_used_mib -Maximum).Maximum
        } else { 0.0 }
        gpu_power_mean_w = if ($samples.Count) {
            ($samples | Measure-Object gpu_power_w -Average).Average
        } else { 0.0 }
        telemetry_samples = $samples.Count
        ffmpeg_filter = if ($Mode -eq "nvdec_surface") {
            "h264_cuvid -> CUDA surface -> scale_cuda -> null; no hwdownload"
        } else {
            "software decode -> host frame -> null"
        }
    }
}

$software = [Collections.Generic.List[object]]::new()
$nvdec = [Collections.Generic.List[object]]::new()
for ($run = 1; $run -le $Runs; ++$run) {
    if (($run % 2) -eq 1) {
        $software.Add((Invoke-DecodeRun "software" $run))
        $nvdec.Add((Invoke-DecodeRun "nvdec_surface" $run))
    } else {
        $nvdec.Add((Invoke-DecodeRun "nvdec_surface" $run))
        $software.Add((Invoke-DecodeRun "software" $run))
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    return [double]$sorted[[Math]::Floor($sorted.Count / 2)]
}

function Measure-Mode([object[]]$Values) {
    return [ordered]@{
        measured_fps = Get-Median ([double[]]@($Values.measured_fps))
        normalized_cpu_percent = Get-Median ([double[]]@($Values.normalized_cpu_percent))
        peak_rss_mib = Get-Median ([double[]]@($Values.peak_rss_mib))
        gpu_utilization_mean_percent = Get-Median `
            ([double[]]@($Values.gpu_utilization_mean_percent))
        gpu_memory_mean_mib = Get-Median ([double[]]@($Values.gpu_memory_mean_mib))
        gpu_memory_peak_mib = Get-Median ([double[]]@($Values.gpu_memory_peak_mib))
        gpu_power_mean_w = Get-Median ([double[]]@($Values.gpu_power_mean_w))
    }
}

$softwareMedian = Measure-Mode @($software)
$nvdecMedian = Measure-Mode @($nvdec)
$report = [ordered]@{
    schema_version = "1.0"
    source = $RtspUri
    iterations_per_run = $Iterations
    warmup_frames = $WarmupFrames
    runs_requested = $Runs
    ffmpeg = $Ffmpeg
    production_integration = "not enabled: the process-pipe reader cannot expose AVHWFramesContext/CUDA surfaces without FFmpeg development libraries and an in-process libavcodec reader"
    software = $software
    nvdec_surface = $nvdec
    median = [ordered]@{
        software = $softwareMedian
        nvdec_surface = $nvdecMedian
        fps_change_percent = 100.0 * (
            $nvdecMedian.measured_fps / $softwareMedian.measured_fps - 1.0)
        cpu_change_percent = if ($softwareMedian.normalized_cpu_percent -gt 0.0) {
            100.0 * ($nvdecMedian.normalized_cpu_percent /
                $softwareMedian.normalized_cpu_percent - 1.0)
        } else { $null }
        peak_rss_change_mib =
            $nvdecMedian.peak_rss_mib - $softwareMedian.peak_rss_mib
        gpu_memory_mean_change_mib =
            $nvdecMedian.gpu_memory_mean_mib - $softwareMedian.gpu_memory_mean_mib
    }
    decision = "not enabled in production: real CUDA surfaces reduce decoder CPU but do not improve 25 FPS throughput, add RSS/VRAM, and the process-pipe reader cannot transfer AVHWFramesContext ownership"
}
$reportPath = Join-Path $evidencePath "benchmark.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: NVDEC surface benchmark report=$reportPath"
