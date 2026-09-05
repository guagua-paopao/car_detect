[CmdletBinding()]
param(
    [string]$EvidencePath = ".\reports\gpu-pipeline-optimization\round3\environment.json",
    [string]$Ffmpeg = "E:\ffmpeg-7.1\ffmpeg-2024-12-27-git-5f38c82536-full_build\bin\ffmpeg.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedEvidence = if ([IO.Path]::IsPathRooted($EvidencePath)) {
    [IO.Path]::GetFullPath($EvidencePath)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidencePath))
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedEvidence) | Out-Null

function Invoke-Version([string]$Executable, [string[]]$Arguments) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        return ((& $Executable @Arguments 2>&1 | Out-String).Trim())
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
}

function Get-Artifact([string]$RelativePath) {
    $path = [IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        return [ordered]@{ path = $RelativePath; present = $false }
    }
    return [ordered]@{
        path = $RelativePath.Replace("\", "/")
        present = $true
        bytes = (Get-Item -LiteralPath $path).Length
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$gpuCsv = (& nvidia-smi.exe `
    --query-gpu=name,driver_version,memory.total,compute_cap `
    --format=csv,noheader,nounits | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi environment query failed" }
$gpu = @($gpuCsv.Split(",") | ForEach-Object { $_.Trim() })
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$opencv = (Get-Command opencv_version.exe -CommandType Application -ErrorAction Stop).Source
$nvcc = (Get-Command nvcc.exe -CommandType Application -ErrorAction Stop).Source
$trtexec = "D:\TensorRT-10.16.1.11\bin\trtexec.exe"
$nvinfer = "D:\TensorRT-10.16.1.11\bin\nvinfer_10.dll"
if (-not (Test-Path -LiteralPath $Ffmpeg -PathType Leaf)) {
    throw "FFmpeg is missing: $Ffmpeg"
}
if (-not (Test-Path -LiteralPath $trtexec -PathType Leaf) -or
    -not (Test-Path -LiteralPath $nvinfer -PathType Leaf)) {
    throw "TensorRT 10.16 runtime is missing"
}

$ffmpegVersion = Invoke-Version $Ffmpeg @("-version")
$ffmpegFirstLine = ($ffmpegVersion -split "`r?`n")[0]
$report = [ordered]@{
    schema_version = "1.0"
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    project_root = $projectRoot
    git_head = ((& git -C $projectRoot rev-parse HEAD) | Out-String).Trim()
    os = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build = $os.BuildNumber
    }
    cpu = [ordered]@{
        name = ([string]$cpu.Name).Trim()
        logical_processors = [int]$cpu.NumberOfLogicalProcessors
    }
    gpu = [ordered]@{
        name = $gpu[0]
        driver = $gpu[1]
        memory_mib = [double]$gpu[2]
        compute_capability = $gpu[3]
        mode = "Windows WDDM"
    }
    cuda = [ordered]@{
        nvcc = $nvcc
        version_output = Invoke-Version $nvcc @("--version")
    }
    tensorrt = [ordered]@{
        version = "10.16.1.11 (trtexec runtime code v101601)"
        trtexec = $trtexec
        runtime_dll = $nvinfer
    }
    cudnn = [ordered]@{
        status = "not installed in the configured CUDA/TensorRT runtime and not linked by this project"
        version = $null
    }
    opencv = [ordered]@{
        executable = $opencv
        version = Invoke-Version $opencv @()
    }
    ffmpeg = [ordered]@{
        executable = [IO.Path]::GetFullPath($Ffmpeg)
        version = $ffmpegFirstLine
        nvdec_experiment = "h264_cuvid -> CUDA surface -> scale_cuda; no hwdownload"
    }
    build = [ordered]@{
        configuration = "Release"
        generator = "Ninja"
        cxx_compiler = "MSVC 19.51.36252.0"
        cxx_release_flags = "/O2 /Ob2 /DNDEBUG"
        cuda_compiler = "NVCC 13.3.73"
        cuda_architectures = "89"
        build_directory = "out/build/gpu-pipeline-round2-baseline"
    }
    formal_benchmark = [ordered]@{
        source = "rtsp://127.0.0.1:18554/vcas-perf"
        source_fps = 25
        warmup_frames = 200
        measured_frames = 2000
        runs = 3
        aggregation = "median of independent runs; A/B order alternated"
        detection_confidence_threshold = 0.25
        detection_nms_iou_threshold = 0.45
        detection_input = "1x3x960x960 FP16 engine"
        attribute_max_batch = 16
        stability_measured_frames = 45000
    }
    artifacts = @(
        Get-Artifact "models/vehicle-det-v1.onnx"
        Get-Artifact "engines/vehicle-det-v1.engine"
        Get-Artifact "models/candidates/vehicle-attr-best-components-256-r1/vehicle-attr-best-components-256-r1.engine"
        Get-Artifact "demo/rtsp_standard/output/vcas_rtsp_demo_60s.mp4"
    )
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resolvedEvidence -Encoding UTF8
Write-Output "PASS: GPU pipeline environment report=$resolvedEvidence"
