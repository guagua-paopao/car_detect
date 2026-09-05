[CmdletBinding()]
param(
    [string]$EvidenceDir =
        ".\reports\gpu-pipeline-optimization\round3\attribute-branches",
    [string]$TrtExec = "D:\TensorRT-10.16.1.11\bin\trtexec.exe",
    [int[]]$BatchSizes = @(1, 2, 4, 8, 16),
    [int]$AuxStreams = 0,
    [switch]$SkipBuild,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
$timingCache = Join-Path $evidencePath "branches.timing.cache"

function Invoke-Trt([string[]]$Arguments, [string]$Log, [string]$Name) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $TrtExec @Arguments 2>&1 | Tee-Object -FilePath $Log | Out-Null
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $saved
    }
    if ($code -ne 0) { throw "$Name failed with exit code $code; see $Log" }
}

function Measure-Values([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    function Percentile([double[]]$Items, [double]$Q) {
        $position = $Q * ($Items.Count - 1)
        $lower = [int][Math]::Floor($position)
        $upper = [int][Math]::Ceiling($position)
        if ($lower -eq $upper) { return [double]$Items[$lower] }
        $fraction = $position - $lower
        return [double]$Items[$lower] * (1.0 - $fraction) +
            [double]$Items[$upper] * $fraction
    }
    return [ordered]@{
        count = $sorted.Count
        mean = [double](($sorted | Measure-Object -Average).Average)
        p50 = Percentile $sorted 0.50
        p95 = Percentile $sorted 0.95
        p99 = Percentile $sorted 0.99
        min = [double]$sorted[0]
        max = [double]$sorted[-1]
    }
}

$engines = [Collections.Generic.List[object]]::new()
foreach ($branch in @("body", "color")) {
    $onnx = Join-Path $evidencePath "vehicle-attribute-$branch.onnx"
    $engine = Join-Path $evidencePath "vehicle-attribute-$branch-aux$AuxStreams.engine"
    $buildLog = Join-Path $evidencePath "$branch-aux$AuxStreams-build.log"
    $buildArgs = @(
        "--onnx=$onnx",
        "--saveEngine=$engine",
        "--minShapes=images:1x3x256x256",
        "--optShapes=images:8x3x256x256",
        "--maxShapes=images:16x3x256x256",
        "--fp16",
        "--builderOptimizationLevel=5",
        "--maxAuxStreams=$AuxStreams",
        "--memPoolSize=workspace:4096",
        "--timingCacheFile=$timingCache",
        "--profilingVerbosity=detailed",
        "--skipInference"
    )
    if (-not $SkipBuild) {
        Invoke-Trt $buildArgs $buildLog "build $branch branch"
    }
    if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
        throw "branch engine is missing: $engine"
    }
    $engines.Add([ordered]@{
        branch = $branch
        onnx = $onnx
        onnx_sha256 = (Get-FileHash $onnx -Algorithm SHA256).Hash.ToLowerInvariant()
        engine = $engine
        engine_sha256 = (Get-FileHash $engine -Algorithm SHA256).Hash.ToLowerInvariant()
        engine_bytes = (Get-Item $engine).Length
        max_aux_streams = $AuxStreams
        build_log = $buildLog
        build_arguments = $buildArgs
    })
}

$profiles = [Collections.Generic.List[object]]::new()
foreach ($descriptor in $engines) {
    foreach ($batch in $BatchSizes) {
        $stem = "$($descriptor.branch)-aux$AuxStreams-batch-$batch"
        $times = Join-Path $evidencePath "$stem-times.json"
        $layers = Join-Path $evidencePath "$stem-layer-profile.json"
        $layerInfo = Join-Path $evidencePath "$stem-layer-info.json"
        $log = Join-Path $evidencePath "$stem-profile.log"
        $iterations = if ($Quick) { 300 } else { 2000 }
        $warmup = if ($Quick) { 300 } else { 1000 }
        $arguments = @(
            "--loadEngine=$($descriptor.engine)",
            "--shapes=images:${batch}x3x256x256",
            "--warmUp=$warmup",
            "--iterations=$iterations",
            "--avgRuns=100",
            "--useCudaGraph",
            "--noDataTransfers",
            "--separateProfileRun",
            "--dumpProfile",
            "--dumpLayerInfo",
            "--profilingVerbosity=detailed",
            "--exportTimes=$times",
            "--exportProfile=$layers",
            "--exportLayerInfo=$layerInfo"
        )
        Invoke-Trt $arguments $log "profile $stem"
        $timings = Get-Content -Raw -LiteralPath $times | ConvertFrom-Json
        $compute = [double[]]@($timings.computeMs)
        $profiles.Add([ordered]@{
            branch = $descriptor.branch
            batch = $batch
            compute_ms = Measure-Values $compute
            throughput_images_per_second =
                1000.0 * $batch /
                [double](($compute | Measure-Object -Average).Average)
            times = $times
            layer_profile = $layers
            layer_info = $layerInfo
            log = $log
            arguments = $arguments
        })
    }
}

$report = [ordered]@{
    schema_version = "1.0"
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    objective = "dependency-preserving body/color branch TensorRT profile"
    source_onnx_sha256 =
        "b470c013acff8cfbe9a7cf7ead8184f8a1516998507cfce4ba342e4ecf4ed86e"
    extraction_report = Join-Path $evidencePath "extraction-report.json"
    timing_cache = $timingCache
    engines = $engines
    profiles = $profiles
}
$reportPath = Join-Path $evidencePath "branch-profile-report.json"
$report | ConvertTo-Json -Depth 12 |
    Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: branch profile report=$reportPath"
