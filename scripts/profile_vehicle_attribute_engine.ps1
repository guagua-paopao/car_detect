[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round3\tensorrt-attribute",
    [string]$TrtExec = "D:\TensorRT-10.16.1.11\bin\trtexec.exe",
    [int[]]$BatchSizes = @(1, 2, 4, 8, 16),
    [switch]$SkipBuild,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$root = if ($ProjectRoot) {
    (Resolve-Path -LiteralPath $ProjectRoot).Path
} else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$evidencePath = if ([IO.Path]::IsPathRooted($EvidenceDir)) {
    [IO.Path]::GetFullPath($EvidenceDir)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $EvidenceDir))
}
New-Item -ItemType Directory -Force -Path $evidencePath | Out-Null
if (-not (Test-Path -LiteralPath $TrtExec -PathType Leaf)) {
    throw "trtexec is missing: $TrtExec"
}
foreach ($batch in $BatchSizes) {
    if ($batch -lt 1 -or $batch -gt 16) {
        throw "BatchSizes entries must be in [1, 16]"
    }
}

$onnx = Join-Path $root (
    "models\candidates\vehicle-attr-best-components-256-r1\" +
    "vehicle-attr-best-components-256-r1.onnx")
$productionEngine = Join-Path $root (
    "models\candidates\vehicle-attr-best-components-256-r1\" +
    "vehicle-attr-best-components-256-r1.engine")
if (-not (Test-Path -LiteralPath $onnx -PathType Leaf) -or
    -not (Test-Path -LiteralPath $productionEngine -PathType Leaf)) {
    throw "Production attribute ONNX or engine is missing"
}

$timingCache = Join-Path $evidencePath "vehicle-attribute.timing.cache"
$candidateRoot = Join-Path $evidencePath "engines"
New-Item -ItemType Directory -Force -Path $candidateRoot | Out-Null

function Invoke-TrtCommand(
    [string[]]$Arguments,
    [string]$LogPath,
    [string]$Description) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $TrtExec @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath | Out-Null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode; see $LogPath"
    }
}

function Build-Candidate(
    [string]$Name,
    [int]$MinimumBatch,
    [int]$OptimumBatch,
    [int]$MaximumBatch,
    [int]$AuxStreams) {
    $engine = Join-Path $candidateRoot "$Name.engine"
    $log = Join-Path $evidencePath "$Name-build.log"
    $arguments = @(
        "--onnx=$onnx",
        "--saveEngine=$engine",
        "--minShapes=images:${MinimumBatch}x3x256x256",
        "--optShapes=images:${OptimumBatch}x3x256x256",
        "--maxShapes=images:${MaximumBatch}x3x256x256",
        "--fp16",
        "--builderOptimizationLevel=5",
        "--maxAuxStreams=$AuxStreams",
        "--memPoolSize=workspace:4096",
        "--timingCacheFile=$timingCache",
        "--profilingVerbosity=detailed",
        "--skipInference"
    )
    Invoke-TrtCommand $arguments $log "TensorRT build $Name"
    if (-not (Test-Path -LiteralPath $engine -PathType Leaf)) {
        throw "TensorRT build did not create $engine"
    }
    return [ordered]@{
        name = $Name
        engine = $engine
        engine_sha256 = (Get-FileHash -LiteralPath $engine -Algorithm SHA256).Hash.ToLowerInvariant()
        engine_bytes = (Get-Item -LiteralPath $engine).Length
        minimum_batch = $MinimumBatch
        optimum_batch = $OptimumBatch
        maximum_batch = $MaximumBatch
        max_aux_streams = $AuxStreams
        build_log = $log
        build_arguments = $arguments
    }
}

function Measure-Values([double[]]$Values) {
    if (-not $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    function Get-Percentile([double[]]$Ordered, [double]$Quantile) {
        $position = $Quantile * ($Ordered.Count - 1)
        $lower = [int][Math]::Floor($position)
        $upper = [int][Math]::Ceiling($position)
        if ($lower -eq $upper) { return [double]$Ordered[$lower] }
        $fraction = $position - $lower
        return [double]$Ordered[$lower] * (1.0 - $fraction) +
            [double]$Ordered[$upper] * $fraction
    }
    return [ordered]@{
        count = $sorted.Count
        mean = [double](($sorted | Measure-Object -Average).Average)
        p50 = Get-Percentile $sorted 0.50
        p95 = Get-Percentile $sorted 0.95
        p99 = Get-Percentile $sorted 0.99
        min = [double]$sorted[0]
        max = [double]$sorted[-1]
    }
}

function Profile-Engine([object]$Descriptor, [int]$Batch) {
    $profileStem = "$($Descriptor.name)-batch-$Batch"
    $times = Join-Path $evidencePath "$profileStem-times.json"
    $layers = Join-Path $evidencePath "$profileStem-layer-profile.json"
    $layerInfo = Join-Path $evidencePath "$profileStem-layer-info.json"
    $log = Join-Path $evidencePath "$profileStem-profile.log"
    $iterations = if ($Quick) { 300 } else { 2000 }
    $warmup = if ($Quick) { 300 } else { 1000 }
    $arguments = @(
        "--loadEngine=$($Descriptor.engine)",
        "--shapes=images:${Batch}x3x256x256",
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
    Invoke-TrtCommand $arguments $log "TensorRT profile $profileStem"
    $timings = Get-Content -Raw -LiteralPath $times | ConvertFrom-Json
    $compute = [double[]]@($timings.computeMs)
    $latency = [double[]]@($timings.latencyMs)
    return [ordered]@{
        name = $Descriptor.name
        batch = $Batch
        compute_ms = Measure-Values $compute
        latency_ms = Measure-Values $latency
        throughput_images_per_second = if ($compute.Count -gt 0) {
            1000.0 * $Batch / [double](($compute | Measure-Object -Average).Average)
        } else { 0.0 }
        times = $times
        layer_profile = $layers
        layer_info = $layerInfo
        log = $log
        arguments = $arguments
    }
}

$engines = [Collections.Generic.List[object]]::new()
$engines.Add([ordered]@{
    name = "production"
    engine = $productionEngine
    engine_sha256 = (Get-FileHash -LiteralPath $productionEngine -Algorithm SHA256).Hash.ToLowerInvariant()
    engine_bytes = (Get-Item -LiteralPath $productionEngine).Length
    minimum_batch = 1
    optimum_batch = 16
    maximum_batch = 16
    max_aux_streams = $null
    build_log = $null
    build_arguments = @()
})

if (-not $SkipBuild) {
    $engines.Add((Build-Candidate "dynamic-aux0" 1 8 16 0))
    $engines.Add((Build-Candidate "dynamic-aux2" 1 8 16 2))
    $engines.Add((Build-Candidate "dynamic-aux4" 1 8 16 4))
    foreach ($batch in $BatchSizes) {
        $engines.Add((Build-Candidate "static-batch-$batch-aux2" $batch $batch $batch 2))
    }
} else {
    foreach ($engineFile in Get-ChildItem -LiteralPath $candidateRoot -Filter "*.engine") {
        if ($engineFile.BaseName -match '^static-batch-(\d+)-aux2$') {
            $batch = [int]$Matches[1]
            $minimum = $batch
            $optimum = $batch
            $maximum = $batch
            $aux = 2
        } else {
            $minimum = 1
            $optimum = 8
            $maximum = 16
            $aux = if ($engineFile.BaseName -match 'aux(\d+)$') {
                [int]$Matches[1]
            } else { $null }
        }
        $engines.Add([ordered]@{
            name = $engineFile.BaseName
            engine = $engineFile.FullName
            engine_sha256 = (Get-FileHash -LiteralPath $engineFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            engine_bytes = $engineFile.Length
            minimum_batch = $minimum
            optimum_batch = $optimum
            maximum_batch = $maximum
            max_aux_streams = $aux
            build_log = Join-Path $evidencePath "$($engineFile.BaseName)-build.log"
            build_arguments = @()
        })
    }
}

$profiles = [Collections.Generic.List[object]]::new()
foreach ($engine in $engines) {
    foreach ($batch in $BatchSizes) {
        if ($batch -lt $engine.minimum_batch -or $batch -gt $engine.maximum_batch) {
            continue
        }
        if ($engine.minimum_batch -eq $engine.maximum_batch -and
            $batch -ne $engine.minimum_batch) {
            continue
        }
        $profiles.Add((Profile-Engine $engine $batch))
    }
}

$report = [ordered]@{
    schema_version = "1.0"
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    objective = "same-ONNX FP16 attribute TensorRT engine matrix"
    trtexec = $TrtExec
    onnx = $onnx
    onnx_sha256 = (Get-FileHash -LiteralPath $onnx -Algorithm SHA256).Hash.ToLowerInvariant()
    model_architecture = "dual independent ConvNeXt-Tiny specialists in one graph"
    input = "NCHW float32 1..16x3x256x256; ImageNet normalization embedded in graph"
    timing_cache = $timingCache
    batches = $BatchSizes
    engines = $engines
    profiles = $profiles
}
$reportPath = Join-Path $evidencePath "profile-report.json"
$report | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Output "PASS: attribute TensorRT matrix report=$reportPath"
