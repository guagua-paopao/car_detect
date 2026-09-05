[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$EvidenceDir = ".\reports\gpu-pipeline-optimization\round2\tensorrt-engine",
    [string]$TrtExec = "D:\TensorRT-10.16.1.11\bin\trtexec.exe"
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

$onnx = Join-Path $root "models\vehicle-det-v1.onnx"
$productionEngine = Join-Path $root "engines\vehicle-det-v1.engine"
$candidateEngine = Join-Path $evidencePath "vehicle-det-v1-fixed-profile.engine"
$timingCache = Join-Path $evidencePath "vehicle-det-v1.timing.cache"
$buildLog = Join-Path $evidencePath "candidate-build.log"

$buildArguments = @(
    "--onnx=$onnx",
    "--saveEngine=$candidateEngine",
    "--fp16",
    "--builderOptimizationLevel=5",
    "--maxAuxStreams=2",
    "--memPoolSize=workspace:4096",
    "--timingCacheFile=$timingCache",
    "--profilingVerbosity=detailed",
    "--skipInference"
)
$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $TrtExec @buildArguments 2>&1 | Tee-Object -FilePath $buildLog | Out-Null
$buildExitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
if ($buildExitCode -ne 0 -or -not (Test-Path -LiteralPath $candidateEngine)) {
    throw "TensorRT fixed-profile candidate build failed; see $buildLog"
}

function Invoke-Profile([string]$Name, [string]$Engine) {
    $profileJson = Join-Path $evidencePath "$Name-layer-profile.json"
    $layersJson = Join-Path $evidencePath "$Name-layer-info.json"
    $log = Join-Path $evidencePath "$Name-profile.log"
    $arguments = @(
        "--loadEngine=$Engine",
        "--warmUp=1000",
        "--duration=10",
        "--iterations=2000",
        "--avgRuns=100",
        "--useCudaGraph",
        "--noDataTransfers",
        "--separateProfileRun",
        "--dumpProfile",
        "--dumpLayerInfo",
        "--profilingVerbosity=detailed",
        "--exportProfile=$profileJson",
        "--exportLayerInfo=$layersJson"
    )
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $TrtExec @arguments 2>&1 | Tee-Object -FilePath $log | Out-Null
    $profileExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($profileExitCode -ne 0) { throw "TensorRT profiling failed for $Name; see $log" }
    return [ordered]@{
        name = $Name
        engine = $Engine
        engine_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Engine).Hash.ToLowerInvariant()
        engine_bytes = (Get-Item -LiteralPath $Engine).Length
        log = $log
        layer_profile = $profileJson
        layer_info = $layersJson
        arguments = $arguments
    }
}

$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$version = (& $TrtExec --version 2>&1 | Out-String).Trim()
$ErrorActionPreference = $savedErrorPreference
$productionProfile = Invoke-Profile "production" $productionEngine
$candidateProfile = Invoke-Profile "fixed-profile-candidate" $candidateEngine
$report = [ordered]@{
    schema_version = "1.0"
    generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    trtexec = $TrtExec
    tensorrt_version_output = $version
    onnx = $onnx
    onnx_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnx).Hash.ToLowerInvariant()
    precision = "fp16"
    input_shape = "1x3x960x960"
    int8 = "not evaluated: no confirmed representative INT8 calibration set/cache for the detection model"
    candidate_build_arguments = $buildArguments
    candidate_build_log = $buildLog
    timing_cache = $timingCache
    production = $productionProfile
    candidate = $candidateProfile
}
$reportPath = Join-Path $evidencePath "profile-report.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: TensorRT engine profile report=$reportPath candidate=$candidateEngine"
