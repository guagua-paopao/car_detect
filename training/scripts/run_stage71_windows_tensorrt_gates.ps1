[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateRoot,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedOnnxExportReportSha256,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedOnnxParityReportSha256,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedTrtParityManifestSha256,
    [string]$TensorRtRoot = "D:\TensorRT-10.16.1.11",
    [string]$Python = "python",
    [string]$Comparator = ""
)

$ErrorActionPreference = "Stop"
$candidate = (Resolve-Path -LiteralPath $CandidateRoot).Path
$trtRoot = (Resolve-Path -LiteralPath $TensorRtRoot).Path
$trtExec = Join-Path $trtRoot "bin\trtexec.exe"
if (-not (Test-Path -LiteralPath $trtExec -PathType Leaf)) {
    throw "trtexec.exe is missing: $trtExec"
}
if (-not $Comparator) {
    $Comparator = Join-Path $PSScriptRoot "compare_stage71_attribute_trt_outputs.py"
}
$comparatorPath = (Resolve-Path -LiteralPath $Comparator).Path
$expectedComparatorSha = "e370b6e0b9713da7faa353127889201b9eb529fd662c6d8fb5dcffeb2948bbc4"
$actualComparatorSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $comparatorPath).Hash.ToLowerInvariant()
if ($actualComparatorSha -ne $expectedComparatorSha) {
    throw "TensorRT comparator SHA256 mismatch"
}

$exportReportPath = Join-Path $candidate "onnx\onnx-export-report.json"
$onnxParityPath = Join-Path $candidate "onnx-parity.json"
$trtParityManifestPath = Join-Path $candidate "trt-parity-inputs\manifest.json"
$pinnedFiles = @(
    @{ Path = $exportReportPath; Expected = $ExpectedOnnxExportReportSha256; Label = "ONNX export report" },
    @{ Path = $onnxParityPath; Expected = $ExpectedOnnxParityReportSha256; Label = "ONNX parity report" },
    @{ Path = $trtParityManifestPath; Expected = $ExpectedTrtParityManifestSha256; Label = "TRT parity manifest" }
)
foreach ($entry in $pinnedFiles) {
    if (-not (Test-Path -LiteralPath $entry.Path -PathType Leaf)) {
        throw "$($entry.Label) is missing: $($entry.Path)"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Path).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Expected.ToLowerInvariant()) {
        throw "$($entry.Label) SHA256 mismatch"
    }
}

$exportReport = Get-Content -Raw -Encoding utf8 -LiteralPath $exportReportPath | ConvertFrom-Json
$onnxParity = Get-Content -Raw -Encoding utf8 -LiteralPath $onnxParityPath | ConvertFrom-Json
$trtManifest = Get-Content -Raw -Encoding utf8 -LiteralPath $trtParityManifestPath | ConvertFrom-Json
if ($exportReport.status -ne "pass_onnx_exported_candidate_only") {
    throw "ONNX export report is not eligible"
}
if ($onnxParity.status -ne "pass" -or $onnxParity.gate -ne $true) {
    throw "PyTorch/ONNX parity did not pass"
}
if ($trtManifest.status -ne "pass_inputs_prepared") {
    throw "TensorRT parity inputs are not eligible"
}
if ($exportReport.policy.frozen_video_used -ne $false -or
    $onnxParity.policy.frozen_video_used -ne $false -or
    $trtManifest.policy.frozen_video_used -ne $false) {
    throw "frozen-video policy violation"
}

$outputRoot = [IO.Path]::GetFullPath((Join-Path $candidate "windows-tensorrt-v1"))
$candidateBoundary = $candidate.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $outputRoot.StartsWith($candidateBoundary, [StringComparison]::OrdinalIgnoreCase)) {
    throw "TensorRT output escapes candidate root"
}
if (Test-Path -LiteralPath $outputRoot) {
    throw "refusing to overwrite TensorRT evidence: $outputRoot"
}
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$models = @("body", "color")
$modelReports = [ordered]@{}
foreach ($model in $models) {
    $onnx = Join-Path $candidate "onnx\$model-specialist.onnx"
    if (-not (Test-Path -LiteralPath $onnx -PathType Leaf)) {
        throw "$model ONNX is missing: $onnx"
    }
    $expectedOnnxSha = $exportReport.exports.$model.onnx_sha256
    $actualOnnxSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnx).Hash.ToLowerInvariant()
    if ($actualOnnxSha -ne $expectedOnnxSha.ToLowerInvariant()) {
        throw "$model ONNX SHA256 mismatch"
    }
    $inputSize = [int]$exportReport.exports.$model.input_size
    if ($inputSize -notin @(224, 256)) {
        throw "unsupported $model input size: $inputSize"
    }
    $engine = Join-Path $outputRoot "$model-specialist.fp16.engine"
    $buildLog = Join-Path $outputRoot "$model-engine-build.log"
    $buildArgs = @(
        "--onnx=$onnx",
        "--saveEngine=$engine",
        "--fp16",
        "--skipInference",
        "--minShapes=images:1x3x${inputSize}x${inputSize}",
        "--optShapes=images:8x3x${inputSize}x${inputSize}",
        "--maxShapes=images:16x3x${inputSize}x${inputSize}"
    )
    & $trtExec @buildArgs *> $buildLog
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $engine -PathType Leaf)) {
        throw "$model TensorRT engine build failed; see $buildLog"
    }

    $raw = Join-Path $candidate "trt-parity-inputs\$model-input.raw"
    $expectedRawSha = $trtManifest.models.$model.input_raw_sha256
    $actualRawSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $raw).Hash.ToLowerInvariant()
    if ($actualRawSha -ne $expectedRawSha.ToLowerInvariant()) {
        throw "$model raw parity input SHA256 mismatch"
    }
    $batchSize = [int]$trtManifest.models.$model.batch_size
    $trtOutput = Join-Path $outputRoot "$model-trtexec-output.json"
    $inferLog = Join-Path $outputRoot "$model-trtexec-inference.log"
    $inferArgs = @(
        "--loadEngine=$engine",
        "--shapes=images:${batchSize}x3x${inputSize}x${inputSize}",
        "--loadInputs=images:$raw",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        "--exportOutput=$trtOutput"
    )
    & $trtExec @inferArgs *> $inferLog
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $trtOutput -PathType Leaf)) {
        throw "$model TensorRT parity inference failed; see $inferLog"
    }

    $parityReport = Join-Path $outputRoot "$model-tensorrt-parity.json"
    & $Python $comparatorPath `
        --parity-manifest $trtParityManifestPath `
        --expected-parity-manifest-sha256 $ExpectedTrtParityManifestSha256 `
        --model $model `
        --trt-output $trtOutput `
        --engine $engine `
        --output $parityReport
    if ($LASTEXITCODE -ne 0) {
        throw "$model TensorRT/ONNX parity failed"
    }
    $parity = Get-Content -Raw -Encoding utf8 -LiteralPath $parityReport | ConvertFrom-Json
    if ($parity.status -ne "pass" -or $parity.gate -ne $true) {
        throw "$model TensorRT/ONNX parity gate failed"
    }

    $timing = Join-Path $outputRoot "$model-timing.json"
    $timingLog = Join-Path $outputRoot "$model-benchmark.log"
    $benchmarkArgs = @(
        "--loadEngine=$engine",
        "--shapes=images:${batchSize}x3x${inputSize}x${inputSize}",
        "--loadInputs=images:$raw",
        "--iterations=100",
        "--warmUp=500",
        "--duration=3",
        "--exportTimes=$timing"
    )
    & $trtExec @benchmarkArgs *> $timingLog
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $timing -PathType Leaf)) {
        throw "$model TensorRT benchmark failed; see $timingLog"
    }
    $modelReports[$model] = [ordered]@{
        onnx = $onnx
        onnx_sha256 = $actualOnnxSha
        engine = $engine
        engine_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $engine).Hash.ToLowerInvariant()
        input_size = $inputSize
        batch_size = $batchSize
        parity_report = $parityReport
        parity_report_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $parityReport).Hash.ToLowerInvariant()
        timing = $timing
        timing_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $timing).Hash.ToLowerInvariant()
    }
}

foreach ($entry in $pinnedFiles) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Path).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Expected.ToLowerInvariant()) {
        throw "$($entry.Label) changed during TensorRT validation"
    }
}
$report = [ordered]@{
    schema_version = "stage71-windows-tensorrt-gates-v1"
    status = "pass_waiting_cpp_and_real_engine_contract"
    candidate_root = $candidate
    tensor_rt_root = $trtRoot
    trtexec_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $trtExec).Hash.ToLowerInvariant()
    models = $modelReports
    gates = [ordered]@{
        body_engine_build = $true
        color_engine_build = $true
        body_onnx_tensorrt_parity = $true
        color_onnx_tensorrt_parity = $true
        timing_evidence_complete = $true
    }
    policy = [ordered]@{
        validation_only_backend_inputs = $true
        test_accessed_by_this_stage = $false
        frozen_video_used = $false
        production_model_modified = $false
        production_registry_modified = $false
        production_config_modified = $false
        deployment_performed = $false
        deployment_paused_by_user = $true
    }
}
$reportPath = Join-Path $outputRoot "windows-tensorrt-report.json"
$report | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: Stage71 TensorRT engines and routed-head parity; report=$reportPath"
