[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$RegistryPath = "models\manifests\model_registry.v1.json",
    [string]$TrtExec = ""
)

$ErrorActionPreference = "Stop"
$root = if ($ProjectRoot) {
    (Resolve-Path -LiteralPath $ProjectRoot).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}
$registryFile = [IO.Path]::GetFullPath((Join-Path $root $RegistryPath))
$registry = Get-Content -Raw -Encoding utf8 -LiteralPath $registryFile | ConvertFrom-Json
if (-not $TrtExec) {
    $command = Get-Command trtexec -ErrorAction SilentlyContinue
    if (-not $command) { throw "trtexec not found; pass -TrtExec explicitly" }
    $TrtExec = $command.Source
}
if (-not (Test-Path -LiteralPath $TrtExec)) {
    throw "trtexec not found: $TrtExec"
}

$reportRoot = Join-Path $root "runtime\reports\model_delivery"
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null

foreach ($artifact in $registry.artifacts) {
    $onnx = [IO.Path]::GetFullPath((Join-Path $root $artifact.files.onnx_path))
    $engine = [IO.Path]::GetFullPath((Join-Path $root $artifact.files.engine_path))
    foreach ($path in @($onnx, $engine)) {
        if (-not $path.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "model path escapes project root: $path"
        }
    }
    if (-not (Test-Path -LiteralPath $onnx)) {
        throw "ONNX file is missing for $($artifact.artifact_id): $onnx"
    }
    $onnxHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $onnx).Hash.ToLowerInvariant()
    if ($artifact.files.onnx_sha256 -and $artifact.files.onnx_sha256 -ne $onnxHash) {
        throw "ONNX SHA256 does not match registry for $($artifact.artifact_id)"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $engine) | Out-Null
    $modelInput = $artifact.input
    $batch = [int]$artifact.deployment.max_batch
    $shape = "images:{0}x3x{1}x{2}" -f $batch, $modelInput.height, $modelInput.width
    $arguments = @(
        "--onnx=$onnx",
        "--saveEngine=$engine",
        "--fp16",
        "--skipInference",
        "--minShapes=images:1x3x$($modelInput.height)x$($modelInput.width)",
        "--optShapes=$shape",
        "--maxShapes=$shape"
    )
    & $TrtExec @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "trtexec failed for $($artifact.artifact_id)"
    }
    $engineHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $engine).Hash.ToLowerInvariant()
    $evidence = [ordered]@{
        schema_version = "1.0"
        artifact_id = $artifact.artifact_id
        onnx_path = $artifact.files.onnx_path
        onnx_sha256 = $onnxHash
        engine_path = $artifact.files.engine_path
        engine_sha256 = $engineHash
        trtexec = $TrtExec
        generated_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $report = Join-Path $reportRoot "$($artifact.artifact_id).engine-build.json"
    $evidence | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $report
    Write-Output "PASS: $($artifact.artifact_id) engine=$engine sha256=$engineHash"
}
