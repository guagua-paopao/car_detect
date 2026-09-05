[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateRoot,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedTensorRtReportSha256,
    [string]$ProjectRoot = "F:\codex\project",
    [string]$BuildDir = ".\out\build\backend-model-validation",
    [string]$VisualStudioRoot = "D:\vs2019"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$candidate = (Resolve-Path -LiteralPath $CandidateRoot).Path
$build = [IO.Path]::GetFullPath((Join-Path $root $BuildDir))
$cache = Join-Path $build "CMakeCache.txt"
$vcvars = Join-Path $VisualStudioRoot "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $VisualStudioRoot `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
foreach ($path in @($cache, $vcvars, $cmake)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "required C++ build input is missing: $path"
    }
}

$trtReportPath = Join-Path $candidate "windows-tensorrt-v1\windows-tensorrt-report.json"
if (-not (Test-Path -LiteralPath $trtReportPath -PathType Leaf)) {
    throw "TensorRT gate report is missing: $trtReportPath"
}
$trtReportSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $trtReportPath).Hash.ToLowerInvariant()
if ($trtReportSha -ne $ExpectedTensorRtReportSha256.ToLowerInvariant()) {
    throw "TensorRT gate report SHA256 mismatch"
}
$trtReport = Get-Content -Raw -Encoding utf8 -LiteralPath $trtReportPath | ConvertFrom-Json
if ($trtReport.status -ne "pass_waiting_cpp_and_real_engine_contract") {
    throw "TensorRT gates did not authorize C++ validation"
}
foreach ($gate in $trtReport.gates.PSObject.Properties) {
    if ($gate.Value -ne $true) {
        throw "TensorRT gate is not green: $($gate.Name)"
    }
}
if ($trtReport.policy.frozen_video_used -ne $false -or
    $trtReport.policy.production_model_modified -ne $false -or
    $trtReport.policy.deployment_performed -ne $false) {
    throw "TensorRT policy violation"
}

$productionFiles = @(
    (Join-Path $root "config\vehicle_analytics.yaml"),
    (Join-Path $root "models\manifests\model_registry.v1.json")
)
$productionHashes = [ordered]@{}
foreach ($path in $productionFiles) {
    $productionHashes[$path] = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
}

function Assert-CandidateFile([string]$Path, [string]$Expected, [string]$Label) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $boundary = $candidate.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label escapes candidate root: $resolved"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA256 mismatch"
    }
    return $resolved
}

$bodyOnnx = Assert-CandidateFile $trtReport.models.body.onnx $trtReport.models.body.onnx_sha256 "body ONNX"
$bodyEngine = Assert-CandidateFile $trtReport.models.body.engine $trtReport.models.body.engine_sha256 "body engine"
$colorOnnx = Assert-CandidateFile $trtReport.models.color.onnx $trtReport.models.color.onnx_sha256 "color ONNX"
$colorEngine = Assert-CandidateFile $trtReport.models.color.engine $trtReport.models.color.engine_sha256 "color engine"

$lines = & cmd.exe /d /s /c "`"call `"$vcvars`" >nul && set`""
if ($LASTEXITCODE -ne 0) {
    throw "failed to import Visual Studio x64 environment"
}
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($line in $lines) {
    $separator = $line.IndexOf("=")
    if ($separator -le 0) { continue }
    $name = $line.Substring(0, $separator)
    if ($seen.Add($name)) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $line.Substring($separator + 1),
            "Process")
    }
}

$cppRoot = Join-Path $candidate "windows-tensorrt-v1\cpp-stage71-v1"
if (Test-Path -LiteralPath $cppRoot) {
    throw "refusing to overwrite Stage71 C++ evidence: $cppRoot"
}
New-Item -ItemType Directory -Path $cppRoot | Out-Null
$configureLog = Join-Path $cppRoot "cmake-configure.log"
& $cmake -S $root -B $build -DVCAS_BUILD_STAGE71_CANDIDATE_TEST=ON *> $configureLog
if ($LASTEXITCODE -ne 0) {
    throw "Stage71 C++ configure failed; see $configureLog"
}
$buildLog = Join-Path $cppRoot "cmake-build.log"
& $cmake --build $build --config Release --target vehicle_stage71_decoupled_real_engine_test *> $buildLog
if ($LASTEXITCODE -ne 0) {
    throw "Stage71 C++ target build failed; see $buildLog"
}
$executable = Join-Path $build "vehicle_stage71_decoupled_real_engine_test.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Stage71 C++ test executable is missing: $executable"
}

function Relative-CandidatePath([string]$Path) {
    $relative = [IO.Path]::GetRelativePath($candidate, $Path)
    if ($relative -eq ".." -or $relative.StartsWith(".." + [IO.Path]::DirectorySeparatorChar)) {
        throw "candidate artifact cannot be expressed below candidate root: $Path"
    }
    return $relative
}

$arguments = @(
    $candidate,
    (Relative-CandidatePath $bodyOnnx),
    $trtReport.models.body.onnx_sha256,
    (Relative-CandidatePath $bodyEngine),
    $trtReport.models.body.engine_sha256,
    [string]$trtReport.models.body.input_size,
    (Relative-CandidatePath $colorOnnx),
    $trtReport.models.color.onnx_sha256,
    (Relative-CandidatePath $colorEngine),
    $trtReport.models.color.engine_sha256,
    [string]$trtReport.models.color.input_size
)
$smokeLog = Join-Path $cppRoot "real-engine-smoke.log"
& $executable @arguments *> $smokeLog
if ($LASTEXITCODE -ne 0) {
    throw "Stage71 C++ real-engine smoke failed; see $smokeLog"
}
$smokeText = Get-Content -Raw -LiteralPath $smokeLog
if ($smokeText -notmatch "PASS: Stage71 decoupled TensorRT engines") {
    throw "Stage71 C++ real-engine smoke did not emit the pass contract"
}

foreach ($entry in $productionHashes.GetEnumerator()) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) {
        throw "production file changed during isolated C++ gate: $($entry.Key)"
    }
}
$currentTrtReportSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $trtReportPath).Hash.ToLowerInvariant()
if ($currentTrtReportSha -ne $trtReportSha) {
    throw "TensorRT report changed during C++ validation"
}

$report = [ordered]@{
    schema_version = "stage71-cpp-real-engine-gate-v1"
    status = "pass"
    candidate_root = $candidate
    tensorrt_report = $trtReportPath
    tensorrt_report_sha256 = $trtReportSha
    executable = $executable
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $executable).Hash.ToLowerInvariant()
    source = (Join-Path $root "tests\vehicle_stage71_decoupled_real_engine_test.cpp")
    source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $root "tests\vehicle_stage71_decoupled_real_engine_test.cpp")).Hash.ToLowerInvariant()
    body_engine_sha256 = $trtReport.models.body.engine_sha256
    color_engine_sha256 = $trtReport.models.color.engine_sha256
    smoke_log = $smokeLog
    smoke_log_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $smokeLog).Hash.ToLowerInvariant()
    gates = [ordered]@{
        candidate_source_contract = $true
        body_real_engine_deserialize = $true
        color_real_engine_deserialize = $true
        two_crop_batch_inference = $true
        decoupled_head_routing = $true
        production_files_unchanged = $true
    }
    policy = [ordered]@{
        test_accessed_by_this_stage = $false
        frozen_video_used = $false
        production_registry_modified = $false
        production_config_modified = $false
        deployment_performed = $false
        deployment_paused_by_user = $true
    }
}
$reportPath = Join-Path $cppRoot "cpp-real-engine-report.json"
$report | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 -LiteralPath $reportPath
Write-Output "PASS: Stage71 C++ contract and real-engine smoke; report=$reportPath"
