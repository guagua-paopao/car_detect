[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$BuildDir = ".\out\build\backend-Release",
    [string]$VisualStudioRoot = "D:\vs2019"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = if ($Root) { (Resolve-Path $Root).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$BuildPath = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $BuildDir))
$CTestExe = Join-Path $VisualStudioRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $CTestExe --test-dir $BuildPath -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "C++ tests failed" }
python (Join-Path $ProjectRoot "tests\web_admin_contract_test.py")
if ($LASTEXITCODE -ne 0) { throw "Web Admin contract test failed" }
foreach ($vehicleTest in @(
        "tests\vehicle_contract_test.py",
        "tests\dataset_manifest_contract_test.py",
        "tests\model_registry_contract_test.py",
        "tests\model_delivery_contract_test.py",
        "tests\vehicle_cascade_contract_test.py",
        "tests\vehicle_storage_api_contract_test.py",
        "tests\vehicle_release_contract_test.py",
        "tests\vehicle_tensorrt_source_contract_test.py"
    )) {
    python (Join-Path $ProjectRoot $vehicleTest)
    if ($LASTEXITCODE -ne 0) {
        throw "Vehicle contract failed: $vehicleTest"
    }
}
Write-Host "PASS: all compact-project tests succeeded." -ForegroundColor Green
