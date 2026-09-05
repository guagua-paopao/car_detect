[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\codex\project",
    [string]$BuildDir = ".\out\build\backend-model-validation",
    [string]$VisualStudioRoot = "D:\vs2019"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$build = [IO.Path]::GetFullPath((Join-Path $root $BuildDir))
$vcvars = Join-Path $VisualStudioRoot "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $VisualStudioRoot `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
foreach ($path in @($vcvars, $cmake, (Join-Path $build "CMakeCache.txt"))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required build input is missing: $path"
    }
}

$lines = & cmd.exe /d /s /c "`"call `"$vcvars`" >nul && set`""
if ($LASTEXITCODE -ne 0) {
    throw "Failed to import the Visual Studio x64 build environment"
}
$seen = [Collections.Generic.HashSet[string]]::new(
    [StringComparer]::OrdinalIgnoreCase)
foreach ($line in $lines) {
    $separator = $line.IndexOf("=")
    if ($separator -le 0) { continue }
    $name = $line.Substring(0, $separator)
    if ($seen.Add($name)) {
        [Environment]::SetEnvironmentVariable(
            $name, $line.Substring($separator + 1), "Process")
    }
}

& $cmake --build $build --config Release `
    --target four_stage_server four_stage_worker
if ($LASTEXITCODE -ne 0) {
    throw "Demo backend build failed"
}
Write-Output "PASS: demo backend=$build"
