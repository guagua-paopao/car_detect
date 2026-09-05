[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$EvidencePath,
    [string]$BuildDir = ".\out\build\backend-Release",
    [string]$OutputDir = ".\out\releases",
    [string]$RegistryPath = ".\models\manifests\model_registry.v1.json",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GitSafeDirectory = $ProjectRoot -replace '\\', '/'
Set-Location $ProjectRoot
$evidenceResolved = (Resolve-Path -LiteralPath $EvidencePath).Path
$registryResolved = (Resolve-Path -LiteralPath $RegistryPath).Path
$buildResolved = (Resolve-Path -LiteralPath $BuildDir).Path
$outputResolved = if ([IO.Path]::IsPathRooted($OutputDir)) {
    [IO.Path]::GetFullPath($OutputDir)
}
else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputDir))
}
if (-not $outputResolved.StartsWith(
        $ProjectRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must stay inside the project workspace."
}
New-Item -ItemType Directory -Force -Path $outputResolved | Out-Null

& $PythonExe (
    Join-Path $ProjectRoot "tools\validate_vehicle_release_evidence.py") `
    $evidenceResolved
if ($LASTEXITCODE -ne 0) {
    throw "Release evidence gate failed."
}
& $PythonExe (
    Join-Path $ProjectRoot "tools\audit_model_delivery.py") `
    --registry $registryResolved `
    --project-root $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Model delivery gate failed."
}

$dirty = @(
    & git -c "safe.directory=$GitSafeDirectory" status --porcelain
)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -gt 0) {
    throw "Release packaging requires a clean Git worktree."
}
$headCommit = (
    & git -c "safe.directory=$GitSafeDirectory" rev-parse HEAD
).Trim()
$evidence = Get-Content -LiteralPath $evidenceResolved -Raw -Encoding UTF8 |
    ConvertFrom-Json
if ($headCommit -ne [string]$evidence.source.commit) {
    throw "Evidence source commit does not match HEAD."
}

$serverExe = Join-Path $buildResolved "four_stage_server.exe"
$workerExe = Join-Path $buildResolved "four_stage_worker.exe"
foreach ($binary in @($serverExe, $workerExe)) {
    if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
        throw "Required release binary is missing: $binary"
    }
}

$releaseId = [string]$evidence.release_id
$stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$stage = Join-Path $outputResolved ".stage_${releaseId}_$stamp"
$stageResolved = [IO.Path]::GetFullPath($stage)
if (-not $stageResolved.StartsWith(
        $outputResolved + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase) -or
    -not (Split-Path $stageResolved -Leaf).StartsWith(".stage_")) {
    throw "Unsafe release staging path."
}
New-Item -ItemType Directory -Force -Path $stageResolved | Out-Null

function Add-ReleaseFile(
    [string]$Source,
    [string]$RelativePath
) {
    if ($RelativePath -notmatch (
            '^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$')) {
        throw "Unsafe release relative path: $RelativePath"
    }
    $sourceResolved = (Resolve-Path -LiteralPath $Source).Path
    $destination = [IO.Path]::GetFullPath((
        Join-Path $stageResolved ($RelativePath -replace '/', '\')))
    if (-not $destination.StartsWith(
            $stageResolved + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Release destination escaped staging: $RelativePath"
    }
    New-Item -ItemType Directory -Force -Path (
        Split-Path $destination -Parent) | Out-Null
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $sourceHash = (Get-FileHash -LiteralPath $sourceResolved `
            -Algorithm SHA256).Hash
        $destinationHash = (Get-FileHash -LiteralPath $destination `
            -Algorithm SHA256).Hash
        if ($sourceHash -ne $destinationHash) {
            throw "Conflicting release file: $RelativePath"
        }
        return
    }
    Copy-Item -LiteralPath $sourceResolved -Destination $destination
}

try {
    Add-ReleaseFile $serverExe "bin/four_stage_server.exe"
    Add-ReleaseFile $workerExe "bin/four_stage_worker.exe"
    foreach ($library in Get-ChildItem -LiteralPath $buildResolved `
            -Filter "*.dll" -File) {
        Add-ReleaseFile $library.FullName ("bin/" + $library.Name)
    }
    foreach ($artifact in @($evidence.artifact_hashes)) {
        $relative = [string]$artifact.path
        $source = [IO.Path]::GetFullPath((
            Join-Path $ProjectRoot ($relative -replace '/', '\')))
        if (-not $source.StartsWith(
                $ProjectRoot + [IO.Path]::DirectorySeparatorChar,
                [StringComparison]::OrdinalIgnoreCase) -or
            -not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Evidence artifact is missing or unsafe: $relative"
        }
        $actual = (Get-FileHash -LiteralPath $source `
            -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$artifact.sha256) {
            throw "Evidence artifact SHA256 mismatch: $relative"
        }
        Add-ReleaseFile $source $relative
    }

    foreach ($relative in @(
            "config/server.yaml",
            "config/worker.yaml",
            "config/cameras.yaml",
            "config/vehicle_analytics.yaml",
            "config/vehicle_labels.v1.json",
            "config/vehicle_release_acceptance.v1.json",
            "models/manifests/model_registry.v1.json",
            "db/postgresql/001_initial_schema.sql",
            "db/postgresql/002_algorithm_service_contract.sql",
            "db/postgresql/003_vehicle_analysis_state.sql",
            "db/postgresql/004_vehicle_events.sql",
            "scripts/start_demo.ps1",
            "scripts/stop_demo.ps1",
            "scripts/backup_runtime.ps1",
            "scripts/restore_runtime.ps1",
            "scripts/new_vehicle_release_evidence.ps1",
            "scripts/package_vehicle_release.ps1",
            "tools/audit_model_delivery.py",
            "tools/compare_model_metrics.py",
            "tools/validate_vehicle_release_evidence.py",
            "docs/operations/VEHICLE_RELEASE_ACCEPTANCE.md",
            "docs/models/MODEL_DELIVERY_STATUS.md",
            "docs/api/VEHICLE_API.md",
            "docs/storage/VEHICLE_EVENT_STORAGE.md"
        )) {
        Add-ReleaseFile (
            Join-Path $ProjectRoot ($relative -replace '/', '\')) $relative
    }
    Add-ReleaseFile $evidenceResolved "evidence/release_evidence.json"

    $registry = Get-Content -LiteralPath $registryResolved `
        -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($artifact in @($registry.artifacts)) {
        foreach ($field in @("onnx_path", "engine_path")) {
            $relative = [string]$artifact.files.$field
            if (-not $relative) {
                throw "$($artifact.artifact_id) is missing files.$field"
            }
            Add-ReleaseFile (
                Join-Path $ProjectRoot ($relative -replace '/', '\')) $relative
        }
        foreach ($field in @("model_card_path", "metrics_path")) {
            $relative = [string]$artifact.provenance.$field
            if (-not $relative) {
                throw "$($artifact.artifact_id) is missing provenance.$field"
            }
            Add-ReleaseFile (
                Join-Path $ProjectRoot ($relative -replace '/', '\')) $relative
        }
    }

    $files = @(Get-ChildItem -LiteralPath $stageResolved -Recurse -File |
        Sort-Object FullName | ForEach-Object {
            [ordered]@{
                path = $_.FullName.Substring($stageResolved.Length).
                    TrimStart([char[]]@('\', '/')) -replace '\\', '/'
                size_bytes = $_.Length
                sha256 = (Get-FileHash -LiteralPath $_.FullName `
                    -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        })
    $manifest = [ordered]@{
        schema_version = "1.0"
        release_id = $releaseId
        created_at = [DateTimeOffset]::UtcNow.ToString("o")
        source_commit = $headCommit
        target_os = "windows"
        engine_portability = "target_host_only"
        files = $files
    }
    $manifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (
            Join-Path $stageResolved "release_manifest.json") -Encoding UTF8

    $archive = Join-Path $outputResolved (
        "${releaseId}_${stamp}_windows_x64.zip")
    Compress-Archive -Path (Join-Path $stageResolved "*") `
        -DestinationPath $archive -CompressionLevel Optimal
    Write-Host (
        "PASS: traceable vehicle release package created: $archive"
    ) -ForegroundColor Green
}
finally {
    if ((Test-Path -LiteralPath $stageResolved) -and
        (Split-Path $stageResolved -Leaf).StartsWith(".stage_") -and
        $stageResolved.StartsWith(
            $outputResolved + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $stageResolved -Recurse -Force
    }
}
