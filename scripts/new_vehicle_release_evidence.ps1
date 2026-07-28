[CmdletBinding()]
param(
    [string]$ReleaseId = "",
    [string]$OutputRoot = ".\reports\m5",
    [string]$RegistryPath = ".\models\manifests\model_registry.v1.json",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GitSafeDirectory = $ProjectRoot -replace '\\', '/'
Set-Location $ProjectRoot
if (-not $ReleaseId) {
    $ReleaseId = "vehicle_candidate_" +
        [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
}
if ($ReleaseId -notmatch '^[A-Za-z0-9_-]{1,160}$') {
    throw "ReleaseId must be a safe identifier."
}

$resolvedOutputRoot = if ([IO.Path]::IsPathRooted($OutputRoot)) {
    [IO.Path]::GetFullPath($OutputRoot)
}
else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputRoot))
}
if (-not $resolvedOutputRoot.StartsWith(
        $ProjectRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputRoot must stay inside the project workspace."
}
$evidenceDir = Join-Path $resolvedOutputRoot $ReleaseId
New-Item -ItemType Directory -Force -Path $evidenceDir | Out-Null

$templatePath = Join-Path $ProjectRoot `
    "api\examples\vehicle_release_evidence.pending.v1.json"
$evidence = Get-Content -LiteralPath $templatePath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$evidence.release_id = $ReleaseId
$evidence.generated_at = [DateTimeOffset]::UtcNow.ToString("o")

$commit = (& git -c "safe.directory=$GitSafeDirectory" rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve the source Git commit."
}
$dirty = @(
    & git -c "safe.directory=$GitSafeDirectory" status --porcelain
).Count -gt 0
$evidence.source.commit = $commit
$evidence.source.dirty = $dirty

$registryResolved = (Resolve-Path -LiteralPath $RegistryPath).Path
$registry = Get-Content -LiteralPath $registryResolved -Raw -Encoding UTF8 |
    ConvertFrom-Json
$labelsPath = Join-Path $ProjectRoot "config\vehicle_labels.v1.json"
$labels = Get-Content -LiteralPath $labelsPath -Raw -Encoding UTF8 |
    ConvertFrom-Json
$vehicleConfigPath = Join-Path $ProjectRoot "config\vehicle_analytics.yaml"
$vehicleConfigText = Get-Content -LiteralPath $vehicleConfigPath `
    -Raw -Encoding UTF8
$vehicleConfig = $vehicleConfigText | ConvertFrom-Json

$evidence.versions.dataset = $null
$evidence.versions.dataset_manifest = $null
$evidence.versions.labels = [string]$labels.labels_version
$evidence.versions.model_registry = [string]$registry.registry_version
$evidence.versions.vehicle_config = [string]$vehicleConfig.config_version
$evidence.versions.database_schema = 4

$hashInputs = @(
    "config/server.yaml",
    "config/worker.yaml",
    "config/cameras.yaml",
    "config/vehicle_analytics.yaml",
    "config/vehicle_labels.v1.json",
    "config/vehicle_release_acceptance.v1.json",
    "models/manifests/model_registry.v1.json",
    "db/postgresql/001_initial_schema.sql",
    "db/postgresql/002_algorithm_service_contract.sql",
    "db/postgresql/003_people_flow_camera_unification.sql",
    "db/postgresql/004_vehicle_events.sql"
)
$hashes = @()
foreach ($relative in $hashInputs) {
    $path = Join-Path $ProjectRoot ($relative -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Release input is missing: $relative"
    }
    $hashes += [ordered]@{
        path = $relative
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).
            Hash.ToLowerInvariant()
    }
}
$evidence.artifact_hashes = @($hashes)

$modelAuditPath = Join-Path $evidenceDir "model_delivery.txt"
$modelAudit = & $PythonExe (
    Join-Path $ProjectRoot "tools\audit_model_delivery.py") `
    --registry $registryResolved `
    --project-root $ProjectRoot `
    --allow-planned 2>&1 | Out-String
$modelAudit = $modelAudit -replace [regex]::Escape($ProjectRoot), "."
$modelAudit | Set-Content -LiteralPath $modelAuditPath -Encoding UTF8
$modelReady = $modelAudit -match (
    '(?m)^PASS: all model payloads, hashes, provenance, and evidence are ready$')
$relativeEvidenceDir = $evidenceDir.Substring($ProjectRoot.Length).
    TrimStart([char[]]@('\', '/')) -replace '\\', '/'
$evidence.artifact_hashes = @($evidence.artifact_hashes) + [ordered]@{
    path = "$relativeEvidenceDir/model_delivery.txt"
    sha256 = (Get-FileHash -LiteralPath $modelAuditPath `
        -Algorithm SHA256).Hash.ToLowerInvariant()
}
$evidence.model_delivery.status = if ($modelReady) {
    "passed"
}
else {
    "pending"
}
$evidence.model_delivery.evidence_path =
    "$relativeEvidenceDir/model_delivery.txt"
if (-not $modelReady) {
    $evidence.notes = @(
        "Model delivery is incomplete; do not promote this record to passed.",
        "Performance, soak, fault, and rollback gates require the target host."
    )
}

$evidencePath = Join-Path $evidenceDir "release_evidence.json"
$evidence | ConvertTo-Json -Depth 16 |
    Set-Content -LiteralPath $evidencePath -Encoding UTF8

& $PythonExe (
    Join-Path $ProjectRoot "tools\validate_vehicle_release_evidence.py") `
    $evidencePath --allow-pending
if ($LASTEXITCODE -ne 0) {
    throw "Generated release evidence does not satisfy the planning contract."
}
Write-Host (
    "PASS: pending M5 evidence created without claiming release readiness: " +
    $evidencePath
) -ForegroundColor Green
