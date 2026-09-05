param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
    [string]$Destination = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path (Split-Path $ProjectRoot -Parent) "vcas-training-code.zip"
}

$resolvedProject = (Resolve-Path -LiteralPath $ProjectRoot).Path
$items = @(
    @{ Source = (Join-Path $resolvedProject "training"); Relative = "training" },
    @{ Source = (Join-Path $resolvedProject "config\vehicle_labels.v1.json"); Relative = "config\vehicle_labels.v1.json" },
    @{ Source = (Join-Path $resolvedProject "models\manifests\model_registry.v1.json"); Relative = "models\manifests\model_registry.v1.json" },
    @{ Source = (Join-Path $resolvedProject "docs\data\ANNOTATION_GUIDE.md"); Relative = "docs\data\ANNOTATION_GUIDE.md" },
    @{ Source = (Join-Path $resolvedProject "tools\validate_dataset_manifest.py"); Relative = "tools\validate_dataset_manifest.py" },
    @{ Source = (Join-Path $resolvedProject "api\schemas\dataset_manifest.v1.schema.json"); Relative = "api\schemas\dataset_manifest.v1.schema.json" }
)

foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath $item.Source)) {
        throw "Required training package input is missing: $($item.Source)"
    }
}

if (Test-Path -LiteralPath $Destination) {
    if (-not $Force) {
        throw "Destination already exists: $Destination. Use -Force to replace it."
    }
    Remove-Item -LiteralPath $Destination
}

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("vcas-training-package-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    foreach ($item in $items) {
        $source = Get-Item -LiteralPath $item.Source
        if ($source.PSIsContainer) {
            Get-ChildItem -LiteralPath $source.FullName -File -Recurse |
                Where-Object {
                    $_.Name -notlike "*.pyc" -and
                    $_.FullName -notmatch "[\\/]__pycache__[\\/]"
                } |
                ForEach-Object {
                    $relativeWithinSource = $_.FullName.Substring($source.FullName.Length).TrimStart("\", "/")
                    $target = Join-Path $staging (Join-Path $item.Relative $relativeWithinSource)
                    New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
                    Copy-Item -LiteralPath $_.FullName -Destination $target
                }
        } else {
            $target = Join-Path $staging $item.Relative
            New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
            Copy-Item -LiteralPath $source.FullName -Destination $target
        }
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $Destination -CompressionLevel Optimal
    $archiveHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    $hashPath = "$Destination.sha256"
    [System.IO.File]::WriteAllText(
        $hashPath,
        "$archiveHash  $([System.IO.Path]::GetFileName($Destination))`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host "PASS: training package created at $Destination"
    Write-Host "PASS: SHA256 written to $hashPath"
}
finally {
    $resolvedStaging = [System.IO.Path]::GetFullPath($staging)
    $resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
    if ($resolvedStaging.StartsWith($resolvedTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
    }
}
