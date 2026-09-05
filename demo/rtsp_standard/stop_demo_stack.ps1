[CmdletBinding()]
param([string]$ProjectRoot = "")

$ErrorActionPreference = "Continue"
$root = if ($ProjectRoot) {
    (Resolve-Path -LiteralPath $ProjectRoot).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
}
$statePath = Join-Path $PSScriptRoot "output\demo_stack_state.json"
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    if ($state.publisher_pid) {
        Stop-Process -Id ([int]$state.publisher_pid) -Force `
            -ErrorAction SilentlyContinue
    }
}
& (Join-Path $root "scripts\stop_demo.ps1") -Root $root
foreach ($name in @(
    "vcas-demo-mediamtx",
    "vcas-demo-redis",
    "vcas-demo-postgres"
)) {
    docker.exe rm -f $name 2>$null | Out-Null
}
Write-Output "PASS: VCAS RTSP demo stack stopped"
