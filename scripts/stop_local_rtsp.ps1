[CmdletBinding()]
param([string]$StatePath = ".\runtime\rtsp-test\state.json")

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedStatePath = if ([IO.Path]::IsPathRooted($StatePath)) {
    [IO.Path]::GetFullPath($StatePath)
} else {
    [IO.Path]::GetFullPath((Join-Path $projectRoot $StatePath))
}
if (-not (Test-Path -LiteralPath $resolvedStatePath -PathType Leaf)) {
    Write-Output "PASS: no local RTSP test state exists"
    return
}

$state = Get-Content -Raw -LiteralPath $resolvedStatePath | ConvertFrom-Json
foreach ($entry in @(
    @{ pid = [int]$state.publisher_pid; expected = [string]$state.publisher_exe },
    @{ pid = [int]$state.media_pid; expected = [string]$state.media_exe }
)) {
    if ($entry.pid -le 0) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($entry.pid)" `
        -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    $actual = [IO.Path]::GetFullPath([string]$process.ExecutablePath)
    $expected = [IO.Path]::GetFullPath($entry.expected)
    if (-not $actual.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to stop reused PID $($entry.pid): executable mismatch"
    }
    Stop-Process -Id $entry.pid -Force -ErrorAction Stop
    Wait-Process -Id $entry.pid -Timeout 5 -ErrorAction SilentlyContinue
}
$state | Add-Member -NotePropertyName stopped_at `
    -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString("o")) -Force
$state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resolvedStatePath -Encoding UTF8
Write-Output "PASS: local RTSP test stream stopped"
