[CmdletBinding()]
param(
    [string]$Root = "",
    [string]$Ffmpeg = "",
    [string]$Ffprobe = ""
)

$ErrorActionPreference = "Stop"
$demoRoot = if ($Root) {
    (Resolve-Path -LiteralPath $Root).Path
}
else {
    (Resolve-Path -LiteralPath $PSScriptRoot).Path
}
$sourceRoot = Join-Path $demoRoot "sources"
$outputRoot = Join-Path $demoRoot "output"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if (-not $Ffmpeg) {
    $command = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
    if (-not $command) { throw "ffmpeg.exe is required" }
    $Ffmpeg = $command.Source
}
if (-not $Ffprobe) {
    $command = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
    if (-not $command) { throw "ffprobe.exe is required" }
    $Ffprobe = $command.Source
}

$clips = @(
    @{ File="01_toll.mp4"; Label="0-12s | TOLL PLAZA | near vehicles / lane occlusion" },
    @{ File="02_intersection.mp4"; Label="12-24s | INTERSECTION | multi-target / pedestrians" },
    @{ File="03_highway.mp4"; Label="24-36s | HIGHWAY | car / bus / truck" },
    @{ File="02_intersection_candidate.mp4"; Label="36-48s | COMPLEX NIGHT TRAFFIC | dense vehicles / lights / occlusion" },
    @{ File="05_night.mp4"; Label="48-60s | NIGHT ROAD | lights / blur" }
)
foreach ($clip in $clips) {
    $path = Join-Path $sourceRoot $clip.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
        (Get-Item -LiteralPath $path).Length -lt 1024) {
        throw "Source video is missing or incomplete: $path"
    }
}

$arguments = @("-y", "-hide_banner")
foreach ($clip in $clips) {
    $arguments += @(
        "-stream_loop", "-1",
        "-i", (Join-Path $sourceRoot $clip.File)
    )
}

$font = "C\:/Windows/Fonts/arial.ttf"
$filters = @()
for ($index = 0; $index -lt $clips.Count; ++$index) {
    $label = $clips[$index].Label.Replace("'", "\'")
    $filters += (
        "[$index`:v]trim=duration=12,setpts=PTS-STARTPTS," +
        "scale=1280:720:force_original_aspect_ratio=increase," +
        "crop=1280:720,setsar=1,fps=25," +
        "drawbox=x=0:y=0:w=iw:h=58:color=black@0.68:t=fill," +
        "drawtext=fontfile='$font':text='$label':" +
        "fontcolor=white:fontsize=28:x=22:y=14[v$index]"
    )
}
$concatInputs = (0..($clips.Count - 1) | ForEach-Object { "[v$_]" }) -join ""
$filters += "${concatInputs}concat=n=5:v=1:a=0[outv]"
$filterComplex = $filters -join ";"

$output = Join-Path $outputRoot "vcas_rtsp_demo_60s.mp4"
$arguments += @(
    "-filter_complex", $filterComplex,
    "-map", "[outv]",
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "21",
    "-pix_fmt", "yuv420p",
    "-r", "25",
    "-g", "25",
    "-movflags", "+faststart",
    "-t", "60",
    $output
)
& $Ffmpeg @arguments
if ($LASTEXITCODE -ne 0) { throw "FFmpeg demo composition failed" }

$probe = & $Ffprobe -v error -select_streams v:0 `
    -show_entries "format=duration:stream=codec_name,width,height,r_frame_rate" `
    -of json $output | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "FFprobe validation failed" }
$duration = [double]$probe.format.duration
if ([Math]::Abs($duration - 60.0) -gt 0.2) {
    throw "Demo duration must be 60 seconds, got $duration"
}
if ($probe.streams[0].codec_name -ne "h264" -or
    [int]$probe.streams[0].width -ne 1280 -or
    [int]$probe.streams[0].height -ne 720) {
    throw "Demo output must be H264 1280x720"
}
Write-Output "PASS: demo video=$output duration=$duration"
