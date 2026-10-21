# Run the full pipeline on one video, write <name>_result.mp4 + <name>_result.json, and open the video.
param([Parameter(Mandatory = $true)][string]$Video, [string]$OutDir = "")
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$Video = (Resolve-Path $Video).Path
if (-not $OutDir) { $OutDir = Join-Path $root "results" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$name = [IO.Path]::GetFileNameWithoutExtension($Video)
$outVideo = Join-Path $OutDir "${name}_result.mp4"
$outJson = Join-Path $OutDir "${name}_result.json"

# bundled weights -> no internet needed
$env:TORCH_HOME = Join-Path $root "weights\torch"
$yolo = (Join-Path $root "weights\yolo11n.pt") -replace '\\', '/'
$device = & "$root\.venv\Scripts\python.exe" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"

Write-Host "Analysing $Video on $device ..." -ForegroundColor Cyan
$sw = [Diagnostics.Stopwatch]::StartNew()
& "$root\.venv\Scripts\abd.exe" --set "tracking.model=$yolo" --set "features.device=$device" --set "train.device=$device" `
    --set features.batch_size=8 infer --ckpt "$root\models\best.pt" --video "$Video" --render "$outVideo" |
    Tee-Object -FilePath $outJson
if ($LASTEXITCODE -ne 0) { throw "abd infer failed" }
# the last line printed is "rendered -> ..."; keep only the JSON in the file
(Get-Content $outJson | Where-Object { $_ -notmatch '^rendered -> ' }) | Set-Content -Encoding utf8 $outJson
Write-Host ("Done in {0:N0}s -> {1}" -f $sw.Elapsed.TotalSeconds, $outVideo) -ForegroundColor Green
Start-Process $outVideo
