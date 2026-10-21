# One-time setup on Windows + NVIDIA GPU. Run from the bundle folder: setup.bat
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "== GPU ==" -ForegroundColor Cyan
try { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv } catch { Write-Warning "nvidia-smi not found: NVIDIA driver missing? The demo will run on CPU (slow)." }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "== Installing uv ==" -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "== Python 3.11 venv ==" -ForegroundColor Cyan
uv venv --python 3.11 .venv
$py = "$root\.venv\Scripts\python.exe"

Write-Host "== PyTorch (CUDA 12.4 build) ==" -ForegroundColor Cyan
uv pip install --python $py torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

Write-Host "== abd + dependencies ==" -ForegroundColor Cyan
uv pip install --python $py -e "$root\code"

Write-Host "== Check ==" -ForegroundColor Cyan
& $py -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'CPU ONLY')"
Write-Host "Setup done. Demo: drag a video onto demo.bat, or run  demo.bat path\to\video.mp4" -ForegroundColor Green
