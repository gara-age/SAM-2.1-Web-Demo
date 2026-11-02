$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot '.cache\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $PSScriptRoot '.cache\pip'
$env:npm_config_cache=Join-Path $PSScriptRoot '.cache\npm'
$env:SAM2_BUILD_CUDA='0'
New-Item -ItemType Directory -Path $env:TEMP,'models','vendor','logs' -Force | Out-Null
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "명령 실패: exit $LASTEXITCODE" } }
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) { python -m venv .venv; Check-Exit }
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip setuptools wheel; Check-Exit
& '.\.venv\Scripts\python.exe' -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121; Check-Exit
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt; Check-Exit
if (!(Test-Path -LiteralPath 'vendor\sam2\.git')) { git clone https://github.com/facebookresearch/sam2.git vendor/sam2; Check-Exit }
git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4; Check-Exit
& '.\.venv\Scripts\python.exe' -m pip install --no-build-isolation -e vendor/sam2; Check-Exit
if (!(Test-Path -LiteralPath 'models\sam2.1_hiera_tiny.pt')) {
    Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt' -OutFile 'models\sam2.1_hiera_tiny.pt.download'
    Move-Item -LiteralPath 'models\sam2.1_hiera_tiny.pt.download' -Destination 'models\sam2.1_hiera_tiny.pt'
}
Push-Location frontend
try { npm.cmd ci; Check-Exit; npm.cmd run build; Check-Exit } finally { Pop-Location }
Write-Host '설치 완료. start.cmd를 실행하세요.' -ForegroundColor Green
