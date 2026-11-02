$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot '.cache\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $PSScriptRoot '.cache\pip'
$env:HF_HOME=Join-Path $PSScriptRoot '.cache\huggingface'
$env:TORCH_HOME=Join-Path $PSScriptRoot '.cache\torch'
$env:npm_config_cache=Join-Path $PSScriptRoot '.cache\npm'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw '먼저 setup.ps1을 실행하세요.' }
if (!(Test-Path -LiteralPath 'frontend\dist\index.html')) { throw '프론트엔드 빌드가 없습니다. setup.ps1을 실행하세요.' }
Write-Host 'SAM 2.1 Vision Console: http://127.0.0.1:8000' -ForegroundColor Green
Write-Host '종료: Ctrl+C / 카메라는 웹 화면에서 시작합니다.'
& '.\.venv\Scripts\python.exe' -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
