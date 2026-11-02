$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:TEMP=Join-Path $PSScriptRoot '.cache\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $PSScriptRoot '.cache\pip'
New-Item -ItemType Directory -Path $env:TEMP,'.cache\downloads','vendor\colmap','data\scans' -Force | Out-Null
& '.\.venv\Scripts\python.exe' -m pip install numpy==1.26.4 pycolmap==4.2.1 plyfile==1.0.3
if ($LASTEXITCODE -ne 0) { throw '3D Python 의존성 설치 실패' }
if (!(Test-Path -LiteralPath 'vendor\colmap\bin\colmap.exe')) {
    $archive=Join-Path $PSScriptRoot '.cache\downloads\colmap-4.2.1-cuda.zip'
    Invoke-WebRequest -Uri 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip' -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath 'vendor\colmap' -Force
}
& '.\vendor\colmap\bin\colmap.exe' -h
if ($LASTEXITCODE -ne 0) { throw 'COLMAP 실행 실패' }
Write-Host '3D 스캔 도구 설치 완료. 서버를 다시 시작하세요.' -ForegroundColor Green
