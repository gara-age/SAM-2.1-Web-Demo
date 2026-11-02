$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot.Replace('\','/')
$linuxRoot='/mnt/'+$projectRoot.Substring(0,1).ToLower()+$projectRoot.Substring(2)
$distribution='Ubuntu-24.04'
$entry=Get-ChildItem -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | ForEach-Object { Get-ItemProperty -LiteralPath $_.PSPath } | Where-Object DistributionName -eq $distribution
if (-not $entry) { throw 'Ubuntu-24.04 WSL 배포판이 없습니다. 먼저 WSL 배포판을 설치하세요.' }
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot '.cache') -Force | Out-Null
if ($entry.BasePath.StartsWith('D:\',[StringComparison]::OrdinalIgnoreCase)) {
    $pythonPath='/opt/sam2trackingweb/sam31/bin/python'
    wsl.exe -d $distribution -u root --exec env 'SAM31_ENV_PATH=/opt/sam2trackingweb/sam31' 'SAM31_PIP_CACHE=/opt/sam2trackingweb/pip-cache' bash ($linuxRoot+'/scripts/setup_sam31.sh')
} else {
    $pythonPath=$linuxRoot+'/.venv-sam31-wsl/bin/python'
    wsl.exe -d $distribution --exec bash ($linuxRoot+'/scripts/setup_sam31.sh')
}
if ($LASTEXITCODE -ne 0) { throw 'SAM 3.1 WSL 설치 실패' }
@{ distribution=$distribution; python=$pythonPath } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot '.cache\sam31-runtime.json') -Encoding utf8
