$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path 'models','vendor' -Force | Out-Null
if (!(Test-Path 'vendor\depth-anything-v2\.git')) {
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git vendor/depth-anything-v2
    if ($LASTEXITCODE -ne 0) { throw 'Depth model source download failed' }
}
git -C vendor/depth-anything-v2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
if ($LASTEXITCODE -ne 0) { throw 'Depth source version checkout failed' }
if (!(Test-Path 'models\depth_anything_v2_vits.pth')) {
    Invoke-WebRequest 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile 'models\depth_anything_v2_vits.pth.download'
    Move-Item -LiteralPath 'models\depth_anything_v2_vits.pth.download' -Destination 'models\depth_anything_v2_vits.pth'
}
Write-Host 'Depth Anything V2 Small ready. Source/model license: Apache-2.0 (vendor/depth-anything-v2/LICENSE).'
