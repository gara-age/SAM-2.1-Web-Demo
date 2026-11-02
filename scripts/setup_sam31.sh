#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"
export TMPDIR="$project_dir/.cache/tmp-linux"
export PIP_CACHE_DIR="${SAM31_PIP_CACHE:-$project_dir/.cache/pip-linux}"
sam31_env="${SAM31_ENV_PATH:-$project_dir/.venv-sam31-wsl}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$project_dir/.cache/downloads" "$project_dir/.cache/triton" "$project_dir/.cache/cuda-linux"
python3 -m venv --without-pip "$sam31_env"
if ! "$sam31_env/bin/python" -m pip --version >/dev/null 2>&1; then
  curl --fail --location https://bootstrap.pypa.io/get-pip.py --output .cache/downloads/get-pip.py
  "$sam31_env/bin/python" .cache/downloads/get-pip.py
fi
"$sam31_env/bin/python" -m pip install 'setuptools<81' wheel
"$sam31_env/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$sam31_env/bin/python" -m pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 scipy einops pycocotools psutil
if [ ! -d vendor/sam3/.git ]; then git clone https://github.com/facebookresearch/sam3.git vendor/sam3; fi
git -C vendor/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
"$sam31_env/bin/python" -m pip install --no-build-isolation -e vendor/sam3
"$sam31_env/bin/python" -m pip check
echo 'SAM 3.1 WSL environment installed. Checkpoints must be downloaded separately after access approval.'
