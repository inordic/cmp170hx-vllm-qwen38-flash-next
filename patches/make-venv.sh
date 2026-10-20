#!/bin/bash
# install-runtime.sh without the apply-patch step (the copied tree is already patched)
# usage: make-venv.sh [TREE [TAG]]   default: /opt/vllm-qfn-n4 n4 (production since 2026-10-20, patches 01-18)
#        make-venv.sh /opt/vllm-qfn-exp exp                    (the night-3 tree, patches 01-07)
set -euo pipefail
export PATH=/opt/vllm-qfn/tools/bin:$PATH UV_CACHE_DIR=/opt/vllm-qfn/.uv-cache
E=${1:-/opt/vllm-qfn-n4}; TAG=${2:-n4}; R=/opt/vllm-qfn/recipe
source <(grep -E "^VLLM_REVISION=|^VLLM_WHEEL=" $R/scripts/common.sh)
[ -x $E/.venv/bin/python ] || uv venv --python 3.12 $E/.venv
uv pip install --python $E/.venv/bin/python --torch-backend=cu130 -c $R/requirements/tested-constraints.txt "$VLLM_WHEEL"
uv pip install --python $E/.venv/bin/python --torch-backend=cu130 -c $R/requirements/tested-constraints.txt -r $E/src/requirements/build/cuda.txt "huggingface-hub==1.32.0"
export VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.29.1rc1.dev402+ga5a30471f.ple1.$TAG VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_WHEEL"
cd $E/src && uv pip install --python $E/.venv/bin/python --reinstall --config-settings editable_mode=compat --no-deps --no-build-isolation -e .
uv pip install --python $E/.venv/bin/python "instanttensor==$(/opt/vllm-qfn/.venv/bin/python -c "import importlib.metadata as m; print(m.version(\"instanttensor\"))")"
echo MAKE-VENV-DONE
