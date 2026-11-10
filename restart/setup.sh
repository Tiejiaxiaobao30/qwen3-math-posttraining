#!/usr/bin/env bash
set -euo pipefail
# Run from repository root, inside a NEW Python 3.11 environment.
python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11"'
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r restart/requirements.txt
python -m pip check
python -c 'import torch; from trl import GRPOConfig, GRPOTrainer; from swanlab.integration.transformers import SwanLabCallback; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
python -m unittest discover -s restart/tests -v
python -m pip freeze > restart-environment.local.txt
echo 'Environment ready. Model/SFT/data and inspect-only are still required.'
