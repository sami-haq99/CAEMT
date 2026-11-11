#!/usr/bin/env bash
# setup_env.sh - one-shot environment for mt_metrics_runner.py on Ubuntu (+ NVIDIA GPU)
#
# Usage:  bash setup_env.sh [venv_dir]        (default: .venv-mtmetrics)
#
# Uses `uv` to provision Python 3.10 without sudo or touching the system Python.
set -euo pipefail

ENV_DIR="${1:-.venv-mtmetrics}"
PYVER="3.10"

# 1) uv (fast installer that can also fetch a specific Python version)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) venv
uv venv --python "$PYVER" "$ENV_DIR"

uv pip install --python "$ENV_DIR/bin/python" "setuptools==80.10.2"

PY="$ENV_DIR/bin/python"

# 3) dependencies (single env for all metrics)
#    - numpy<2 and transformers<5 are precautionary pins: unbabel-comet, the MetricX repo and the
#      BLEURT port were written against the 4.x / numpy 1.x APIs. Relax them if everything works.
uv pip install --python "$PY" \
  "numpy<2" pandas tqdm \
  torch \
  "transformers>=4.40,<5" sentencepiece protobuf datasets accelerate \
  "sacrebleu>=2.4" \
  bert-score \
  unbabel-comet \
  openai

# BLEURT-20 (PyTorch port of the official checkpoint; avoids a TensorFlow install)
uv pip install --python "$PY" "git+https://github.com/lucadiliello/bleurt-pytorch.git"

# 4) MetricX-24: official repo, called by the runner via `python -m metricx24.predict`
if [ ! -d metricx ]; then
  git clone --depth 1 https://github.com/google-research/metricx.git metricx
fi

# 5) sanity check
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
import sacrebleu, bert_score, comet, bleurt_pytorch, transformers  # noqa: F401
print("imports OK, transformers", transformers.__version__)
EOF

cat <<EOF

Done. Next:
  source $ENV_DIR/bin/activate
  export OPENAI_API_KEY=...        # only needed for GEMBA-ESA
  python mt_metrics_runner.py --input example_testset.csv --output scores.json \\
         --src-lang en --tgt-lang de --metrics bleu,chrf,ter --limit 4   # quick smoke test
  python mt_metrics_runner.py --input your_testset.csv --output scores.json \\
         --src-lang en --tgt-lang de                                      # all metrics
EOF
