#!/usr/bin/env bash
# setup_env_qe.sh - environment for mt_qe_metrics_runner.py (reference-free / QE metrics) on Ubuntu + NVIDIA GPU
#
# Usage:  bash setup_env_qe.sh [main_venv_dir]        (default: .venv-mtqe)
#
# Creates up to three isolated Python 3.10 environments, because these projects pin different dependencies:
#   .venv-mtqe          main env: MetricX-23-QE, CometKiwi, xCOMET-XL, TransQuest, CLIPScore (+ the runner)
#   .venv-sentinel      Sentinel-src-mqm (its package is a COMET fork; the runner calls it as a subprocess)
#   .venv-metametrics   official MetaMetrics repo (optional; set SKIP_METAMETRICS=1 to skip)
# Optional installs fail softly: you still get a working main env.
set -uo pipefail

ENV_DIR="${1:-.venv-mtqe}"
PYVER="3.10"
WARN=()

warn() { echo "WARNING: $*" >&2; WARN+=("$*"); }

# 1) uv (installs a private Python 3.10 without sudo)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "uv install failed"; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) main environment (hard requirement)
uv venv --python "$PYVER" "$ENV_DIR" || exit 1
PY="$ENV_DIR/bin/python"
#  numpy<2 / transformers<5 are precautionary pins (unbabel-comet, the MetricX repo and CLIP code were
#  written against 4.x / numpy 1.x). Relax them if everything works.
uv pip install --python "$PY" \
  "numpy<2" pandas tqdm pillow \
  torch \
  "transformers>=4.40,<5" sentencepiece protobuf datasets accelerate \
  "unbabel-comet>=2.2.0" \
  || { echo "main env install failed"; exit 1; }

# 3) MetricX repo (provides metricx23/ and metricx24/ predict scripts)
if [ ! -d metricx ]; then
  git clone --depth 1 https://github.com/google-research/metricx.git metricx || warn "could not clone metricx"
fi

# 4) Sentinel-src-mqm in its own venv
if [ ! -d guardians-mt-eval ]; then
  git clone --depth 1 https://github.com/SapienzaNLP/guardians-mt-eval.git guardians-mt-eval \
    || warn "could not clone guardians-mt-eval"
fi
if [ -d guardians-mt-eval ]; then
  uv venv --python "$PYVER" .venv-sentinel \
    && uv pip install --python .venv-sentinel/bin/python -e guardians-mt-eval \
    && .venv-sentinel/bin/python -c "from sentinel_metric import download_model, load_from_checkpoint" \
    || warn "Sentinel install/import failed - follow the install steps in the guardians-mt-eval README inside .venv-sentinel (runner auto-uses .venv-sentinel/bin/python)"
fi

# 5) Official MetaMetrics repo (training / official pipeline) in its own venv - optional
if [ "${SKIP_METAMETRICS:-0}" != "1" ]; then
  if [ ! -d metametrics ]; then
    git clone --recurse-submodules https://github.com/meta-metrics/metametrics.git metametrics \
      || warn "could not clone metametrics"
  fi
  if [ -d metametrics ]; then
    uv venv --python "$PYVER" .venv-metametrics \
      && uv pip install --python .venv-metametrics/bin/python -e metametrics \
      || warn "MetaMetrics install failed - see metametrics/README.md (Python >= 3.10, 'pip install -e .')"
  fi
fi

# 6) sanity checks
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
import comet, transformers, PIL  # noqa: F401
print("imports OK, transformers", transformers.__version__)
PYEOF

# 7) Hugging Face access for gated models
if [ -z "${HF_TOKEN:-}" ] && ! "$ENV_DIR/bin/python" -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  warn "Not logged in to Hugging Face. CometKiwi (Unbabel/wmt22-cometkiwi-da) and Unbabel/XCOMET-XL are gated:
         1) open each model page and accept the licence, 2) run: $ENV_DIR/bin/huggingface-cli login   (or export HF_TOKEN=...)"
fi

echo
echo "Done. Next:"
echo "  source $ENV_DIR/bin/activate"
echo "  python mt_qe_metrics_runner.py --input example_testset.csv --output qe_scores.json \\"
echo "         --src-lang en --tgt-lang de --metrics transquest,metricx23_qe --limit 4     # smoke test"
echo "  python mt_qe_metrics_runner.py --input your_testset.csv --output qe_scores.json --src-lang en --tgt-lang de"
if [ ${#WARN[@]} -gt 0 ]; then
  echo; echo "Things that need your attention:"; printf '  - %s\n' "${WARN[@]}"
fi
