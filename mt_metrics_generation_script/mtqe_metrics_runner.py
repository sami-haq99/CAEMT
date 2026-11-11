#!/usr/bin/env python3
"""
mt_qe_metrics_runner.py
=======================
Reference-free (QE) metrics for the same CSV test set / JSON output format as mt_metrics_runner.py.
It reuses that module (keep both files in the same folder), so every metric there (bleu, comet22, ...)
is also available here via --metrics.

QE metrics (CLI key -> output name)
    metricx23_qe -> MetricX-23-QE-XL     google/metricx-23-qe-xl-v2p0   (0-25, LOWER is better)
    cometkiwi    -> CometKiwi-22         Unbabel/wmt22-cometkiwi-da     (gated on HF)
    xcomet       -> xCOMET-XL-QE         Unbabel/XCOMET-XL              (gated on HF; also stores error spans)
    sentinel_src -> Sentinel-src-mqm     sapienzanlp/sentinel-src-mqm   (SOURCE ONLY, see note)
    transquest   -> TransQuest           surrey-nlp/MonoTQ-XLMV
    clipscore    -> CLIPScore            openai/clip-vit-base-patch32   (needs an image column)
    metametrics  -> MetaMetrics          weighted combination of the metrics above (needs a weights file)

Notes
  * Sentinel-src-mqm only looks at the SOURCE segment (it is a "sentinel"/sanity-check metric, not a
    quality estimator of the translation). It gives the same score to every system translating the
    same source - that is expected.
  * CLIPScore: CSV needs an image column (default name "image", paths optionally relative to
    --clip-image-dir). The default OpenAI CLIP is English-only; for non-English targets pass an
    HF-compatible multilingual CLIP via --clip-model.
  * MetaMetrics is a weighted sum of component metrics whose weights are *learned from human scores*.
    Here it is applied with weights YOU supply (--metametrics-weights, JSON, see class docstring);
    it is not the official pretrained pipeline. Component metrics must be part of the same run.
  * CometKiwi and xCOMET-XL are gated: accept the licence on their HF pages, then `huggingface-cli login`
    (or export HF_TOKEN).

Usage
    python mt_qe_metrics_runner.py --input test.csv --output qe_scores.json --src-lang en --tgt-lang de
    python mt_qe_metrics_runner.py --input test.csv --output qe_scores.json --src-lang en --tgt-lang de \
        --metrics metricx23_qe,cometkiwi,metametrics --metametrics-weights weights.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import mt_metrics_runner as base
from mt_metrics_runner import BaseMetric, MTMetricsRunner, RunnerConfig

log = logging.getLogger("mt_qe_metrics")


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
@dataclass
class QEConfig(RunnerConfig):
    batch_size: int = 8

    # MetricX-23-QE (official repo CLI; shares the ./metricx clone with MetricX-24)
    metricx23_model: str = "google/metricx-23-qe-xl-v2p0"
    metricx23_tokenizer: Optional[str] = None   # auto: google/mt5-{large,xl,xxl}
    metricx23_max_len: int = 1024
    metricx23_batch_size: int = 1               # keep 1: the official predict.py does not pad, batches >1 crash

    # COMET-family (unbabel-comet)
    cometkiwi_model: str = "Unbabel/wmt22-cometkiwi-da"   # or Unbabel/wmt23-cometkiwi-da-xl / -xxl
    xcomet_model: str = "Unbabel/XCOMET-XL"
    xcomet_use_reference: bool = False                   # False = QE mode (source + MT only)

    # Sentinel (own package -> run in a subprocess, optionally in a different venv)
    sentinel_model: str = "sapienzanlp/sentinel-src-mqm"
    sentinel_python: Optional[str] = None                # e.g. .venv-sentinel/bin/python

    # TransQuest
    transquest_model: str = "surrey-nlp/MonoTQ-XLMV"
    transquest_max_len: int = 128

    # CLIPScore
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_image_column: str = "image"
    clip_image_dir: str = ""
    clip_weight: float = 2.5
    clip_prefix: str = "A photo depicts "

    # MetaMetrics (weighted combination)
    metametrics_weights: Optional[str] = None


def _gpus(cfg: RunnerConfig) -> int:
    return 1 if cfg.device.startswith("cuda") else 0


# ----------------------------------------------------------------------------------------
# MetricX-23-QE
# ----------------------------------------------------------------------------------------
def run_metricx_cli(repo, module, tokenizer, model, max_len, batch_size, rows, qe) -> List[float]:
    """Call the official `python -m <module>` predict script and return its predictions."""
    repo = Path(repo).resolve()
    pkg = module.split(".")[0]
    if not (repo / pkg).is_dir():
        raise RuntimeError(f"MetricX repo not found at {repo} (missing '{pkg}/'). Run setup_env_qe.sh or: "
                           "git clone https://github.com/google-research/metricx.git")
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with open(inp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cmd = [sys.executable, "-c", base.METRICX_LAUNCHER, module, "--tokenizer", tokenizer, "--model_name_or_path", model,
               "--max_input_length", str(max_len), "--batch_size", str(batch_size),
               "--input_file", str(inp), "--output_file", str(outp)]
        if qe:
            cmd.append("--qe")
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
        log.info("MetricX command: %s", " ".join(cmd[:1] + ["<launcher>"] + cmd[3:]))
        rc = subprocess.run(cmd, env=env, cwd=str(repo)).returncode
        if rc != 0 or not outp.exists():
            raise RuntimeError(f"{module} failed (exit code {rc}); see output above")
        with open(outp, encoding="utf-8") as f:
            return [float(json.loads(line)["prediction"]) for line in f if line.strip()]


class MetricX23QEMetric(BaseMetric):
    key, name = "metricx23_qe", "MetricX-23-QE"

    def __init__(self, cfg):
        super().__init__(cfg)
        m = re.search(r"-(large|xl|xxl)-", cfg.metricx23_model)
        self.size = m.group(1) if m else None
        self.out_name = f"MetricX-23-QE-{self.size.upper()}" if self.size else "MetricX-23-QE"
        self.outputs = {self.out_name: False}   # error score in [0, 25]: LOWER is better

    def score(self, srcs, refs, hyps):
        c = self.cfg
        tok = c.metricx23_tokenizer or (f"google/mt5-{self.size}" if self.size else None)
        if not tok:
            raise ValueError("Cannot infer mT5 tokenizer from model name; pass --metricx23-tokenizer")
        rows = [{"source": s, "hypothesis": h} for s, h in zip(srcs, hyps)]
        preds = run_metricx_cli(c.metricx_repo, "metricx23.predict", tok, c.metricx23_model,
                                c.metricx23_max_len, c.metricx23_batch_size, rows, qe=True)
        return {self.out_name: preds}

    def info(self):
        return {"model": self.cfg.metricx23_model, "scale": "0-25, lower is better"}


# ----------------------------------------------------------------------------------------
# CometKiwi / xCOMET (unbabel-comet)
# ----------------------------------------------------------------------------------------
def _load_comet(model_name: str):
    from comet import download_model, load_from_checkpoint
    try:
        path = download_model(model_name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download {model_name}. CometKiwi and XCOMET are gated: accept the licence on the "
            f"model's Hugging Face page, then run `huggingface-cli login` (or export HF_TOKEN). "
            f"Original error: {e}") from e
    return load_from_checkpoint(path)


def _kiwi_name(model: str) -> str:
    n = model.lower()
    if "xxl" in n:
        return "CometKiwi-23-XXL"
    if "-xl" in n:
        return "CometKiwi-23-XL"
    if "wmt22" in n:
        return "CometKiwi-22"
    return "CometKiwi"


class CometKiwiMetric(BaseMetric):
    key, name = "cometkiwi", "CometKiwi"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_name = _kiwi_name(cfg.cometkiwi_model)
        self.outputs = {self.out_name: True}

    def score(self, srcs, refs, hyps):
        model = _load_comet(self.cfg.cometkiwi_model)
        data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
        out = model.predict(data, batch_size=self.cfg.batch_size, gpus=_gpus(self.cfg), progress_bar=True)
        return {self.out_name: [float(x) for x in out.scores]}

    def info(self):
        return {"model": self.cfg.cometkiwi_model}


class XCometMetric(BaseMetric):
    key, name = "xcomet", "xCOMET"

    def __init__(self, cfg):
        super().__init__(cfg)
        m = re.search(r"(XXL|XL)$", cfg.xcomet_model, re.I)
        size = m.group(1).upper() if m else ""
        self.out_name = "xCOMET" + (f"-{size}" if size else "") + ("" if cfg.xcomet_use_reference else "-QE")
        self.outputs = {self.out_name: True}

    def score(self, srcs, refs, hyps):
        c = self.cfg
        model = _load_comet(c.xcomet_model)
        if c.xcomet_use_reference:
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        else:
            data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
        out = model.predict(data, batch_size=c.batch_size, gpus=_gpus(c), progress_bar=True)
        spans = getattr(getattr(out, "metadata", None), "error_spans", None)
        if spans is not None:
            self.extra["xCOMET_error_spans"] = [json.dumps(s, ensure_ascii=False, default=str) for s in spans]
        return {self.out_name: [float(x) for x in out.scores]}

    def info(self):
        return {"model": self.cfg.xcomet_model, "reference_used": self.cfg.xcomet_use_reference}


# ----------------------------------------------------------------------------------------
# Sentinel-src-mqm (separate package `sentinel_metric`; run in a subprocess so it can live in its own venv)
# ----------------------------------------------------------------------------------------
SENTINEL_WORKER = """
import json, sys
from sentinel_metric import download_model, load_from_checkpoint
inp, outp, model_name, bs, gpus = sys.argv[1:6]
with open(inp, encoding="utf-8") as f:
    srcs = json.load(f)
model = load_from_checkpoint(download_model(model_name))
out = model.predict([{"src": s} for s in srcs], batch_size=int(bs), gpus=int(gpus))
with open(outp, "w", encoding="utf-8") as f:
    json.dump([float(x) for x in out.scores], f)
"""


class SentinelSrcMQMMetric(BaseMetric):
    key, name = "sentinel_src", "Sentinel-src-mqm"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"Sentinel-src-mqm": True}

    def score(self, srcs, refs, hyps):
        c = self.cfg
        uniq = list(dict.fromkeys(srcs))          # source-only metric: score each distinct source once
        # default: the isolated venv that setup_env_qe.sh creates for Sentinel, else the current interpreter
        local_venv = Path(".venv-sentinel/bin/python")
        py = c.sentinel_python or (str(local_venv.resolve()) if local_venv.exists() else sys.executable)
        with tempfile.TemporaryDirectory() as tmp:
            inp, outp = Path(tmp) / "src.json", Path(tmp) / "scores.json"
            inp.write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
            cmd = [py, "-c", SENTINEL_WORKER, str(inp), str(outp), c.sentinel_model,
                   str(c.batch_size), str(_gpus(c))]
            rc = subprocess.run(cmd).returncode
            if rc != 0 or not outp.exists():
                raise RuntimeError(
                    f"Sentinel worker failed (exit code {rc}) using {py}. Is `sentinel_metric` installed there? "
                    "See setup_env_qe.sh (it creates .venv-sentinel if it cannot share the main env); "
                    "then pass --sentinel-python .venv-sentinel/bin/python")
            by_src = dict(zip(uniq, json.loads(outp.read_text(encoding="utf-8"))))
        return {"Sentinel-src-mqm": [by_src[s] for s in srcs]}

    def info(self):
        return {"model": self.cfg.sentinel_model, "note": "uses the source segment only; ignores the MT output"}


# ----------------------------------------------------------------------------------------
# TransQuest (MonoTransQuest-style cross-encoder: source/target pair -> regression score)
# ----------------------------------------------------------------------------------------
class TransQuestMetric(BaseMetric):
    key, name = "transquest", "TransQuest"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"TransQuest": True}

    def score(self, srcs, refs, hyps):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        c = self.cfg
        tok = AutoTokenizer.from_pretrained(c.transquest_model)
        model = AutoModelForSequenceClassification.from_pretrained(c.transquest_model).to(c.device).eval()
        if model.config.num_labels != 1:
            raise RuntimeError(f"{c.transquest_model} has num_labels={model.config.num_labels}; "
                               "expected a single-output regression head")
        n, bs = len(hyps), c.batch_size
        order = sorted(range(n), key=lambda i: len(srcs[i]) + len(hyps[i]))
        scores = [float("nan")] * n
        with torch.inference_mode():
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                enc = tok([srcs[i] for i in idx], [hyps[i] for i in idx], padding=True, truncation=True,
                          max_length=c.transquest_max_len, return_tensors="pt").to(c.device)
                out = model(**enc).logits.flatten().float().cpu().tolist()
                for i, s in zip(idx, out):
                    scores[i] = s
        return {"TransQuest": scores}

    def info(self):
        return {"model": self.cfg.transquest_model,
                "note": "scale depends on the model's training labels; compare/rank, don't read as absolute"}


# ----------------------------------------------------------------------------------------
# CLIPScore (Hessel et al. 2021): w * max(cos(image, "A photo depicts " + candidate), 0)
# ----------------------------------------------------------------------------------------
class CLIPScoreMetric(BaseMetric):
    key, name = "clipscore", "CLIPScore"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"CLIPScore": True}

    @staticmethod
    def _tensor(x):
        import torch
        if torch.is_tensor(x):
            return x
        return getattr(x, "pooler_output", None) if getattr(x, "pooler_output", None) is not None else x[0]

    def score(self, srcs, refs, hyps):
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        c = self.cfg
        col = c.clip_image_column
        if col not in self.columns_data:
            raise RuntimeError(f"CLIPScore needs an image column '{col}' in the CSV (set --clip-image-column)")
        paths = [str(Path(c.clip_image_dir) / p) if c.clip_image_dir else p for p in self.columns_data[col]]
        missing = sorted({p for p in paths if not Path(p).is_file()})
        if missing:
            raise RuntimeError(f"{len(missing)} image file(s) not found, e.g. {missing[:3]}")

        proc = CLIPProcessor.from_pretrained(c.clip_model)
        model = CLIPModel.from_pretrained(c.clip_model).to(c.device).eval()
        bs = c.batch_size

        img_emb: Dict[str, "torch.Tensor"] = {}
        uniq = list(dict.fromkeys(paths))                     # images repeat across systems: embed once
        with torch.inference_mode():
            for s in range(0, len(uniq), bs):
                chunk = uniq[s:s + bs]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(c.device)
                f = F.normalize(self._tensor(model.get_image_features(pixel_values=px)).float(), dim=-1).cpu()
                for p, e in zip(chunk, f):
                    img_emb[p] = e

            scores: List[float] = []
            for s in range(0, len(hyps), bs):
                texts = [c.clip_prefix + h for h in hyps[s:s + bs]]
                enc = proc.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                     return_tensors="pt").to(c.device)
                t = F.normalize(self._tensor(model.get_text_features(**enc)).float(), dim=-1).cpu()
                im = torch.stack([img_emb[p] for p in paths[s:s + bs]])
                cos = (t * im).sum(-1)
                scores.extend((c.clip_weight * torch.clamp(cos, min=0)).tolist())
        return {"CLIPScore": scores}

    def info(self):
        return {"model": self.cfg.clip_model, "weight": self.cfg.clip_weight, "prefix": self.cfg.clip_prefix,
                "note": "reference-free image-text similarity of the MT output"}


# ----------------------------------------------------------------------------------------
# MetaMetrics-style weighted combination of metrics computed earlier in the same run
# ----------------------------------------------------------------------------------------
class MetaMetricsMetric(BaseMetric):
    """
    Weighted sum of min-max-normalised component metrics (lower-is-better metrics are flipped), divided
    by the sum of the weights, so the result lies in [0, 1] and higher is better.

    Weights file (JSON):
        {
          "weights": {"MetricX-23-QE-XL": 0.6, "CometKiwi-22": 0.3, "xCOMET-XL-QE": 0.1},
          "ranges":  {"TransQuest": [-3, 3]}          # optional; overrides the defaults below
        }
    Keys must equal the output names in the JSON (see "metric_info"). Default ranges: MetricX 0-25;
    CometKiwi/xCOMET/COMET/Sentinel 0-1; CLIPScore 0-clip_weight; anything else: min/max of this data set.

    This reproduces the *form* of MetaMetrics (a calibrated weighted combination), with weights you
    supply - e.g. learned with the official repo (github.com/meta-metrics/metametrics) on human scores.
    It is not the official pretrained pipeline.
    """
    key, name = "metametrics", "MetaMetrics"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"MetaMetrics": True}
        self._used: dict = {}

    def _default_range(self, n: str):
        if n.startswith("MetricX"):
            return 0.0, 25.0
        if n.startswith(("CometKiwi", "xCOMET", "COMET", "Sentinel")):
            return 0.0, 1.0
        if n == "CLIPScore":
            return 0.0, float(self.cfg.clip_weight)
        return None

    def score(self, srcs, refs, hyps):
        path = self.cfg.metametrics_weights
        if not path:
            raise RuntimeError("MetaMetrics needs --metametrics-weights <weights.json> (see class docstring)")
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        weights, ranges = spec["weights"], spec.get("ranges", {})
        missing = [k for k in weights if k not in self.prior_scores]
        if missing:
            raise RuntimeError(f"Component metric(s) {missing} were not scored earlier in this run "
                               f"(available: {list(self.prior_scores)}). Add them to --metrics.")
        total = float(sum(weights.values()))
        combo = np.zeros(len(srcs))
        for n, w in weights.items():
            vals = np.array(self.prior_scores[n], dtype=float)
            rng = ranges.get(n) or self._default_range(n) or (float(np.nanmin(vals)), float(np.nanmax(vals)))
            lo, hi = float(rng[0]), float(rng[1])
            norm = np.clip((vals - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(vals)
            if not self.prior_info[n].get("higher_is_better", True):
                norm = 1.0 - norm
            combo += float(w) * norm
            self._used[n] = {"weight": w, "range": [lo, hi],
                             "flipped": not self.prior_info[n].get("higher_is_better", True)}
        return {"MetaMetrics": (combo / total).tolist()}

    def info(self):
        return {"components": self._used, "scale": "0-1, higher is better",
                "note": "weighted combination with user-supplied weights; not the official pretrained pipeline"}


# ----------------------------------------------------------------------------------------
# Registry + CLI
# ----------------------------------------------------------------------------------------
QE_METRICS = {m.key: m for m in (MetricX23QEMetric, CometKiwiMetric, XCometMetric, SentinelSrcMQMMetric,
                                 TransQuestMetric, CLIPScoreMetric, MetaMetricsMetric)}
QE_ORDER = ["metricx23_qe", "cometkiwi", "xcomet", "sentinel_src", "transquest", "clipscore", "metametrics"]
base.METRICS.update(QE_METRICS)   # the base runner looks metrics up in this dict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV test set")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--src-lang", required=True, help="source language code, e.g. en")
    p.add_argument("--tgt-lang", required=True, help="target language code, e.g. de")
    p.add_argument("--metrics", default="all",
                   help=f"comma-separated keys from {','.join(QE_ORDER)} (or any key of mt_metrics_runner). "
                        "'all' = every QE metric that can run (metametrics only if --metametrics-weights is given)")
    p.add_argument("--limit", type=int, help="only score the first N rows (smoke test)")
    p.add_argument("--omit-text", action="store_true")
    for col in ("srctxt", "reftxt", "tgttxt", "sys", "seg"):
        p.add_argument(f"--col-{col}", default=col, help=f"CSV column holding '{col}'")
    # every remaining QEConfig field becomes a flag automatically (--metricx23-model, --clip-model, ...)
    for f in dataclasses.fields(QEConfig):
        if f.name in ("src_lang", "tgt_lang"):
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(flag, action="store_true", default=default)
        elif isinstance(default, int):
            p.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(flag, type=float, default=default)
        else:
            p.add_argument(flag, default=default)
    return p


def main(argv: Optional[Sequence[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    cfg = QEConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(QEConfig)
                      if getattr(a, f.name, None) is not None})
    columns = {k: getattr(a, f"col_{k}") for k in ("srctxt", "reftxt", "tgttxt", "sys", "seg")}

    requested = [m.strip() for m in a.metrics.split(",") if m.strip()]
    run_all = requested == ["all"]
    keys = list(QE_ORDER) if run_all else requested

    # runner needs the CSV to decide on image-dependent / weight-dependent metrics
    runner = MTMetricsRunner(["bleu"], cfg, columns=columns, include_text=not a.omit_text)  # placeholder keys
    df = runner.load_testset(a.input, limit=a.limit)
    if run_all:
        if cfg.clip_image_column not in df.columns:
            log.warning("No '%s' column -> skipping CLIPScore", cfg.clip_image_column)
            keys.remove("clipscore")
        if not cfg.metametrics_weights:
            keys.remove("metametrics")
    if "metametrics" in keys:                       # combiner must run after its components
        keys = [k for k in keys if k != "metametrics"] + ["metametrics"]

    runner = MTMetricsRunner(keys, cfg, columns=columns, include_text=not a.omit_text)
    runner.input_path = str(a.input)
    log.info("Loaded %d segments, device=%s, metrics=%s", len(df), cfg.device, keys)
    results = runner.run(df, output_path=a.output)

    print("\nOverall:")
    for name, val in results["overall"].items():
        print(f"  {name:<20} {val if val is None else round(val, 4)}")
    if results["errors"]:
        print("\nFailed metrics:")
        for name, err in results["errors"].items():
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
