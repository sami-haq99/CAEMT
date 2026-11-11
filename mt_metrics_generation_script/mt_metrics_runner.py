#!/usr/bin/env python3
"""
mt_metrics_runner.py
====================
Run several MT evaluation metrics over a CSV test set and record the scores in a JSON file.

Metrics (CLI key -> output name)
    bleu       -> BLEU                      (sacreBLEU)
    chrf       -> chrF                      (sacreBLEU, chrF2)
    ter        -> TER                       (sacreBLEU; lower is better)
    bertscore  -> BERTScore-F1 / -P / -R    (bert_score)
    comet22    -> COMET-22                  (Unbabel/wmt22-comet-da)
    bleurt20   -> BLEURT-20                 (PyTorch port: lucadiliello/BLEURT-20)
    metricx24  -> MetricX-24                (official google-research/metricx CLI; 0-25, lower is better)
    gemba_esa  -> GEMBA-ESA                 (LLM via OpenAI-compatible API; 0-100)

Input CSV columns (names can be overridden with --col-* flags):
    source, reference, target, system, segment
    - target  = the MT output being evaluated
    - system  = MT system name (used for per-system aggregation)
    - segment = segment id (carried through to the JSON untouched)

Usage
    python mt_metrics_runner.py --input test.csv --output scores.json --src-lang en --tgt-lang de
    python mt_metrics_runner.py --input test.csv --output scores.json --src-lang en --tgt-lang de \
        --metrics bleu,chrf,ter,comet22 --limit 50

Library usage
    from mt_metrics_runner import MTMetricsRunner, RunnerConfig
    cfg = RunnerConfig(src_lang="en", tgt_lang="de")
    runner = MTMetricsRunner(["bleu", "chrf", "comet22"], cfg)
    df = runner.load_testset("test.csv")
    results = runner.run(df, output_path="scores.json")
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("mt_metrics")

LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "cs": "Czech",
    "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "he": "Hebrew", "hi": "Hindi",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "is": "Icelandic", "fi": "Finnish",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "bg": "Bulgarian", "hr": "Croatian", "sk": "Slovak", "sl": "Slovenian",
    "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "fa": "Persian", "bn": "Bengali", "ga": "Irish",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.lower().split("-")[0].split("_")[0], code)


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
@dataclass
class RunnerConfig:
    src_lang: str = "en"
    tgt_lang: str = "de"
    device: str = field(default_factory=_default_device)
    batch_size: int = 16

    # lexical metrics
    bleu_tokenize: str = "auto"         # auto: zh->"zh", ja->"ja-mecab", ko->"ko-mecab", else "13a"

    # BERTScore: None -> bert_score's default model for tgt_lang (roberta-large for en,
    # bert-base-multilingual-cased otherwise). Set e.g. "xlm-roberta-large" to override.
    bertscore_model: Optional[str] = None

    # neural metrics
    comet_model: str = "Unbabel/wmt22-comet-da"
    bleurt_model: str = "lucadiliello/BLEURT-20"
    metricx_model: str = "google/metricx-24-hybrid-xl-v2p6"
    metricx_tokenizer: Optional[str] = None   # auto: google/mt5-{large,xl,xxl} from model name
    metricx_repo: str = "./metricx"           # clone of github.com/google-research/metricx
    metricx_max_len: int = 1536
    metricx_batch_size: int = 1               # keep 1: the official predict.py does not pad, batches >1 crash
    metricx_qe: bool = False                  # True -> reference-free (QE) mode

    # GEMBA-ESA (LLM judge; any OpenAI-compatible endpoint)
    gemba_model: str = "gpt-4o"               # change to whatever model you have access to
    gemba_base_url: Optional[str] = None      # e.g. http://localhost:8000/v1 for vLLM / Ollama
    gemba_workers: int = 8
    gemba_use_reference: bool = False         # GEMBA-ESA is normally reference-free
    gemba_cache: str = "gemba_cache.json"


# ----------------------------------------------------------------------------------------
# Metric base class
# ----------------------------------------------------------------------------------------
class BaseMetric:
    key = "base"
    name = "base"
    aggregation = "mean of segment scores"

    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        # output name -> True if higher is better
        self.outputs: Dict[str, bool] = {}
        # extra per-segment details (e.g. LLM annotations): name -> list[str]
        self.extra: Dict[str, List[str]] = {}
        # filled in by the runner before score(): every CSV column, and scores of metrics run earlier
        self.columns_data: Dict[str, List[str]] = {}
        self.prior_scores: Dict[str, List[float]] = {}
        self.prior_info: Dict[str, dict] = {}

    def score(self, srcs: List[str], refs: List[str], hyps: List[str]) -> Dict[str, List[float]]:
        raise NotImplementedError

    def corpus_score(self, out_name: str, srcs, refs, hyps) -> Optional[float]:
        """Optional corpus-level override. Return None to fall back to the mean of segment scores."""
        return None

    def info(self) -> dict:
        return {}


# ----------------------------------------------------------------------------------------
# Lexical metrics (sacreBLEU)
# ----------------------------------------------------------------------------------------
class _SacreMetric(BaseMetric):
    aggregation = "corpus-level (sacreBLEU)"
    higher_is_better = True

    def _make(self, segment_level: bool):
        raise NotImplementedError

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {self.name: self.higher_is_better}
        self.seg_metric = self._make(True)
        self.corpus_metric = self._make(False)

    def score(self, srcs, refs, hyps):
        vals = [float(self.seg_metric.sentence_score(h, [r]).score) for h, r in zip(hyps, refs)]
        return {self.name: vals}

    def corpus_score(self, out_name, srcs, refs, hyps):
        return float(self.corpus_metric.corpus_score(list(hyps), [list(refs)]).score)

    def info(self):
        try:
            return {"signature": str(self.corpus_metric.get_signature())}
        except Exception:
            return {}


def _lang_base(code: str) -> str:
    return code.lower().split("-")[0].split("_")[0]


def _resolve_bleu_tokenize(cfg) -> str:
    if cfg.bleu_tokenize != "auto":
        return cfg.bleu_tokenize
    return {"zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab"}.get(_lang_base(cfg.tgt_lang), "13a")


class BLEUMetric(_SacreMetric):
    key, name = "bleu", "BLEU"

    def _make(self, segment_level):
        from sacrebleu.metrics import BLEU
        return BLEU(tokenize=_resolve_bleu_tokenize(self.cfg), effective_order=segment_level)


class CHRFMetric(_SacreMetric):
    key, name = "chrf", "chrF"

    def _make(self, segment_level):
        from sacrebleu.metrics import CHRF
        return CHRF()


class TERMetric(_SacreMetric):
    key, name = "ter", "TER"
    higher_is_better = False

    def _make(self, segment_level):
        from sacrebleu.metrics import TER
        # unsegmented CJK text needs sacreBLEU's Asian-support tokenisation, otherwise TER is meaningless
        asian = _lang_base(self.cfg.tgt_lang) in ("zh", "ja")
        return TER(normalized=asian, asian_support=asian)


# ----------------------------------------------------------------------------------------
# BERTScore
# ----------------------------------------------------------------------------------------
class BERTScoreMetric(BaseMetric):
    key, name = "bertscore", "BERTScore"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"BERTScore-F1": True, "BERTScore-P": True, "BERTScore-R": True}
        self._hash = None

    def score(self, srcs, refs, hyps):
        from bert_score import BERTScorer
        scorer = BERTScorer(
            model_type=self.cfg.bertscore_model,
            lang=self.cfg.tgt_lang,
            batch_size=self.cfg.batch_size,
            device=self.cfg.device,
            rescale_with_baseline=False,
        )
        try:
            self._hash = scorer.hash
        except Exception:
            pass
        P, R, F = scorer.score(list(hyps), list(refs), batch_size=self.cfg.batch_size)
        return {
            "BERTScore-F1": F.tolist(),
            "BERTScore-P": P.tolist(),
            "BERTScore-R": R.tolist(),
        }

    def info(self):
        return {"hash": self._hash} if self._hash else {}


# ----------------------------------------------------------------------------------------
# COMET-22
# ----------------------------------------------------------------------------------------
class Comet22Metric(BaseMetric):
    key, name = "comet22", "COMET-22"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"COMET-22": True}

    def score(self, srcs, refs, hyps):
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(self.cfg.comet_model))
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        gpus = 1 if self.cfg.device.startswith("cuda") else 0
        out = model.predict(data, batch_size=self.cfg.batch_size, gpus=gpus, progress_bar=True)
        return {"COMET-22": [float(x) for x in out.scores]}

    def info(self):
        return {"model": self.cfg.comet_model}


# ----------------------------------------------------------------------------------------
# BLEURT-20 (PyTorch port of the official checkpoint)
# ----------------------------------------------------------------------------------------
class BLEURT20Metric(BaseMetric):
    key, name = "bleurt20", "BLEURT-20"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"BLEURT-20": True}

    def score(self, srcs, refs, hyps):
        import torch
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        tok = BleurtTokenizer.from_pretrained(self.cfg.bleurt_model)
        model = BleurtForSequenceClassification.from_pretrained(self.cfg.bleurt_model)
        model.to(self.cfg.device).eval()

        n, bs = len(hyps), self.cfg.batch_size
        order = sorted(range(n), key=lambda i: len(refs[i]) + len(hyps[i]))  # less padding
        scores = [float("nan")] * n
        with torch.inference_mode():
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                enc = tok([refs[i] for i in idx], [hyps[i] for i in idx], padding="longest",
                          truncation=True, max_length=512, return_tensors="pt").to(self.cfg.device)
                out = model(**enc).logits.flatten().float().cpu().tolist()
                for i, s in zip(idx, out):
                    scores[i] = s
        return {"BLEURT-20": scores}

    def info(self):
        return {"model": self.cfg.bleurt_model}



# MetricX's custom model builds its decoder with is_encoder_decoder=False. With recent transformers (>= ~4.5x)
# that makes the decoder create ONE shared cache for self- and cross-attention, so the encoder keys get
# appended to the single decoder token ("size of tensor a (21) must match size of tensor b (20)").
# MetricX runs exactly one decoder step and needs no cache, so we run the official predict script
# unchanged except that models are loaded with use_cache=False.
METRICX_LAUNCHER = """
import importlib, runpy, sys
module = sys.argv.pop(1)                        # e.g. metricx24.predict
models = importlib.import_module(module.split(".")[0] + ".models")
cls = models.MT5ForRegression
_orig = cls.from_pretrained.__func__
def _patched(c, *a, **k):
    k.setdefault("use_cache", False)
    return _orig(c, *a, **k)
cls.from_pretrained = classmethod(_patched)
runpy.run_module(module, run_name="__main__", alter_sys=True)
"""

# ----------------------------------------------------------------------------------------
# MetricX-24 (calls the official `python -m metricx24.predict` so numbers match the reference impl.)
# ----------------------------------------------------------------------------------------
class MetricX24Metric(BaseMetric):
    key, name = "metricx24", "MetricX-24"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_name = "MetricX-24-QE" if cfg.metricx_qe else "MetricX-24"
        self.outputs = {self.out_name: False}  # error score in [0, 25]: LOWER is better

    def _tokenizer(self) -> str:
        if self.cfg.metricx_tokenizer:
            return self.cfg.metricx_tokenizer
        m = re.search(r"-(large|xl|xxl)-", self.cfg.metricx_model)
        if not m:
            raise ValueError("Cannot infer mT5 tokenizer from model name; pass --metricx-tokenizer")
        return f"google/mt5-{m.group(1)}"

    def score(self, srcs, refs, hyps):
        repo = Path(self.cfg.metricx_repo).resolve()
        if not (repo / "metricx24").is_dir():
            raise RuntimeError(
                f"MetricX repo not found at {repo}. Run setup_env.sh or: "
                "git clone https://github.com/google-research/metricx.git"
            )
        with tempfile.TemporaryDirectory() as tmp:
            inp, outp = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
            with open(inp, "w", encoding="utf-8") as f:
                for s, h, r in zip(srcs, hyps, refs):
                    f.write(json.dumps({"source": s, "hypothesis": h, "reference": r},
                                       ensure_ascii=False) + "\n")
            cmd = [
                sys.executable, "-c", METRICX_LAUNCHER, "metricx24.predict",
                "--tokenizer", self._tokenizer(),
                "--model_name_or_path", self.cfg.metricx_model,
                "--max_input_length", str(self.cfg.metricx_max_len),
                "--batch_size", str(self.cfg.metricx_batch_size),
                "--input_file", str(inp),
                "--output_file", str(outp),
            ]
            if self.cfg.metricx_qe:
                cmd.append("--qe")
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
            log.info("MetricX-24 command: %s", " ".join(cmd[:1] + ["<launcher>"] + cmd[3:]))
            rc = subprocess.run(cmd, env=env, cwd=str(repo)).returncode
            if rc != 0 or not outp.exists():
                raise RuntimeError(f"metricx24.predict failed (exit code {rc}); see output above")
            with open(outp, encoding="utf-8") as f:
                preds = [float(json.loads(line)["prediction"]) for line in f if line.strip()]
        return {self.out_name: preds}

    def info(self):
        return {"model": self.cfg.metricx_model, "qe_mode": self.cfg.metricx_qe,
                "scale": "0-25, lower is better"}


# ----------------------------------------------------------------------------------------
# GEMBA-ESA (LLM-as-judge, Error Span Annotation protocol)
#
# NOTE: these two prompts are an implementation of the ESA protocol (1. mark minor/major error
# spans, 2. give a 0-100 score given those spans). They are NOT guaranteed to be byte-identical to
# the prompts in Microsoft's official GEMBA release. For numbers you intend to report as
# "GEMBA-ESA", diff these against the official prompt and edit the constants below if needed.
# ----------------------------------------------------------------------------------------
ESA_HEADER = """Your task is to identify machine translation errors and assess the quality of the translation.

{src_lang} source:
```{source}```
{tgt_lang} translation:
```{target}```
{reference_block}"""

ESA_REFERENCE_BLOCK = """{tgt_lang} reference translation (for guidance only):
```{reference}```
"""

ESA_ANNOTATE_INSTR = """
Based on the source segment and the translation, identify every error span in the translation and mark its severity:
- minor: the error does not impede understanding (e.g. slightly awkward wording, small grammar, spelling or punctuation issues)
- major: the error changes the meaning, makes the text hard to understand, or is a serious accuracy/fluency problem (including omissions, additions and untranslated text)
Copy each error span verbatim from the translation (for an omission, quote the nearest affected span). Output one error per line in the format:
<severity> | <error span> | <short explanation>
If the translation contains no errors, output exactly: no-error"""

ESA_SCORE_INSTR = """
Error annotations for the translation:
{annotations}

Taking the annotations into account, rate the overall quality of the translation on a continuous scale from 0 to 100, where 0 means "no meaning preserved" and 100 means "perfect translation". Respond with the number only."""


class GembaESAMetric(BaseMetric):
    key, name = "gemba_esa", "GEMBA-ESA"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = {"GEMBA-ESA": True}
        self._lock = threading.Lock()
        self._cache: Dict[str, str] = {}
        self._cache_path = Path(cfg.gemba_cache)
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Could not read GEMBA cache %s; starting empty", self._cache_path)
        self._client = self._make_client()

    def _make_client(self):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key and not self.cfg.gemba_base_url:
            raise RuntimeError("GEMBA-ESA needs OPENAI_API_KEY (or --gemba-base-url for a local server)")
        return OpenAI(base_url=self.cfg.gemba_base_url, api_key=key or "EMPTY")

    def _chat(self, prompt: str) -> str:
        key = hashlib.sha1(f"{self.cfg.gemba_model}\n{prompt}".encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        kwargs = dict(model=self.cfg.gemba_model, temperature=0, max_tokens=1024,
                      messages=[{"role": "user", "content": prompt}])
        for attempt in range(6):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                with self._lock:
                    self._cache[key] = text
                return text
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # newer models reject some legacy parameters; adapt and retry
                if "max_tokens" in msg and "max_completion_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in msg and kwargs.pop("temperature", None) is not None:
                    continue
                log.warning("GEMBA call failed (attempt %d/6): %s", attempt + 1, msg[:200])
                time.sleep(min(2 ** attempt, 30))
        return ""

    def _save_cache(self):
        with self._lock:
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._cache_path)

    @staticmethod
    def _parse_score(text: str) -> float:
        m = re.search(r"\d+(?:\.\d+)?", text or "")
        return min(100.0, max(0.0, float(m.group()))) if m else float("nan")

    def score(self, srcs, refs, hyps):
        from tqdm import tqdm
        c = self.cfg
        sl, tl = lang_name(c.src_lang), lang_name(c.tgt_lang)

        def header(i):
            ref_block = ESA_REFERENCE_BLOCK.format(tgt_lang=tl, reference=refs[i]) if c.gemba_use_reference else ""
            return ESA_HEADER.format(src_lang=sl, tgt_lang=tl, source=srcs[i], target=hyps[i],
                                     reference_block=ref_block)

        def run(i):
            ann = self._chat(header(i) + ESA_ANNOTATE_INSTR)
            if not ann:
                return "", float("nan")
            score_text = self._chat(header(i) + ESA_SCORE_INSTR.format(annotations=ann))
            return ann, self._parse_score(score_text)

        try:
            with ThreadPoolExecutor(max_workers=c.gemba_workers) as ex:
                results = list(tqdm(ex.map(run, range(len(srcs))), total=len(srcs), desc="GEMBA-ESA"))
        finally:
            self._save_cache()
        self.extra["GEMBA-ESA_annotation"] = [a for a, _ in results]
        return {"GEMBA-ESA": [s for _, s in results]}

    def info(self):
        return {"model": self.cfg.gemba_model, "reference_used": self.cfg.gemba_use_reference,
                "scale": "0-100, higher is better", "prompts": "ESA-style two-step (see source)"}


METRICS = {m.key: m for m in (
    BLEUMetric, CHRFMetric, TERMetric, BERTScoreMetric,
    Comet22Metric, BLEURT20Metric, MetricX24Metric, GembaESAMetric,
)}
DEFAULT_ORDER = ["bleu", "chrf", "ter", "bertscore", "comet22", "bleurt20", "metricx24", "gemba_esa"]


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
def _mean(vals) -> float:
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def _clean(obj):
    """Make an object JSON-safe (NaN/inf -> null, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _versions() -> Dict[str, str]:
    out = {}
    for pkg in ("torch", "transformers", "sacrebleu", "bert-score", "unbabel-comet",
                "bleurt-pytorch", "openai", "pandas", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    out["python"] = sys.version.split()[0]
    return out


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ----------------------------------------------------------------------------------------
# Wrapper class
# ----------------------------------------------------------------------------------------
class MTMetricsRunner:
    DEFAULT_COLUMNS = {"srctxt": "srctxt", "reftxt": "reftxt", "tgttxt": "tgttxt",
                       "sys": "sys", "seg": "seg"}

    def __init__(self, metrics: Optional[Sequence[str]] = None, config: Optional[RunnerConfig] = None,
                 columns: Optional[Dict[str, str]] = None, include_text: bool = True):
        self.cfg = config or RunnerConfig()
        keys = list(metrics) if metrics and list(metrics) != ["all"] else list(DEFAULT_ORDER)
        unknown = [k for k in keys if k not in METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s) {unknown}; choose from {sorted(METRICS)}")
        self.metric_keys = keys
        self.cols = {**self.DEFAULT_COLUMNS, **(columns or {})}
        self.include_text = include_text
        self._reset_state()

    def _reset_state(self):
        self.df: Optional[pd.DataFrame] = None
        self.seg_scores: Dict[str, List[float]] = {}
        self.seg_extra: Dict[str, List[str]] = {}
        self.metric_info: Dict[str, dict] = {}
        self.agg: Dict[str, Dict[str, float]] = {}
        self.errors: Dict[str, str] = {}
        self.timing: Dict[str, float] = {}
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- data ---------------------------------------------------------------------------
    def load_testset(self, path: str, limit: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in self.cols.values() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing column(s) {missing} in {path}; found {list(df.columns)}")
        if limit:
            df = df.head(limit)
        df = df.reset_index(drop=True)
        for k in ("srctxt", "reftxt", "tgttxt"):
            n_empty = int((df[self.cols[k]].str.strip() == "").sum())
            if n_empty:
                log.warning("%d empty value(s) in column '%s'", n_empty, self.cols[k])
        self.input_path = str(path)
        return df

    # -- main loop ----------------------------------------------------------------------
    def run(self, df: pd.DataFrame, output_path: Optional[str] = None) -> dict:
        self._reset_state()
        self.df = df
        c = self.cols
        srcs, refs, hyps = (df[c[k]].tolist() for k in ("srctxt", "reftxt", "tgttxt"))
        systems = df[c["sys"]].tolist()

        self.groups: Dict[str, List[int]] = {"ALL": list(range(len(df)))}
        for name, idx in df.groupby(c["sys"], sort=False).indices.items():
            self.groups[str(name)] = [int(i) for i in idx]
        self.agg = {g: {} for g in self.groups}

        for key in self.metric_keys:
            cls = METRICS[key]
            t0 = time.time()
            log.info("=== %s ===", cls.name)
            metric = None
            try:
                metric = cls(self.cfg)
                metric.columns_data = {col: df[col].tolist() for col in df.columns}
                metric.prior_scores, metric.prior_info = self.seg_scores, self.metric_info
                scores = metric.score(srcs, refs, hyps)
                for out_name, vals in scores.items():
                    if len(vals) != len(df):
                        raise RuntimeError(f"{out_name}: got {len(vals)} scores for {len(df)} segments")
                    self.seg_scores[out_name] = [float(v) for v in vals]
                    for g, idx in self.groups.items():
                        cs = metric.corpus_score(out_name, [srcs[i] for i in idx],
                                                 [refs[i] for i in idx], [hyps[i] for i in idx])
                        self.agg[g][out_name] = cs if cs is not None else _mean([vals[i] for i in idx])
                    # after aggregation: sacreBLEU signatures are only available once a corpus was scored
                    self.metric_info[out_name] = {
                        "higher_is_better": metric.outputs.get(out_name, True),
                        "aggregation": metric.aggregation,
                        **metric.info(),
                    }
                self.seg_extra.update(metric.extra)
            except Exception as e:  # noqa: BLE001  - one failing metric must not kill the run
                log.exception("%s failed", cls.name)
                self.errors[cls.name] = f"{type(e).__name__}: {e}"
            finally:
                del metric
                _free_gpu()
            self.timing[cls.name] = round(time.time() - t0, 1)
            if output_path:
                self.save(output_path)  # checkpoint after every metric

        results = self.results()
        if output_path:
            self.save(output_path)
            log.info("Wrote %s", output_path)
        return results

    # -- output -------------------------------------------------------------------------
    def results(self) -> dict:
        df, c = self.df, self.cols
        segments = []
        for i in range(len(df)):
            seg = {"segment": df[c["seg"]].iloc[i], "system": df[c["sys"]].iloc[i]}
            if self.include_text:
                seg.update(source=df[c["srctxt"]].iloc[i], reference=df[c["reftxt"]].iloc[i],
                           target=df[c["tgttxt"]].iloc[i])
            seg["scores"] = {name: vals[i] for name, vals in self.seg_scores.items()}
            if self.seg_extra:
                seg["details"] = {name: vals[i] for name, vals in self.seg_extra.items()}
            segments.append(seg)

        return _clean({
            "meta": {
                "input": getattr(self, "input_path", None),
                "started_utc": self.started,
                "n_segments": len(df),
                "systems": [g for g in self.groups if g != "ALL"],
                "src_lang": self.cfg.src_lang,
                "tgt_lang": self.cfg.tgt_lang,
                "metrics_requested": self.metric_keys,
                "runtime_seconds": self.timing,
                "versions": _versions(),
                "config": asdict(self.cfg),
            },
            "metric_info": self.metric_info,
            "overall": self.agg.get("ALL", {}),
            "system_scores": {g: v for g, v in self.agg.items() if g != "ALL"},
            "errors": self.errors,
            "segments": segments,
        })

    def save(self, path: str):
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(self.results(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV test set")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--src-lang", required=True, help="source language code, e.g. en")
    p.add_argument("--tgt-lang", required=True, help="target language code, e.g. de")
    p.add_argument("--metrics", default="all",
                   help=f"comma-separated subset of {','.join(DEFAULT_ORDER)} (default: all)")
    p.add_argument("--limit", type=int, help="only score the first N rows (smoke test)")
    p.add_argument("--omit-text", action="store_true", help="do not copy source/reference/target into the JSON")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bleu-tokenize", default="auto")
    p.add_argument("--bertscore-model", default=None)
    p.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    p.add_argument("--bleurt-model", default="lucadiliello/BLEURT-20")
    p.add_argument("--metricx-model", default="google/metricx-24-hybrid-xl-v2p6")
    p.add_argument("--metricx-tokenizer", default=None)
    p.add_argument("--metricx-repo", default="./metricx")
    p.add_argument("--metricx-max-len", type=int, default=1536)
    p.add_argument("--metricx-batch-size", type=int, default=1)
    p.add_argument("--metricx-qe", action="store_true")
    p.add_argument("--gemba-model", default="gpt-4o")
    p.add_argument("--gemba-base-url", default=None)
    p.add_argument("--gemba-workers", type=int, default=8)
    p.add_argument("--gemba-use-reference", action="store_true")
    p.add_argument("--gemba-cache", default="gemba_cache.json")
    for col in ("srctxt", "reftxt", "tgttxt", "sys", "seg"):
        p.add_argument(f"--col-{col}", default=col, help=f"CSV column holding '{col}'")
    return p


def main(argv: Optional[Sequence[str]] = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = build_parser().parse_args(argv)
    cfg = RunnerConfig(
        src_lang=a.src_lang, tgt_lang=a.tgt_lang, batch_size=a.batch_size,
        bleu_tokenize=a.bleu_tokenize, bertscore_model=a.bertscore_model,
        comet_model=a.comet_model, bleurt_model=a.bleurt_model,
        metricx_model=a.metricx_model, metricx_tokenizer=a.metricx_tokenizer, metricx_repo=a.metricx_repo,
        metricx_max_len=a.metricx_max_len, metricx_batch_size=a.metricx_batch_size, metricx_qe=a.metricx_qe,
        gemba_model=a.gemba_model, gemba_base_url=a.gemba_base_url, gemba_workers=a.gemba_workers,
        gemba_use_reference=a.gemba_use_reference, gemba_cache=a.gemba_cache,
    )
    if a.device:
        cfg.device = a.device
    columns = {k: getattr(a, f"col_{k}") for k in ("srctxt", "reftxt", "tgttxt", "sys", "seg")}
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    runner = MTMetricsRunner(metrics, cfg, columns=columns, include_text=not a.omit_text)
    df = runner.load_testset(a.input, limit=a.limit)
    log.info("Loaded %d segments, device=%s", len(df), cfg.device)
    results = runner.run(df, output_path=a.output)

    print("\nOverall:")
    for name, val in results["overall"].items():
        print(f"  {name:<14} {val if val is None else round(val, 4)}")
    if results["errors"]:
        print("\nFailed metrics:")
        for name, err in results["errors"].items():
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
