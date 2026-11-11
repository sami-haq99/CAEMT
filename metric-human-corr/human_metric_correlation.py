import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------- settings ----------
INPUT_GLOB    = "/home/sami/mmt-eval/MMT-Human-Eval/MMT-eval-analysis /campaign_analysis/qc_out/en-cs/scores/judgments_good_z.csv"        # annotation files
SCORES_FILE   = "/home/sami/mmt-eval/MMT-Human-Eval/MMT-eval-analysis /metric-human-corr/jina-results/aya_cs_results.csv"  # Final_Score file (1 system, 1 language, one row per segment)
LANG          = "cs"                # language of SCORES_FILE (value in the 'lang' column)
SYSTEM        = "aya"               # system of SCORES_FILE (value in the 'sys' column)
SEG_ID_OFFSET = 1                   # seg id = row position + offset (0 -> first data row is seg 0, 1 -> seg 1)
# ------------------------------

# 1. read and merge all annotation files
files = sorted(glob.glob(INPUT_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched {INPUT_GLOB}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Read {len(files)} files, {len(df)} rows")

if LANG is not None:
    df = df[df["lang"] == LANG]
    print(f"Kept {len(df)} rows for lang = {LANG}")

for c in ["score", "z", "seconds"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 2. aggregate: one row per system + segment
agg = (
    df.groupby(["sys", "seg"], as_index=False)
      .agg(
          n_records=("score", "size"),
          n_workers=("worker", "nunique"),
          avg_score=("score", "mean"),
          avg_z=("z", "mean"),
          avg_seconds=("seconds", "mean"),
          type=("type", "first"),
          lang=("lang", "first"),
          img=("img", "first"),
          srctxt=("srctxt", "first"),
          tgttxt=("tgttxt", "first"),
      )
      .sort_values(["sys", "seg"])
)

# 3. nested dictionary {system: {seg: {...}}} (all systems, kept in memory only)
result = {}
for r in agg.itertuples(index=False):
    result.setdefault(r.sys, {})[int(r.seg)] = {
        "n_records": int(r.n_records),
        "n_workers": int(r.n_workers),
        "avg_score": float(r.avg_score),
        "avg_z": float(r.avg_z),
        "avg_seconds": float(r.avg_seconds),
        "type": r.type,
        "img": r.img,
        "srctxt": r.srctxt,
        "tgttxt": r.tgttxt,
    }

# 4. keep only the system that the Final_Score file belongs to
if SYSTEM not in result:
    raise ValueError(f"System '{SYSTEM}' not found. Available systems: {sorted(result)}")
agg_sys = agg[agg["sys"] == SYSTEM]
print(f"System '{SYSTEM}': {len(agg_sys)} aggregated segments")

# 5. read Final_Score file; the row position is the segment id
fs = pd.read_csv(SCORES_FILE)
fs.columns = fs.columns.str.strip()
fs["seg"] = fs.index + SEG_ID_OFFSET
fs = fs[["seg", "Image", "Final_Score_K2", "Final_Score_k1", "Final_Score_k0"]]

# 6. match on segment id (aggregated data may lack some segments)
merged = agg_sys.merge(fs, on="seg", how="inner")

missing_in_agg = sorted(set(fs["seg"]) - set(agg_sys["seg"]))
not_in_scores  = sorted(set(agg_sys["seg"]) - set(fs["seg"]))
print(f"\nSegments in Final_Score file: {len(fs)}")
print(f"Segments missing from aggregated data: {len(missing_in_agg)}")
print(f"Segments used for correlation: {len(merged)}")
if not_in_scores:
    print(f"WARNING: {len(not_in_scores)} aggregated segment ids not found in Final_Score file "
          f"(check SEG_ID_OFFSET, LANG and SYSTEM): {not_in_scores[:10]}")

# alignment check: the image names should agree if the segment ids line up
img_match = (merged["img"] == merged["Image"]).mean()
print(f"Image name agreement after matching on seg id: {img_match:.1%}")
if img_match < 0.9:
    print("WARNING: low agreement; try the other SEG_ID_OFFSET (0 or 1), or check LANG.")

# 7. Pearson correlation
def pearson(d, col, score_col="Final_Score_K2"):
    d = d[[score_col, col]].dropna()
    if len(d) < 3 or d[score_col].nunique() < 2 or d[col].nunique() < 2:
        return np.nan, np.nan, len(d)
    r, p = pearsonr(d[score_col], d[col])
    return r, p, len(d)

rows = []
for col in ["Final_Score_K2", "Final_Score_k1", "Final_Score_k0"]:
    for s in ["avg_score", "avg_z"]:
        r, p, n = pearson(merged, s, score_col=col)
        rows.append({"system": SYSTEM, "lang": LANG, "metric": col,
                     "pearson_r": r, "p_value": p, "n_segments": n, "score_type": s})

corr = pd.DataFrame(rows)
print("\n", corr.to_string(index=False))