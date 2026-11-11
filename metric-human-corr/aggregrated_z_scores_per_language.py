import glob
import json
import pandas as pd

# ---------- settings ----------
l_pair = "en-fr"                     # language pair (used for filtering)
BASE_DIR   = f"/home/sami/mmt-eval/MMT-Human-Eval/MMT-eval-analysis/campaign_analysis/qc_out/{l_pair}/scores"  # directory containing all judgment files
INPUT_GLOB = f"{BASE_DIR}/judgments_good_z.csv"          # e.g. "data/en-*.csv"
BY_LANG    = False                 # True -> group by lang + sys + seg, dict becomes {lang: {sys: {seg: ...}}}
# ------------------------------

OUTPUT_DIR = "human-aggr-scores"
# 1. read and merge all files
files = sorted(glob.glob(INPUT_GLOB))
if not files:
    raise FileNotFoundError(f"No files matched {INPUT_GLOB}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"Read {len(files)} files, {len(df)} rows")

for c in ["score", "z", "seconds"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 2. warn if segment IDs may collide across languages
if not BY_LANG and (df.groupby(["sys", "seg"])["lang"].nunique() > 1).any():
    print("WARNING: some (sys, seg) pairs occur in more than one language. "
          "Set BY_LANG = True to keep them separate.")

# 3. aggregate: one row per system + segment
keys = ["lang", "sys", "seg"] if BY_LANG else ["sys", "seg"]

agg = (
    df.groupby(keys, as_index=False)
      .agg(
          n_records=("score", "size"),     # number of rows combined
          n_workers=("worker", "nunique"), # distinct annotators
          avg_score=("score", "mean"),
          avg_z=("z", "mean"),
          avg_seconds=("seconds", "mean"),
          type=("type", "first"),
          img=("img", "first"),
          srctxt=("srctxt", "first"),
          tgttxt=("tgttxt", "first"),
          **({} if BY_LANG else {"lang": ("lang", "first")}),
      )
      .sort_values(keys)
)

# 4. nested dictionary
def make_entry(r):
    return {
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

result = {}
for r in agg.itertuples(index=False):
    node = result
    if BY_LANG:
        node = node.setdefault(r.lang, {})
    node = node.setdefault(r.sys, {})
    node[int(r.seg)] = make_entry(r)

# 5. save
agg.to_csv(f"{OUTPUT_DIR}/aggregated_scores_{l_pair}.csv", index=False, encoding="utf-8-sig")
with open(f"{OUTPUT_DIR}/aggregated_scores_{l_pair}.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# quick checks
print(agg[["sys", "seg", "n_records", "avg_score", "avg_z"]].head(10))
print("Systems:", list(result) if not BY_LANG else {l: list(s) for l, s in result.items()})
print("Total rows in:", len(df), "| sum of n_records:", agg["n_records"].sum())  # these should match