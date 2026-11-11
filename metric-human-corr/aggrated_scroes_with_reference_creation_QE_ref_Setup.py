import pandas as pd

langs = ["de", "fr", "cs", "ar", "zh"]  # Replace with your actual languages

REF_DIR = "/home/sami/mmt-eval/MMT-Human-Eval/data/comute-translations/"

for lang in langs:
    csv_in = f"human-aggr-scores/aggregated_scores_en-{lang}.csv"
    ref_in = f"{REF_DIR}ref.{lang}"
    ref_bad_in = f"{REF_DIR}ref_bad.{lang}"
    csv_out = f"human-aggr-scores/aggregated_scores_en-{lang}_with_ref_QE_fix.csv"

    # Read line by line (keep empty lines so numbering stays aligned)
    with open(ref_in, encoding="utf-8") as f:
        refs = [line.rstrip("\r\n") for line in f]

    with open(ref_bad_in, encoding="utf-8") as f:
        ref_bads = [line.rstrip("\r\n") for line in f]

    df = pd.read_csv(csv_in, encoding="utf-8")

    # seg is the 1-based line number in the reference files
    def get_line(lines, seg):
        idx = int(seg) - 1
        if 0 <= idx < len(lines):
            return lines[idx]
        return ""  # seg out of range

    df["reftxt"] = df["seg"].apply(lambda s: get_line(refs, s))
    df["ref_badtxt"] = df["seg"].apply(lambda s: get_line(ref_bads, s))

    # Warn if any segment IDs don't have a matching line
    missing_ref = (df["seg"] < 1) | (df["seg"] > len(refs))
    missing_bad = (df["seg"] < 1) | (df["seg"] > len(ref_bads))
    if missing_ref.any():
        print(f"Warning [{lang}]: {missing_ref.sum()} rows have a seg ID outside 1..{len(refs)} (ref)")
    if missing_bad.any():
        print(f"Warning [{lang}]: {missing_bad.sum()} rows have a seg ID outside 1..{len(ref_bads)} (ref_bad)")

    # For BAD system rows, replace tgttxt with the matching ref_bad line
    is_bad = df["sys"].astype(str).str.strip() == "BAD"
    replace_mask = is_bad & ~missing_bad   # don't overwrite with "" if seg is out of range
    df.loc[replace_mask, "tgttxt"] = df.loc[replace_mask, "ref_badtxt"]
    print(f"[{lang}] Replaced tgttxt for {replace_mask.sum()} BAD rows "
          f"({is_bad.sum()} BAD rows total)")

    df.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {csv_out}")