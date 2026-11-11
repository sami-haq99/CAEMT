import pandas as pd

lang = [ "de", "fr", "cs", "ar", "zh"]  # Replace with your actual languages

REF_DIR = "/home/sami/mmt-eval/MMT-Human-Eval/data/comute-translations/"  # Directory containing reference files

for lang in lang:
    csv_in = f"human-aggr-scores/aggregated_scores_en-{lang}.csv"          # your CSV file
    ref_in = f"{REF_DIR}ref.{lang}"      # your reference file
    ref_bad_in = f"{REF_DIR}ref_bad.{lang}"  # your bad reference file
    csv_out = f"human-aggr-scores/aggregated_scores_en-{lang}_with_ref.csv"

    # Read the reference file line by line (keep empty lines so numbering stays aligned)
    with open(ref_in, encoding="utf-8") as f:
        refs = [line.rstrip("\r\n") for line in f]

    # Read the bad reference file line by line (keep empty lines so numbering stays aligned)
    with open(ref_bad_in, encoding="utf-8") as f:
        ref_bads = [line.rstrip("\r\n") for line in f]

    # Read the CSV
    df = pd.read_csv(csv_in, encoding="utf-8")

    # seg is the 1-based line number in reference.txt
    def get_ref(seg):
        idx = int(seg) - 1
        if 0 <= idx < len(refs):
            return refs[idx]
        return ""  # seg out of range

    def get_ref_bad(seg):
        idx = int(seg) - 1
        if 0 <= idx < len(ref_bads):
            return ref_bads[idx]
        return ""  # seg out of range

    df["reftxt"] = df["seg"].apply(get_ref)
    df["ref_badtxt"] = df["seg"].apply(get_ref_bad)

    # Warn if any segment IDs don't have a matching line
    missing = (df["seg"] < 1) | (df["seg"] > len(refs))
    if missing.any():
        print(f"Warning: {missing.sum()} rows have a seg ID outside 1..{len(refs)}")

    # Write the result
    df.to_csv(csv_out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {csv_out}")