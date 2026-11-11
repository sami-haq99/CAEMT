#!/usr/bin/env bash
# Run reference-based and QE metrics over one CSV per language pair.


#SBATCH --gres=gpu:a100
#SBATCH -p compute
#SBATCH -J metrics
#SBATCH --mem=100000
#SBATCH -t 23:59:59
#SBATCH -o metrics-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie

set -u
PAIRS=(en-de en-zh en-cs en-ar en-fr)
B_DIR="/home/shaq/metrics"
DATA_DIR="${DATA_DIR:-/home/shaq/mmss/data/comute}"
OUT_DIR="${OUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"
STAGES="${STAGES:-ref qe}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

REF_PY="${REF_PY:-B_DIR/.venv-mtmetrics/bin/python}"
REF_RUNNER="${REF_RUNNER:-mt_metrics_runner.py}"
QE_PY="${QE_PY:-$B_DIR/.venv-mtqe/bin/python}"
QE_RUNNER="${QE_RUNNER:-mtqe_metrics_runner.py}"

BERTSCORE_MODEL="${BERTSCORE_MODEL:-}"

# Metrics for each stage
REF_METRICS="bleu,chrf,ter,comet22,bertscore,bleurt20,metricx24"
QE_METRICS="cometkiwi,xcomet,transquest,metricx23_qe"




csv_for() {
    echo "$DATA_DIR/aggregated_scores_$1_with_ref_QE_fix.csv"
}

mkdir -p "$OUT_DIR" "$LOG_DIR"

for pair in "${PAIRS[@]}"; do
    src="${pair%%-*}"
    tgt="${pair##*-}"
    csv="$(csv_for "$pair")"

    if [ ! -f "$csv" ]; then
        echo "!! $csv not found - skipping $pair" >&2
        continue
    fi

    for stage in $STAGES; do
        out="$OUT_DIR/${pair}_${stage}.json"

        if [ "$SKIP_EXISTING" = "1" ] && [ -s "$out" ]; then
            echo "-- $out exists, skipping"
            continue
        fi

        extra=()

        if [ "$stage" = "ref" ]; then
            py="$REF_PY"
            runner="$REF_RUNNER"
            metrics="$REF_METRICS"

            extra+=(--metrics "$metrics")

            if [ -n "$BERTSCORE_MODEL" ]; then
                extra+=(--bertscore-model "$BERTSCORE_MODEL")
            fi

        elif [ "$stage" = "qe" ]; then
            py="$QE_PY"
            runner="$QE_RUNNER"
            metrics="$QE_METRICS"

            extra+=(--metrics "$metrics")

        else
            echo "!! Unknown stage: $stage" >&2
            continue
        fi

        echo "=== $pair [$stage] ($src -> $tgt) $(date '+%F %T') ==="
        echo "Metrics: $metrics"

        "$py" "$runner" \
            --input "$csv" \
            --output "$out" \
            --src-lang "$src" \
            --tgt-lang "$tgt" \
            "${extra[@]}" \
            2>&1 | tee "$LOG_DIR/${pair}_${stage}.log"

    done
done

# ---- Summary ----------------------------------------------------------------
echo
echo "=== Summary ==="

"$REF_PY" - "$OUT_DIR" <<'PYEOF'
import glob, json, sys

files = sorted(glob.glob(sys.argv[1] + "/*.json"))

if not files:
    print("no result files found")

for f in files:
    try:
        with open(f, encoding="utf-8") as fp:
            r = json.load(fp)

        ok = list(r.get("overall", {}))
        bad = r.get("errors", {})

        print(
            f"{f}: {len(ok)} score columns"
            + (f" | FAILED: {', '.join(bad)}" if bad else " | no errors")
        )

    except Exception as e:
        print(f"{f}: ERROR reading output: {e}")
PYEOF
