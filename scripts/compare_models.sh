#!/usr/bin/env bash
# 5-fold CV for each input stream x head variant, then a table sorted by end-to-end macro-F1.
# usage: scripts/compare_models.sh [frame person both]
set -euo pipefail
cd "$(dirname "$0")/.."
MODES=("$@"); [ $# -eq 0 ] && MODES=(frame person both)

# name | overrides  (best head settings from the frame-only sweep + the original default)
VARIANTS=(
  "tf2|model.n_layers=2"
  "tf1|model.n_layers=1 model.d_model=128 model.dropout=0.5"
  "notf|model.n_layers=0 model.d_model=128 model.dropout=0.5"
  "notf_mean|model.n_layers=0 model.d_model=128 model.dropout=0.5 model.behavior_pool=mean"
)

for mode in "${MODES[@]}"; do
  for v in "${VARIANTS[@]}"; do
    name=${v%%|*}; args=()
    for kv in ${v#*|}; do args+=(--set "$kv"); done
    echo "cv $mode/$name ..."
    .venv/bin/abd --set "model.input=$mode" "${args[@]}" cv --name "cmp_${mode}_${name}" --folds 5 >/dev/null 2>&1 \
      || echo "  FAILED (person features missing? run extract with streams [frame, person])"
  done
done

.venv/bin/python - <<'PY'
import json, glob
rows = []
for f in glob.glob("outputs/runs/cmp_*/cv_summary.json"):
    d = json.load(open(f)); g = lambda k: (d[k]["mean"], d[k]["std"])
    rows.append((f.split("/")[2][4:], g("anomaly.roc_auc"), g("anomaly.balanced_accuracy"),
                 g("behavior.accuracy"), g("end_to_end.accuracy"), g("end_to_end.macro_f1")))
rows.sort(key=lambda r: -r[5][0])
print(f"\n{'model':<22}{'AUC':>14}{'bal.acc':>14}{'behavior':>14}{'e2e acc':>14}{'e2e F1':>14}")
for r in rows:
    print(f"{r[0]:<22}" + "".join(f"{m:>8.3f}±{s:.3f}" for m, s in r[1:]))
print("\nbest by end-to-end macro-F1:", rows[0][0] if rows else "-")
PY
