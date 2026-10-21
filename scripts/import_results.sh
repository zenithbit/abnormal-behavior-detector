#!/usr/bin/env bash
# Unpack abd_results.zip from Colab into outputs/ (features, tracks, runs, manifest, splits).
# usage: scripts/import_results.sh abd_results.zip
set -euo pipefail
ZIP=${1:?abd_results.zip path}
cd "$(dirname "$0")/.."
OUT=outputs
if [ -d "$OUT/features" ] || [ -d "$OUT/runs" ]; then
  BACKUP="$OUT.bak.$(date +%Y%m%d-%H%M%S)"
  echo "existing $OUT -> $BACKUP"
  mv "$OUT" "$BACKUP"
fi
mkdir -p "$OUT"
unzip -q "$ZIP" -d "$OUT"
echo "features: $(find "$OUT/features" -name '*.npz' | wc -l | tr -d ' ') videos"
echo "runs:     $(ls "$OUT/runs" 2>/dev/null | tr '\n' ' ')"
for f in "$OUT"/runs/*/cv_summary.json; do [ -e "$f" ] && echo "cv summary: $f"; done
true
