#!/usr/bin/env bash
# Unzip the Ano-AAD Kaggle archive and lay it out as <dest>/normal and <dest>/abnormal/<Class>.
# usage: scripts/prepare_ano_aad.sh archive.zip data/Ano-AAD
set -euo pipefail
ZIP=${1:?archive.zip path}
DEST=${2:-data/Ano-AAD}
RAW="$(dirname "$DEST")/raw"

mkdir -p "$RAW" "$DEST"
unzip -q -n "$ZIP" -d "$RAW"
SRC="$RAW/Anomalous Action Detection Dataset( Ano-AAD)"
[ -d "$DEST/abnormal" ] || mv "$SRC/abnormal class" "$DEST/abnormal"
[ -d "$DEST/normal" ]   || mv "$SRC/normal class - normal videos/Normal Videos" "$DEST/normal"
rm -rf "$RAW"
echo "$(find "$DEST" -name '*.mp4' | wc -l | tr -d ' ') videos in $DEST"
