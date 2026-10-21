#!/usr/bin/env bash
# Build dist/abd_demo.zip for the Windows+NVIDIA demo machine.
# usage: scripts/pack_demo.sh <run_dir with best.pt> <video> [video ...]
#   the videos are copied as live-demo inputs AND pre-rendered here as a fallback.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:?run dir, e.g. outputs/runs/baseline}; shift
[ -f "$RUN/best.pt" ] || { echo "no $RUN/best.pt"; exit 1; }
B=dist/abd_demo
rm -rf "$B" dist/abd_demo.zip
mkdir -p "$B/code" "$B/models" "$B/weights/torch/hub/checkpoints" "$B/samples/inputs" "$B/samples/rendered"

git archive HEAD | tar -x -C "$B/code"
[ -z "$(git status --porcelain -- src configs pyproject.toml)" ] || echo "warning: uncommitted code changes are NOT in the bundle"
cp demo/setup.ps1 demo/demo.ps1 demo/setup.bat demo/demo.bat demo/README_DEMO.md "$B/"
cp "$RUN/best.pt" "$B/models/"
for f in metrics.json history.json; do [ -f "$RUN/$f" ] && cp "$RUN/$f" "$B/models/"; done
for d in outputs/runs/cv_*; do [ -f "$d/cv_summary.json" ] && cp "$d/cv_summary.json" "$B/models/$(basename "$d")_summary.json"; done
cp ~/.cache/torch/hub/checkpoints/s3d-*.pth "$B/weights/torch/hub/checkpoints/"
cp yolo11n.pt "$B/weights/"

for v in "$@"; do
  # every class folder has 01.mp4, 02.mp4 ... -> prefix with the class folder
  name=$(basename "$(dirname "$v")")_$(basename "${v%.*}")
  name=${name// /_}
  cp "$v" "$B/samples/inputs/$name.mp4"
  echo "pre-rendering $name ..."
  .venv/bin/abd --set features.batch_size=2 infer --ckpt "$RUN/best.pt" --video "$B/samples/inputs/$name.mp4" \
      --render "$B/samples/rendered/${name}_result.mp4" 2>/dev/null | grep -v '^rendered -> ' > "$B/samples/rendered/${name}_result.json"
done

(cd dist && zip -qr abd_demo.zip abd_demo)
echo "dist/abd_demo.zip ($(du -h dist/abd_demo.zip | cut -f1))"
