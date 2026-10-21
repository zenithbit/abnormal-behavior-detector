#!/usr/bin/env bash
# Package the committed code (no data, no outputs) for upload to Google Drive: dist/abd_code.zip
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
git archive --format=zip -o dist/abd_code.zip HEAD
echo "dist/abd_code.zip ($(du -h dist/abd_code.zip | cut -f1)) from commit $(git rev-parse --short HEAD)"
[ -z "$(git status --porcelain -- src configs scripts pyproject.toml)" ] || echo "warning: uncommitted changes are NOT included"
