#!/usr/bin/env bash
# Download HAM10000 from Kaggle into data/.
# Requires: kaggle CLI configured (~/.kaggle/kaggle.json with API token).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

kaggle datasets download kmader/skin-lesion-analysis-toward-melanoma-detection -p data/
unzip -n data/skin-lesion-analysis-toward-melanoma-detection.zip -d data/

echo "Done. Verify: ls data/HAM10000_metadata.csv"
