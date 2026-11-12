#!/bin/sh
# The raw data, pinned. Nothing derived from it is in this repo: you build it with the next steps.
#   PurpleLlama CyberSecEval (MIT): MITRE attack prompts and their benign look-alikes (FRR set)
#   CodeAlpaca-20k: ordinary coding requests (the HF card says CC BY 4.0)
set -e
WORK="${HIDDEN_INTENT_WORK:-work}"
mkdir -p "$WORK/data"
cd "$WORK/data"
if [ ! -d PurpleLlama ]; then
  git clone --filter=blob:none --no-checkout https://github.com/meta-llama/PurpleLlama.git
  cd PurpleLlama
  git sparse-checkout set CybersecurityBenchmarks/datasets/mitre CybersecurityBenchmarks/datasets/mitre_frr
  git checkout -q f64eebd
  cd ..
fi
[ -f code_alpaca.json ] || curl -sL -o code_alpaca.json \
  https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json
python3 - <<'PY'
import json
print("PurpleLlama MITRE:", len(json.load(open("PurpleLlama/CybersecurityBenchmarks/datasets/mitre/mitre_benchmark_100_per_category_with_augmentation.json"))))
print("PurpleLlama FRR:  ", len(json.load(open("PurpleLlama/CybersecurityBenchmarks/datasets/mitre_frr/mitre_frr.json"))))
print("CodeAlpaca:       ", len(json.load(open("code_alpaca.json"))))
PY
