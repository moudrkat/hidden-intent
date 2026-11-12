#!/bin/sh
# The whole thing, in order. About two hours on one 16 GB GPU; step 2 and the text-guard comparison take longest.
set -e
cd "$(dirname "$0")"
export HIDDEN_INTENT_WORK="${HIDDEN_INTENT_WORK:-work}"
sh pipeline/0_get_data.sh
python pipeline/1_rewrite.py
python pipeline/2_augment.py
python pipeline/3_train.py --anthropic 1
python pipeline/4_export.py
python pipeline/5_threshold.py
python eval/keyword_filter.py
python eval/long_context.py
python eval/czech_transfer.py
python eval/vs_text_guard.py
cp "$HIDDEN_INTENT_WORK"/results/*.json results/ 2>/dev/null || true
cp "$HIDDEN_INTENT_WORK"/probe.json results/ 2>/dev/null || true
