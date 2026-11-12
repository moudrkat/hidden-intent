# hidden-intent

**Train a probe that reads a coding model's intent before it writes a line.**

A linear probe on the residual stream of Qwen2.5-Coder-1.5B-Instruct that tells malicious coding requests from
benign ones, from the user's message alone, before the first token of the answer. It follows the production
recipes of Google (Gemini) and Anthropic (Constitutional Classifiers++) at the scale of one 16 GB GPU, and every
number below comes out of `run_all.sh`.

The exported probe is one vector and a bias, and it loads into
[brainscope](https://github.com/moudrkat/brainscope) as it is: the probe strip then reads every request live.

> **Status: research code.** It is a working recipe and an honest set of measurements on one small model and one
> dataset, not a finished safety product.

## Results

One run of `run_all.sh` (Qwen2.5-Coder-1.5B-Instruct, hidden_states[14], one 16 GB GPU). The files are in
[`results/`](results/): labels and scores, no request texts.

**Test split, by form of the request** (AUC; missed attacks at the threshold picked on validation):

| probe | English | Czech | story-like | three steps |
|---|---|---|---|---|
| keywords → TF-IDF + logistic regression | 0.999 · 6 % | 0.994 · 26 % | 0.987 · 19 % | 0.990 · 21 % |
| linear mean (exported) | 1.000 · 0 % | 0.998 · 5 % | 0.994 · 11 % | 0.995 · 12 % |
| Rolling Attention | 1.000 · 1 % | 0.998 · 7 % | 0.997 · 9 % | 0.996 · 16 % |
| all layers (Anthropic) | 1.000 · 3 % | 0.996 · 6 % | 0.996 · 9 % | 0.993 · 10 % |

**The exported probe at its deployment threshold** (1 in 100 normal English requests flagged on validation):

| | attacks caught | normal flagged | hard negatives flagged |
|---|---|---|---|
| probe, English | 100 % | 2 % | 2 % |
| probe, Czech | 99 % | 2 % | 5 % |
| keyword list, English | 52 % | 1 % | 79 % |
| keyword list, Czech | 21 % | 0 % | 25 % |

**Against a text classifier** (missed attacks, verdict "Unsafe"; "Unsafe or Controversial" in brackets):

| | English | Czech | story-like | three steps | cost per request |
|---|---|---|---|---|---|
| Qwen3Guard-Gen-0.6B | 45 % (30 %) | 29 % (18 %) | 94 % (78 %) | 83 % (74 %) | 107 ms |
| Qwen3Guard-Gen-4B | 50 % (32 %) | 58 % (27 %) | 93 % (75 %) | 91 % (74 %) | 173 ms |
| the probe | 0 % | 5 % | 11 % | 12 % | 0.013 ms on top of the coding model's own 11.7 ms forward pass |

Qwen3Guard is a general safety classifier, not trained for this task; the comparison shows what an
off-the-shelf text guard does here, not how good it can be.

**Long context: the mean does not survive it.** Each English test request after N tokens of real Python code,
in one user message. Missed attacks / false alarms on normal requests:

| tokens before the request | 0 | 512 | 2 048 | 8 192 |
|---|---|---|---|---|
| linear mean (exported) | 0 % / 1 % | 100 % / 0 % | 100 % / 0 % | 100 % / 0 % |
| same direction, best 10-token window | 4 % / 0 % | 12 % / 2 % | 16 % / 6 % | 13 % / 28 % |
| Rolling Attention | 1 % / 0 % | 8 % / 1 % | 14 % / 2 % | 15 % / 5 % |

Averaged over the whole message, a short request drowns in the code around it. Read the user's text on its
own, or use Rolling Attention, which was built for this (Google trains it on long contexts; this one saw only
short requests).

**Trained in English only, tested in Czech** (10 bootstrap resamples, median missed Czech attacks): layer 10 7 %,
layer 14 20 %, ranging 4–52 % between resamples at layer 14. Czech in the training data is what makes the
Czech numbers above stable.

## Run it

```bash
pip install -r requirements.txt
sh run_all.sh            # about two hours on one 16 GB GPU; everything lands in work/
```

Or step by step:

| step | script | what it does |
|---|---|---|
| 0 | `pipeline/0_get_data.sh` | PurpleLlama CyberSecEval (MIT) at a pinned commit, and CodeAlpaca-20k |
| 1 | `pipeline/1_rewrite.py` | one rewriter (Qwen3-4B-Instruct) turns every prompt of every class into a short casual request, English, plus Czech for val/test |
| 2 | `pipeline/2_augment.py` | the reformulations a probe must survive: a story-like version without security words, Czech, and the request split into three innocent-looking steps |
| 3 | `pipeline/3_train.py` | the probes: Google's linear-mean and Rolling Attention, Anthropic's all-layers probe (`--anthropic 1`), and a TF-IDF baseline |
| 4 | `pipeline/4_export.py` | the linear-mean probe as one direction and a bias: `probe.json` |
| 5 | `pipeline/5_threshold.py` | the threshold: 1 in 100 normal requests flagged, set on the validation split |
| eval | `eval/*.py` | keyword filter, long context, English-to-Czech transfer, a text classifier (Qwen3Guard) |

Three classes, split 60/20/20 by group so near-duplicates never straddle train and test:

- **attack**: MITRE ATT&CK requests from CyberSecEval
- **hard negative**: CyberSecEval's false-refusal set, requests that sound risky and are benign
- **normal**: ordinary coding requests from CodeAlpaca

All three go through the same rewriter, so the style of a request cannot give its class away.

## Load it into brainscope

```bash
jq '{probes: [{name: "intent", vector: .direction, bias: .bias, hidden_state: .hidden_state,
               threshold: .threshold, ema: 0.2}]}' work/probe.json \
  | curl -s localhost:8010/probes -H 'Content-Type: application/json' -d @-
```

## What is not here, on purpose

- **The data.** The rewritten attacks are complete malicious requests, reworded to get past keyword filters. You
  build them yourself with steps 0–2; nothing derived from them is committed, and the score files keep labels and
  scores, not texts.
- **The papers' scale.** 10 seeds instead of 100, minibatches, the prompt only (no responses), one model.

## Data and licences

- PurpleLlama CyberSecEval datasets: MIT ([meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama), commit `f64eebd`).
- CodeAlpaca-20k: the Hugging Face card says CC BY 4.0; the original repository may restrict use to research.
- Qwen2.5-Coder-1.5B-Instruct, Qwen3-4B-Instruct-2507, Qwen3Guard-Gen: Apache-2.0.
- The long-context filler is the Python standard library of the interpreter that runs the test (PSF licence).

## Standing on shoulders

- Kramar et al. (Google DeepMind), probes for Gemini's cyber misuse detection, 2026: the linear-mean and Rolling
  Attention probes, the threshold objective, the long-context test, the reformulation training data.
- Anthropic, Constitutional Classifiers++, 2026: the all-layers probe with a sliding-window mean and a
  softmax-weighted loss.
- Meta, CyberSecEval (PurpleLlama): the attack and false-refusal prompts.

## License

MIT © Kateřina Fajmanová
