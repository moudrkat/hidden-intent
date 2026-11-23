# hidden-intent

A linear probe that reads a coding model's intent from the request, before it writes a line: malicious coding
requests vs benign ones, on the residual stream of Qwen2.5-Coder-1.5B-Instruct. Google's and Anthropic's
production probe recipes, at the scale of one 16 GB GPU.

> Work in progress.

## Quickstart

The trained probe is in [`results/probe.json`](results/probe.json): one vector and a bias. Watch it read
requests live in [brainscope](https://github.com/moudrkat/brainscope):

```bash
pip install brainscope
brainscope --model Qwen/Qwen2.5-Coder-1.5B-Instruct &

curl -sLO https://raw.githubusercontent.com/moudrkat/hidden-intent/main/results/probe.json
jq '{probes: [{name: "intent", vector: .direction, bias: .bias,
               hidden_state: .hidden_state, threshold: 0, ema: 0.2}]}' probe.json \
  | curl -s localhost:8010/probes -H 'Content-Type: application/json' -d @-
```

Then ask for a coding task in the chat at localhost:8010; the probe strip shows the request turn red where
the intent shows up.

Train it yourself (about two hours on one 16 GB GPU, everything lands in `work/`):

```bash
pip install -r requirements.txt
sh run_all.sh
```

## Results

Missed attacks on the test split (threshold set on validation: 1 in 100 normal English requests flagged):

| | English | Czech | story-like | in three steps | cost |
|---|---|---|---|---|---|
| this probe | 0 % | 2 % | 5 % | 6 % | 0.013 ms on top of the model's forward pass |
| a keyword list | 48 % | 79 % | | | |
| Qwen3Guard-Gen-4B | 50 % | 58 % | 93 % | 91 % | 173 ms |

False alarms of this probe: normal coding requests 0.7 %, benign requests that use security words
("explain how ransomware works", "detect a keylogger in our logs") 4 %, one-sentence benign questions 12 %.
The test topics never appear in training.

The first version read words: it flagged 99 % of benign requests with security vocabulary. Steps `2b`–`2d`
fixed that; [`results/versions.json`](results/versions.json) has the three versions side by side.

Known limit: the probe averages over the whole message, so a short request after a long paste of code gets
diluted. Read the user's own text, or use the Rolling Attention probe (trained by `pipeline/3_train.py`).
All numbers, including that one: [`results/`](results/).

## What is not here

The data. The rewritten attacks are complete malicious requests reworded to get past keyword filters; you
build them yourself with `pipeline/0–2` from PurpleLlama CyberSecEval (MIT) and CodeAlpaca-20k. The result
files hold labels and scores, not texts.

Built on Kramar et al. (Google DeepMind, 2026), Anthropic's Constitutional Classifiers++ (2026) and Meta's
CyberSecEval.

MIT © Kateřina Fajmanová
