"""Step 1: one rewriter for all three classes, so style cannot give the class away.

Every source prompt becomes the short casual request a developer would type into a coding assistant (English), and
val/test also get a Czech version. Classes: attack (MITRE), hard negative (FRR: sounds risky, is benign), normal
(CodeAlpaca). Split 60/20/20 by group, so near-duplicates never straddle train and test. Writes guard_data.json."""
import json, random, re, sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import REWRITER, args, device, load, write_json

a = args(__doc__, n_normal=(int, 1500, "CodeAlpaca requests to use"), batch=(int, 48, "rewrite batch size"))
random.seed(0)
P = a.work / "data/PurpleLlama/CybersecurityBenchmarks/datasets"
src = []
for d in json.load(open(P / "mitre/mitre_benchmark_100_per_category_with_augmentation.json")):
    src.append({"src": d["base_prompt"], "y": 1, "kind": "attack", "group": "mitre:" + d["ttp_id_name_mapping"]["TTP_ID"]})
for i, d in enumerate(json.load(open(P / "mitre_frr/mitre_frr.json"))):
    src.append({"src": d["mutated_prompt"], "y": 0, "kind": "hardneg", "group": f"frr:{i // 10}"})
alpaca = json.load(open(a.work / "data/code_alpaca.json"))
random.shuffle(alpaca)
for i, d in enumerate(alpaca[:a.n_normal]):
    src.append({"src": d["instruction"] + (" " + d["input"] if d["input"] else ""), "y": 0, "kind": "normal", "group": f"alpaca:{i // 10}"})
for k in ["attack", "hardneg", "normal"]:
    groups = sorted({r["group"] for r in src if r["kind"] == k})
    random.shuffle(groups)
    n = len(groups)
    split = {g: ("train" if i < .6 * n else "val" if i < .8 * n else "test") for i, g in enumerate(groups)}
    for r in src:
        if r["kind"] == k:
            r["split"] = split[r["group"]]

tok, model = load(REWRITER)
tok.padding_side = "left"
INSTR = {
    "en": "You are helping build a dataset for a safety classifier. Rewrite the following request as the short, casual message a developer would type into an AI coding assistant: 1-2 sentences, plain words, no role-play framing, no preamble, keep the technical intent exactly (do not soften or sharpen it). Output only the rewritten message.\n\nRequest:\n",
    "cs": "Pomáháš vytvořit dataset pro bezpečnostní klasifikátor. Přepiš následující požadavek jako krátkou neformální zprávu, kterou by vývojář napsal česky do AI asistenta pro programování: 1-2 věty, běžná slova, žádné hraní rolí, žádný úvod, technický záměr zachovej přesně (nezmírňuj ani nezostřuj). Vypiš jen přepsanou zprávu.\n\nPožadavek:\n",
}
REFUSE = re.compile(r"(I can'?t|I cannot|I'm sorry|I am sorry|nemohu|Omlouvám|can't help|cannot assist)", re.I)
jobs = [(r, "en") for r in src] + [(r, "cs") for r in src if r["split"] != "train"]
for i in range(0, len(jobs), a.batch):
    b = jobs[i:i + a.batch]
    texts = [tok.apply_chat_template([{"role": "user", "content": INSTR[l] + r["src"][:1500]}], tokenize=False,
                                     add_generation_prompt=True) for r, l in b]
    enc = tok(texts, return_tensors="pt", padding=True).to(device())
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=120, do_sample=False)
    for (r, l), row in zip(b, g):
        t = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().strip('"')
        r[l] = None if (REFUSE.search(t) or len(t) < 15) else t
    print(f"rewrite {i + len(b)}/{len(jobs)}", flush=True)
out, drop = [], Counter()
for r in src:
    for l in ["en", "cs"]:
        if l not in r:
            continue
        if r[l] is None:
            drop[(r["kind"], l)] += 1
            continue
        out.append({"text": r[l], "lang": l, "y": r["y"], "kind": r["kind"], "group": r["group"], "split": r["split"]})
print("dropped (rewriter refused):", dict(drop))
print("rows:", len(out), dict(Counter((o["kind"], o["lang"]) for o in out)))
write_json(a.work / "guard_data.json", out, indent=0)
