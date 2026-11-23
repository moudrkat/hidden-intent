"""Step 2d: the rewriter softens some attacks while shortening them ('Create harmless, non-persistent system service code.').
Keep a short attack only if Qwen3Guard-Gen-4B still calls it Unsafe or Controversial. Short benign rows are not filtered
(the guard fears security words too; that is exactly what the probe has to unlearn). Writes guard_data_aug3f.json."""
import re, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import args, device, load, read_json, write_json

a = args(__doc__, guard=(str, "Qwen/Qwen3Guard-Gen-4B", "judge"))
rows = read_json(a.work / "guard_data_aug3.json")
tok, m = load(a.guard)
keep, drop = [], []
for i, r in enumerate(rows):
    if not (r["lang"] == "short" and r["kind"] == "attack"):
        keep.append(r); continue
    ids = tok(tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True),
              return_tensors="pt").input_ids.to(device())
    with torch.no_grad(): g = m.generate(ids, max_new_tokens=24, do_sample=False)
    s = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    (keep if s and s.group(1) != "Safe" else drop).append(r)
    if len(drop) + sum(x["lang"] == "short" and x["kind"] == "attack" for x in keep) % 100 == 0: print("judged", i, flush=True)
write_json(a.work / "guard_data_aug3f.json", keep)
sa = [r for r in keep if r["lang"] == "short" and r["kind"] == "attack"]
print("short attacks kept:", len(sa), "dropped:", len(drop), {s: sum(r["split"] == s for r in sa) for s in ("train", "val", "test")})
print("KEPT e.g.:", *[r["text"][:110] for r in sa[:6]], sep="\n  ")
print("DROPPED e.g.:", *[r["text"][:110] for r in drop[:6]], sep="\n  ")
