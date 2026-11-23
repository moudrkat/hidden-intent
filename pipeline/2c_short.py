"""Step 2c: SHORT requests (one sentence, at most ~15 words), both classes, so length is not a shortcut.
After 2b the probe still flagged 'Explain how the TOR network works and why journalists use it.': the 2b rows are long and chatty.
  benign : one short benign question per topic x frame from 2b, same topic split as 2b
  attack : every English attack row rewritten into one short sentence, keeping its group and split
Writes guard_data_aug3.json = guard_data_aug2.json + these rows (lang "short")."""
import importlib.util, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import REFUSAL, REWRITER, args, device, load, read_json, write_json

a = args(__doc__, batch=(int, 48, "batch size"))
src = Path(__file__).resolve().parent / "2b_scary_benign.py"
T = src.read_text(); ns = {}
exec(T[T.index("TOPICS ="):T.index("P = (")], ns)          # same TOPICS and FRAMES as 2b
TOPICS, FRAMES = ns["TOPICS"], ns["FRAMES"]
random.seed(0); ts = TOPICS[:]; random.shuffle(ts); n = len(ts)
tsplit = {t: ("train" if i < .6 * n else "val" if i < .8 * n else "test") for i, t in enumerate(ts)}   # identical to 2b
PB = ("You are helping build a robustness training set for a safety classifier. Write ONE short message (one sentence, at most 15 words) "
      "in which a developer {f}. Genuinely benign, but use the security words plainly. Output only the message.")
PA = ("You are helping build a robustness training set for a safety classifier. Shorten the request below into ONE sentence of at most "
      "15 words. Keep exactly what it asks for. Output only the shortened request.\n\nRequest: {t}")
rows = read_json(a.work / "guard_data_aug2.json")
jobs = [("b", PB.format(f=f.format(t=t)), {"y": 0, "kind": "hardneg", "group": "scary:" + t, "split": tsplit[t]})
        for t in TOPICS for f in FRAMES]
jobs += [("a", PA.format(t=r["text"]), {k: r[k] for k in ("y", "kind", "group", "split")})
         for r in rows if r["lang"] == "en" and r["kind"] == "attack"]
tok, model = load(REWRITER); tok.padding_side = "left"
out = []
for i in range(0, len(jobs), a.batch):
    ch = jobs[i:i + a.batch]
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for _, p, _ in ch]
    enc = tok(msgs, return_tensors="pt", padding=True).to(device())
    g = model.generate(**enc, max_new_tokens=50, do_sample=False)
    for (_, _, meta), s in zip(ch, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
        s = s.strip().strip('"').split("\n")[0]
        if s and not REFUSAL.search(s): out.append({"text": s, "lang": "short", **meta})
    print("short", i + len(ch), "/", len(jobs), flush=True)
write_json(a.work / "guard_data_aug3.json", rows + out)
print("short rows:", {(k, s): sum(r["kind"] == k and r["split"] == s for r in out) for k in ("attack", "hardneg") for s in ("train", "val", "test")})
