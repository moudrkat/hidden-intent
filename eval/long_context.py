"""Google's long-context test: each English test request put after N tokens of real Python code (the standard library of
the Python running this script, PSF licence), as one user message. Three ways to read the same probe direction:
  mean     the exported linear probe as trained: the mean over all the user's tokens
  window   the same per-token logits, the highest mean over any 10 consecutive tokens
  rolling  Google's Rolling Attention probe (trained, step 3)
Each threshold is picked on the short English val split (as deployed). Reports missed attacks and false alarms per N."""
import glob, random, sys, sysconfig
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import LAYER, TARGET, args, device, load, pick_threshold, read_json, user_prefix, write_json
from hidden_intent.probes import RollingAttn
from hidden_intent.score import Probe, max_window

a = args(__doc__, data=(str, "guard_data_aug.json", "dataset in the work dir"), lengths=(str, "0,512,2048,8192", "code tokens before the request"),
         limit=(int, 0, "test requests per length, 0 = all"))
rows = read_json(a.work / a.data)
probe = Probe(a.work / "probe.json")
tok, model = load(TARGET)
norm = torch.load(a.work / "probes/norm.pt", map_location=device())
rnet = RollingAttn(norm["mu"].shape[0]).to(device())
rnet.load_state_dict(torch.load(a.work / "probes/RollingAttn.pt", map_location=device())); rnet.eval()
A = user_prefix(tok)
enc = lambda t: tok(t, add_special_tokens=False).input_ids[:384]


@torch.no_grad()
def scores(ids):
    h = model(torch.tensor([A + ids], device=device()), output_hidden_states=True).hidden_states[LAYER][0, len(A):].float()
    lg = h @ probe.v + probe.bias
    X = ((h - norm["mu"]) / norm["sd"])[None]
    return {"mean": float(lg.mean()), "window": max_window(lg), "rolling": float(rnet(X, torch.ones(1, X.shape[1], dtype=torch.bool, device=device())))}


val = [r for r in rows if r["split"] == "val" and r["lang"] == "en"]
test = [r for r in rows if r["split"] == "test" and r["lang"] == "en"]
if a.limit:
    test = random.Random(0).sample(test, min(a.limit, len(test)))
vk = np.array([r["kind"] for r in val]); vs = [scores(enc(r["text"])) for r in val]
thr = {k: pick_threshold(vk, np.array([s[k] for s in vs])) for k in ["mean", "window", "rolling"]}
code = ""
for f in sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/**/*.py", recursive=True))[:400]:
    try:
        code += open(f).read() + "\n"
    except Exception:
        pass
code_ids = tok(code[:3_000_000], add_special_tokens=False).input_ids
sep = tok("\n\n", add_special_tokens=False).input_ids
tk = np.array([r["kind"] for r in test])
res = {"thresholds": thr, "n_test": len(test), "by_length": {}}
print("tokens | reader  | missed attacks | false alarms hard neg | false alarms normal", flush=True)
for N in [int(x) for x in a.lengths.split(",")]:
    rnd = random.Random(N)
    S = []
    for r in test:
        o = rnd.randrange(0, len(code_ids) - N - 1)
        S.append(scores((code_ids[o:o + N] + sep if N else []) + enc(r["text"])))
    for k in ["mean", "window", "rolling"]:
        s = np.array([x[k] for x in S]); t = thr[k]
        row = {"missed": float((s[tk == "attack"] <= t).mean()), "fp_hardneg": float((s[tk == "hardneg"] > t).mean()),
               "fp_normal": float((s[tk == "normal"] > t).mean())}
        res["by_length"][f"{N}/{k}"] = row
        print(f"{N:6d} | {k:7s} | {row['missed']:.3f} | {row['fp_hardneg']:.3f} | {row['fp_normal']:.3f}", flush=True)
(a.work / "results").mkdir(exist_ok=True)
write_json(a.work / "results/long_context.json", res, indent=1)
