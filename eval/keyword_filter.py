"""The probe against what most people would try first: a list of 50-odd security words (malware, exploit, keylogger,
backdoor, steal, bypass…). Test split, English and Czech; the probe at its deployment threshold (step 5)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import KEYWORDS, TARGET, args, load, read_json, write_json
from hidden_intent.score import Probe

a = args(__doc__, data=(str, "guard_data.json", "dataset in the work dir"))
rows = read_json(a.work / a.data)
probe = Probe(a.work / "probe.json")
tok, model = load(TARGET)
out = {"threshold": probe.threshold}
for lang in ["en", "cs"]:
    t = {k: [r["text"] for r in rows if r["split"] == "test" and r["lang"] == lang and r["kind"] == k] for k in ["attack", "normal", "hardneg"]}
    p = {k: np.mean([probe.mean(tok, model, x) > probe.threshold for x in v]) for k, v in t.items()}
    w = {k: np.mean([bool(KEYWORDS.search(x)) for x in v]) for k, v in t.items()}
    out[lang] = {"probe": {k: float(v) for k, v in p.items()}, "keywords": {k: float(v) for k, v in w.items()}}
    for name, d in (("probe   ", p), ("keywords", w)):
        print(f"[{lang}] {name} flags attacks {d['attack']:.0%} | normal {d['normal']:.0%} | hard negatives {d['hardneg']:.0%}")
(a.work / "results").mkdir(exist_ok=True)
write_json(a.work / "results/keyword_filter.json", out, indent=1)
