"""Trained in English only, tested in Czech: how stable is it? A linear-mean probe (logistic regression on the mean of
the user's tokens) at layers 10 and 14, retrained on 10 bootstrap resamples of the English train split; threshold from
English val. Reports Czech missed attacks per resample and the median."""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import CZECH_REFUSAL, TARGET, args, load, read_json, user_states, weighted_error, write_json

a = args(__doc__, data=(str, "guard_data.json", "dataset in the work dir"), seeds=(int, 10, "bootstrap resamples"))
rows = [r for r in read_json(a.work / a.data) if not (r["lang"] == "cs" and r["kind"] == "attack" and CZECH_REFUSAL.search(r["text"]))]
tok, model = load(TARGET)
LAYERS = (10, 14)
F = np.stack([torch.stack([h.float().mean(0) for h in user_states(tok, model, r["text"], LAYERS)]).cpu().numpy() for r in rows])
y = np.array([r["y"] for r in rows]); kind = np.array([r["kind"] for r in rows])
sel = lambda sp, l: np.array([r["split"] == sp and r["lang"] == l for r in rows])
tr, va, te_en, te_cs = sel("train", "en"), sel("val", "en"), sel("test", "en"), sel("test", "cs")
out = {}
for li, L in enumerate(LAYERS):
    X = F[:, li]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4; Z = (X - mu) / sd; runs = []
    for seed in range(a.seeds):
        idx = np.random.default_rng(seed).choice(np.where(tr)[0], tr.sum(), replace=True)
        c = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", random_state=seed).fit(Z[idx], y[idx])
        s = c.decision_function(Z)
        t = min(np.unique(s[va]), key=lambda t: weighted_error(kind[va], s[va], t))
        runs.append(dict(cs_missed=float((s[te_cs & (kind == "attack")] <= t).mean()),
                         cs_auc=float(roc_auc_score(y[te_cs & (kind != "normal")], s[te_cs & (kind != "normal")])),
                         en_missed=float((s[te_en & (kind == "attack")] <= t).mean()),
                         cs_fp_normal=float((s[te_cs & (kind == "normal")] > t).mean())))
    out[f"L{L}"] = runs
    med = lambda k: float(np.median([r[k] for r in runs]))
    print(f"L{L}: Czech missed attacks per resample {[round(r['cs_missed'], 3) for r in runs]} | median {med('cs_missed'):.3f} | "
          f"Czech AUC (attack vs hard neg) median {med('cs_auc'):.3f} | English missed median {med('en_missed'):.3f}", flush=True)
(a.work / "results").mkdir(exist_ok=True)
write_json(a.work / "results/czech_transfer.json", out, indent=1)
