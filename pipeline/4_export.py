"""Step 4: fold the linear-mean probe into one direction and a bias on raw residual states:
per-token logit = direction·h + bias at hidden_states[LAYER] of the user's tokens; the mean over the user's tokens is
the probe's logit (> 0 = attack). This file loads in brainscope as it is (vector, bias, hidden_state)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_intent.common import LAYER, TARGET, args, write_json

a = args(__doc__)
sd_ = torch.load(a.work / "probes/LinearMean.pt", map_location="cpu")
norm = torch.load(a.work / "probes/norm.pt", map_location="cpu")
w, b = sd_["w.weight"][0].float(), sd_["w.bias"][0].float()
v = w / norm["sd"]
bias = float(b - (w * (norm["mu"] / norm["sd"])).sum())
write_json(a.work / "probe.json", {
    "model": TARGET, "hidden_state": LAYER, "direction": v.tolist(), "bias": bias,
    "note": f"per-token score = direction·h + bias on hidden_states[{LAYER}] of the user's tokens; "
            "mean over the user's tokens = the probe's logit; > 0 = attack"})
print(f"probe.json: hidden_state {LAYER}, |direction| {float(v.norm()):.3f}, bias {bias:+.3f}")
