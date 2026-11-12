"""Scoring a request with the exported probe (probe.json)."""
import torch

from .common import device, read_json, user_states


class Probe:
    def __init__(self, path):
        p = read_json(path)
        self.layer, self.bias = p["hidden_state"], p["bias"]
        self.v = torch.tensor(p["direction"], device=device()).float()
        self.threshold = p.get("threshold")

    def token_logits(self, tok, model, text, system=None, maxtok=384):
        h = user_states(tok, model, text, (self.layer,), system, maxtok)[0].float()
        return h @ self.v + self.bias

    def mean(self, tok, model, text, system=None):
        return float(self.token_logits(tok, model, text, system).mean())


def max_window(logits: torch.Tensor, w: int = 10) -> float:
    """The highest mean over any w consecutive tokens: a short attack inside a long request."""
    if logits.numel() < w:
        return float(logits.mean())
    return float(logits.unfold(0, w, 1).mean(1).max())
