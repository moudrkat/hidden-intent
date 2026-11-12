"""The two Google probe heads (Kramar et al. 2026), on z-scored residual states [batch, tokens, hidden]."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMean(nn.Module):
    """One weight vector; the request's score is the mean of the per-token logits."""
    def __init__(s, d):
        super().__init__(); s.w = nn.Linear(d, 1)

    def forward(s, X, m):
        return (s.w(X).squeeze(-1) * m).sum(1) / m.sum(1)


class RollingAttn(nn.Module):
    """MLP 2x100 ReLU, then per head a softmax-weighted mean of values over a causal window of w tokens;
    the score is the sum over heads of the maximum over time."""
    def __init__(s, d, H=10, w=10, h=100):
        super().__init__()
        s.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        s.q = nn.Linear(h, H, bias=False); s.v = nn.Linear(h, H, bias=False); s.b = nn.Parameter(torch.zeros(1)); s.w = w

    def forward(s, X, m):
        y = s.mlp(X); a = s.q(y); v = s.v(y)
        a = a.masked_fill(~m[..., None], -1e4); a = a - a.amax(1, keepdim=True).detach()
        e = torch.exp(a) * m[..., None]
        k = torch.ones(1, 1, s.w, device=X.device)

        def roll(z):
            B, T, H = z.shape; z = z.permute(0, 2, 1).reshape(B * H, 1, T)
            return F.conv1d(F.pad(z, (s.w - 1, 0)), k).reshape(B, H, T).permute(0, 2, 1)
        vbar = roll(e * v) / (roll(e) + 1e-9)
        vbar = vbar.masked_fill(~m[..., None], -1e4)
        return vbar.amax(1).sum(-1) + s.b
