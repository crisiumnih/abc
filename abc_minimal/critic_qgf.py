"""QGF guidance: load the saved GT-advantage critic and expose a differentiable Q(s, a1).

Used by dit.py sample_actions: g = grad_{a1} Q(s_pooled, a1).  a1 is the policy's 30-step chunk in
norm_stats-normalized space (same space the critic was trained on). All transforms are differentiable
w.r.t. a1, so autograd.grad flows through the critic MLP (NOT through predict_velocity -- v is
stop-gradiented when forming a1, matching the QGF reference apply_jacobian=False).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from abc_minimal.dit import DiTPolicy  # noqa: F401  (kept for module locality)
from abc_minimal.critic_train import MLP


class QGFGuidance:
    def __init__(self, ckpt_path: str, device):
        d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.device = device
        self.chunk_len = int(d["chunk_len"])
        self.mlp = MLP(int(d["in_dim"])).to(device)
        self.mlp.load_state_dict(d["mlp"]); self.mlp.eval()
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.s_mean, self.s_std = t(d["s_mean"]), t(d["s_std"])
        self.pca_mean, self.pca_W = t(d["pca_mean"]), t(d["pca_W"])
        self.pcapost_mean, self.pcapost_std = t(d["pcapost_mean"]), t(d["pcapost_std"])
        self.a_mean, self.a_std = t(d["a_mean"]), t(d["a_std"])

    def _s_feat(self, s_pooled):
        z = (s_pooled.to(self.device).float() - self.s_mean) / self.s_std
        p = (z - self.pca_mean) @ self.pca_W
        return (p - self.pcapost_mean) / self.pcapost_std

    def q(self, s_pooled, a1):
        """s_pooled (B,1536); a1 (B,chunk_len,14) normalized -> Q (B,). Differentiable wrt a1."""
        s_pca = self._s_feat(s_pooled)
        a_flat = a1.reshape(a1.shape[0], -1).to(self.device).float()
        a_z = (a_flat - self.a_mean) / self.a_std
        return self.mlp(torch.cat([s_pca, a_z], dim=-1))

    def grad(self, s_pooled, a1):
        a1g = a1.detach().to(self.device).float().requires_grad_(True)
        with torch.enable_grad():
            q = self.q(s_pooled.detach(), a1g)
            (g,) = torch.autograd.grad(q.sum(), a1g)
        return g, q.detach()


if __name__ == "__main__":
    # standalone smoke: load critic, random s/a1, confirm finite nonzero grad
    import argparse, numpy as np
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = QGFGuidance(a.ckpt, dev)
    s = torch.randn(4, 1536, device=dev); a1 = torch.randn(4, G.chunk_len, 14, device=dev)
    g, q = G.grad(s, a1)
    print(f"q={q.flatten().tolist()}  g finite={torch.isfinite(g).all().item()} "
          f"|g|mean={g.abs().mean().item():.4e} nonzero={(g.abs()>0).any().item()}")
