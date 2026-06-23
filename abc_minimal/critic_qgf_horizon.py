"""Action-ranking BY HORIZON for the saved policy-aligned QGF critic (interpretation guard).

Did mid-horizon signal survive the switch from the 37-step interval critic (mid rank 0.70) to the
30-step normalized policy-aligned critic? Computes state-conditioned action-ranking + corr + AUC by
horizon on the SAME val split used to train/save the critic.
"""
import numpy as np
import torch

from abc_minimal.critic_train import return_to_go, pearson, auc, action_ranking
from abc_minimal.critic_train_save import episode_split_generic
from abc_minimal.critic_qgf import QGFGuidance

SCR = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad"

d = np.load(f"{SCR}/chunk_table_policy.npz")
s = d["s"].astype(np.float32); a = d["a_chunk"].astype(np.float32)
reward = d["reward"].astype(np.float32)
cidx, hor, ep, bucket = d["chunk_idx"], d["horizon_frac"], d["episode_id"], d["bucket"]
G = return_to_go(reward, ep, cidx, 0.99).astype(np.float32)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = QGFGuidance(f"{SCR}/critic_qgf.pt", dev)
with torch.no_grad():
    pred = g.q(torch.tensor(s, device=dev), torch.tensor(a, device=dev)).cpu().numpy()
    s_pca = g._s_feat(torch.tensor(s, device=dev)).cpu().numpy()

split = episode_split_generic(ep, bucket, seed=0, frac=(0.9, 0.1))
va = np.array([split[int(e)] for e in ep]) == "val"
bins = {"early(<0.3)": hor < 0.3, "mid(0.3-0.7)": (hor >= 0.3) & (hor < 0.7),
        "late(>=0.7)": hor >= 0.7, "ALL": np.ones(len(hor), bool)}

print(f"policy-aligned critic (30-step normalized) — VAL action-ranking by horizon")
print(f"{'horizon':<16}{'rank':>8}{'npairs':>9}{'corr':>9}{'AUC(g/b)':>10}{'n':>7}")
for h, hm in bins.items():
    sel = va & hm
    rank, npairs = action_ranking(pred[sel], G[sel], s_pca[sel], ep[sel])
    bk = bucket[sel]; keep = (bk == 0) | (bk == 1)
    a_uc = auc(pred[sel][keep], bk[keep]) if keep.sum() else float("nan")
    print(f"{h:<16}{rank:>8.3f}{npairs:>9}{pearson(pred[sel], G[sel]):>9.3f}{a_uc:>10.3f}{int(sel.sum()):>7}")
print("\n(greenlight ref: 37-step interval critic had mid action-ranking ~0.70)")
