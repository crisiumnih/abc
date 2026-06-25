"""Test the data-imbalance hypothesis: does the critic's action-gradient quality track data density
across current-bottle-count levels, and does class-balancing the retrain strengthen it?

(a) action-ranking (state-conditioned) binned by CURRENT bottle count, using the saved critic (val split).
(b) retrain with inverse-frequency class weights over bottle levels; re-check ranking by level.
"""
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from abc_minimal.critic_train import return_to_go, standardize_fit, std_actions_fit, MLP, pearson, action_ranking
from abc_minimal.critic_train_save import episode_split_generic
from abc_minimal.critic_qgf import QGFGuidance

SCR = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad"
BUN = Path("/data2/experiemnts/transfer_bundle")
LEVELS = [("0-1", lambda c: c <= 1), ("2", lambda c: c == 2), ("3", lambda c: c == 3),
          ("4", lambda c: c == 4), ("5", lambda c: c == 5)]


def current_bottles(ep, hor, ms):
    rows = list(csv.DictReader(open(BUN / "labels.csv")))
    cache = {}
    def b(ei):
        if ei not in cache:
            r = rows[ei]; cache[ei] = np.load(BUN / "abc_eval/gt_eval" / r["shard"] / f"{r['world']}_bottles.npy")
        return cache[ei]
    return np.array([int(b(int(e))[min(int(round(h * ms)), len(b(int(e))) - 1)]) for e, h in zip(ep, hor)])


def rank_by_level(pred, G, s_pca, ep, cur, mask, tag):
    print(f"\n[{tag}] action-ranking by current bottle count (val):")
    print(f"  {'level':<6}{'n':>6}{'rank':>8}{'corr':>8}{'npairs':>8}")
    for name, f in LEVELS:
        sel = mask & f(cur)
        if sel.sum() < 8:
            print(f"  {name:<6}{int(sel.sum()):>6}{'  --':>8}"); continue
        r, npairs = action_ranking(pred[sel], G[sel], s_pca[sel], ep[sel])
        print(f"  {name:<6}{int(sel.sum()):>6}{r:>8.3f}{pearson(pred[sel], G[sel]):>8.3f}{npairs:>8}")


def train_weighted(X, y, w, Xva, yva, device, epochs=400, bs=256, lr=1e-3, wd=1e-4, patience=40):
    torch.manual_seed(0)
    m = MLP(X.shape[1]).to(device); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Xt, yt, wt = (torch.tensor(z, device=device) for z in (X, y, w))
    Xv, yv = torch.tensor(Xva, device=device), torch.tensor(yva, device=device)
    n = len(X); best = (1e9, None); bad = 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            loss = (wt[idx] * (m(Xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = ((m(Xv) - yv) ** 2).mean().item()
        if vl < best[0] - 1e-8:
            best = (vl, {k: v.detach().clone() for k, v in m.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(best[1]); return m


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(f"{SCR}/chunk_table_policy.npz")
    s = d["s"].astype(np.float32); a_flat = d["a_chunk"].reshape(len(s), -1).astype(np.float32)
    m_flat = np.repeat(d["a_mask"], d["a_chunk"].shape[2], 1).astype(np.float32)
    reward = d["reward"].astype(np.float32); cidx = d["chunk_idx"]; hor = d["horizon_frac"]
    ep = d["episode_id"]; bucket = d["bucket"]; ms = int(d["max_steps"])
    G = return_to_go(reward, ep, cidx, 0.99).astype(np.float32)
    cur = current_bottles(ep, hor, ms)

    split = episode_split_generic(ep, bucket, seed=0, frac=(0.9, 0.1))
    where = np.array([split[int(e)] for e in ep]); tr, va = where == "train", where == "val"
    dens = {name: int(f(cur).sum()) for name, f in LEVELS}
    print(f"data density by level (all rows): {dens}")

    # (a) saved critic
    g = QGFGuidance(f"{SCR}/critic_qgf.pt", dev)
    with torch.no_grad():
        pred0 = g.q(torch.tensor(s, device=dev), torch.tensor(d["a_chunk"], device=dev)).cpu().numpy()
        s_pca0 = g._s_feat(torch.tensor(s, device=dev)).cpu().numpy()
    rank_by_level(pred0, G, s_pca0, ep, cur, va, "(a) SAVED critic")

    # (b) class-balanced retrain (inverse-freq over levels)
    lvl = np.full(len(s), -1)
    for i, (name, f) in enumerate(LEVELS):
        lvl[f(cur)] = i
    freq = np.array([max((lvl[tr] == i).sum(), 1) for i in range(len(LEVELS))], float)
    w_lvl = (freq.sum() / (len(LEVELS) * freq))            # inverse-freq, mean ~1
    weights = np.where(lvl >= 0, w_lvl[lvl.clip(0)], 1.0).astype(np.float32)

    s_mu, s_sd = standardize_fit(s[tr]); s_z = (s - s_mu) / s_sd
    pmean = s_z[tr].mean(0); _, _, Vt = np.linalg.svd(s_z[tr] - pmean, full_matrices=False)
    s_p = (s_z - pmean) @ Vt[:128].T.astype(np.float32)
    s_pca = ((s_p - s_p[tr].mean(0)) / (s_p[tr].std(0) + 1e-6)).astype(np.float32)
    a_mu, a_sd = std_actions_fit(a_flat[tr], m_flat[tr]); a_z = ((a_flat - a_mu) / a_sd) * m_flat
    base = np.array([G[tr & (cidx == k)].mean() if (tr & (cidx == k)).any() else 0.0
                     for k in range(int(cidx.max()) + 1)], np.float32)
    adv = (G - base[cidx]).astype(np.float32)
    X = np.concatenate([s_pca, a_z], 1).astype(np.float32)
    print(f"\nclass weights by level {[round(float(x),2) for x in w_lvl]} (upweights rare 5, downweights dense 0-1)")
    m = train_weighted(X[tr], adv[tr], weights[tr], X[va], adv[va], dev)
    with torch.no_grad():
        predb = m(torch.tensor(X, device=dev)).cpu().numpy()
    rank_by_level(predb, G, s_pca, ep, cur, va, "(b) CLASS-BALANCED retrain")


if __name__ == "__main__":
    main()
