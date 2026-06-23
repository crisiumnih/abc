"""Train the GT-advantage critic on the policy-aligned table and SAVE it for QGF.

Saves a self-contained checkpoint with the MLP + every preprocessing transform QGF needs:
  s:  s_pooled(1536) -> (s-s_mean)/s_std -> ( . - pca_mean) @ pca_W -> (. - pcapost_mean)/pcapost_std
  a:  a1(30,14) flatten(420) -> (a-a_mean)/a_std            (a1 already in norm_stats-normalized space)
  Q = MLP([s_pca(128) ; a_std(420)])
Target = GT-advantage: G_k = sum gamma^j reward; advantage = G - per-chunk-position baseline.
(grad_a advantage == grad_a G, so guidance is identical to return-to-go.)
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from abc_minimal.critic_train import (
    return_to_go, standardize_fit, std_actions_fit, MLP, train_one, pearson, auc, action_ranking,
)

SCR = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad"


def episode_split_generic(ep, bucket, seed=0, frac=(0.9, 0.1)):
    rng = np.random.default_rng(seed)
    bb = {int(e): int(bucket[ep == e][0]) for e in np.unique(ep)}
    assign = {}
    for b in sorted(set(bb.values())):
        ids = [e for e in bb if bb[e] == b]; rng.shuffle(ids)
        ntr = int(round(frac[0] * len(ids)))
        for e in ids[:ntr]: assign[e] = "train"
        for e in ids[ntr:]: assign[e] = "val"
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=f"{SCR}/chunk_table_policy.npz")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{SCR}/critic_qgf.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.table)
    s = d["s"].astype(np.float32)
    a_flat = d["a_chunk"].reshape(len(s), -1).astype(np.float32)
    m_flat = np.repeat(d["a_mask"], d["a_chunk"].shape[2], axis=1).astype(np.float32)
    reward = d["reward"].astype(np.float32)
    cidx, hor, ep, bucket = d["chunk_idx"], d["horizon_frac"], d["episode_id"], d["bucket"]

    G = return_to_go(reward, ep, cidx, args.gamma).astype(np.float32)
    base = np.zeros(int(cidx.max()) + 1, np.float32)

    split = episode_split_generic(ep, bucket, args.seed)
    where = np.array([split[int(e)] for e in ep]); tr, va = where == "train", where == "val"
    for k in range(len(base)):
        sel = tr & (cidx == k); base[k] = G[sel].mean() if sel.any() else 0.0
    adv = (G - base[cidx]).astype(np.float32)

    # s transforms (TRAIN-fit)
    s_mu, s_sd = standardize_fit(s[tr]); s_z = (s - s_mu) / s_sd
    pca_mean = s_z[tr].mean(0)
    _, _, Vt = np.linalg.svd(s_z[tr] - pca_mean, full_matrices=False)
    pca_W = Vt[:args.pca].T.astype(np.float32)
    s_p = (s_z - pca_mean) @ pca_W
    pcapost_mu, pcapost_sd = standardize_fit(s_p[tr])
    s_pca = (s_p - pcapost_mu) / pcapost_sd
    # action transforms
    a_mu, a_sd = std_actions_fit(a_flat[tr], m_flat[tr]); a_z = ((a_flat - a_mu) / a_sd) * m_flat

    X = np.concatenate([s_pca, a_z], 1).astype(np.float32)
    model = train_one(X[tr], adv[tr], X[va], adv[va], args.seed, device)

    # sanity on val
    with torch.no_grad():
        pv = model(torch.tensor(X[va], device=device)).cpu().numpy()
    print(f"rows={len(s)} train={tr.sum()} val={va.sum()}  in_dim={X.shape[1]}")
    print(f"[val] corr(pred,adv)={pearson(pv, adv[va]):.3f}  "
          f"good-vs-bad AUC={auc(pv[(bucket[va]!=-1)], bucket[va][bucket[va]!=-1]):.3f}")
    s_pca_va = s_pca[va]
    racc, npairs = action_ranking(pv, G[va], s_pca_va, ep[va])
    print(f"[val] state-conditioned action-ranking={racc:.3f} (npairs={npairs})")

    torch.save({
        "mlp": model.state_dict(), "in_dim": X.shape[1], "pca_dim": args.pca,
        "s_mean": s_mu, "s_std": s_sd, "pca_mean": pca_mean, "pca_W": pca_W,
        "pcapost_mean": pcapost_mu, "pcapost_std": pcapost_sd,
        "a_mean": a_mu, "a_std": a_sd, "chunk_len": int(d["a_chunk"].shape[1]),
        "gamma": args.gamma, "target": "gt_advantage",
        "action_mean": d["action_mean"], "action_std": d["action_std"], "max_steps": int(d["max_steps"]),
    }, args.out)
    print(f"saved critic -> {args.out}")


if __name__ == "__main__":
    main()
