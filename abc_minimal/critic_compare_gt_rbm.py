"""Q_gt vs Q_rbm — per-chunk comparison harness (ready to run once GT is available).

Trains the SAME critic + baselines twice on the SAME states/actions/splits, differing only in the
per-chunk REWARD source, and compares them by horizon on the CLEAN metrics:
  - corr(pred, value target)
  - state-conditioned action ranking (ranks better-progress action at similar s)  [the metric grad_a Q needs]
  - good-vs-bad AUC is reported diagnostic-only (CONTAMINATED: bucket=f(policy action)).

Reward sources:
  reward_rbm : Robometer-progress delta   (current, saturates mid/late)
  reward_gt  : GT bottle-count/6 delta     (from patched eval_policy per-step bottles)

Expected dual-reward table (npz) keys:
  s (N,1536), a_chunk (N,L,14), a_mask (N,L), chunk_idx (N,), horizon_frac (N,),
  episode_id (N,), bucket (N,), reward_rbm (N,), reward_gt (N,)
  [optional: rbm_progress (N,), gt_progress (N,) for reference]

Build it by re-running the patched eval_policy.py (saves world_*_bottles.npy + seed), re-featurizing
s on the 32-frame grid for those rollouts, running Robometer on the same videos, then emitting the
two reward columns from the matched per-chunk progress curves.

The headline output is Δ = (gt − rbm) for Q(s,a) per horizon: does GT sharpen the MID action gradient?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from abc_minimal.critic_train import (
    return_to_go, standardize_fit, std_actions_fit, train_one,
    pearson, auc, action_ranking, episode_split,
)

SCR = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad"
HBINS = lambda hor: {"early(<0.3)": hor < 0.3, "mid(0.3-0.7)": (hor >= 0.3) & (hor < 0.7), "late(>=0.7)": hor >= 0.7}


def advantage_of(G, tr, cidx):
    base = np.zeros(int(cidx.max()) + 1, np.float32)
    for k in range(len(base)):
        sel = tr & (cidx == k)
        base[k] = G[sel].mean() if sel.any() else 0.0
    return (G - base[cidx]).astype(np.float32)


def build_feats(s, a_flat, m_flat, cidx, tr, pca_dim):
    s_mu, s_sd = standardize_fit(s[tr]); s_z = (s - s_mu) / s_sd
    a_mu, a_sd = std_actions_fit(a_flat[tr], m_flat[tr]); a_z = ((a_flat - a_mu) / a_sd) * m_flat
    mean_tr = s_z[tr].mean(0)
    _, _, Vt = np.linalg.svd(s_z[tr] - mean_tr, full_matrices=False)
    s_pca = ((s_z - mean_tr) @ Vt[:pca_dim].T).astype(np.float32)
    s_pca = (s_pca - s_pca[tr].mean(0)) / (s_pca[tr].std(0) + 1e-6)
    cidx_norm = (cidx / cidx.max()).astype(np.float32)[:, None]
    return {
        "Q(s,a)":        np.concatenate([s_pca, a_z], 1),
        "Q_time":        np.concatenate([cidx_norm, a_z], 1),
        "Q_action_only": a_z,
    }, s_pca


def evaluate_source(name, reward, feats, s_pca, tr, va, te, ep, cidx, hor, bucket,
                    gamma, target, seeds, device):
    """Train all models on this reward's value target; return metrics-by-horizon dict."""
    G = return_to_go(reward, ep, cidx, gamma).astype(np.float32)
    G_rtg = G.copy()  # progress-outcome used to rank actions
    tgt = advantage_of(G, tr, cidx) if target == "advantage" else G
    hbins = HBINS(hor); horder = list(hbins) + ["ALL"]
    out = {m: {h: {"corr": [], "rank": [], "auc": []} for h in horder} for m in feats}
    s_pca_te, ep_te, G_te = s_pca[te], ep[te], G_rtg[te]
    for m, X in feats.items():
        for seed in range(seeds):
            model = train_one(X[tr], tgt[tr], X[va], tgt[va], seed, device)
            with torch.no_grad():
                pred = model(torch.tensor(X[te], device=device)).cpu().numpy()
            for h in horder:
                hm = np.ones(len(hor), bool) if h == "ALL" else hbins[h]
                it = np.ones(te.sum(), bool) if h == "ALL" else hm[te]
                out[m][h]["corr"].append(pearson(pred[it], tgt[te & hm]))
                out[m][h]["auc"].append(auc(pred[it], bucket[te & hm]))
                r, _ = action_ranking(pred[it], G_te[it], s_pca_te[it], ep_te[it])
                out[m][h]["rank"].append(r)
    agg = lambda l: float(np.nanmean(l)) if np.any(np.isfinite(l)) else float("nan")
    return {m: {h: {k: agg(v[k]) for k in v} for h, v in hd.items()} for m, hd in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=f"{SCR}/chunk_table_gt.npz", help="dual-reward table (reward_rbm + reward_gt)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target", choices=["rtg", "advantage"], default="advantage")
    ap.add_argument("--cap", type=int, default=37)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default=f"{SCR}/critic_gt_vs_rbm.json")
    args = ap.parse_args()

    d = np.load(args.table)
    missing = [k for k in ("s", "a_chunk", "a_mask", "chunk_idx", "horizon_frac",
                           "episode_id", "bucket", "reward_rbm", "reward_gt") if k not in d.files]
    if missing:
        raise SystemExit(f"table missing keys {missing}. Build a dual-reward table first "
                         f"(patched eval -> per-step bottles -> reward_gt; Robometer -> reward_rbm).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = d["s"].astype(np.float32)
    C = args.cap
    a_chunk = d["a_chunk"][:, :C, :].astype(np.float32); mk = d["a_mask"][:, :C].astype(np.float32)
    assert mk.min() == 1.0, "cap exceeds an interval -> padding leak"
    a_flat = a_chunk.reshape(len(s), -1); m_flat = np.repeat(mk, a_chunk.shape[2], 1)
    cidx, hor, ep, bucket = d["chunk_idx"], d["horizon_frac"], d["episode_id"], d["bucket"]

    split = episode_split(args.split_seed); where = np.array([split[e] for e in ep])
    tr, va, te = where == "train", where == "val", where == "test"
    feats, s_pca = build_feats(s, a_flat, m_flat, cidx, tr, args.pca)

    res = {}
    for src in ("rbm", "gt"):
        res[src] = evaluate_source(src, d[f"reward_{src}"].astype(np.float32), feats, s_pca,
                                   tr, va, te, ep, cidx, hor, bucket, args.gamma, args.target,
                                   args.seeds, device)

    hbins = HBINS(hor); horder = list(hbins) + ["ALL"]
    print(f"{'='*86}\nQ_gt vs Q_rbm  (target={args.target}, gamma={args.gamma}, PCA-{args.pca})  seeds={args.seeds}\n{'='*86}")
    for metric, label in [("rank", "STATE-CONDITIONED action ranking  [what grad_a Q needs; >0.5 good]"),
                          ("corr", "corr(pred, value target)"),
                          ("auc", "good-vs-bad AUC [contaminated, diagnostic]")]:
        print(f"\n--- {label} ---  (Q(s,a) only; Δ = gt − rbm)")
        print(f"{'horizon':<16}{'rbm':>10}{'gt':>10}{'Δ(gt-rbm)':>12}")
        for h in horder:
            rb = res["rbm"]["Q(s,a)"][h][metric]; gt = res["gt"]["Q(s,a)"][h][metric]
            print(f"{h:<16}{rb:>10.3f}{gt:>10.3f}{gt-rb:>+12.3f}")

    # full per-model tables too
    for src in ("rbm", "gt"):
        print(f"\n[{src}] action-ranking by model x horizon:")
        print(f"{'model':<15}" + "".join(f"{h:>16}" for h in horder))
        for m in feats:
            print(f"{m:<15}" + "".join(f"{res[src][m][h]['rank']:>16.3f}" for h in horder))

    # headline verdict: does GT lift MID action-ranking above 0.5 and beat baselines?
    mid = "mid(0.3-0.7)"
    gt_mid = res["gt"]["Q(s,a)"][mid]["rank"]
    gt_mid_beats = (gt_mid > res["gt"]["Q_time"][mid]["rank"] + 0.01) and (gt_mid > res["gt"]["Q_action_only"][mid]["rank"] + 0.01)
    print(f"\n{'='*86}\nMID action-ranking with GT reward: Q(s,a)={gt_mid:.3f} "
          f"({'>0.5 & beats baselines' if (gt_mid > 0.5 and gt_mid_beats) else 'still weak'})  "
          f"vs rbm={res['rbm']['Q(s,a)'][mid]['rank']:.3f}\n{'='*86}")

    Path(args.out).write_text(json.dumps({"rbm": res["rbm"], "gt": res["gt"],
                                          "target": args.target, "gamma": args.gamma}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
