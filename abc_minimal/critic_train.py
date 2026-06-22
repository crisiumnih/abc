"""Chunked critic v2 — VALUE target (Monte-Carlo return-to-go), not one-step effect.

v1 regressed Q -> one-step Robometer delta r_k. That target barely separates good/bad
(per-step deltas overlap; good-vs-bad AUC < 0.5), so the critic learned a non-discriminative
signal. v2 regresses Q -> discounted return-to-go  G_k = sum_{j>=k} gamma^(j-k) r_j  (gamma=0.99),
i.e. remaining-progress-to-go -- cumulative, discriminative, matching what the s-probe validated.

Fixes from v1:
  - target = return-to-go (value), gamma=0.99
  - s compressed via PCA-128 (fit on TRAIN only) -- raw 1536-d overfits on 2170 rows
  - a_chunk capped to first C=37 steps (= min interval) -> NO padding -> kills the
    pad-length clock leak (every chunk has exactly C real steps)
  - gt_final stays diagnostic-only, never an input

Baselines (predict the SAME G target): Q_time(norm_chunk_idx, a) and Q_action_only(a).
Validation broken out by horizon. Pass = Q(s,a) beats BOTH baselines in EARLY and MID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCR = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad"


def episode_split(seed=0):
    rng = np.random.default_rng(seed)
    split = {}
    for base in (0, 50):
        ids = np.arange(base, base + 50); rng.shuffle(ids)
        for i in ids[:35]: split[i] = "train"
        for i in ids[35:42]: split[i] = "val"
        for i in ids[42:50]: split[i] = "test"
    return split


def return_to_go(reward, episode_id, chunk_idx, gamma):
    """G_k = sum_{j>=k} gamma^(j-k) r_j, computed per episode in chunk order."""
    G = np.zeros_like(reward)
    for e in np.unique(episode_id):
        m = np.where(episode_id == e)[0]
        order = m[np.argsort(chunk_idx[m])]
        acc = 0.0
        for i in order[::-1]:
            acc = reward[i] + gamma * acc
            G[i] = acc
    return G


def standardize_fit(x):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-6] = 1.0
    return mu, sd


def std_actions_fit(a_flat, m_flat):
    cnt = m_flat.sum(0); cnt[cnt < 1] = 1
    mu = (a_flat * m_flat).sum(0) / cnt
    sd = np.sqrt(((a_flat - mu) ** 2 * m_flat).sum(0) / cnt); sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, seed, device, epochs=400, bs=256, lr=1e-3, wd=1e-4, patience=40):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device); yva_t = torch.tensor(yva, device=device)
    n = len(Xtr); best = (1e9, None); bad = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(model(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva_t), yva_t).item()
        if vloss < best[0] - 1e-8:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best[1]); return model


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def auc(scores, labels):
    pos, neg = (labels == 1), (labels == 0)
    if pos.sum() == 0 or neg.sum() == 0: return float("nan")
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def action_ranking(pred, value, s_std, ep_ids, sim_frac=0.15, gap_q=0.5, max_anchor=400, knn=12):
    """State-conditioned better-action ranking: among pairs at SIMILAR state (small ||s_i-s_j||,
    different episodes) with a real value gap (|Δvalue| above gap_q quantile), fraction where Q
    orders them by value. Ranked by the continuous progress value -- NOT the bucket -> uncontaminated.
    Returns (accuracy, n_pairs)."""
    m = len(pred)
    if m < 10:
        return float("nan"), 0
    # subsample anchors for O(m*knn) cost
    anchors = np.arange(m) if m <= max_anchor else np.linspace(0, m - 1, max_anchor).astype(int)
    pairs = []
    for i in anchors:
        d = ((s_std - s_std[i]) ** 2).sum(1)
        d[ep_ids == ep_ids[i]] = np.inf          # only cross-episode pairs
        nn = np.argpartition(d, min(knn, m - 1))[:knn]
        nn = nn[np.isfinite(d[nn])]
        for j in nn:
            if j > i:
                pairs.append((i, j))
    if not pairs:
        return float("nan"), 0
    pairs = np.array(pairs)
    dv = value[pairs[:, 0]] - value[pairs[:, 1]]
    dp = pred[pairs[:, 0]] - pred[pairs[:, 1]]
    thr = np.quantile(np.abs(dv), gap_q)
    keep = np.abs(dv) > max(thr, 1e-6)
    if keep.sum() == 0:
        return float("nan"), 0
    acc = float(np.mean(np.sign(dp[keep]) == np.sign(dv[keep])))
    return acc, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=f"{SCR}/chunk_table.npz")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target", choices=["rtg", "advantage", "terminal"], default="rtg")
    ap.add_argument("--cap", type=int, default=37, help="first C action steps (=min interval -> no pad)")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default=f"{SCR}/critic_report_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.table)
    s = d["s"].astype(np.float32)
    C = args.cap
    a_chunk = d["a_chunk"][:, :C, :].astype(np.float32)        # cap -> no padding
    m = d["a_mask"][:, :C].astype(np.float32)
    assert m.min() == 1.0, "cap exceeds an interval -> padding leak not eliminated"
    a_flat = a_chunk.reshape(len(s), -1)
    m_flat = np.repeat(m, a_chunk.shape[2], axis=1)
    reward = d["reward"].astype(np.float32); rbm = d["rbm"].astype(np.float32)
    cidx = d["chunk_idx"]; hor = d["horizon_frac"]; ep = d["episode_id"]; bucket = d["bucket"]
    cidx_norm = (cidx / cidx.max()).astype(np.float32)[:, None]

    # VALUE target
    G = return_to_go(reward, ep, cidx, args.gamma).astype(np.float32)
    G_rtg = G.copy()  # raw return-to-go = "progress outcome" used to rank actions (a)

    split = episode_split(args.split_seed)
    where = np.array([split[e] for e in ep])
    tr, va, te = where == "train", where == "val", where == "test"

    # optional advantage target: subtract per-chunk-position time baseline (fit on TRAIN).
    # Removes the time-to-go trend the clock exploits; leaves grad_a Q unchanged for QGF.
    if args.target == "advantage":
        base = np.zeros(int(cidx.max()) + 1, np.float32)
        for k in range(len(base)):
            sel = tr & (cidx == k)
            base[k] = G[sel].mean() if sel.any() else 0.0
        G = (G - base[cidx]).astype(np.float32)
    elif args.target == "terminal":
        # non-saturating value: discounted eventual outcome = final_progress * gamma^(K-1-k).
        # (stand-in for GT terminal/time-to-success; NOTE: not action-sensitive -> QGF caveat)
        K = int(cidx.max()) + 1
        fin = {}
        for e in np.unique(ep):
            o = np.where(ep == e)[0]; last = o[np.argmax(cidx[o])]
            fin[e] = rbm[last] + reward[last]
        G = np.array([fin[ep[i]] * (args.gamma ** (K - 1 - cidx[i])) for i in range(len(ep))], np.float32)
    print(f"target={args.target} gamma={args.gamma}  a_chunk capped to {C} steps (pad eliminated, mask.min={m.min():.0f})")
    print(f"rows: train={tr.sum()} val={va.sum()} test={te.sum()} (episodes 70/14/16)")
    print(f"G: mean={G.mean():.3f} std={G.std():.3f}  (v1 one-step r: mean={reward.mean():.3f} std={reward.std():.3f})")

    # standardizers (TRAIN only)
    s_mu, s_sd = standardize_fit(s[tr]); s_z = (s - s_mu) / s_sd
    a_mu, a_sd = std_actions_fit(a_flat[tr], m_flat[tr]); a_z = ((a_flat - a_mu) / a_sd) * m_flat
    # PCA-128 on s, fit on TRAIN only
    mean_tr = s_z[tr].mean(0)
    _, _, Vt = np.linalg.svd(s_z[tr] - mean_tr, full_matrices=False)
    Pk = Vt[:args.pca].T
    s_pca = ((s_z - mean_tr) @ Pk).astype(np.float32)
    s_pca = (s_pca - s_pca[tr].mean(0)) / (s_pca[tr].std(0) + 1e-6)

    feats = {
        "Q(s,a)":        np.concatenate([s_pca, a_z], 1),   # primary: PCA-128 s + capped action
        "Q_time":        np.concatenate([cidx_norm, a_z], 1),
        "Q_action_only": a_z,
        "Q_s_only":      s_pca,                              # diagnostic
        "Q(s_raw,a)":    np.concatenate([s_z, a_z], 1),      # diagnostic: raw 1536-d (overfit check)
    }

    hbins = {"early(<0.3)": hor < 0.3, "mid(0.3-0.7)": (hor >= 0.3) & (hor < 0.7), "late(>=0.7)": hor >= 0.7}
    horder = list(hbins) + ["ALL"]

    # target's OWN good-vs-bad AUC (sanity: should be > 0.5, unlike v1 one-step delta)
    print("\n[sanity] target G good-vs-bad AUC by horizon (was <0.5 for one-step r in v1):")
    print("  " + "  ".join(f"{h}={auc(G[te & (hm if h!='ALL' else np.ones(len(hor),bool))], bucket[te & (hm if h!='ALL' else np.ones(len(hor),bool))]):.3f}"
                            for h, hm in list(hbins.items()) + [("ALL", None)]))

    s_pca_te = s_pca[te]; ep_te = ep[te]; G_rtg_te = G_rtg[te]
    results = {name: {h: {"corr": [], "auc": [], "rank": []} for h in horder} for name in feats}
    for name, X in feats.items():
        for seed in range(args.seeds):
            model = train_one(X[tr], G[tr], X[va], G[va], seed, device)
            with torch.no_grad():
                pred = model(torch.tensor(X[te], device=device)).cpu().numpy()
            for h in horder:
                hm = np.ones(len(hor), bool) if h == "ALL" else hbins[h]
                idx_test = np.ones(te.sum(), bool) if h == "ALL" else hm[te]
                results[name][h]["corr"].append(pearson(pred[idx_test], G[te & hm]))
                results[name][h]["auc"].append(auc(pred[idx_test], bucket[te & hm]))
                racc, npairs = action_ranking(pred[idx_test], G_rtg_te[idx_test],
                                              s_pca_te[idx_test], ep_te[idx_test])
                results[name][h]["rank"].append(racc)
                results[name][h]["npairs"] = npairs

    def agg(l):
        a = np.array([x for x in l if np.isfinite(x)])
        return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), 0.0)
    summary = {n: {h: {k: agg(v[k]) for k in ("corr", "auc", "rank")} for h, v in hd.items()}
               for n, hd in results.items()}

    print(f"\n{'='*82}\nCRITIC v2  VALUE target ({args.target}, gamma={args.gamma})  seeds={args.seeds}\n{'='*82}")
    for metric, label in [("rank", "(a) STATE-CONDITIONED action ranking  [clean: ranks better-progress action @ similar s; >0.5 good]"),
                          ("corr", "Pearson corr(pred, value target)  [clean pass metric]"),
                          ("auc", "good-vs-bad AUC  [CONTAMINATED: bucket=f(action); diagnostic only]")]:
        print(f"\n--- {label} ---")
        print(f"{'model':<15}" + "".join(f"{h:>18}" for h in horder))
        for name in feats:
            print(f"{name:<15}" + "".join(f"{summary[name][h][metric][0]:>11.3f}+/-{summary[name][h][metric][1]:.2f}" for h in horder))

    print("\nnpairs used for action-ranking by horizon: " +
          "  ".join(f"{h}={results['Q(s,a)'][h].get('npairs', 0)}" for h in horder))

    print(f"\n{'='*82}\nVERDICT (pass = Q(s,a) beats BOTH baselines in EARLY & MID on the CLEAN metrics)\n{'='*82}")
    ok = {"corr": True, "rank": True}
    for h in ["early(<0.3)", "mid(0.3-0.7)"]:
        for metric in ("corr", "rank"):
            q = summary["Q(s,a)"][h][metric][0]
            qt = summary["Q_time"][h][metric][0]; qa = summary["Q_action_only"][h][metric][0]
            beats = (q > qt + 0.01) and (q > qa + 0.01)
            ok[metric] = ok[metric] and beats
            print(f"  {h:<14} [{metric}] Q(s,a)={q:.3f}  Q_time={qt:.3f}  Q_action={qa:.3f}  -> {'beats both' if beats else 'no'}")
    overall = ok["corr"]  # corr is the primary clean pass metric (AUC dropped as contaminated)
    print(f"\n  beats-both early&mid: corr={ok['corr']}  rank={ok['rank']}  (AUC dropped: contaminated)")
    print(f">>> {'PASS' if overall else 'PARTIAL/FAIL'} (on clean corr metric) <<<")

    Path(args.out).write_text(json.dumps(
        {"gamma": args.gamma, "cap": C, "pca": args.pca, "target": args.target, "summary": summary,
         "G_mean": float(G.mean()), "G_std": float(G.std()),
         "pass_corr": bool(ok["corr"]), "pass_rank": bool(ok["rank"]), "pass": bool(overall)}, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
