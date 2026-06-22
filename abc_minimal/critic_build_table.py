"""Build the chunked-bandit critic table on the Robometer 32-frame grid.

Chunk grain = Robometer interval. Per episode (32 robometer samples) we emit 31
transition rows k = 0..30:
  s_k         : (1536,)  cached mean-pooled build_vision_tokens at frame_k  (from probe cache)
  a_chunk_k   : (L,14)   ALL actions executed over [frame_k, frame_{k+1}), padded to fixed L
  a_mask_k    : (L,)     1 for real steps, 0 for padding
  rbm[k]      : float    robometer progress at sample k
  reward r_k  : float    rbm[k+1] - rbm[k]            (raw delta; clipping decided after plotting)
  gt_final    : float    final bottle-count / 6 (per-episode scalar; per-chunk GT unavailable)
  chunk_idx k : int      0..30
  horizon_frac: float    k / 30   (early<0.3 / mid / late>0.7)
  episode_id  : int      cache episode id (0..49 good, 50..99 bad)
  bucket      : int      1 good / 0 bad
  is_expert   : bool      always False here (expert episodes registered in manifest, not used)

Alignment: the probe cache samples frames linspace(0, n-1, 32) and Robometer scores 32
samples over the same video, so cache row k and robometer progress[k] are the same moment
(index-to-index, NO interpolation). Step 0 guards this (good final >> bad final, progress rises).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CACHE = "/data/tmp/claude-1000/-data2-experiemnts/e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad/probe_feats_full.npz"
COLLECTED = Path("/data2/abc/outputs/collected")
RBM = {"good": Path("/data2/abc_data/robometer_view_top_good"),
       "bad": Path("/data2/abc_data/robometer_view_top_bad")}
EXPERT_DIR = Path("/data2/abc_data/train")
ACTION_SLICE = slice(14, 28)  # states_actions columns 14:28 = action


def load_cache():
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


def ep_frames(cache, ep_id):
    m = cache["episode_id"] == ep_id
    order = np.argsort(cache["frame_idx"][m])
    return cache["frame_idx"][m][order], cache["feat_pooled"][m][order]  # (32,), (32,1536)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/tmp/claude-1000/-data2-experiemnts/"
                    "e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad/chunk_table.npz")
    ap.add_argument("--plotdir", default="/data/tmp/claude-1000/-data2-experiemnts/"
                    "e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad")
    args = ap.parse_args()

    cache = load_cache()
    n_samples = 32

    # --- pass 1: collect raw intervals + robometer + guard stats, find max interval length ---
    episodes = []  # dicts
    for bucket, label in (("good", 1), ("bad", 0)):
        for e in range(50):
            ep_id = e if bucket == "good" else 50 + e
            frames, feats = ep_frames(cache, ep_id)
            assert len(frames) == n_samples, f"{ep_id} has {len(frames)} frames"
            sa = np.load(COLLECTED / bucket / f"episode_{e:03d}" / "states_actions.npy")
            meta = json.loads((COLLECTED / bucket / f"episode_{e:03d}" / "metadata.json").read_text())
            rbm = np.load(RBM[bucket] / "per_episode" / f"episode_{e:03d}.npz")
            prog = rbm["progress"].astype(np.float32)
            assert len(prog) == n_samples, f"rbm {ep_id} len {len(prog)}"
            episodes.append(dict(ep_id=ep_id, bucket=label, frames=frames, feats=feats,
                                 sa=sa, prog=prog, gt_final=float(meta["max_bottles"]) / 6.0))

    L = max(int(ep["frames"][k + 1] - ep["frames"][k])
            for ep in episodes for k in range(n_samples - 1))
    print(f"max reward-interval length over all episodes: L={L} steps")

    # --- guard ---
    g = np.array([ep["prog"][-1] for ep in episodes if ep["bucket"] == 1])
    b = np.array([ep["prog"][-1] for ep in episodes if ep["bucket"] == 0])
    print(f"[guard] rbm final mean: good={g.mean():.3f} bad={b.mean():.3f} "
          f"(expect ~0.683 / ~0.473)")
    frac_rising = np.mean([ep["prog"][-1] > ep["prog"][0] for ep in episodes])
    print(f"[guard] frac episodes with progress[-1] > progress[0]: {frac_rising:.2f}")

    # --- pass 2: build rows ---
    S, A, M, RBMv, R, GT, CIDX, HOR, EID, BUC = ([] for _ in range(10))
    for ep in episodes:
        frames, feats, sa, prog = ep["frames"], ep["feats"], ep["sa"], ep["prog"]
        T = sa.shape[0]
        for k in range(n_samples - 1):  # 31 transitions
            f0, f1 = int(frames[k]), int(frames[k + 1])
            acts = sa[f0:min(f1, T), ACTION_SLICE].astype(np.float32)  # (len,14)
            a = np.zeros((L, 14), dtype=np.float32)
            mask = np.zeros((L,), dtype=np.float32)
            n = acts.shape[0]
            a[:n] = acts
            mask[:n] = 1.0
            S.append(feats[k]); A.append(a); M.append(mask)
            RBMv.append(prog[k]); R.append(prog[k + 1] - prog[k]); GT.append(ep["gt_final"])
            CIDX.append(k); HOR.append(k / (n_samples - 2)); EID.append(ep["ep_id"]); BUC.append(ep["bucket"])

    out = dict(
        s=np.asarray(S, np.float32), a_chunk=np.asarray(A, np.float32), a_mask=np.asarray(M, np.float32),
        rbm=np.asarray(RBMv, np.float32), reward=np.asarray(R, np.float32), gt_final=np.asarray(GT, np.float32),
        chunk_idx=np.asarray(CIDX, np.int64), horizon_frac=np.asarray(HOR, np.float32),
        episode_id=np.asarray(EID, np.int64), bucket=np.asarray(BUC, np.int64),
        is_expert=np.zeros(len(S), np.bool_), L=np.int64(L),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    r = out["reward"]
    print(f"\nrows={len(r)}  episodes=100  chunks/ep=31  a_chunk shape=(N,{L},14)")
    print(f"reward (raw, pre-clip): mean={r.mean():.4f} std={r.std():.4f} "
          f"min={r.min():.4f} max={r.max():.4f}")
    for q in (1, 5, 50, 95, 99):
        print(f"  p{q:02d} = {np.percentile(r, q):+.4f}")
    outside = np.mean((r < -0.25) | (r > 0.25))
    print(f"  frac |r|>0.25 = {outside:.3f}  -> {'WILD, clip recommended' if outside > 0.02 else 'tame, no clip'}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(r, bins=80, color="steelblue"); ax[0].set_title(f"reward r_k (raw)  n={len(r)}")
        ax[0].axvline(-0.25, c="r", ls="--"); ax[0].axvline(0.25, c="r", ls="--"); ax[0].set_yscale("log")
        for lab, c in ((1, "green"), (0, "crimson")):
            ax[1].hist(r[out["bucket"] == lab], bins=60, alpha=0.5, label=("good" if lab else "bad"), color=c)
        ax[1].set_title("reward by bucket"); ax[1].legend(); ax[1].set_yscale("log")
        p = Path(args.plotdir) / "reward_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=110)
        print(f"\nsaved reward plot -> {p}")
    except Exception as ex:
        print("plot skipped:", ex)

    # manifest: register 100 policy (used) + 25 expert (NOT used)
    policy = [dict(episode_id=ep["ep_id"],
                   bucket=("good" if ep["bucket"] == 1 else "bad"),
                   source="policy", use_in_training=True,
                   path=str(COLLECTED / ("good" if ep["bucket"] == 1 else "bad")
                            / f"episode_{(ep['ep_id'] if ep['bucket']==1 else ep['ep_id']-50):03d}"))
              for ep in episodes]
    expert_dirs = sorted(d for d in EXPERT_DIR.glob("episode_*") if d.is_dir())[:25]
    expert = [dict(episode_id=f"expert_{i:02d}", bucket="expert", source="expert",
                   use_in_training=False, path=str(d)) for i, d in enumerate(expert_dirs)]
    manifest = dict(grain="robometer_32frame", chunks_per_episode=31, action_window_L=L,
                    reward="rbm_progress[k+1]-rbm_progress[k]",
                    n_policy=len(policy), n_expert=len(expert),
                    note="expert episodes registered but NOT used in this run",
                    episodes=policy + expert)
    mpath = Path(args.out).with_name("critic_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"saved manifest ({len(policy)} policy used, {len(expert)} expert registered/unused) -> {mpath}")
    print(f"saved table -> {args.out}")


if __name__ == "__main__":
    main()
