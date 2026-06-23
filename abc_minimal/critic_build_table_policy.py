"""Policy-aligned GT critic table for QGF (decision-point granularity, NORMALIZED action chunks).

Unlike the Robometer-grid table, this aligns to the policy's own decision points so the critic's
action input matches the diffusion variable a_1 that QGF guides:

Per episode, at decision frames f = 0, 15, 30, ...  (stride = execute_chunk_dim = 15):
  s            : (1536,) mean-pool(build_vision_tokens) at video frame f
  a_chunk      : (30,14) the next 30 executed actions, NORMALIZED via norm_stats["actions"]
                 (matches the policy's chunk_length=30 diffusion variable, same normalized space)
  a_mask       : (30,)   1 for real steps, 0 for padding (only near episode end)
  reward       : gt_progress(f+15) - gt_progress(f),  gt_progress = bottles/6   (GT, executed horizon)
  chunk_idx    : decision index 0,1,2,...
  horizon_frac : f / MAX_STEPS (1800)   -> SAME bins as eval scope gate (early<0.3/mid/late)
  episode_id, bucket (1 good>=5 / 0 bad<=2 / -1 mid), world_seed

This is the critic QGF queries: Q(s, a_1) with a_1 the 30-step normalized chunk.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from abc_minimal.critic_featurize import load_model, pooled_s, split_frame
from abc_minimal.dit import load_pretrained  # noqa: F401  (model loaded in load_model)
from abc_minimal.preprocess import parse_norm_stats, normalize

BUNDLE = Path("/data2/experiemnts/transfer_bundle")
STRIDE = 15          # execute_chunk_dim
CHUNK = 30           # policy chunk_length
MAX_STEPS = 1800     # horizon denominator (matches eval scope gate)
ACTION_SLICE = slice(14, 28)
BUCKET_MAP = {"good": 1, "bad": 0, "mid": -1, "success": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data2/abc/cache/bottles_75k.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="/data/tmp/claude-1000/-data2-experiemnts/"
                    "e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad/chunk_table_policy.npz")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BUNDLE / "labels.csv")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    nstats = parse_norm_stats(ckpt["norm_stats"])
    a_stats = nstats["actions"]  # dict mean/std, shape (14,)
    print(f"labels.csv: {len(rows)} episodes; action norm mean[:3]={np.round(a_stats['mean'][:3],3)}")

    S, A, M, R, CI, HO, EID, BUC, SEED = ([] for _ in range(9))
    for ei, r in enumerate(rows):
        shard, world = r["shard"], r["world"]
        base = BUNDLE / "abc_eval" / "gt_eval" / shard
        sa = np.load(base / f"{world}_states_actions.npy")
        bot = np.load(base / f"{world}_bottles.npy").astype(np.float32)
        reader = imageio.get_reader(str(base / f"{world}.mp4"))
        T = min(sa.shape[0], reader.count_frames(), len(bot))
        gtp = bot / 6.0
        bucket = BUCKET_MAP.get(r["bucket"], -1); seed = int(r["world_seed"])

        frames = list(range(0, T - 1, STRIDE))
        fr = [split_frame(reader.get_data(int(f))) for f in frames]
        reader.close()
        feats = np.concatenate([pooled_s(model, fr[b:b + args.batch], device)
                                for b in range(0, len(fr), args.batch)], axis=0)  # (nf,1536)

        for j, f in enumerate(frames):
            acts_raw = sa[f:f + CHUNK, ACTION_SLICE].astype(np.float32)         # (<=30,14) raw
            acts_n = normalize(acts_raw, a_stats).astype(np.float32)             # normalized (policy space)
            a = np.zeros((CHUNK, 14), np.float32); msk = np.zeros((CHUNK,), np.float32)
            n = min(acts_n.shape[0], CHUNK); a[:n] = acts_n[:n]; msk[:n] = 1.0
            f_next = min(f + STRIDE, T - 1)
            S.append(feats[j]); A.append(a); M.append(msk)
            R.append(float(gtp[f_next] - gtp[f]))
            CI.append(j); HO.append(f / MAX_STEPS); EID.append(ei); BUC.append(bucket); SEED.append(seed)
        print(f"ep {ei:03d} ({shard}/{world}) frames={len(frames)} bucket={bucket:+d}", flush=True)

    out = dict(s=np.asarray(S, np.float32), a_chunk=np.asarray(A, np.float32), a_mask=np.asarray(M, np.float32),
               reward=np.asarray(R, np.float32), chunk_idx=np.asarray(CI, np.int64),
               horizon_frac=np.asarray(HO, np.float32), episode_id=np.asarray(EID, np.int64),
               bucket=np.asarray(BUC, np.int64), world_seed=np.asarray(SEED, np.int64),
               action_mean=a_stats["mean"].astype(np.float32), action_std=a_stats["std"].astype(np.float32),
               chunk_len=np.int64(CHUNK), max_steps=np.int64(MAX_STEPS))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    rr = out["reward"]
    print(f"\nsaved {args.out}")
    print(f"rows={len(S)} episodes={len(rows)} a_chunk=(N,{CHUNK},14) normalized")
    print(f"reward(GT delta): mean={rr.mean():+.4f} std={rr.std():.4f} p01={np.percentile(rr,1):+.3f} p99={np.percentile(rr,99):+.3f}")
    print(f"horizon spread: early(<0.3)={int((out['horizon_frac']<0.3).sum())} "
          f"mid={int(((out['horizon_frac']>=0.3)&(out['horizon_frac']<0.7)).sum())} late={int((out['horizon_frac']>=0.7).sum())}")


if __name__ == "__main__":
    main()
