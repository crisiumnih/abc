"""Build the DUAL-REWARD chunk table from the transfer_bundle (114 episodes, GT + Robometer).

Robometer 32-frame grid (same as before). Per episode, 31 transition rows k=0..30:
  s_k          : (1536,) mean-pool(build_vision_tokens) at video frame_k (re-featurized here)
  a_chunk_k    : (L,14)  actions over [frame_k, frame_{k+1}), padded to L; a_mask_k marks real steps
  gt_progress  : bottles[frame_k]/6        (per-step GT bottle count, sampled at grid)
  rbm_progress : robometer progress[k]
  reward_gt    : gt_progress[k+1]  - gt_progress[k]
  reward_rbm   : rbm_progress[k+1] - rbm_progress[k]
  chunk_idx, horizon_frac, episode_id (0..113), bucket (1 good>=5 / 0 bad<=2 / -1 mid), world_seed

Frame grid = linspace(0, n-1, 32) with n=min(states_rows, video_frames) -- matches how Robometer
sampled 32 frames over the same video, so index k <-> progress[k] (no interpolation).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from abc_minimal.critic_featurize import load_model, pooled_s, split_frame, sample_frame_indices

BUNDLE = Path("/data2/experiemnts/transfer_bundle")
N_SAMPLES = 32
ACTION_SLICE = slice(14, 28)
BUCKET_MAP = {"good": 1, "bad": 0, "mid": -1, "success": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data2/abc/cache/bottles_75k.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="/data/tmp/claude-1000/-data2-experiemnts/"
                    "e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad/chunk_table_gt.npz")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(BUNDLE / "labels.csv")))
    print(f"labels.csv: {len(rows)} episodes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device)

    # pass 1: gather per-episode arrays + frame grids; find max interval L
    eps = []
    for ei, r in enumerate(rows):
        shard, world = r["shard"], r["world"]
        base = BUNDLE / "abc_eval" / "gt_eval" / shard
        sa = np.load(base / f"{world}_states_actions.npy")
        bot = np.load(base / f"{world}_bottles.npy").astype(np.float32)
        rbm = np.load(BUNDLE / "robometer_eval" / "out_run" / "per_episode" /
                      f"ep_{shard}_{world}.npz", allow_pickle=True)["progress"].astype(np.float32)
        assert len(rbm) == N_SAMPLES, f"{shard}/{world} rbm len {len(rbm)}"
        reader = imageio.get_reader(str(base / f"{world}.mp4"))
        nframes = min(sa.shape[0], reader.count_frames(), len(bot))
        frames = np.array(sample_frame_indices(nframes, N_SAMPLES))
        assert len(frames) == N_SAMPLES
        eps.append(dict(ei=ei, reader=reader, sa=sa, bot=bot, rbm=rbm, frames=frames,
                        bucket=BUCKET_MAP.get(r["bucket"], -1), seed=int(r["world_seed"]),
                        max_bottles=int(r["max_bottles"])))

    L = max(int(e["frames"][k + 1] - e["frames"][k]) for e in eps for k in range(N_SAMPLES - 1))
    intervals = [int(e["frames"][k + 1] - e["frames"][k]) for e in eps for k in range(N_SAMPLES - 1)]
    print(f"interval steps: min={min(intervals)} median={int(np.median(intervals))} max(L)={L}")

    # pass 2: featurize s at grid frames + build rows
    S, A, M, RG, RR, GP, RP, CI, HO, EID, BUC, SEED = ([] for _ in range(12))
    for e in eps:
        reader, sa, bot, rbm, frames = e["reader"], e["sa"], e["bot"], e["rbm"], e["frames"]
        fr = [split_frame(reader.get_data(int(f))) for f in frames]
        reader.close()
        feats = np.concatenate([pooled_s(model, fr[b:b + args.batch], device)
                                for b in range(0, len(fr), args.batch)], axis=0)  # (32,1536)
        gtp = bot[frames] / 6.0
        for k in range(N_SAMPLES - 1):
            f0, f1 = int(frames[k]), int(frames[k + 1])
            acts = sa[f0:f1, ACTION_SLICE].astype(np.float32)
            a = np.zeros((L, 14), np.float32); msk = np.zeros((L,), np.float32)
            n = min(acts.shape[0], L); a[:n] = acts[:n]; msk[:n] = 1.0
            S.append(feats[k]); A.append(a); M.append(msk)
            GP.append(gtp[k]); RP.append(rbm[k])
            RG.append(gtp[k + 1] - gtp[k]); RR.append(rbm[k + 1] - rbm[k])
            CI.append(k); HO.append(k / (N_SAMPLES - 2)); EID.append(e["ei"]); BUC.append(e["bucket"]); SEED.append(e["seed"])
        print(f"ep {e['ei']:03d} bucket={e['bucket']:+d} done", flush=True)

    out = dict(s=np.asarray(S, np.float32), a_chunk=np.asarray(A, np.float32), a_mask=np.asarray(M, np.float32),
               reward_gt=np.asarray(RG, np.float32), reward_rbm=np.asarray(RR, np.float32),
               gt_progress=np.asarray(GP, np.float32), rbm_progress=np.asarray(RP, np.float32),
               chunk_idx=np.asarray(CI, np.int64), horizon_frac=np.asarray(HO, np.float32),
               episode_id=np.asarray(EID, np.int64), bucket=np.asarray(BUC, np.int64),
               world_seed=np.asarray(SEED, np.int64), L=np.int64(L))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)

    b = out["bucket"]
    nb = {v: int((np.array([eps[i]["bucket"] for i in range(len(eps))]) == v).sum()) for v in (1, 0, -1)}
    print(f"\nsaved {args.out}")
    print(f"rows={len(S)} episodes={len(eps)} a_chunk=(N,{L},14)  buckets good={nb[1]} bad={nb[0]} mid={nb[-1]}")
    for src in ("reward_gt", "reward_rbm"):
        r = out[src]; print(f"{src}: mean={r.mean():+.4f} std={r.std():.4f} p01={np.percentile(r,1):+.3f} p99={np.percentile(r,99):+.3f} |r|>0.25={(np.abs(r)>0.25).mean():.3f}")


if __name__ == "__main__":
    main()
