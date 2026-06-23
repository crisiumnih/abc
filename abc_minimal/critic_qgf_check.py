"""Differentiability + SIGN check for QGF guidance, BEFORE any closed-loop eval.

Confirms on a real observation:
  (1) g = grad_{a1} Q(s, a1) is finite and non-zero in the eager path (autograd flows through critic).
  (2) SIGN: guided sample_actions (weight>0) yields an action with HIGHER Q than unguided (weight=0),
      i.e. positive 1/beta increases Q under ABC's dt<0 convention (validates the minus-sign step).
If (1) fails -> differentiability blocker. If (2) fails -> sign is backwards.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from abc_minimal.critic_featurize import load_model, split_frame
from abc_minimal.critic_qgf import QGFGuidance
from abc_minimal.preprocess import resize_pad_normalize, parse_norm_stats, normalize
import imageio.v2 as imageio

BUNDLE = Path("/data2/experiemnts/transfer_bundle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data2/abc/cache/bottles_75k.pt")
    ap.add_argument("--critic", default="/data/tmp/claude-1000/-data2-experiemnts/"
                    "e13cbe9d-be59-4b9a-8186-693ee5df9bb7/scratchpad/critic_qgf.pt")
    ap.add_argument("--weights", default="0.5,1,2,4")
    ap.add_argument("--mode", choices=["qgf", "qfql"], default="qgf")
    ap.add_argument("--frame", type=int, default=300)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), dev)  # fp32 eager
    guidance = QGFGuidance(args.critic, dev)
    nstats = parse_norm_stats(torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)["norm_stats"])

    # one real observation (a mid-episode frame of the first good episode)
    row = next(r for r in csv.DictReader(open(BUNDLE / "labels.csv")) if r["bucket"] == "good")
    base = BUNDLE / "abc_eval" / "gt_eval" / row["shard"]
    sa = np.load(base / f"{row['world']}_states_actions.npy")
    reader = imageio.get_reader(str(base / f"{row['world']}.mp4"))
    f = min(args.frame, reader.count_frames() - 1, sa.shape[0] - 1)
    cams = split_frame(reader.get_data(f)); reader.close()
    images = {c: resize_pad_normalize(cams[c]).unsqueeze(0).to(dev) for c in model.camera_keys}
    state = torch.from_numpy(normalize(sa[f, :14].astype(np.float32), nstats["state"])[None]).float().to(dev)
    batch = {"state": state, "images": images,
             "task_vec_clip": torch.zeros(1, 512, device=dev),
             "actions": torch.zeros(1, model.chunk_length, model.action_dim, device=dev)}

    with torch.no_grad():
        s_pooled = model.build_vision_tokens(images).mean(dim=1).float()

    # (1) finite/nonzero grad at a random a1
    a1 = torch.randn(1, model.chunk_length, model.action_dim, device=dev)
    g, q = guidance.grad(s_pooled, a1)
    print(f"(1) DIFFERENTIABILITY: g finite={torch.isfinite(g).all().item()} "
          f"|g|mean={g.abs().mean().item():.4e} nonzero={(g.abs()>0).any().item()}  Q(rand a1)={q.item():+.4f}")

    # (2) sign: guided (w>0) raises Q of the produced action vs unguided
    noise = torch.randn(1, model.chunk_length, model.action_dim, device=dev)
    with torch.no_grad():
        out0 = model.sample_actions(batch, num_steps=10, noise=noise, guidance=None, guidance_weight=0.0)
        q0 = guidance.q(s_pooled, out0).item()
    print(f"(2) SIGN check  mode={args.mode}  (unguided Q={q0:+.4f}):")
    ok = True
    for w in [float(x) for x in args.weights.split(",")]:
        with torch.no_grad():
            outw = model.sample_actions(batch, num_steps=10, noise=noise, guidance=guidance,
                                        guidance_weight=w, guidance_mode=args.mode)
            qw = guidance.q(s_pooled, outw).item()
            dnorm = (outw - out0).norm().item()
        flag = "OK(raises Q)" if qw > q0 else "BACKWARDS(lowers Q)"
        ok = ok and (qw > q0)
        print(f"   1/beta={w:>4}: Q={qw:+.4f}  ΔQ={qw-q0:+.4f}  |Δaction|={dnorm:.3f}  -> {flag}")
    print(f"\n>>> {'PASS' if ok else 'FAIL (sign backwards -> flip the guidance sign)'} <<<")


if __name__ == "__main__":
    main()
