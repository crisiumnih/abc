"""Featurize ABC eval episodes into a pooled critic state s for the s-validation probe.

s = mean-pool over the 36 vision tokens returned by DiTPolicy.build_vision_tokens,
computed on the cached, preprocessed observation in an UN-GRAPHED forward pass
(no CUDA graph, no reliance on hooks under fast_inference=True).

For each sampled frame of each episode we store:
  - feat_pooled : (1536,)  mean-pooled build_vision_tokens output  -> candidate critic state s
  - robot_state : (14,)    proprioceptive state (baseline probe input)
  - label       : 1 for good bucket, 0 for bad bucket (episode-level outcome)
  - episode_id  : integer, unique per episode (for episode-held-out splits)
  - frame_idx   : frame index within the episode

Episodes live in <collected>/{good,bad}/episode_*/ with:
  combined_camera-images-rgb.mp4  (cameras concatenated horizontally: top|left|right, each 224 wide)
  states_actions.npy              ((T, 28) = 14 proprio state + 14 action)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from abc_minimal.config import DiTConfig
from abc_minimal.dit import DiTPolicy, load_pretrained
from abc_minimal.preprocess import resize_pad_normalize

CAM_ORDER = ("top", "left", "right")  # horizontal tile order in the combined video
TILE_W = 224                          # each camera tile width (camera_width)


def split_frame(frame_hwc: np.ndarray) -> dict[str, np.ndarray]:
    """(H, 672, 3) HWC uint8 -> {cam: (3, H, 224) CHW uint8}."""
    out = {}
    for i, cam in enumerate(CAM_ORDER):
        tile = frame_hwc[:, i * TILE_W:(i + 1) * TILE_W, :]
        out[cam] = np.ascontiguousarray(tile.transpose(2, 0, 1))
    return out


def sample_frame_indices(n_frames: int, n_sample: int) -> list[int]:
    if n_sample >= n_frames:
        return list(range(n_frames))
    # evenly spaced over the whole episode
    return list(np.linspace(0, n_frames - 1, n_sample).round().astype(int))


@torch.no_grad()
def pooled_s(model: DiTPolicy, frames_split: list[dict[str, np.ndarray]], device) -> np.ndarray:
    """Run build_vision_tokens on a batch of frames; mean-pool 36 tokens -> (B, 1536)."""
    images = {}
    for cam in model.camera_keys:
        imgs = [resize_pad_normalize(fs[cam]) for fs in frames_split]  # each (3,224,224) fp32
        images[cam] = torch.stack(imgs, dim=0).to(device)
    vt = model.build_vision_tokens(images)        # (B, 36, 1536), UN-GRAPHED
    pooled = vt.float().mean(dim=1)               # mean-pool over 36 tokens -> (B, 1536)
    return pooled.cpu().numpy()


def load_model(ckpt_path: Path, device, dino_bf16: bool = True) -> DiTPolicy:
    model = DiTPolicy(DiTConfig()).to(device)
    load_pretrained(model, str(ckpt_path))
    model.eval()
    if dino_bf16 and device.type == "cuda":
        model.img_backbone.set_bfloat16(True)  # halves DINO activation memory; casts back to fp32
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default="/data2/abc/outputs/collected")
    ap.add_argument("--ckpt", default="/data2/abc/cache/bottles_75k.pt")
    ap.add_argument("--n-episodes", type=int, default=10, help="per bucket")
    ap.add_argument("--n-frames", type=int, default=16, help="sampled frames per episode")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.ckpt), device)

    feats, states, labels, ep_ids, frame_idxs = [], [], [], [], []
    ep_counter = 0
    for bucket, label in (("good", 1), ("bad", 0)):
        ep_dirs = sorted((Path(args.collected) / bucket).glob("episode_*"))[: args.n_episodes]
        for ep_dir in ep_dirs:
            sa = np.load(ep_dir / "states_actions.npy")          # (T, 28)
            reader = imageio.get_reader(str(ep_dir / "combined_camera-images-rgb.mp4"))
            n_frames = min(sa.shape[0], reader.count_frames())
            idxs = sample_frame_indices(n_frames, args.n_frames)
            frames_split = [split_frame(reader.get_data(i)) for i in idxs]
            reader.close()

            for b in range(0, len(frames_split), args.batch):
                chunk = frames_split[b:b + args.batch]
                chunk_idxs = idxs[b:b + args.batch]
                pooled = pooled_s(model, chunk, device)
                feats.append(pooled)
                states.append(sa[chunk_idxs, :14].astype(np.float32))
                labels.extend([label] * len(chunk))
                ep_ids.extend([ep_counter] * len(chunk))
                frame_idxs.extend(chunk_idxs)
            print(f"{bucket} {ep_dir.name}: {len(idxs)} frames (ep_id={ep_counter})", flush=True)
            ep_counter += 1

    feat_pooled = np.concatenate(feats, axis=0)
    robot_state = np.concatenate(states, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        feat_pooled=feat_pooled,
        robot_state=robot_state,
        label=np.asarray(labels, dtype=np.int64),
        episode_id=np.asarray(ep_ids, dtype=np.int64),
        frame_idx=np.asarray(frame_idxs, dtype=np.int64),
    )
    print(f"\nsaved {out}")
    print(f"feat_pooled {feat_pooled.shape}  robot_state {robot_state.shape}  "
          f"n={len(labels)}  episodes={ep_counter}  pos={int(np.sum(labels))}")


if __name__ == "__main__":
    main()
