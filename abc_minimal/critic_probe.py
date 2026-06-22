"""s-validation probe: can a LINEAR probe on the candidate critic state s predict
the episode outcome bucket (good=1 / bad=0)?

Compares two probe inputs on identical frames and identical episode-held-out splits:
  - pooled-s    : mean-pooled build_vision_tokens (1536-d)  -> the candidate critic state
  - robot_state : proprioceptive state (14-d)               -> baseline (expected near chance)

Splits are GROUPED BY EPISODE (GroupKFold) so frames from a test episode never appear
in training -- otherwise within-episode frame correlation inflates accuracy.

Pass criterion: pooled-s clearly beats chance (>~65-70%) AND beats robot_state.

Reports both frame-level and episode-level (per-episode majority vote) accuracy.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch


def grouped_folds(groups: np.ndarray, k: int, seed: int = 0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    assign = np.array([fold_of[g] for g in groups])
    for f in range(k):
        test = assign == f
        yield ~test, test


def train_logreg(X, y, l2=1e-2, iters=500, lr=0.5):
    """L2-regularized logistic regression via full-batch LBFGS. X already standardized."""
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=iters, line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        logits = Xt @ w + b
        loss = bce(logits, yt) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def predict(X, w, b):
    return (X @ w + b) > 0.0


def run_probe(name, feats, labels, episodes, k=5, l2=1e-2):
    frame_accs, ep_accs = [], []
    for tr, te in grouped_folds(episodes, k):
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
        Xtr, Xte = (feats[tr] - mu) / sd, (feats[te] - mu) / sd
        w, b = train_logreg(Xtr, labels[tr].astype(np.float32), l2=l2)
        pred = predict(Xte, w, b)
        frame_accs.append((pred == labels[te]).mean())

        # episode-level: majority vote of frame predictions per test episode
        for ep in np.unique(episodes[te]):
            m = episodes[te] == ep
            vote = pred[m].mean() > 0.5
            ep_accs.append(float(vote == labels[te][m][0]))

    fa, ea = np.array(frame_accs), np.array(ep_accs)
    print(f"  {name:<12} frame-acc = {fa.mean():.3f} +/- {fa.std():.3f}   "
          f"episode-acc = {ea.mean():.3f}  (n_ep_test={len(ea)})")
    return fa.mean(), ea.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()

    d = np.load(args.data)
    feat_pooled = d["feat_pooled"].astype(np.float32)
    robot_state = d["robot_state"].astype(np.float32)
    label = d["label"]
    episode = d["episode_id"]

    n = len(label)
    chance = max(label.mean(), 1 - label.mean())
    print(f"data: n={n} frames, episodes={len(np.unique(episode))}, "
          f"pos={int(label.sum())}/{n}, majority-class chance={chance:.3f}")
    print(f"GroupKFold by episode, k={args.k}, l2={args.l2}\n")

    s_frame, s_ep = run_probe("pooled-s", feat_pooled, label, episode, args.k, args.l2)
    r_frame, r_ep = run_probe("robot_state", robot_state, label, episode, args.k, args.l2)

    print("\n=== VERDICT ===")
    print(f"pooled-s    episode-acc {s_ep:.3f}  (frame {s_frame:.3f})")
    print(f"robot_state episode-acc {r_ep:.3f}  (frame {r_frame:.3f})")
    print(f"chance      {chance:.3f}")
    beats_chance = s_frame > 0.65
    beats_state = s_frame > r_frame + 0.02
    verdict = "PASS" if (beats_chance and beats_state) else "FAIL"
    print(f"\npooled-s beats ~0.65 chance: {beats_chance} | beats robot_state: {beats_state}")
    print(f">>> {verdict} <<<")
    if verdict == "FAIL":
        print("If FAIL: escalate to DINOv3 patch tokens (Candidate 1), pooled.")


if __name__ == "__main__":
    main()
