#!/usr/bin/env python3
"""Bucket ABC eval worlds into good/ and bad/ by per-world success, up to N each.

Reads collect_*/summary.json, copies each world's video + states_actions.npy into
collected/{good,bad}/episode_<k>/ until N of each are gathered. Writes an index.json.
"""
import argparse, json, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-glob", default="collect_*")
    ap.add_argument("--out", default="/data2/abc/outputs/collected")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--mode", choices=["success", "bottles"], default="success",
                    help="success: strict 6/6; bottles: good>=good_thr, bad<=bad_thr by max bottles")
    ap.add_argument("--good-thr", type=int, default=5)
    ap.add_argument("--bad-thr", type=int, default=2)
    args = ap.parse_args()

    def maxb(w):
        return int(w.get("final_task_eval", {}).get("max_bottles_in_bin_so_far", -1))

    def is_good(w):
        return bool(w["success"]) if args.mode == "success" else maxb(w) >= args.good_thr

    def is_bad(w):
        return (not w["success"]) if args.mode == "success" else maxb(w) <= args.bad_thr

    summaries = sorted(Path("/data2/abc/outputs").glob(f"{args.collect_glob}/summary.json"))
    worlds = []
    for s in summaries:
        d = json.loads(s.read_text())
        for w in d.get("worlds", []):
            w["_src"] = str(s.parent)
            worlds.append(w)
    print(f"read {len(summaries)} summaries, {len(worlds)} total worlds")
    n_succ = sum(bool(w["success"]) for w in worlds)
    print(f"success_rate={n_succ}/{len(worlds)} = {n_succ/max(len(worlds),1):.2f}")

    out = Path(args.out)
    buckets = {"good": [w for w in worlds if is_good(w)], "bad": [w for w in worlds if is_bad(w)]}
    index = {}
    for name, ws in buckets.items():
        bdir = out / name
        if bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir(parents=True)
        kept = []
        for k, w in enumerate(ws[: args.n]):
            ep = bdir / f"episode_{k:03d}"
            ep.mkdir()
            vp, sap = w.get("video_path"), w.get("states_actions_path")
            if vp and Path(vp).exists():
                shutil.copy(vp, ep / "combined_camera-images-rgb.mp4")
            if sap and Path(sap).exists():
                shutil.copy(sap, ep / "states_actions.npy")
            meta = {kk: w[kk] for kk in ("world_index", "world_seed", "success", "final_success",
                                          "reward", "steps", "states_actions_shape") if kk in w}
            meta["max_bottles_in_bin"] = w.get("final_task_eval", {}).get("max_bottles_in_bin_so_far")
            meta["num_active_bottles"] = w.get("final_task_eval", {}).get("num_active_bottles")
            (ep / "metadata.json").write_text(json.dumps(meta, indent=2))
            kept.append(meta)
        index[name] = {"count": len(kept), "episodes": kept}
        status = "OK" if len(kept) >= args.n else f"SHORT (need {args.n - len(kept)} more)"
        print(f"{name}: kept {len(kept)}/{args.n}  [{status}]  -> {bdir}")
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {out/'index.json'}")


if __name__ == "__main__":
    main()
