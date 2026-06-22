#!/usr/bin/env python3
"""Build final good/bad ABC episode buckets.

GOOD = all 6/6 successes + the shortest-to-5 5/6 episodes (ranked by the first step the
episode reached 5 bottles), capped at N. Uses summary.json (needs per-chunk bottles).
BAD  = <=2/6 episodes (clearest failures: fewest bottles first), capped at N. Uses summaries
       plus a fallback parse of worker logs so worlds from not-yet-finished workers count.

Each kept episode -> collected/<good|bad>/episode_NNN/{combined_camera-images-rgb.mp4,
states_actions.npy, metadata.json}. Writes index.json.
"""
import json, re, shutil, glob
from pathlib import Path

OUT = Path("/data2/abc/outputs/collected")
N = 50


def reach_step(w):
    mb = w["final_task_eval"]["max_bottles_in_bin_so_far"]
    if w.get("success"):
        return int(w["steps"])
    for c in w.get("chunk_metrics", []):
        if c.get("max_bottles", 0) >= mb:
            return (c["chunk"] + 1) * 15
    return 1800


def load_summary_worlds():
    ws = []
    for f in glob.glob("/data2/abc/outputs/collect_bc_*/summary.json"):
        d = json.load(open(f))
        for w in d["worlds"]:
            w["_dir"] = str(Path(f).parent)
            ws.append(w)
    return ws


def _ep_files(seed, idx):
    d = Path(f"/data2/abc/outputs/collect_bc_{seed}")
    return {"_dir": str(d), "world_index": idx,
            "video_path": str(d / f"world_{idx:03d}.mp4"),
            "states_actions_path": str(d / f"world_{idx:03d}_states_actions.npy")}


def load_log_good():
    """All 6/6 successes across worker logs (carry steps); these may not be in summaries yet."""
    out = []
    pat = re.compile(r"world=(\d+) done success=True bottles=6/6 steps=(\d+)")
    for lf in glob.glob("/tmp/abc_bc_*.log"):
        seed = re.search(r"abc_bc_(\d+)\.log", lf).group(1)
        for m in pat.finditer(open(lf).read()):
            idx, steps = int(m.group(1)), int(m.group(2))
            out.append({**_ep_files(seed, idx), "success": True, "maxb": 6,
                        "steps": steps, "reach": steps})
    return out


def load_log_bad():
    """All <=2/6 worlds across worker logs (covers not-yet-summarised workers)."""
    out = []
    pat = re.compile(r"world=(\d+) done success=False bottles=(\d)/\d")
    for lf in glob.glob("/tmp/abc_bc_*.log"):
        seed = re.search(r"abc_bc_(\d+)\.log", lf).group(1)
        for m in pat.finditer(open(lf).read()):
            idx, mb = int(m.group(1)), int(m.group(2))
            if mb <= 2:
                out.append({**_ep_files(seed, idx), "max_bottles": mb, "success": False})
    return out


def copy_ep(w, ep, extra):
    ep.mkdir(parents=True, exist_ok=True)
    vp, sap = w.get("video_path"), w.get("states_actions_path")
    if vp and Path(vp).exists():
        shutil.copy(vp, ep / "combined_camera-images-rgb.mp4")
    if sap and Path(sap).exists():
        shutil.copy(sap, ep / "states_actions.npy")
    meta = {"world_index": w.get("world_index"), "world_seed": w.get("world_seed"),
            "success": bool(w.get("success")), **extra}
    (ep / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    worlds = load_summary_worlds()
    for w in worlds:
        w["maxb"] = w["final_task_eval"]["max_bottles_in_bin_so_far"]
        w["reach"] = reach_step(w)

    # ALL 6/6 successes from logs (authoritative, incl. workers without summaries yet).
    succ = sorted(load_log_good(), key=lambda w: w["steps"])
    # 5/6 from summaries (need per-chunk metrics to rank by time-to-reach-5).
    five = sorted([w for w in worlds if not w["success"] and w["maxb"] == 5], key=lambda w: w["reach"])
    good = (succ + five)[:N]

    # bad: <=2/6 from logs, dedup by (dir,index), fewest bottles first.
    badmap = {}
    for w in load_log_bad():
        badmap[(w["_dir"], w["world_index"])] = w
    bad = sorted(badmap.values(), key=lambda w: w["max_bottles"])[:N]

    if OUT.exists():
        shutil.rmtree(OUT)
    idx = {"good": [], "bad": []}
    for k, w in enumerate(good):
        idx["good"].append(copy_ep(w, OUT / "good" / f"episode_{k:03d}",
                                   {"max_bottles": int(w["maxb"]), "steps": int(w["steps"]),
                                    "reach_step": int(w["reach"]),
                                    "kind": "6/6_success" if w["success"] else "5/6_fast"}))
    for k, w in enumerate(bad):
        idx["bad"].append(copy_ep(w, OUT / "bad" / f"episode_{k:03d}",
                                  {"max_bottles": int(w["max_bottles"])}))
    (OUT / "index.json").write_text(json.dumps(idx, indent=2))

    n6 = sum(1 for w in good if w["success"]); n5 = len(good) - n6
    print(f"GOOD kept {len(good)}/{N}  ({n6} x 6/6 + {n5} x shortest-5/6)"
          + ("" if len(good) >= N else "  [SHORT]"))
    if good and not good[-1]["success"]:
        print(f"  5/6 reach-step cutoff (longest kept): {good[-1]['reach']} steps")
    print(f"BAD  kept {len(bad)}/{N}" + ("" if len(bad) >= N else f"  [SHORT by {N-len(bad)}]"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
