#!/usr/bin/env python3
"""Generate a labelled dataset of G1 attack motions in PBHC's motion format.

Run from the PBHC repository root (it reads the G1 model from description/robots/g1/):

    python attack_gen/generate_attacks.py --out motion_data/attacks --per_attack 50

Output layout
    <out>/<attack>/<attack>_0000.pkl   one clip per file, joblib dict {stem: motion}
                                       (PBHC directory mode, or pass one file to train on it)
    <out>/attacks_all.pkl              every clip in one joblib dict {stem: motion} (PBHC file mode)
    <out>/index.json                   per-clip labels, timings, targets and quality metrics
    <out>/index.csv                    the same, flat
    <out>/defender_guard.pkl           the canonical defender guard pose as a static 2 s motion

Each motion dict has PBHC's keys (root_trans_offset, root_rot (xyzw), dof (23), pose_aa (27x3),
smpl_joints, fps, contact_mask (T x 2)) plus two extra keys PBHC ignores:
    attack_meta  dict of labels/timings/targets (see README)
    phase        (T,) int8 per-frame phase: 0 guard, 1 strike (onset..impact), 2 retract, 3 guard
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_attacks.attacks import ATTACKS  # noqa: E402

_G = {}


def _init(xml, dof_axis_path, fps):
    from g1_attacks.generate import G1Kinematics, make_defender
    kin = G1Kinematics(xml)
    _G.update(kin=kin, dfd=make_defender(kin), dof_axis=np.load(dof_axis_path).astype(np.float32), fps=fps)


def _work(job):
    from g1_attacks.generate import make_clip, to_pbhc
    name, idx, seed = job
    rng = np.random.default_rng(seed)
    log = []
    for _ in range(5):
        try:
            qs, contact, meta, phase, traj, chk = make_clip(_G["kin"], _G["dfd"], name, rng, fps=_G["fps"], log=log)
            break
        except RuntimeError:   # all attempts rejected: keep drawing from the same stream
            continue
    else:
        raise RuntimeError(f"{name} #{idx}: no acceptable clip after 30 attempts")
    motion = to_pbhc(qs, contact, _G["fps"], _G["dof_axis"])
    stem = f"{name}_{idx:04d}"
    meta = dict(meta, clip=stem, seed=int(seed), num_frames=int(len(qs)),
                quality={k: float(v) for k, v in chk.items()})
    motion["attack_meta"] = meta
    motion["phase"] = phase
    return stem, motion, [r[3] for r in log]


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 5) for x in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pbhc_root", default=".", help="PBHC repository root")
    ap.add_argument("--out", default="motion_data/attacks")
    ap.add_argument("--per_attack", type=int, default=50)
    ap.add_argument("--attacks", default=",".join(ATTACKS), help="comma-separated subset of: " + ",".join(ATTACKS))
    ap.add_argument("--fps", type=int, default=50, help="PBHC's policy runs at 50 Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--no_combined", action="store_true", help="skip writing attacks_all.pkl")
    args = ap.parse_args()

    xml = os.path.join(args.pbhc_root, "description/robots/g1/g1_23dof_lock_wrist.xml")
    dof_axis = os.path.join(args.pbhc_root, "description/robots/g1/dof_axis.npy")
    for p in (xml, dof_axis):
        if not os.path.isfile(p):
            sys.exit(f"missing {p}; run from the PBHC repo root or pass --pbhc_root")
    names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    bad = set(names) - set(ATTACKS)
    if bad:
        sys.exit(f"unknown attacks: {sorted(bad)}")

    jobs = []
    for name in names:   # per-attack seed streams: a clip does not depend on which other attacks are generated
        seeds = np.random.SeedSequence([args.seed, ATTACKS.index(name)]).generate_state(args.per_attack)
        jobs += [(name, i, int(s)) for i, s in enumerate(seeds)]

    os.makedirs(args.out, exist_ok=True)
    allm, index, rejects = {}, [], {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(xml, dof_axis, args.fps)) as pool:
        for n, (stem, motion, rej) in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), 1):
            name = motion["attack_meta"]["attack"]
            d = os.path.join(args.out, name)
            os.makedirs(d, exist_ok=True)
            joblib.dump({stem: motion}, os.path.join(d, stem + ".pkl"))
            allm[stem] = motion
            index.append(_jsonable(motion["attack_meta"]))
            for r in rej:
                for reason in r:
                    rejects[reason] = rejects.get(reason, 0) + 1
            if n % 10 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)

    index.sort(key=lambda m: m["clip"])
    # canonical defender guard pose (clip frame of a D=0 placement), as a 2 s static PBHC motion
    from g1_attacks.generate import G1Kinematics, make_defender, to_pbhc
    kin = G1Kinematics(xml)
    q = make_defender(kin).q_local
    n = 2 * args.fps
    guard = to_pbhc(np.repeat(q[None], n, 0), np.ones((n, 2)), args.fps, np.load(dof_axis).astype(np.float32))
    joblib.dump({"defender_guard": guard}, os.path.join(args.out, "defender_guard.pkl"))
    if not args.no_combined:
        joblib.dump({k: allm[k] for k in sorted(allm)}, os.path.join(args.out, "attacks_all.pkl"))
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(dict(attacks=names, label_ids={a: ATTACKS.index(a) for a in names}, fps=args.fps,
                       rejected_attempts=rejects, clips=index), f, indent=1)
    cols = ["clip", "attack", "label", "striker", "target_region", "target_link", "num_frames", "fps",
            "onset_frame", "impact_frame", "full_extension_frame", "recovered_frame", "distance",
            "speed_scale", "miss_distance", "impact_speed", "peak_striker_speed"]
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["target_point_x", "target_point_y", "target_point_z"])
        for m in index:
            w.writerow([m[c] for c in cols] + m["target_point"])
    print(f"wrote {len(index)} clips to {args.out} in {time.time() - t0:.0f}s; rejected attempts: {rejects}")


if __name__ == "__main__":
    main()
