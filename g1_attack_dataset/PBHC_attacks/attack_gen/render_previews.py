#!/usr/bin/env python3
"""Render preview GIFs of generated attacks (attacker in orange, canonical defender in blue,
target point in green). Needs an OpenGL context: on a headless machine use
    MUJOCO_GL=glfw xvfb-run -a python attack_gen/render_previews.py --data motion_data/attacks
or MUJOCO_GL=egl with a GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import joblib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_attacks.attacks import ATTACKS  # noqa: E402
from g1_attacks.generate import G1Kinematics, make_defender  # noqa: E402
from g1_attacks.render import Renderer  # noqa: E402


def qpos(motion, f):
    x, y, z, w = motion["root_rot"][f]
    return np.r_[motion["root_trans_offset"][f], [w, x, y, z], motion["dof"][f]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbhc_root", default=".")
    ap.add_argument("--data", default="motion_data/attacks")
    ap.add_argument("--out", default=None, help="default: <data>/previews")
    ap.add_argument("--clips", default=None, help="comma-separated clip names (default: first of each attack)")
    ap.add_argument("--views", default="side,oblique")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--stride", type=int, default=2, help="render every n-th frame")
    args = ap.parse_args()
    xml = os.path.join(args.pbhc_root, "description/robots/g1/g1_23dof_lock_wrist.xml")
    out = args.out or os.path.join(args.data, "previews")
    os.makedirs(out, exist_ok=True)
    index = json.load(open(os.path.join(args.data, "index.json")))
    clips = args.clips.split(",") if args.clips else [
        next(c["clip"] for c in index["clips"] if c["attack"] == a) for a in ATTACKS if a in index["attacks"]]
    kin = G1Kinematics(xml)
    dfd = make_defender(kin)
    rd = Renderer(xml, args.size, args.size)
    views = args.views.split(",")
    sheet = []
    for clip in clips:
        attack = clip.rsplit("_", 1)[0]
        m = joblib.load(os.path.join(args.data, attack, clip + ".pkl"))[clip]
        meta = m["attack_meta"]
        qd = dfd.q_world(float(meta["distance"]))
        frames = []
        keep = set(range(0, len(m["dof"]), args.stride)) | {meta["onset_frame"], meta["impact_frame"]}
        idx = sorted(keep)
        for f in idx:
            row = [rd.frame(qpos(m, f), qd, meta["target_point"], view=v) for v in views]
            im = Image.fromarray(np.concatenate(row, axis=1))
            dr = ImageDraw.Draw(im)
            ph = ["guard", "STRIKE", "retract", "guard"][int(m["phase"][f])]
            hit = " IMPACT" if abs(f - meta["impact_frame"]) <= 1 else ""
            dr.text((6, 6), f"{clip}  t={f / m['fps']:.2f}s  {ph}{hit}", fill=(255, 255, 255))
            frames.append(np.array(im))
        dur = [1000 * ((idx[i + 1] if i + 1 < len(idx) else idx[i] + args.stride) - idx[i]) / m["fps"] for i in range(len(idx))]
        imageio.mimsave(os.path.join(out, clip + ".gif"), frames, duration=dur, loop=0)
        k, on = meta["impact_frame"], meta["onset_frame"]
        pick = lambda f: frames[idx.index(min(idx, key=lambda x: abs(x - f)))]
        sheet.append(np.concatenate([pick(on), pick((on + k) // 2), pick(k)], axis=1))
        print("wrote", os.path.join(out, clip + ".gif"))
    Image.fromarray(np.concatenate(sheet, axis=0)).save(os.path.join(out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
