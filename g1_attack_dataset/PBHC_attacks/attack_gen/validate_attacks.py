#!/usr/bin/env python3
"""Summarise a generated attack dataset and (optionally) load it through PBHC's own motion libraries.

    python attack_gen/validate_attacks.py --data motion_data/attacks            # summary table
    python attack_gen/validate_attacks.py --data motion_data/attacks --pbhc     # + PBHC load/FK check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np


def summary(index):
    clips = index["clips"]
    rows = []
    for a in index["attacks"]:
        cs = [c for c in clips if c["attack"] == a]
        q = lambda k: np.array([c["quality"][k] for c in cs])
        v = lambda k: np.array([c[k] for c in cs])
        dur = v("num_frames") / index["fps"]
        lead = (v("impact_frame") - v("onset_frame")) / index["fps"]
        rows.append([a, len(cs),
                     f"{v('distance').min():.2f}-{v('distance').max():.2f}",
                     f"{1000 * lead.mean():.0f} ({1000 * lead.min():.0f}-{1000 * lead.max():.0f})",
                     f"{dur.mean():.2f}",
                     f"{100 * v('miss_distance').mean():.1f} / {100 * v('miss_distance').max():.1f}",
                     f"{v('impact_speed').mean():.1f}",
                     f"{v('peak_striker_speed').max():.1f}",
                     f"{q('max_dof_vel').max():.1f}",
                     f"{1000 * q('max_foot_slip').max():.0f}",
                     f"{100 * q('min_com_margin').min():.1f}",
                     f"{100 * np.median(q('min_zmp_margin')):.0f}",
                     f"{1000 * q('max_self_penetration').max():.0f}"])
    hdr = ["attack", "n", "D [m]", "onset->impact ms", "len s", "miss cm avg/max", "v_imp m/s",
           "v_peak", "max qd rad/s", "foot slip mm", "min CoM margin cm", "median min ZMP margin cm", "self-pen mm"]
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = lambda r: "| " + " | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) + " |"
    print(line(hdr))
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print(line(r))
    print("rejected attempts during generation:", index.get("rejected_attempts"))


def pbhc_check(pbhc_root, data):
    """Load every clip with PBHC's MotionLibRobot ('origin', multi-motion) and one with the WJX
    single-motion library, and compare their forward kinematics with MuJoCo's."""
    import joblib
    import mujoco
    import torch
    from omegaconf import OmegaConf
    sys.path[:0] = [pbhc_root, os.path.join(pbhc_root, "humanoidverse/isaac_utils")]
    try:
        import open3d  # noqa: F401
    except ImportError:  # only needed by PBHC for mesh export, not for FK
        o3d = types.ModuleType("open3d")
        o3d.io = types.SimpleNamespace(read_triangle_mesh=lambda f: None)
        sys.modules["open3d"] = o3d
    cwd = os.getcwd()
    data = os.path.abspath(data)
    os.chdir(pbhc_root)
    try:
        cfg = OmegaConf.load("humanoidverse/config/robot/g1/g1_23dof_lock_wrist.yaml").robot.motion
        m = mujoco.MjModel.from_xml_path("description/robots/g1/g1_23dof_lock_wrist.xml")
        d = mujoco.MjData(m)
        names = list(cfg.body_names) + [e.joint_name for e in cfg.extend_config]
        allfile = os.path.join(data, "attacks_all.pkl")
        src_all = joblib.load(allfile)

        def compare(ml, n):
            worst = 0.0
            keys = np.atleast_1d(ml.curr_motion_keys)
            for mid in range(n):
                key = keys[mid]
                key = os.path.basename(key)[:-4] if key.endswith(".pkl") else key
                src = src_all[key]
                for f in range(0, len(src["dof"]), 3):
                    st = ml.get_motion_state(torch.tensor([mid]), torch.tensor([f / src["fps"]], dtype=torch.float32))
                    d.qpos[:] = np.r_[src["root_trans_offset"][f], src["root_rot"][f][[3, 0, 1, 2]], src["dof"][f]]
                    mujoco.mj_kinematics(m, d)
                    pm = st["rg_pos_t"][0].numpy()
                    for bi, bn in enumerate(names):
                        if bn.endswith("hand_link"):
                            b = d.body(bn.replace("hand", "elbow")); p = b.xpos + b.xmat.reshape(3, 3) @ [0.25, 0, 0]
                        elif bn == "head_link":
                            b = d.body("torso_link"); p = b.xpos + b.xmat.reshape(3, 3) @ [0, 0, 0.42]
                        else:
                            p = d.body(bn).xpos
                        worst = max(worst, float(np.abs(p - pm[bi]).max()))
            return worst

        from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
        cfg.motion_file = allfile
        n = len(src_all)
        ml = MotionLibRobot(cfg, num_envs=n, device="cpu")
        ml.load_motions(random_sample=False)
        print(f"MotionLibRobot (file mode): {ml._num_unique_motions} clips, contact mask: {ml.has_contact_mask}, "
              f"max FK error {1000 * compare(ml, n):.3f} mm")
        first = sorted(src_all)[0]
        from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
        cfg.motion_file = os.path.join(data, first.rsplit("_", 1)[0], first + ".pkl")
        ml = MotionLibRobotWJX(cfg, num_envs=1, device="cpu")
        ml.load_motions(random_sample=False)
        print(f"MotionLibRobotWJX (single clip {first}): max FK error {1000 * compare(ml, 1):.3f} mm")
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="motion_data/attacks")
    ap.add_argument("--pbhc_root", default=".")
    ap.add_argument("--pbhc", action="store_true", help="also load the clips with PBHC's motion libraries")
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.data, "index.json")))
    summary(index)
    if args.pbhc:
        pbhc_check(os.path.abspath(args.pbhc_root), args.data)


if __name__ == "__main__":
    main()
