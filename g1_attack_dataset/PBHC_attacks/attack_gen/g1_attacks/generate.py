"""Turn attack choreographies into PBHC-format G1 motion clips, with quality checks."""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from .attacks import (ATTACKS, ATTACK_ID, D_MIN, D_RANGE, STRIKER, TARGET_REGION,  # noqa: F401
                      Choreo, Defender, Stance, build_attack, guard_targets)
from .ik import DEFAULT_DOF, DOF_LOWER, DOF_UPPER, G1Kinematics  # noqa: F401

FOOT_RECT = np.array([[-0.066, -0.036], [0.142, -0.036], [0.142, 0.036], [-0.066, 0.036]])  # sole, ankle frame


def solve_guard(kin: G1Kinematics, st: Stance, g):
    ch = Choreo(st, g)
    q = np.zeros(30); q[2] = 0.79; q[3] = 1.0; q[7:] = DEFAULT_DOF
    for _ in range(3):
        q = kin.solve(q, ch.ik_targets(kin, 0.0), iters=200, tol=1e-7)
    return q, ch


def make_defender(kin):
    st = Stance()
    q, _ = solve_guard(kin, st, guard_targets(st))
    return Defender(kin, q)


def striker_point(kin, name):
    s = STRIKER[name]
    return kin.hand("left") if s == "left_hand" else kin.hand("right") if s == "right_hand" else kin.shin()


def sample_params(name, rng):
    return dict(
        stance=Stance.sample(rng, kick=(name == "calf_kick")),
        D=float(rng.uniform(*D_RANGE[name])),
        aim=rng.uniform(-0.012, 0.012, 2),       # aim noise (y, z) in metres
        onset=float(rng.uniform(0.35, 0.9)),     # time spent in guard before the attack starts
        speed=float(rng.uniform(0.85, 1.2)),     # >1 = faster
        tail=float(rng.uniform(0.35, 0.7)),      # guard time after recovery
        seed=int(rng.integers(1 << 31)),
    )


def generate_clip(kin, defender, name, params, fps=50, iters=40, w_prev=0.3):
    st = params["stance"]
    g = guard_targets(st)
    q0, _ = solve_guard(kin, st, g)
    D = params["D"]
    target = defender.target_world(name, D) + np.r_[0.0, params["aim"]]
    onset, speed, tail = params["onset"], params["speed"], params["tail"]
    ch, extra = build_attack(name, st, g, target, np.random.default_rng(params["seed"]), onset, speed, tail)
    T_end = ch.events["recovered"] + tail
    n = int(np.floor(T_end * fps)) + 1
    ts = np.arange(n) / fps

    qs = np.zeros((n, 30))
    strike = np.zeros((n, 3)); com = np.zeros((n, 3)); feet = np.zeros((n, 2, 3)); fyaw = np.zeros((n, 2))
    hands = np.zeros((n, 2, 3)); head = np.zeros((n, 3))
    contact = np.zeros((n, 2))
    q = q0.copy()
    for i, t in enumerate(ts):
        q = kin.solve(q, ch.ik_targets(kin, t), iters=iters, q_prev=q.copy(), w_prev=w_prev, tol=1e-6)
        qs[i] = q
        strike[i] = striker_point(kin, name)
        com[i] = kin.com()
        for j, side in enumerate(("left", "right")):
            b = kin.d.body(f"{side}_ankle_roll_link")
            feet[i, j] = b.xpos
            fyaw[i, j] = np.arctan2(b.xmat[3], b.xmat[0])
            hands[i, j] = kin.hand(side)
        head[i] = kin.head()
        contact[i] = [1.0, ch.rf_contact(t)]

    _qd = defender.q_world(D)
    meta = dict(
        attack=name, label=ATTACK_ID[name], striker=STRIKER[name], target_region=TARGET_REGION[name],
        target_link=defender.target_local[name][0],
        target_local_offset=defender.target_local[name][1].astype(np.float32),
        target_point=target.astype(np.float32),
        defender_root_pos=_qd[:3].astype(np.float32),
        defender_root_rot=np.r_[_qd[4:7], _qd[3:4]].astype(np.float32),   # xyzw
        defender_root_yaw=np.float32(R.from_quat(np.r_[_qd[4:7], _qd[3:4]]).as_euler("ZYX")[0]),
        defender_dof=_qd[7:].astype(np.float32),
        distance=np.float32(D), speed_scale=np.float32(speed), fps=fps,
        onset_time=np.float32(ch.events["onset"]), recovered_time=np.float32(ch.events["recovered"]),
    )
    # impact frame = closest approach of the striking point to the target. Strikes are aimed a few
    # cm *through* the target, so this is when the striker crosses it (still moving fast), shortly
    # before full extension.
    lo = int((ch.events["onset"]) * fps); hi = min(n, int((ch.events["impact"] + 0.06) * fps) + 1)
    dist = np.linalg.norm(strike - target, axis=1)
    k = lo + int(np.argmin(dist[lo:hi]))
    meta["impact_frame"] = int(k)
    meta["impact_time"] = np.float32(k / fps)
    meta["full_extension_frame"] = int(round(ch.events["impact"] * fps))
    meta["onset_frame"] = int(round(ch.events["onset"] * fps))
    meta["recovered_frame"] = int(round(ch.events["recovered"] * fps))
    meta["impact_point"] = strike[k].astype(np.float32)
    meta["miss_distance"] = np.float32(dist[k])
    vel = np.gradient(strike, 1.0 / fps, axis=0)
    meta["impact_velocity"] = vel[k].astype(np.float32)
    meta["impact_speed"] = np.float32(np.linalg.norm(vel[k]))
    meta["peak_striker_speed"] = np.float32(np.linalg.norm(vel, axis=1).max())
    # per-frame phase label: 0 guard, 1 wind-up/strike (onset..impact), 2 retract, 3 guard after
    phase = np.zeros(n, np.int8)
    phase[meta["onset_frame"]:k + 1] = 1
    phase[k + 1:meta["recovered_frame"] + 1] = 2
    phase[meta["recovered_frame"] + 1:] = 3
    traj = dict(strike=strike, com=com, feet=feet, fyaw=fyaw, contact=contact, hands=hands, head=head, ts=ts)
    return qs, contact, meta, phase, traj


# --------------------------------------------------------------------------------------
# PBHC export
# --------------------------------------------------------------------------------------
def to_pbhc(qs, contact, fps, dof_axis):
    root_trans = qs[:, :3].astype(np.float32)
    wxyz = qs[:, 3:7]
    root_rot = np.concatenate([wxyz[:, 1:], wxyz[:, :1]], axis=1)  # xyzw, as PBHC
    root_rot *= np.sign(root_rot[:, 3:4] + 1e-12)
    dof = qs[:, 7:].astype(np.float32)
    root_aa = R.from_quat(root_rot).as_rotvec()
    T = len(qs)
    pose_aa = np.concatenate([root_aa[:, None], dof_axis[None] * dof[:, :, None], np.zeros((T, 3, 3))], axis=1)
    return {
        "root_trans_offset": root_trans,
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof,
        "root_rot": root_rot.astype(np.float32),
        "smpl_joints": np.zeros_like(pose_aa, dtype=np.float32),
        "fps": int(fps),
        "contact_mask": contact.astype(np.float64),
    }


# --------------------------------------------------------------------------------------
# quality checks
# --------------------------------------------------------------------------------------
def _support_hull(feet, fyaw, contact):
    pts = []
    for j in range(2):
        if contact[j] > 0.5:
            c, s = np.cos(fyaw[j]), np.sin(fyaw[j])
            Rm = np.array([[c, -s], [s, c]])
            pts.append(feet[j, :2] + FOOT_RECT @ Rm.T)
    pts = np.concatenate(pts)
    return pts[ConvexHull(pts).vertices]


def _signed_margin(poly, p):
    """>0 inside (distance to nearest edge), <0 outside. poly is CCW."""
    m = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)  # outward normal for CCW
        m = min(m, -np.dot(p - a, nrm))
    return m


def check_clip(kin, qs, traj, fps):
    n = len(qs)
    dof = qs[:, 7:]
    dv = np.gradient(dof, 1.0 / fps, axis=0)
    com = traj["com"]
    from scipy.signal import savgol_filter
    win = max(5, int(round(0.1 * fps)) | 1)   # ~0.1 s window: ignores single-frame noise
    acc = savgol_filter(com[:, :2], win, 3, deriv=2, delta=1.0 / fps, axis=0, mode="nearest")
    zmp = com[:, :2] - (com[:, 2:3] / 9.81) * acc
    com_m, zmp_m = np.zeros(n), np.zeros(n)
    for i in range(n):
        hull = _support_hull(traj["feet"][i], traj["fyaw"][i], traj["contact"][i])
        com_m[i] = _signed_margin(hull, com[i, :2])
        zmp_m[i] = _signed_margin(hull, zmp[i])
    slip = 0.0   # largest drift of a planted foot within one contiguous stance segment
    for j in range(2):
        c = traj["contact"][:, j] > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            seg = traj["feet"][a:b, j]
            # ignore the first 3 frames after touchdown (the foot settles)
            seg = seg[3:] if a > 0 and len(seg) > 3 else seg
            slip = max(slip, float(np.abs(seg - seg[0]).max()))
    # self-penetration between non-adjacent bodies
    m, d = kin.m, kin.d
    pen = 0.0
    for i in range(0, n, 2):
        d.qpos[:] = qs[i]
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == 0 or b2 == 0:
                continue
            pen = max(pen, -c.dist)
    return dict(
        max_dof_vel=float(np.abs(dv).max()),
        min_limit_margin=float(min((dof - DOF_LOWER).min(), (DOF_UPPER - dof).min())),
        min_com_margin=float(com_m.min()),
        min_zmp_margin=float(zmp_m.min()),
        max_foot_slip=float(slip),
        min_foot_z=float(traj["feet"][..., 2].min()),
        max_self_penetration=float(pen),
    )


ACCEPT = dict(max_miss=0.03, max_miss_kick=0.045, min_impact_speed=1.0, max_foot_slip=0.02,
              max_self_penetration=0.015, min_com_margin=0.0, max_dof_vel=30.0)


def accepted(meta, chk):
    tol = ACCEPT["max_miss_kick"] if meta["attack"] == "calf_kick" else ACCEPT["max_miss"]
    reasons = []
    if meta["miss_distance"] > tol: reasons.append("miss")
    if meta["impact_speed"] < ACCEPT["min_impact_speed"]: reasons.append("slow_impact")
    if chk["max_foot_slip"] > ACCEPT["max_foot_slip"]: reasons.append("foot_slip")
    if chk["max_self_penetration"] > ACCEPT["max_self_penetration"]: reasons.append("self_collision")
    if chk["min_com_margin"] < ACCEPT["min_com_margin"]: reasons.append("com_outside_support")
    if chk["max_dof_vel"] > ACCEPT["max_dof_vel"]: reasons.append("dof_vel")
    return reasons


def make_clip(kin, defender, name, rng, fps=50, max_tries=6, log=None):
    """Sample until a clip passes the checks. If the strike falls short, the defender is moved
    closer by the shortfall and the same clip is re-solved once before resampling."""
    for attempt in range(max_tries):
        params = sample_params(name, rng)
        for adjust in range(3):
            qs, contact, meta, phase, traj = generate_clip(kin, defender, name, params, fps=fps)
            chk = check_clip(kin, qs, traj, fps)
            bad = accepted(meta, chk)
            if not bad:
                meta["attempts"] = attempt + 1
                return qs, contact, meta, phase, traj, chk
            if log is not None:
                log.append((name, attempt, adjust, bad, float(meta["miss_distance"]), float(meta["impact_speed"])))
            if not ({"miss", "slow_impact"} & set(bad)) or set(bad) - {"miss", "slow_impact"}:
                break
            # strike fell short: bring the defender closer by the shortfall along x (+5 cm overshoot),
            # but never closer than D_MIN (the two robots' guards are already touching there)
            short = float(meta["target_point"][0] - traj["strike"][meta["impact_frame"]][0])
            D_new = params["D"] - min(max(short, 0.0) + 0.05, 0.08)
            if D_new < D_MIN[name]:
                break
            params = dict(params, D=D_new)
    raise RuntimeError(f"could not generate an acceptable {name} clip")
