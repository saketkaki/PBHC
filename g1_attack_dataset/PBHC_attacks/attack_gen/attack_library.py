"""Batched replay of the generated attacks for a defender-training environment.

The attacker is meant to be driven kinematically (a "ghost"): each step, write the root state and
joint state returned by `AttackLibrary.state(...)` into the attacker actor (e.g. with IsaacGym's
set_actor_root_state_tensor_indexed / set_dof_state_tensor_indexed), and read the labels,
impact timing and target from the same library for rewards and termination.

Frames
    Clips are stored in the *clip frame*: the attacker starts near the origin facing +x and the
    canonical defender (a G1 in guard) faces it along -x. Its root (pelvis) pose in the clip frame
    is attack_meta["defender_root_pos"/"defender_root_rot"/"defender_root_yaw"], and its joint
    angles are attack_meta["defender_dof"], so you can reset your defender into exactly that pose.
    `place_relative_to_defender` maps a clip state into the world so that the clip's defender root
    lands on your defender robot's root position and heading (normally sampled at episode reset).

Quaternions are xyzw everywhere (PBHC / IsaacGym convention).

Example
    lib = AttackLibrary("motion_data/attacks/attacks_all.pkl", device="cuda")
    ids = lib.sample(num_envs)                       # balanced over attack classes
    t = torch.zeros(num_envs, device="cuda")
    ...
    s = lib.state(ids, t)                            # dict of (num_envs, ...) tensors
    pos, rot, lin_vel, ang_vel = lib.place_relative_to_defender(ids, s, def_xy, def_yaw)
    # attacker dof_pos = s["dof_pos"], dof_vel = s["dof_vel"]
    t += env_dt
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import joblib
import numpy as np
import torch

ATTACKS = ["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut", "body_shot", "calf_kick"]


def _quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def _quat_rotate(q, v):  # xyzw, rotate v by q
    qv, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def _yaw_quat(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


def _slerp(q0, q1, t):
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    small = s < 1e-6
    w0 = torch.where(small, 1 - t, torch.sin((1 - t) * theta) / s.clamp(min=1e-6))
    w1 = torch.where(small, t, torch.sin(t * theta) / s.clamp(min=1e-6))
    q = w0 * q0 + w1 * q1
    return q / q.norm(dim=-1, keepdim=True)


class AttackLibrary:
    def __init__(self, path: str, device="cpu", attacks: Optional[Sequence[str]] = None):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True))
            files = [f for f in files if os.path.basename(f) != "attacks_all.pkl"]
            data = {}
            for f in files:
                data.update(joblib.load(f))
        else:
            data = joblib.load(path)
        keys = sorted(k for k, m in data.items() if attacks is None or m["attack_meta"]["attack"] in attacks)
        assert keys, "no clips found"
        self.keys = keys
        self.device = device
        ms = [data[k] for k in keys]
        self.meta = [m["attack_meta"] for m in ms]
        fps = {int(m["fps"]) for m in ms}
        assert len(fps) == 1, f"mixed fps {fps}"
        self.fps = fps.pop()
        self.dt = 1.0 / self.fps
        N, T = len(ms), max(len(m["dof"]) for m in ms)
        self.num_clips, self.max_frames = N, T

        def pad(name, dim, dtype=np.float32):
            out = np.zeros((N, T) + dim, dtype)
            for i, m in enumerate(ms):
                x = np.asarray(m[name])
                out[i, :len(x)] = x
                out[i, len(x):] = x[-1]          # hold the last frame
            return torch.as_tensor(out, device=device)

        self.root_pos = pad("root_trans_offset", (3,))
        self.root_rot = pad("root_rot", (4,))
        self.dof_pos = pad("dof", (23,))
        self.contact = pad("contact_mask", (2,))
        self.phase = pad("phase", (), np.int64)
        self.length = torch.tensor([len(m["dof"]) for m in ms], device=device)
        # finite-difference velocities (world frame)
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.dt, dim=1)[0]
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.dt, dim=1)[0]
        q = self.root_rot
        dq = torch.gradient(q, spacing=self.dt, dim=1)[0]
        qc = q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=device)
        self.root_ang_vel = 2.0 * _quat_mul(dq, qc)[..., :3]

        f = lambda k, dt=torch.float32: torch.tensor([m[k] for m in self.meta], dtype=dt, device=device)
        self.label = f("label", torch.long)
        self.onset_frame = f("onset_frame", torch.long)
        self.impact_frame = f("impact_frame", torch.long)
        self.recovered_frame = f("recovered_frame", torch.long)
        self.distance = f("distance")
        self.def_root_pos = torch.tensor(np.stack([m["defender_root_pos"] for m in self.meta]), device=device)
        self.def_root_rot = torch.tensor(np.stack([m["defender_root_rot"] for m in self.meta]), device=device)
        self.def_root_yaw = f("defender_root_yaw")
        self.def_dof = torch.tensor(np.stack([m["defender_dof"] for m in self.meta]), device=device)
        self.target_point = torch.tensor(np.stack([m["target_point"] for m in self.meta]), device=device)
        self.target_local = torch.tensor(np.stack([m["target_local_offset"] for m in self.meta]), device=device)
        self.target_link = [m["target_link"] for m in self.meta]
        self.striker = [m["striker"] for m in self.meta]
        self.attack = [m["attack"] for m in self.meta]
        self._by_label = {l: torch.nonzero(self.label == l).squeeze(-1) for l in self.label.unique().tolist()}

    # ------------------------------------------------------------------ sampling
    def sample(self, n, labels: Optional[Sequence[int]] = None, balanced=True, generator=None):
        """Clip ids. balanced=True draws an attack class uniformly, then a clip within it."""
        labs = list(self._by_label) if labels is None else list(labels)
        if balanced:
            cls = torch.tensor(labs, device=self.device)[
                torch.randint(len(labs), (n,), device=self.device, generator=generator)]
            ids = torch.empty(n, dtype=torch.long, device=self.device)
            for l in labs:
                sel = cls == l
                pool = self._by_label[l]
                ids[sel] = pool[torch.randint(len(pool), (int(sel.sum()),), device=self.device, generator=generator)]
            return ids
        pool = torch.cat([self._by_label[l] for l in labs])
        return pool[torch.randint(len(pool), (n,), device=self.device, generator=generator)]

    # ------------------------------------------------------------------ playback
    def duration(self, ids):
        return (self.length[ids] - 1).float() * self.dt

    def state(self, ids, t):
        """Interpolated clip state at time t (seconds since clip start); t is clamped to the clip."""
        L = self.length[ids]
        ft = (t / self.dt).clamp(min=0.0)
        f0 = torch.minimum(ft.floor().long(), L - 1)
        f1 = torch.minimum(f0 + 1, L - 1)
        a = (ft - f0.float()).clamp(0.0, 1.0)
        lerp = lambda x: x[ids, f0] + a.view(-1, *([1] * (x.dim() - 2))) * (x[ids, f1] - x[ids, f0])
        near = torch.where(a < 0.5, f0, f1)
        return dict(
            root_pos=lerp(self.root_pos), root_rot=_slerp(self.root_rot[ids, f0], self.root_rot[ids, f1], a[:, None]),
            root_lin_vel=lerp(self.root_lin_vel), root_ang_vel=lerp(self.root_ang_vel),
            dof_pos=lerp(self.dof_pos), dof_vel=lerp(self.dof_vel),
            contact=self.contact[ids, near], phase=self.phase[ids, near], frame=near,
            time_to_impact=(self.impact_frame[ids] - ft) * self.dt,
            done=f0 >= L - 1,
        )

    def place_relative_to_defender(self, ids, s, defender_xy, defender_yaw):
        """World root state for the attacker such that the clip's canonical defender root lands on
        (defender_xy, defender_yaw). Returns (pos, rot_xyzw, lin_vel, ang_vel)."""
        qz = _yaw_quat(defender_yaw - self.def_root_yaw[ids])
        p = s["root_pos"].clone()
        p[:, :2] -= self.def_root_pos[ids, :2]
        pos = _quat_rotate(qz, p)
        pos[:, :2] += defender_xy
        rot = _quat_mul(qz, s["root_rot"])
        return pos, rot, _quat_rotate(qz, s["root_lin_vel"]), _quat_rotate(qz, s["root_ang_vel"])

    def target_world(self, ids, defender_xy, defender_yaw):
        """Canonical target point (clip frame) mapped like place_relative_to_defender. For hit
        detection on a moving defender prefer: target = link_pose(target_link) * target_local."""
        qz = _yaw_quat(defender_yaw - self.def_root_yaw[ids])
        p = self.target_point[ids].clone()
        p[:, :2] -= self.def_root_pos[ids, :2]
        p = _quat_rotate(qz, p)
        p[:, :2] += defender_xy
        return p
