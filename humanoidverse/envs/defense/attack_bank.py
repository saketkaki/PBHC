"""Attack clips of the G1 attack dataset (g1_attack_dataset/PBHC_attacks) for defender training.

The attacker is a kinematic ghost: it is not an actor in the simulator and never collides. At load time every clip is
run once through the robot's forward kinematics (Humanoid_Batch, as in the motion library), so at run time the attacker
is a lookup of body positions in the clip frame, placed in the world relative to the defender.

Clip frame (dataset README): the attacker starts near the origin facing +x; the canonical defender, a G1 in the guard of
defender_guard.pkl, stands at attack_meta['defender_root_pos'] with heading attack_meta['defender_root_yaw'].
"""
import glob
import os
import sys

import joblib
import numpy as np
import torch

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

ATTACKS = ["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut", "body_shot", "calf_kick"]
NUM_PHASES = 4  # attack_meta phase: 0 guard, 1 strike (onset to impact), 2 retract, 3 guard


def load_joblib_compat(path):
    """ joblib.load that also reads pickles written with NumPy 2 (which reference the module numpy._core) under NumPy 1,
    as the released attack clips are. """
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if 'numpy._core' not in str(e):
            raise
    import numpy.core.multiarray, numpy.core.numeric  # noqa: E401,F401
    for name in ('', '.multiarray', '.numeric'):
        sys.modules.setdefault('numpy._core' + name, sys.modules['numpy.core' + name])
    return joblib.load(path)


def closest_point_on_segment(p, a, b):
    """ Point of the segments a-b closest to the points p, all [..., 3] (broadcast). """
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
    return a + t.clamp(0.0, 1.0)[..., None] * ab


def point_segment_distance(p, a, b):
    """ Distance from points p to segments a-b, all [..., 3] (broadcast). """
    return (p - closest_point_on_segment(p, a, b)).norm(dim=-1)


def sweep_segment_closest(p0, p1, a, b, samples=5):
    """ Closest approach of the swept path p0 -> p1 [N, 3] to the segments a-b [N, S, 3], sampling the sweep: distance
    [N], the segment index [N] and the point on that segment [N, 3]. """
    s = torch.linspace(0.0, 1.0, samples, device=p0.device)
    path = p0[:, None] + s[None, :, None] * (p1 - p0)[:, None]  # [N, samples, 3]
    q = closest_point_on_segment(path[:, :, None], a[:, None], b[:, None])  # [N, samples, S, 3]
    dist, i = (path[:, :, None] - q).norm(dim=-1).flatten(1).min(dim=1)
    n, S = torch.arange(len(p0), device=p0.device), a.shape[1]
    return dist, i % S, q[n, i // S, i % S]


def sweep_segment_distance(p0, p1, a, b, samples=5):
    """ Smallest distance between the swept path p0 -> p1 [N, 3] and the segments a-b [N, S, 3], sampling the sweep. """
    return sweep_segment_closest(p0, p1, a, b, samples)[0]


class AttackBank:
    def __init__(self, path, motion_cfg, device, attacks=None, keypoints=()):
        if os.path.isdir(path):
            data = {}
            for f in sorted(glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True)):
                if os.path.basename(f) not in ("attacks_all.pkl", "defender_guard.pkl"):
                    data.update(load_joblib_compat(f))
        else:
            data = load_joblib_compat(path)
        keys = sorted(k for k, m in data.items() if 'attack_meta' in m and (attacks is None or m['attack_meta']['attack'] in attacks))
        assert keys, f"no attack clips in {path}"
        clips = [data[k] for k in keys]
        meta = [m['attack_meta'] for m in clips]
        fps = {int(m['fps']) for m in clips}
        assert len(fps) == 1, f"mixed fps {fps}"
        self.keys, self.device = keys, device
        self.dt = 1.0 / fps.pop()
        self.num_clips = N = len(clips)
        self.max_frames = T = max(len(m['dof']) for m in clips)

        # forward kinematics of every clip in the clip frame (bodies of the robot's motion config, extended bodies last)
        fk = Humanoid_Batch(motion_cfg)
        self.body_names = list(fk.body_names_augment)
        self.body_parents = fk._parents.tolist()
        body_pos = np.zeros((N, T, len(self.body_names), 3), np.float32)
        body_rot = np.zeros((N, T, len(self.body_names), 4), np.float32)
        for i, m in enumerate(clips):
            pose_aa = torch.as_tensor(np.asarray(m['pose_aa']), dtype=torch.float32)[None]
            trans = torch.as_tensor(np.asarray(m['root_trans_offset']), dtype=torch.float32)[None]
            out = fk.fk_batch(pose_aa, trans)
            pos, rot = out.global_translation_extend[0].numpy(), out.global_rotation_extend[0].numpy()
            body_pos[i, :len(pos)], body_pos[i, len(pos):] = pos, pos[-1]  # hold the last frame
            body_rot[i, :len(rot)], body_rot[i, len(rot):] = rot, rot[-1]
        self.body_pos = torch.as_tensor(body_pos, device=device)  # [N, T, B, 3]
        self.body_rot = torch.as_tensor(body_rot, device=device)  # [N, T, B, 4] xyzw, for drawing the attacker as a G1

        # striking point: fist = extended hand link (elbow + 0.25 m); shin = 75 % of the way from the right knee to the ankle
        idx = self.body_names.index
        striker = np.zeros((N, T, 3), np.float32)
        for i, m in enumerate(meta):
            if m['striker'] == 'right_shin':
                knee, ankle = body_pos[i, :, idx('right_knee_link')], body_pos[i, :, idx('right_ankle_pitch_link')]
                striker[i] = knee + 0.75 * (ankle - knee)
            else:
                striker[i] = body_pos[i, :, idx(m['striker'] + '_link')]
        self.striker_pos = torch.as_tensor(striker, device=device)  # [N, T, 3]
        self.keypoint_ids = torch.tensor([idx(k) for k in keypoints], dtype=torch.long, device=device)

        phase = np.zeros((N, T), np.int64)
        for i, m in enumerate(clips):
            phase[i, :len(m['phase'])], phase[i, len(m['phase']):] = m['phase'], m['phase'][-1]
        self.phase = torch.as_tensor(phase, device=device)

        f = lambda k, dtype=torch.float32: torch.tensor([m[k] for m in meta], dtype=dtype, device=device)
        self.label = f('label', torch.long)
        self.num_classes = len(ATTACKS)
        assert int(self.label.max()) < self.num_classes
        self.length = torch.tensor([len(m['dof']) for m in clips], dtype=torch.long, device=device)
        self.onset_frame = f('onset_frame', torch.long)
        self.impact_frame = f('impact_frame', torch.long)
        self.recovered_frame = f('recovered_frame', torch.long)
        self.def_root_xy = torch.tensor(np.stack([np.asarray(m['defender_root_pos'])[:2] for m in meta]), dtype=torch.float32, device=device)
        self.def_root_yaw = f('defender_root_yaw')
        self.target_link = [m['target_link'] for m in meta]
        self.target_local = torch.tensor(np.stack([m['target_local_offset'] for m in meta]), dtype=torch.float32, device=device)
        self.attack_names = [m['attack'] for m in meta]

        # balanced sampling: an attack class uniformly, then a clip of that class
        order = torch.argsort(self.label, stable=True)
        counts = torch.bincount(self.label, minlength=self.num_classes)
        self._sorted_ids = order
        self._present_classes = torch.nonzero(counts).squeeze(-1)
        self._class_start = torch.cumsum(counts, 0) - counts
        self._class_count = counts

    def sample(self, n):
        cls = self._present_classes[torch.randint(len(self._present_classes), (n,), device=self.device)]
        offset = (torch.rand(n, device=self.device) * self._class_count[cls]).long().clamp(max=self._class_count[cls] - 1)
        return self._sorted_ids[self._class_start[cls] + offset]

    def duration(self, ids):
        return (self.length[ids] - 1).float() * self.dt

    def frame(self, ids, t):
        """ Continuous frame index at time t [s] (clamped to the clip). """
        return (t / self.dt).clamp(min=0.0).minimum((self.length[ids] - 1).float())

    def _lerp(self, x, ids, t):
        ft = self.frame(ids, t)
        f0 = ft.floor().long()
        f1 = torch.minimum(f0 + 1, self.length[ids] - 1)
        a = (ft - f0.float()).view(-1, *([1] * (x.dim() - 2)))
        return x[ids, f0] + a * (x[ids, f1] - x[ids, f0])

    def bodies(self, ids, t):
        """ Clip-frame body positions [n, B, 3] at time t. """
        return self._lerp(self.body_pos, ids, t)

    def body_rots(self, ids, t):
        """ Clip-frame body rotations [n, B, 4] (xyzw) at time t, interpolated linearly and normalized. """
        ft = self.frame(ids, t)
        f0 = ft.floor().long()
        f1 = torch.minimum(f0 + 1, self.length[ids] - 1)
        q0, q1 = self.body_rot[ids, f0], self.body_rot[ids, f1]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = q0 + (ft - f0.float())[:, None, None] * (q1 - q0)
        return q / q.norm(dim=-1, keepdim=True)

    def to_world_rot(self, rots, ids, anchor_yaw):
        """ Rotate clip-frame rotations [n, ..., 4] (xyzw) like to_world. """
        half = 0.5 * (anchor_yaw - self.def_root_yaw[ids]).view((-1,) + (1,) * (rots.dim() - 2))
        cz, sz = torch.cos(half), torch.sin(half)  # yaw quaternion (0, 0, sz, cz) times rots
        x, y, z, w = rots.unbind(-1)
        return torch.stack([cz * x - sz * y, cz * y + sz * x, cz * z + sz * w, cz * w - sz * z], dim=-1)

    def striker(self, ids, t):
        return self._lerp(self.striker_pos, ids, t)

    def standoff_weight(self, ids, t):
        """ Share [n] of the standoff applied at time t: 1 in guard, easing to 0 at the impact frame (the attacker steps
        in during the strike, so the striker lands where the clip aimed it) and back to 1 by the recovered frame. """
        f = self.frame(ids, t)
        onset, impact, recovered = self.onset_frame[ids].float(), self.impact_frame[ids].float(), self.recovered_frame[ids].float()
        step_in = ((impact - f) / (impact - onset).clamp(min=1.0)).clamp(0.0, 1.0)
        step_out = ((f - impact) / (recovered - impact).clamp(min=1.0)).clamp(0.0, 1.0)
        s = torch.where(f <= impact, step_in, step_out)
        return s * s * (3.0 - 2.0 * s)  # smoothstep: no jump in the attacker's velocity at the onset or recovery

    def phase_at(self, ids, t):
        return self.phase[ids, self.frame(ids, t).round().long()]

    def to_world(self, points, ids, anchor, anchor_yaw):
        """ Map clip-frame points [n, ..., 3] so that the clip's canonical defender root lands on anchor [n, 3] (x, y and
        ground height) with heading anchor_yaw [n]. """
        yaw = anchor_yaw - self.def_root_yaw[ids]
        shape = (-1,) + (1,) * (points.dim() - 2)
        cos, sin = torch.cos(yaw).view(shape), torch.sin(yaw).view(shape)
        x = points[..., 0] - self.def_root_xy[ids, 0].view(shape)
        y = points[..., 1] - self.def_root_xy[ids, 1].view(shape)
        return torch.stack([cos * x - sin * y + anchor[:, 0].view(shape),
                            sin * x + cos * y + anchor[:, 1].view(shape),
                            points[..., 2] + anchor[:, 2].view(shape)], dim=-1)
