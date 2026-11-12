"""Counterattack clips of the defense env (env.config.defense.counter): kung-fu motions the defender performs against the
attacker after a blocked or evaded attack. The clips are made by scripts/make_defense_counters.py; each starts and ends
in the guard of defender_guard.pkl, at its root position and heading, so the reference stays continuous when the env
switches from the held guard to a clip and back.

PBHC's WJX motion library holds one motion, so every clip gets its own library; the env's reference of the robots that
are countering is overwritten with the state of their clip (write_states).
"""
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.envs.defense.attack_bank import load_joblib_compat
from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

STRIKERS = ['left_hand_link', 'right_hand_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shin', 'right_shin']
SHIN = 0.75  # a shin striker is this share of the way from the knee link to the ankle pitch link


def striker_positions(pos, body, name):
    """ Positions [..., 3] of the striker `name` from body positions [..., B, 3]; body(name) -> index. """
    if name.endswith('_shin'):
        side = name.split('_')[0]
        knee, ankle = pos[..., body(side + '_knee_link'), :], pos[..., body(side + '_ankle_pitch_link'), :]
        return knee + SHIN * (ankle - knee)
    return pos[..., body(name), :]


class CounterBank:
    def __init__(self, clip_dir, names, motion_cfg, device):
        self.names, self.device = list(names), device
        self.num_clips = len(self.names)
        self.hold_shift = False  # True: the aim's shift stays until the clip ends (the env then moves the guard, counter.reanchor)
        fk = Humanoid_Batch(motion_cfg)
        body = list(fk.body_names_augment).index
        self.libs, duration, striker, impact_time, impact_point, start_xy, blend, extension, leg = [], [], [], [], [], [], [], [], []
        for name in self.names:
            path = os.path.join(clip_dir, name + '.pkl')
            assert os.path.isfile(path), f'{path} is missing: run scripts/make_defense_counters.py'
            clip = next(iter(load_joblib_compat(path).values()))
            meta, fps = clip['counter_meta'], int(clip['fps'])
            cfg = OmegaConf.create(OmegaConf.to_container(motion_cfg, resolve=True))
            cfg.motion_file = path
            lib = MotionLibRobotWJX(cfg, num_envs=1, device=device)
            lib.load_motions(random_sample=False)
            self.libs.append(lib)

            # the strike: the clip's labelled strike time (full extension of a punch, highest point of a kicking foot)
            pos = fk.fk_batch(torch.as_tensor(np.asarray(clip['pose_aa']), dtype=torch.float32)[None],
                              torch.as_tensor(np.asarray(clip['root_trans_offset']), dtype=torch.float32)[None]
                              ).global_translation_extend[0]  # [T, B, 3]
            frame = min(int(round(meta['strike_time'] * fps)), len(pos) - 1)
            duration.append((len(pos) - 1) / fps)
            striker.append(STRIKERS.index(meta['striker']))
            impact_time.append(frame / fps)
            hitter = striker_positions(pos, body, meta['striker'])  # [T, 3]
            impact_point.append(hitter[frame])
            leg.append(meta.get('target', 'trunk') == 'leg')
            start_xy.append(pos[0, 0, :2])
            extension.append(float((hitter[frame, :2] - pos[frame, 0, :2]).norm()))
            blend.append(list(meta.get('aim_span', (meta['motion_start'], meta['motion_end']))))
        self.duration = torch.tensor(duration, dtype=torch.float, device=device)
        self.striker = torch.tensor(striker, dtype=torch.long, device=device)  # index into STRIKERS
        self.impact_time = torch.tensor(impact_time, dtype=torch.float, device=device)
        self.impact_point = torch.stack(impact_point).to(device)  # [K, 3] in the clip (= guard file) frame, z above flat ground
        self.target_leg = torch.tensor(leg, dtype=torch.bool, device=device)  # [K] aimed at the attacker's leg, not its trunk
        self.extension = torch.tensor(extension, dtype=torch.float, device=device)  # [K] striker to the root at the strike, horizontal
        self.start_xy = torch.stack(start_xy).to(device)  # [K, 2] root at the clip start (the guard's)
        self.motion_span = torch.tensor(blend, dtype=torch.float, device=device)  # [K, 2] the aim eases in until the first time, out from the second

    def aim_weight(self, clip_ids, times, hold=False):
        """ Share [n] of a counter's aim transform applied at time t in the clip: it eases in while the clip blends from
        the guard into the motion, is 1 during the motion and eases out while the clip blends back to the guard (hold: it does not ease out). """
        start, end, duration = self.motion_span[clip_ids, 0], self.motion_span[clip_ids, 1], self.duration[clip_ids]
        out = torch.ones_like(times) if hold else (duration - times) / (duration - end)
        s = torch.where(times < start, times / start, torch.where(times > end, out, torch.ones_like(times)))
        s = s.clamp(0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)

    def write_states(self, res, rows, clip_ids, times, offset, aim_yaw=None, aim_shift=None):
        """ Overwrite rows (bool mask or indices into the batch of res) of a motion state dict with the state of each
        row's clip at its time [s] (clamped to the clip). offset [n, 3] or None as for get_motion_state.
        aim_yaw [n], aim_shift [n, 2]: the clip is turned by aim_yaw about its start root and moved by aim_shift (world
        xy), eased in and out with aim_weight, so that its strike lands where the env aims it. """
        rows = rows.nonzero(as_tuple=False).flatten() if rows.dtype == torch.bool else rows
        for k, lib in enumerate(self.libs):
            sel = (clip_ids == k).nonzero(as_tuple=False).flatten()
            if len(sel) == 0:
                continue
            ids = torch.zeros(len(sel), dtype=torch.long, device=self.device)
            tk = times[sel].clamp(0.0, float(self.duration[k]))
            state = lib.get_motion_state(ids, tk, offset=None if offset is None else offset[sel])
            if aim_yaw is not None and offset is not None:
                w = self.aim_weight(clip_ids[sel], tk)
                w_shift = self.aim_weight(clip_ids[sel], tk, hold=True) if self.hold_shift else w
                self._aim(state, self.start_xy[k] + offset[sel, :2], w * aim_yaw[sel], w_shift[:, None] * aim_shift[sel])
            for key, value in state.items():
                res[key][rows[sel]] = value

    @staticmethod
    def _aim(state, pivot, yaw, shift):
        """ Turn a motion state by yaw [n] about the vertical through pivot [n, 2] and move it by shift [n, 2]. """
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        half_cos, half_sin = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
        done = set()
        for key in ('root_pos', 'rg_pos', 'rg_pos_t', 'root_vel', 'root_ang_vel', 'body_vel', 'body_ang_vel', 'body_vel_t',
                    'body_ang_vel_t', 'root_rot', 'rb_rot', 'rg_rot_t'):
            if key not in state or id(state[key]) in done:  # rg_pos_t can be the same tensor as rg_pos
                continue
            done.add(id(state[key]))
            x = state[key]
            shape = (-1,) + (1,) * (x.dim() - 2)
            if key in ('root_rot', 'rb_rot', 'rg_rot_t'):  # yaw quaternion (0, 0, sin, cos) times the rotation (xyzw)
                cz, sz = half_cos.view(shape), half_sin.view(shape)
                qx, qy, qz, qw = x.unbind(-1)
                state[key] = torch.stack([cz * qx - sz * qy, cz * qy + sz * qx, cz * qz + sz * qw, cz * qw - sz * qz], dim=-1)
                continue
            c, s = cos.view(shape), sin.view(shape)
            position = key in ('root_pos', 'rg_pos', 'rg_pos_t')
            px, py = (pivot[:, 0].view(shape), pivot[:, 1].view(shape)) if position else (0.0, 0.0)
            vx, vy = x[..., 0] - px, x[..., 1] - py
            out = x.clone()
            out[..., 0] = c * vx - s * vy + px + (shift[:, 0].view(shape) if position else 0.0)
            out[..., 1] = s * vx + c * vy + py + (shift[:, 1].view(shape) if position else 0.0)
            state[key] = out
