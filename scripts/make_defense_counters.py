"""Build the counterattack clips of the defense env (env.config.defense.counter, DEFENSE_MANEUVERS.md Section 10) from
- the strikes of the attack dataset: one clip per attack class (jab, cross, lead_hook, rear_hook, lead_uppercut,
  rear_uppercut, body_shot, calf_kick), the clip of the class whose striking distance is the class median. They start
  and end in the guard and keep the feet planted, so the blends are short (--strike-blend);
- the kung-fu motions in example/motion_data (horse_stance_punch, hooks_punch, side_kick, roundhouse_kick).

After the defender blocks or evades an attack it performs one of these clips against the attacker. A clip is

    guard -> [blend-in] -> the kung-fu motion, turned and moved so that it starts at the guard's root position and
    heading -> [blend-out] -> guard

so the reference is continuous when the env switches from the held guard (defender_guard.pkl) to the clip and back.
The env aims the clip at the attacker (a yaw about the clip's start and a shift, eased in and out over the blends).
The blends interpolate the root position (linear), the root rotation (slerp) and the joint angles (linear) with a
smoothstep; the feet of the blended reference slide (the guard is staggered, the motions start with the feet side by side).

Each clip is written to its own file (PBHC's WJX motion library loads one motion per file) with an extra key
`counter_meta`, ignored by the motion library:
    name          the motion
    striker       body that lands the strike (a hand link, an ankle roll link, or left_shin / right_shin: 75 % of the way
                  from the knee to the ankle)
    target        what it is aimed at on the attacker: trunk (pelvis to head) or leg
    strike_time   [s] in the clip, when the strike lands: the full extension of a punch, the highest point of a kicking foot
    motion_start, motion_end   [s] of the motion inside the clip (between the blends)
    aim_span      [s] the env's aim eases in until the first time and out from the second

usage (from the repo root, env `pbhc` active; needs the attack dataset's defender_guard.pkl):
  python scripts/make_defense_counters.py [--out g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_counters]
         [--blend-in 1.5] [--blend-out 1.5]
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from humanoidverse.envs.defense.attack_bank import load_joblib_compat

# motion file, striking body, time of the strike in the original motion [s]
KUNGFU = {
    'horse_stance_punch': ('example/motion_data/Horse-stance_punch.pkl', 'right_hand_link', 2.87),
    'hooks_punch': ('example/motion_data/Hooks_punch.pkl', 'right_hand_link', 2.23),
    'side_kick': ('example/motion_data/Side_kick.pkl', 'right_ankle_roll_link', 2.33),
    'roundhouse_kick': ('example/motion_data/Roundhouse_kick.pkl', 'left_ankle_roll_link', 2.00),
}
STRIKER_LINK = {'left_hand': 'left_hand_link', 'right_hand': 'right_hand_link', 'right_shin': 'right_shin', 'left_shin': 'left_shin'}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--guard', default='g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_guard.pkl')
parser.add_argument('--out', default='g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_counters')
parser.add_argument('--blend-in', type=float, default=1.5)
parser.add_argument('--blend-out', type=float, default=1.5)
parser.add_argument('--strike-blend', type=float, default=0.3, help='blend in and out of the dataset strikes [s]')
parser.add_argument('--attacks', default='g1_attack_dataset/PBHC_attacks/motion_data/attacks', help='attack dataset (index.json)')
args = parser.parse_args()


def first(path):
    return next(iter(load_joblib_compat(path).values()))


def yaw_of(quat):  # xyzw
    x, y, z, w = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def blend(pos0, rot0, dof0, pos1, rot1, dof1, n):
    """ n frames from pose 0 (excluded) towards pose 1 (excluded), smoothstep """
    s = np.arange(1, n + 1) / (n + 1)
    s = s * s * (3 - 2 * s)
    rot = Slerp([0, 1], R.from_quat([rot0, rot1]))(s).as_quat()
    return pos0 + s[:, None] * (pos1 - pos0), rot, dof0 + s[:, None] * (dof1 - dof0)


guard = first(args.guard)
g_pos, g_rot, g_dof = (np.asarray(guard[k], np.float64)[0] for k in ('root_trans_offset', 'root_rot', 'dof'))
dof_axis = np.load('description/robots/g1/dof_axis.npy').astype(np.float32)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# name: (motion file, striker, strike time [s], blend in [s], blend out [s], target, aim over the whole wind-up)
COUNTERS = {name: (path, striker, t, args.blend_in, args.blend_out, 'trunk', False) for name, (path, striker, t) in KUNGFU.items()}
index_path = Path(args.attacks) / 'index.json'
if index_path.is_file():
    index = json.loads(index_path.read_text())
    for attack in index['attacks']:
        clips = [c for c in index['clips'] if c['attack'] == attack]
        median = np.median([c['distance'] for c in clips])
        c = min(clips, key=lambda c: (abs(c['distance'] - median), c['miss_distance']))
        COUNTERS[attack] = (str(Path(args.attacks) / attack / (c['clip'] + '.pkl')), STRIKER_LINK[c['striker']], c['impact_time'],
                            args.strike_blend, args.strike_blend, 'leg' if c['target_region'] == 'leg' else 'trunk', True)
else:
    print(f'{index_path} is missing: only the kung-fu clips are built')

for name, (path, striker, strike_time, blend_in, blend_out, target, aim_windup) in COUNTERS.items():
    m = first(path)
    fps = int(m['fps'])
    pos, rot, dof = (np.asarray(m[k], np.float64) for k in ('root_trans_offset', 'root_rot', 'dof'))
    contact = np.asarray(m['contact_mask'], np.float64)
    # turn the motion about its start so that it starts with the guard's heading, at the guard's root xy
    turn = R.from_euler('z', yaw_of(g_rot) - yaw_of(rot[0]))
    pos = turn.apply(pos - pos[0]) + pos[0]
    pos[:, :2] += g_pos[:2] - pos[0, :2]
    rot = (turn * R.from_quat(rot)).as_quat()

    n_in, n_out = int(round(blend_in * fps)), int(round(blend_out * fps))
    pos_in, rot_in, dof_in = blend(g_pos, g_rot, g_dof, pos[0], rot[0], dof[0], n_in)
    pos_out, rot_out, dof_out = blend(pos[-1], rot[-1], dof[-1], g_pos, g_rot, g_dof, n_out)
    pos = np.concatenate([g_pos[None], pos_in, pos, pos_out, g_pos[None]])
    rot = np.concatenate([g_rot[None], rot_in, rot, rot_out, g_rot[None]])
    dof = np.concatenate([g_dof[None], dof_in, dof, dof_out, g_dof[None]])
    contact = np.concatenate([np.ones((n_in + 1, 2)), contact, np.ones((n_out + 1, 2))])
    rot *= np.sign(rot[:, 3:4] + 1e-12)
    for i in range(1, len(rot)):  # no sign flips between frames
        if np.dot(rot[i], rot[i - 1]) < 0:
            rot[i] *= -1

    T = len(pos)
    pose_aa = np.concatenate([R.from_quat(rot).as_rotvec()[:, None], dof_axis[None] * dof[:, :, None], np.zeros((T, 3, 3))], axis=1)
    start = (n_in + 1) / fps
    end = start + (len(m['dof']) - 1) / fps
    # the env's aim (turn and shift towards the attacker) eases in until aim_span[0] and out from aim_span[1]: over the
    # blends of a kung-fu clip; for a dataset strike from the clip's start to just before the strike and from just after
    # it to the clip's end (the blends are too short to move the body in)
    aim_span = (start + strike_time - 0.1, min(start + strike_time + 0.3, end)) if aim_windup else (start, end)
    clip = {
        'root_trans_offset': pos.astype(np.float32), 'pose_aa': pose_aa.astype(np.float32), 'dof': dof.astype(np.float32),
        'root_rot': rot.astype(np.float32), 'smpl_joints': np.zeros_like(pose_aa, dtype=np.float32), 'fps': fps,
        'contact_mask': contact,
        'counter_meta': dict(name=name, striker=striker, target=target, strike_time=start + strike_time, motion_start=start,
                             motion_end=end, aim_span=aim_span, source=path),
    }
    joblib.dump({f'counter_{name}': clip}, out_dir / f'{name}.pkl')
    print(f'{name}: {T} frames, {(T - 1) / fps:.2f} s, strike by {striker} at {start + strike_time:.2f} s -> {out_dir / (name + ".pkl")}')
