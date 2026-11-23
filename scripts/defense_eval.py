"""Evaluate a defense checkpoint (env.config.defense, DEFENSE_MANEUVERS.md): attack outcomes per class, episode survival,
guard tracking and the belief head.

Every env runs --episodes / --num-envs episodes of --episode-s seconds with the training terminations (falls, any body
more than the training threshold from the guard reference). Attacks are counted as the env scores them:
  hit / block / evade   the outcome rules of the env (hit_radius, block_radius)
  interrupted           the episode ended by a fall or drift after the attack's onset and before its outcome
Blocks are also described by the contact's distance to the target and the share made with the left forearm, and every
step by the share with a forearm inside the attacker (defense_attacker_overlap > 0).
Rates are shares of the resolved attacks; interrupted attacks are reported separately, so a policy that falls instead of
getting hit does not look better (check fall_rate and interrupted together).
Counterattacks (env.config.defense.counter): per clip, the counters started, the share whose strike landed on the attacker
and the share performed to the end (the rest ended with the episode: a fall, drift from the reference or the time-out).
Belief head (actor's encoder on the observed keypoints, against the privileged labels), over all steps of all episodes:
  class accuracy per attack phase (0 guard, 1 strike, 2 retract, 3 guard), phase accuracy, time-to-impact MAE during
  the strike phase.
Smoothness (class Smoothness): action rate and jerk, joint acceleration and the share of joint-velocity power above 4 Hz
(legs, waist, arms), each next to the same numbers for the reference the robot tracks, plus the raw sums the penalty
rewards use, to size their scales.
--blind sets env.config.defense.blind_actor, hiding the attacker from the actor (an unconditioned test of a conditioned policy; the critic, rewards and scoring
are unchanged, and the belief metrics are computed on what the actor sees):
  freeze  the attacker_keypoints the actor sees are held from the last step at which the attacker stood in guard (more
          than threat_lead_time before an onset), so it sees an attacker that never strikes
  zero    the attacker_keypoints are zeros (out of distribution: no attacker at all, rather than an idle one)

usage (from the repo root, env `pbhc` active):
  python scripts/defense_eval.py CKPT [--name default] [--episodes 256] [--num-envs 256] [--seed 0] [--episode-s 20]
         [--blind {none,freeze,zero}] [--env-from RUN] [--terrain {flat,step,tilt,bumps} --difficulty D] [key=value ...]   e.g. env.config.defense.block_radius=0 env.config.defense.placement_noise.yaw_deg=30
                                "env.config.defense.attacks=[cross,rear_hook]"
--env-from RUN (a defense run's directory, checkpoint or config.yaml) evaluates a checkpoint trained without the defense
env, e.g. a plain PBHC guard tracker: the env (attacks, standoff, terminations) comes from RUN, the policy's observations,
robot and algorithm from CKPT's own config. The policy just doesn't see the attacker unless its observations include it.
Results go to CKPT_DIR/metrics/defense_eval/<ckpt name>/<name>_seed<seed>.json.
"""
import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('ckpt')
parser.add_argument('--name', default='default')
parser.add_argument('--episodes', type=int, default=256)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--episode-s', type=float, default=20.0)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--blind', choices=['none', 'freeze', 'zero'], default='none')
parser.add_argument('--env-from', default=None)
parser.add_argument('--terrain', default=None)  # flat, step, tilt or bumps: one terrain type for all envs (terrain runs)
parser.add_argument('--difficulty', type=float, default=1.0)  # with --terrain: 1 = 10 cm steps, 12 deg tilt, 4 cm bumps
parser.add_argument('--diag', action='store_true')  # what was happening just before each fall or drift end
args, overrides = parser.parse_known_args()

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import isaacgym  # noqa: F401,E402  (must be imported before torch)
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
import humanoidverse.utils.config_utils  # noqa: F401,E402  (registers the `eval` resolver)
from humanoidverse.utils.helpers import pre_process_config  # noqa: E402
from humanoidverse.envs.defense.attack_bank import ATTACKS, NUM_PHASES  # noqa: E402
from humanoidverse.envs.defense.defense import PENDING  # noqa: E402
from humanoidverse.envs.defense.counter_bank import SHIN, STRIKERS  # noqa: E402

ckpt = Path(args.ckpt)
config = OmegaConf.load(ckpt.parent / 'config.yaml')
if config.get('eval_overrides') is not None:
    config = OmegaConf.merge(config, config.eval_overrides)
if args.env_from is not None:
    src = Path(args.env_from)
    src = src / 'config.yaml' if src.is_dir() else src if src.suffix == '.yaml' else src.parent / 'config.yaml'
    env_config = OmegaConf.load(src)
    if env_config.get('eval_overrides') is not None:
        env_config = OmegaConf.merge(env_config, env_config.eval_overrides)
    config.env = env_config.env  # its obs, rewards and robot interpolate to CKPT's
config = OmegaConf.merge(config, OmegaConf.from_dotlist(
    ['headless=True', f'num_envs={args.num_envs}', f'env.config.max_episode_length_s={args.episode_s}',
     'env.config.defense.debug_draw=False', f'env.config.defense.blind_actor={args.blind}']
    + ([] if args.terrain is None else [  # a run trained with +exp=defense_terrain, on one terrain type and difficulty
        'terrain.curriculum=False', f'terrain.terrain_types=[{args.terrain}]', 'terrain.terrain_proportions=[1.0]',
        f'terrain.fixed_difficulty={args.difficulty if args.terrain != "flat" else 0}']) + overrides))
pre_process_config(config)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env = instantiate(config.env, device=args.device)
algo = instantiate(config.algo, env=env, device=args.device, log_dir=None)
algo.setup()
algo.load(str(ckpt))
algo._eval_mode()
env.set_is_evaluating()
policy = algo._get_inference_policy()
actor = algo.actor
has_belief = hasattr(actor, 'belief_loss')

N, dev = env.num_envs, env.device
bank = env._attack_bank
C = bank.num_classes
episodes_per_env = -(-args.episodes // N)
episodes_done = torch.zeros(N, dtype=torch.long, device=dev)
counted = torch.ones(N, dtype=torch.bool, device=dev)
outcomes = torch.zeros(C, 4, device=dev)  # hit, block, evade, interrupted
block_stats = torch.zeros(C, 2, device=dev)  # sum of contact-to-target distances, blocks with the left forearm
combo_outcomes = torch.zeros(3, device=dev)  # hit, block, evade of the second strikes of combos (defense.combo)

# count the outcomes exactly as the env scores and logs them
log_outcomes = env._log_defense_outcomes
def count_outcomes(labels, event):
    resolved = (event > 0) & counted
    outcomes.index_put_((labels[resolved], event[resolved] - 1), torch.ones(int(resolved.sum()), device=dev), accumulate=True)
    blocked = (event == 2) & counted  # _record_outcomes ran earlier in this step
    stats = torch.stack([(env._outcome_point - env._outcome_target).norm(dim=-1), (env._outcome_arm_id == 0).float()], dim=1)
    block_stats.index_add_(0, labels[blocked], stats[blocked])
    second = resolved & getattr(env, '_combo_second', torch.zeros_like(resolved))
    combo_outcomes.index_put_((event[second] - 1,), torch.ones(int(second.sum()), device=dev), accumulate=True)
    log_outcomes(labels, event)
env._log_defense_outcomes = count_outcomes

reset_states = env._reset_robot_states_callback
def count_interrupted(env_ids, target_states=None):
    ids = env._attack_ids[env_ids]
    live = (env._attack_status[env_ids] == PENDING) & (bank.frame(ids, env._attack_t[env_ids]) >= bank.onset_frame[ids]) \
        & counted[env_ids] & ~env.time_out_buf[env_ids].bool()
    labels = bank.label[ids[live]]
    outcomes.index_put_((labels, torch.full_like(labels, 3)), torch.ones(len(labels), device=dev), accumulate=True)
    reset_states(env_ids, target_states)
env._reset_robot_states_callback = count_interrupted

episode_counts = {'survived': 0, 'fall': 0, 'drift': 0}
diag = {'n': 0, 'counter': {}, 'phase': [0] * NUM_PHASES, 'two_levels_end': 0, 'two_levels_start': 0,
        'time': [0, 0, 0, 0], 'attack': {}, 'early': 0, 'early_two_start': 0, 'early_counter': 0, 'starts': 0, 'starts_two': 0, 'early_overhang': 0, 'starts_overhang': 0}
last = {}
start_two = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
start_overhang = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)  # a foot across an edge in the first 0.2 s


def feet_two_levels():
    """ [N] bool: the robot's feet stand on ground more than 3 cm apart in height (across a step edge). """
    feet = env.simulator._rigid_body_pos[:, env.feet_indices, :2]
    h = env._terrain_height_at(feet)
    return (h[:, 0] - h[:, 1]).abs() > 0.03


class Smoothness:
    """ Jitter of the robot's joints next to the reference it tracks (the guard and counter clips), over the alive steps
    of counted episodes from SKIP steps after a reset on. Per joint, RMS over steps and joints:
      action rate / jerk   first / second difference of the policy action per control step (action units)
      joint acc            change of joint velocity per control step / dt (rad/s^2), robot and reference
      HF share             share of the joint-velocity power (1 s Hann windows, 1 Hz bins, DC left out) above HF_HZ,
                           robot and reference, for the legs, the waist and the arms
    plus the raw per-step sums the penalty rewards use (sum of squares over joints: action_rate, dof_acc, torques) and the
    mean mechanical power sum |tau * qdot| (W). """
    W, HF_HZ, SKIP = 50, 4.0, 5

    def __init__(self):
        D = env.num_dof
        self.dt = env.dt
        self.prev = {k: torch.zeros(N, D, device=dev) for k in ('a1', 'a2', 'v1', 'r1')}
        self.age = torch.zeros(N, dtype=torch.long, device=dev)
        self.buf = torch.zeros(2, self.W, N, D, device=dev)  # robot / reference joint velocity, last W steps
        self.t = 0
        self.s = {k: 0. for k in ('n', 'n2', 'act_rate', 'act_jerk', 'acc', 'ref_acc', 'vel', 'ref_vel', 'pen_action_rate',
                                  'pen_dof_acc', 'pen_torques', 'power')}
        names = list(env.dof_names)
        self.groups = {g: torch.tensor([i for i, n in enumerate(names) if any(k in n for k in keys)], device=dev)
                       for g, keys in (('legs', ('hip', 'knee', 'ankle')), ('waist', ('waist', 'torso')),
                                       ('arms', ('shoulder', 'elbow', 'wrist')))}
        f = torch.fft.rfftfreq(self.W, self.dt).to(dev)
        self.band, self.hf = f > 0.5, f > self.HF_HZ
        self.window = torch.hann_window(self.W, periodic=False, device=dev).view(self.W, 1, 1)
        self.power = {g: torch.zeros(2, 2, device=dev) for g in self.groups}  # [robot, ref] x [HF, all]
        self.series = {g: 0 for g in self.groups}  # windows x joints summed into self.power

    def step(self, alive):
        a, v = env.actions, env.simulator.dof_vel
        r = env.dif_joint_velocities + v  # the reference joint velocity
        self.age = torch.where(alive, self.age + 1, torch.zeros_like(self.age))
        ok, ok2 = alive & (self.age > self.SKIP), alive & (self.age > self.SKIP + 1)
        p, s, n = self.prev, self.s, int(ok.sum())
        if n:
            da, acc, racc = (a - p['a1'])[ok], ((v - p['v1']) / self.dt)[ok], ((r - p['r1']) / self.dt)[ok]
            s['n'] += n
            s['act_rate'] += da.square().mean(dim=1).sum().item()
            s['acc'] += acc.square().mean(dim=1).sum().item()
            s['ref_acc'] += racc.square().mean(dim=1).sum().item()
            s['vel'] += v[ok].square().mean(dim=1).sum().item()
            s['ref_vel'] += r[ok].square().mean(dim=1).sum().item()
            s['pen_action_rate'] += da.square().sum(dim=1).sum().item()
            s['pen_dof_acc'] += acc.square().sum(dim=1).sum().item()
            s['pen_torques'] += env.torques[ok].square().sum(dim=1).sum().item()
            s['power'] += (env.torques[ok] * v[ok]).abs().sum(dim=1).sum().item()
        if ok2.any():
            s['n2'] += int(ok2.sum())
            s['act_jerk'] += (a - 2 * p['a1'] + p['a2'])[ok2].square().mean(dim=1).sum().item()
        self.buf[0, self.t % self.W], self.buf[1, self.t % self.W] = v, r
        self.t += 1
        if self.t % self.W == 0:
            full = self.age > self.SKIP + self.W  # alive through the whole window
            if full.any():
                x = self.buf[:, :, full]  # [2, W, n, D]
                X = torch.fft.rfft((x - x.mean(dim=1, keepdim=True)) * self.window, dim=1).abs().square()  # [2, F, n, D]
                for g, idx in self.groups.items():
                    Pg = X[..., idx].sum(dim=(2, 3))  # [2, F]
                    self.power[g] += torch.stack([Pg[:, self.hf].sum(dim=1), Pg[:, self.band].sum(dim=1)], dim=1)
                    self.series[g] += int(full.sum()) * len(idx)
        p['a2'], p['a1'], p['v1'], p['r1'] = p['a1'], a.clone(), v.clone(), r.clone()

    def result(self):
        s, n = self.s, max(self.s['n'], 1)
        rms = lambda k, m=n: (s[k] / m) ** 0.5
        out = {'action_rate_rms': rms('act_rate'), 'action_jerk_rms': rms('act_jerk', max(s['n2'], 1)),
               'dof_acc_rms': rms('acc'), 'ref_dof_acc_rms': rms('ref_acc'), 'dof_vel_rms': rms('vel'), 'ref_dof_vel_rms': rms('ref_vel'),
               'penalty_action_rate_raw': s['pen_action_rate'] / n, 'penalty_dof_acc_raw': s['pen_dof_acc'] / n,
               'penalty_torques_raw': s['pen_torques'] / n, 'mech_power_w': s['power'] / n, 'steps': s['n']}
        norm = self.W ** 2 * float(self.window.square().mean()) / 2  # one-sided Parseval, undoing the Hann window
        for g, P in self.power.items():
            hf = (P[:, 0] / P[:, 1].clamp(min=1e-9)).tolist()
            out[f'hf_share_{g}'], out[f'ref_hf_share_{g}'] = hf
            rms = (P[:, 0] / max(self.series[g], 1) / norm).sqrt().tolist()  # RMS joint speed above HF_HZ (rad/s)
            out[f'hf_vel_rms_{g}'], out[f'ref_hf_vel_rms_{g}'] = rms
        return out


class CounterQuality:
    """ How fast and how clean the counters are, per clip, over the counters performed to their end in counted episodes
    (robot next to its reference clip):
      peak speed   of the striking hand / foot / shin within hit_window of the clip's strike time [m/s]
      lag          time of the robot's peak minus the reference's [s] (> 0: the strike comes late)
      jerk         RMS third time derivative of the striker's path over the clip [m/s^3] (a jagged strike is high)
      joint vel    RMS speed of the striking limb's joints (arm: shoulder, elbow; leg: hip, knee, ankle), robot / reference,
                   and the RMS error between them [rad/s] """
    def __init__(self):
        self.on = bool(getattr(env, '_counter_enabled', False))
        if not self.on:
            return
        self.bank = bank = env._counter_bank
        self.T = int(float(bank.duration.max()) / env.dt) + 4
        self.pos = torch.zeros(2, N, self.T, 3, device=dev)  # robot, reference
        self.jv = torch.zeros(3, N, self.T, device=dev)  # per step, over the limb's joints: sum v^2, ref^2, error^2
        self.cost = torch.zeros(5, N, self.T, device=dev)  # per step, the limb's: torque-limit penalty raw, steps near the limit, action rate^2
        self.idx = torch.zeros(N, dtype=torch.long, device=dev)
        self.prev_active = torch.zeros(N, dtype=torch.bool, device=dev)
        self.prev_act = torch.zeros(N, env.num_dof, device=dev)  # the policy action of the previous step (env.last_actions is updated before we see it)
        names = list(env.dof_names)
        limb = lambda side, parts: [i for i, n in enumerate(names) if n.startswith(side) and any(p in n for p in parts)]
        arm, leg = ('shoulder', 'elbow', 'wrist'), ('hip', 'knee', 'ankle')
        masks = {'left_hand_link': limb('left', arm), 'right_hand_link': limb('right', arm),
                 'left_ankle_roll_link': limb('left', leg), 'right_ankle_roll_link': limb('right', leg),
                 'left_shin': limb('left', leg), 'right_shin': limb('right', leg)}
        self.mask = torch.zeros(len(STRIKERS), env.num_dof, device=dev)
        for k, name in enumerate(STRIKERS):
            self.mask[k, masks[name]] = 1.0
        K = bank.num_clips
        self.acc = {k: torch.zeros(K, device=dev) for k in
                    ('n', 'peak', 'ref_peak', 'lag', 'abs_lag', 'jerk2', 'ref_jerk2', 'jv2', 'ref_jv2', 'err2', 'jsteps', 'tlim', 'tsat', 'arate2', 'steps', 'dwell', 'prox')}
        w = float(env._counter_hit_window)
        self.win = ((bank.impact_time - w) / env.dt).long().clamp(min=0), ((bank.impact_time + w) / env.dt).long()

    def ref_strikers(self):
        p, nb = env.ref_body_pos_extend, env.num_bodies  # extended bodies after the rigid ones: left hand, right hand, head
        knee, ankle = p[:, env._knee_ids], p[:, env._ankle_pitch_ids]
        return torch.cat([p[:, [nb, nb + 1]], p[:, env._ankle_ids], knee + SHIN * (ankle - knee)], dim=1)

    def step(self, dones, counted):
        if not self.on:
            return
        prev_act, self.prev_act = self.prev_act, env.actions.clone()
        active = env._counter_active
        ended = self.prev_active & ~active & ~dones & counted
        if ended.any():
            self.finish(ended.nonzero(as_tuple=False).flatten())
        self.idx[~active | dones] = 0
        ids = (active & ~dones & counted).nonzero(as_tuple=False).flatten()
        self.prev_active = active & ~dones
        if len(ids) == 0:
            return
        s = self.bank.striker[env._counter_clip[ids]]
        i = self.idx[ids].clamp(max=self.T - 1)
        self.pos[0, ids, i] = env._defender_strikers()[ids, s]
        self.pos[1, ids, i] = self.ref_strikers()[ids, s]
        v = env.simulator.dof_vel[ids]
        r = env.dif_joint_velocities[ids] + v
        m = self.mask[s]
        self.jv[0, ids, i] = (v.square() * m).sum(-1)
        self.jv[1, ids, i] = (r.square() * m).sum(-1)
        self.jv[2, ids, i] = ((r - v).square() * m).sum(-1)
        tau, lim = env.torques[ids].abs(), env.torque_limits
        soft = float(getattr(env, 'soft_torque_curriculum_value', env.config.rewards.reward_limit.soft_torque_limit))
        over = (tau - lim * soft).clip(min=0., max=1.)
        self.cost[0, ids, i] = (over * m).sum(-1)
        self.cost[1, ids, i] = ((over > 0).float() * m).sum(-1)
        self.cost[2, ids, i] = ((env.actions[ids] - prev_act[ids]).square() * m).sum(-1)
        # dwell: steps in the hit window with the striker within the landing radius of the target, and the proximity
        # reward the env pays on that step (counter.proximity_sigma kernel, before its scale)
        near = env._counter_window[ids] * (env._counter_dist[ids] < env._counter_clip_radius[env._counter_clip[ids]]).float()
        self.cost[3, ids, i] = near
        self.cost[4, ids, i] = env._counter_window[ids] * torch.exp(-((env._counter_dist[ids] - env._counter_depth[ids]).clamp(min=0.0) / env._counter_sigma_now[ids]) ** 2)
        self.idx[ids] += 1

    def finish(self, ids):
        L, clip = self.idx[ids], env._counter_clip[ids]
        keep = L >= 10
        ids, L, clip = ids[keep], L[keep], clip[keep]
        if len(ids) == 0:
            return
        dt, T = env.dt, self.T
        P = self.pos[:, ids]  # [2, n, T, 3]
        t = torch.arange(T, device=dev)
        valid = t[None] < L[:, None]
        speed = (P[:, :, 1:] - P[:, :, :-1]).norm(dim=-1) / dt  # [2, n, T-1]
        lo, hi = self.win[0][clip], self.win[1][clip]
        in_win = (t[None, 1:] >= lo[:, None]) & (t[None, 1:] <= hi[:, None]) & valid[:, 1:]
        peak, arg = (speed * in_win).max(dim=-1)  # [2, n]
        lag = (arg[0] - arg[1]).float() * dt
        j = P[:, :, 3:] - 3 * P[:, :, 2:-1] + 3 * P[:, :, 1:-2] - P[:, :, :-3]
        vj = valid[:, 3:].float()
        jerk2 = (j.square().sum(-1) * vj).sum(-1) / vj.sum(-1).clamp(min=1) / dt ** 6  # [2, n]
        jv = (self.jv[:, ids] * valid).sum(-1)  # [3, n]
        cost = (self.cost[:, ids] * valid).sum(-1)  # [3, n]
        nj = self.mask[self.bank.striker[clip]].sum(-1) * L
        a = self.acc
        for k, val in (('n', torch.ones_like(lag)), ('peak', peak[0]), ('ref_peak', peak[1]), ('lag', lag), ('abs_lag', lag.abs()),
                       ('jerk2', jerk2[0]), ('ref_jerk2', jerk2[1]), ('jv2', jv[0]), ('ref_jv2', jv[1]), ('err2', jv[2]), ('jsteps', nj),
                       ('tlim', cost[0]), ('tsat', cost[1]), ('arate2', cost[2]), ('steps', L.float()), ('dwell', cost[3]), ('prox', cost[4])):
            a[k].index_add_(0, clip, val)

    def result(self):
        if not self.on:
            return None
        a = {k: v.tolist() for k, v in self.acc.items()}
        def one(sel):
            n = max(sum(a['n'][k] for k in sel), 1)
            js = max(sum(a['jsteps'][k] for k in sel), 1)
            tot = lambda key: sum(a[key][k] for k in sel)
            return {'counters': int(sum(a['n'][k] for k in sel)), 'peak_speed': tot('peak') / n, 'ref_peak_speed': tot('ref_peak') / n,
                    'lag_s': tot('lag') / n, 'abs_lag_s': tot('abs_lag') / n,
                    'jerk_rms': (tot('jerk2') / n) ** 0.5, 'ref_jerk_rms': (tot('ref_jerk2') / n) ** 0.5,
                    'limb_vel_rms': (tot('jv2') / js) ** 0.5, 'ref_limb_vel_rms': (tot('ref_jv2') / js) ** 0.5,
                    'limb_vel_err_rms': (tot('err2') / js) ** 0.5,
                    'limb_torque_limit_raw_per_step': tot('tlim') / max(tot('steps'), 1),  # limits_torque's raw term, the limb's share
                    'limb_near_torque_limit_share': tot('tsat') / js,  # share of limb joint-steps above the soft torque limit
                    'limb_action_rate_rms': (tot('arate2') / js) ** 0.5,
                    'dwell_s': tot('dwell') / n * env.dt,  # per counter: time the striker sits within the landing radius in the hit window
                    'proximity_kernel_sum': tot('prox') / n}  # per counter: sum of the proximity kernel (x scale x dt = its reward)
        K = self.bank.num_clips
        out = {self.bank.names[k]: one([k]) for k in range(K) if a['n'][k] > 0}
        out['all'] = one(list(range(K)))
        return out


cq = CounterQuality()
smooth = Smoothness()
lengths = []
body_err_sum, err_steps = 0., 0
overlap_steps = 0.
belief_correct = torch.zeros(NUM_PHASES, device=dev)  # class accuracy per attack phase
belief_steps = torch.zeros(NUM_PHASES, device=dev)
phase_correct, tti_abs_sum, tti_steps = 0., 0., 0
with torch.no_grad():
    obs = env.reset_all()
    while (episodes_done < episodes_per_env).any():
        counted[:] = episodes_done < episodes_per_env
        if has_belief:
            P = actor.actor_module.num_phases
            _, belief = actor.actor_module.encode(obs['actor_obs'][counted])
            critic = obs['critic_obs'][counted]
            label, phase = critic[:, actor.label_slice].argmax(dim=-1), critic[:, actor.phase_slice].argmax(dim=-1)
            correct = (belief[:, :C].argmax(dim=-1) == label).float()
            belief_correct.index_add_(0, phase, correct)
            belief_steps.index_add_(0, phase, torch.ones_like(correct))
            phase_correct += (belief[:, C:C + P].argmax(dim=-1) == phase).float().sum().item()
            striking = phase == 1
            tti_abs_sum += (belief[striking, C + P] - critic[striking, actor.tti_index]).abs().sum().item()
            tti_steps += int(striking.sum())

        if args.diag:
            bank = env._attack_bank
            last = {'counter': env._counter_active.clone() if env._counter_enabled else torch.zeros_like(counted),
                    'clip': env._counter_clip.clone() if env._counter_enabled else torch.zeros_like(episodes_done),
                    'phase': bank.phase_at(env._attack_ids, env._attack_t), 'label': bank.label[env._attack_ids],
                    'two': feet_two_levels(), 't': env.episode_length_buf.clone()}
            start_two = torch.where(env.episode_length_buf <= 1, last['two'], start_two)
            start_overhang = torch.where(env.episode_length_buf <= 1, torch.zeros_like(start_overhang), start_overhang)
            start_overhang |= (env.episode_length_buf <= 10) & (env._feet_off_level > 0)
            new = (env.episode_length_buf == 1) & counted
            diag['starts'] += int(new.sum()); diag['starts_two'] += int((new & last['two']).sum())
            at10 = (env.episode_length_buf == 10) & counted
            diag['starts_overhang'] += int((at10 & start_overhang).sum()); diag['starts10'] = diag.get('starts10', 0) + int(at10.sum())
        obs, _, dones, _ = env.step({'actions': policy(obs['actor_obs'])})
        dones = dones.bool()
        alive = ~dones & counted  # terminal states are reset already
        smooth.step(alive)
        cq.step(dones, counted)
        if alive.any():
            body_err_sum += (env.ref_body_pos_extend[alive] - env._rigid_body_pos_extend[alive]).norm(dim=-1).mean(dim=-1).sum().item() * 1e3
            err_steps += int(alive.sum())
            overlap_steps += float((env._attacker_overlap[alive] > 0).sum())
        by = env.reset_buf_terminate_by
        for i in torch.nonzero(dones & counted).flatten().tolist():
            if 'gravity' in by and bool(by['gravity'][i]):
                episode_counts['fall'] += 1
            elif bool(env.time_out_buf[i]):
                episode_counts['survived'] += 1
            else:
                episode_counts['drift'] += 1
            if args.diag and not bool(env.time_out_buf[i]):
                diag['n'] += 1
                name = env.config.defense.counter.clips_all[int(last['clip'][i])] if bool(last['counter'][i]) else 'none'
                diag['counter'][name] = diag['counter'].get(name, 0) + 1
                diag['phase'][int(last['phase'][i])] += 1
                a = ATTACKS[int(last['label'][i])]
                diag['attack'][a] = diag['attack'].get(a, 0) + 1
                diag['two_levels_end'] += int(last['two'][i])
                diag['two_levels_start'] += int(start_two[i])
                t = float(last['t'][i]) * env.dt
                diag['time'][0 if t < 4 else 1 if t < 10 else 2 if t < 20 else 3] += 1
                if t < 4:
                    diag['early'] += 1; diag['early_two_start'] += int(start_two[i]); diag['early_counter'] += int(bool(last['counter'][i]))
                    diag['early_overhang'] += int(start_overhang[i])
            lengths.append(int(env.last_episode_length_buf[i]) * env.dt)
        episodes_done += (dones & counted).long()

n_ep = sum(episode_counts.values())
per_class = {}
for c in range(C):
    n = outcomes[c].tolist()
    resolved = max(sum(n[:3]), 1)
    per_class[ATTACKS[c]] = {'attacks': int(sum(n)), 'hit': n[0] / resolved, 'block': n[1] / resolved, 'evade': n[2] / resolved,
                             'interrupted': n[3] / max(sum(n), 1),
                             'block_target_dist_m': float(block_stats[c, 0]) / max(n[1], 1), 'block_left_arm': float(block_stats[c, 1]) / max(n[1], 1)}
tot = outcomes.sum(dim=0).tolist()
resolved = max(sum(tot[:3]), 1)
result = {
    'checkpoint': str(ckpt), 'name': args.name, 'seed': args.seed, 'blind': args.blind, 'env_from': args.env_from, 'terrain': args.terrain,
    'difficulty': args.difficulty if args.terrain is not None else None, 'overrides': overrides,
    'episodes': n_ep, 'episode_s': args.episode_s,
    'survival_rate': episode_counts['survived'] / n_ep, 'fall_rate': episode_counts['fall'] / n_ep,
    'drift_rate': episode_counts['drift'] / n_ep, 'mean_episode_s': float(np.mean(lengths)),
    'guard_body_err_mm': body_err_sum / max(err_steps, 1),
    'attacks': int(sum(tot)), 'hit_rate': tot[0] / resolved, 'block_rate': tot[1] / resolved, 'evade_rate': tot[2] / resolved,
    'interrupted_rate': tot[3] / max(sum(tot), 1),
    'block_target_dist_m': float(block_stats[:, 0].sum()) / max(tot[1], 1), 'block_left_arm': float(block_stats[:, 1].sum()) / max(tot[1], 1),
    'attacker_overlap_rate': overlap_steps / max(err_steps, 1),
    'block_max_target_dist': env._block_max_target_dist,
    'per_class': per_class,
    'smoothness': smooth.result(),
    'counter_quality': cq.result(),
}
if combo_outcomes.sum() > 0:
    n = combo_outcomes.tolist()
    result['combo_second'] = {'attacks': int(sum(n)), 'hit': n[0] / sum(n), 'block': n[1] / sum(n), 'evade': n[2] / sum(n)}
counters = None
if getattr(env, '_counter_enabled', False):  # counterattacks: started, landed on the attacker, performed to the end
    totals = env._counter_totals.tolist()
    counters = {name: {'started': int(n[0]), 'landed': n[1] / max(n[0], 1), 'finished': n[2] / max(n[0], 1)}
                for name, n in zip(env._counter_bank.names, totals) if n[0] > 0}
    started = max(sum(n[0] for n in totals), 1)
    counters['all'] = {'started': int(sum(n[0] for n in totals)), 'landed': sum(n[1] for n in totals) / started,
                       'finished': sum(n[2] for n in totals) / started}
    result['counters'] = counters
if has_belief:
    result['belief'] = {
        'class_acc_by_phase': (belief_correct / belief_steps.clamp(min=1)).tolist(),
        'class_acc': float(belief_correct.sum() / belief_steps.sum().clamp(min=1)),
        'phase_acc': phase_correct / max(float(belief_steps.sum()), 1),
        'tti_mae_strike_s': tti_abs_sum / max(tti_steps, 1),
    }
out = ckpt.parent / 'metrics' / 'defense_eval' / ckpt.stem / f'{args.name}_seed{args.seed}.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))

if args.diag and diag['n']:
    n = diag['n']
    pct = lambda k: f"{100 * k / n:.0f}%"
    print(f"DIAG {n} early ends: during a counter {pct(n - diag['counter'].get('none', 0))} "
          f"({', '.join(f'{k} {pct(v)}' for k, v in sorted(diag['counter'].items(), key=lambda kv: -kv[1]) if k != 'none')})")
    print(f"DIAG attack phase at the end (guard/strike/retract/guard): {' / '.join(pct(v) for v in diag['phase'])}; "
          f"attack class: {', '.join(f'{k} {pct(v)}' for k, v in sorted(diag['attack'].items(), key=lambda kv: -kv[1]))}")
    print(f"DIAG feet on two levels: at the end {pct(diag['two_levels_end'])}, at the episode start {pct(diag['two_levels_start'])}; "
          f"ended at 0-4 / 4-10 / 10-20 / 20-30 s: {' / '.join(pct(v) for v in diag['time'])}")
    e = max(diag['early'], 1)
    print(f"DIAG all episode starts on two levels: {100 * diag['starts_two'] / max(diag['starts'], 1):.0f}% of {diag['starts']}; "
          f"ends in the first 4 s: {diag['early']}, of them started on two levels {100 * diag['early_two_start'] / e:.0f}%, "
          f"during a counter {100 * diag['early_counter'] / e:.0f}%")
    print(f"DIAG a foot across an edge in the first 0.2 s: {100 * diag['starts_overhang'] / max(diag.get('starts10', 0), 1):.0f}% of all "
          f"episodes, {100 * diag['early_overhang'] / e:.0f}% of the ends in the first 4 s")
print(f"\nRESULT {args.name} seed {args.seed}{'' if args.blind == 'none' else f' (blind: {args.blind})'}: {n_ep} episodes | survived {result['survival_rate']:.1%} | "
      f"fall {result['fall_rate']:.1%} | drift {result['drift_rate']:.1%} | mean length {result['mean_episode_s']:.1f} s | "
      f"guard body err {result['guard_body_err_mm']:.1f} mm | forearm in the attacker {result['attacker_overlap_rate']:.1%} of steps")
print(f"blocks count within {result['block_max_target_dist']} m of the target; blk dist = contact to target, L arm = share with the left forearm")
print(f"{'attack':15s} {'n':>5s} {'hit':>6s} {'block':>6s} {'evade':>6s} {'interr':>7s} {'blk dist':>9s} {'L arm':>6s}")
for name, r in list(per_class.items()) + [('all', {'attacks': result['attacks'], 'hit': result['hit_rate'], 'block': result['block_rate'],
                                                   'evade': result['evade_rate'], 'interrupted': result['interrupted_rate'],
                                                   'block_target_dist_m': result['block_target_dist_m'], 'block_left_arm': result['block_left_arm']})]:
    blk = f"{r['block_target_dist_m'] * 100:7.1f}cm {r['block_left_arm']:6.0%}" if r['block'] > 0 else f"{'-':>9s} {'-':>6s}"
    print(f"{name:15s} {r['attacks']:5d} {r['hit']:6.1%} {r['block']:6.1%} {r['evade']:6.1%} {r['interrupted']:7.1%} {blk}")
if 'combo_second' in result:
    r = result['combo_second']
    print(f"{'combo 2nd':15s} {r['attacks']:5d} {r['hit']:6.1%} {r['block']:6.1%} {r['evade']:6.1%}")
if counters:
    print(f"{'counter':20s} {'n':>5s} {'landed':>7s} {'finished':>9s}")
    for name, r in counters.items():
        print(f"{name:20s} {r['started']:5d} {r['landed']:7.1%} {r['finished']:9.1%}")
m = result['smoothness']
print(f"smoothness (robot / reference): action rate {m['action_rate_rms']:.3f}, jerk {m['action_jerk_rms']:.3f} | "
      f"joint acc {m['dof_acc_rms']:.1f} / {m['ref_dof_acc_rms']:.1f} rad/s^2 | joint vel {m['dof_vel_rms']:.2f} / {m['ref_dof_vel_rms']:.2f} rad/s | "
      f"power >{Smoothness.HF_HZ:.0f} Hz legs {m['hf_share_legs']:.0%} / {m['ref_hf_share_legs']:.0%}, "
      f"waist {m['hf_share_waist']:.0%} / {m['ref_hf_share_waist']:.0%}, arms {m['hf_share_arms']:.0%} / {m['ref_hf_share_arms']:.0%} | "
      f"mech power {m['mech_power_w']:.0f} W")
print(f"joint speed RMS above {Smoothness.HF_HZ:.0f} Hz (rad/s, robot / reference): legs {m['hf_vel_rms_legs']:.3f} / {m['ref_hf_vel_rms_legs']:.3f}, "
      f"waist {m['hf_vel_rms_waist']:.3f} / {m['ref_hf_vel_rms_waist']:.3f}, arms {m['hf_vel_rms_arms']:.3f} / {m['ref_hf_vel_rms_arms']:.3f}")
print(f"penalty raw sums per step: action_rate {m['penalty_action_rate_raw']:.3g}, dof_acc {m['penalty_dof_acc_raw']:.3g}, "
      f"torques {m['penalty_torques_raw']:.3g}")
if result['counter_quality']:
    print(f"{'counter quality':16s} {'n':>5s} {'peak m/s rob/ref':>17s} {'lag ms':>7s} {'|lag| ms':>8s} {'jerk rob/ref':>14s} {'limb vel rob/ref':>17s} {'vel err':>8s}")
    for name, q in result['counter_quality'].items():
        print(f"cq {name:13s} {q['counters']:5d} {q['peak_speed']:8.2f} / {q['ref_peak_speed']:5.2f} {1e3 * q['lag_s']:7.0f} {1e3 * q['abs_lag_s']:8.0f} "
              f"{q['jerk_rms']:6.0f} / {q['ref_jerk_rms']:5.0f} {q['limb_vel_rms']:8.2f} / {q['ref_limb_vel_rms']:5.2f} {q['limb_vel_err_rms']:8.2f}")
    q = result['counter_quality']['all']
    print(f"counter cost (striking limb): torque-limit penalty raw {q['limb_torque_limit_raw_per_step']:.3f} per step, "
          f"{q['limb_near_torque_limit_share']:.1%} of joint-steps above the soft torque limit, action rate RMS {q['limb_action_rate_rms']:.3f}")
    print(f"counter dwell: striker within the landing radius {1e3 * q['dwell_s']:.0f} ms per counter of the "
          f"{2e3 * env._counter_hit_window:.0f} ms hit window; proximity kernel sum {q['proximity_kernel_sum']:.1f} per counter")
if has_belief:
    b = result['belief']
    print(f"belief: class acc {b['class_acc']:.1%} (by phase guard/strike/retract/guard: "
          f"{' / '.join(f'{a:.0%}' for a in b['class_acc_by_phase'])}) | phase acc {b['phase_acc']:.1%} | "
          f"tti MAE during strike {b['tti_mae_strike_s'] * 1e3:.0f} ms")
print(f'wrote {out}')
