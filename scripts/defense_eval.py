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

# count the outcomes exactly as the env scores and logs them
log_outcomes = env._log_defense_outcomes
def count_outcomes(labels, event):
    resolved = (event > 0) & counted
    outcomes.index_put_((labels[resolved], event[resolved] - 1), torch.ones(int(resolved.sum()), device=dev), accumulate=True)
    blocked = (event == 2) & counted  # _record_outcomes ran earlier in this step
    stats = torch.stack([(env._outcome_point - env._outcome_target).norm(dim=-1), (env._outcome_arm_id == 0).float()], dim=1)
    block_stats.index_add_(0, labels[blocked], stats[blocked])
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
}
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
if counters:
    print(f"{'counter':20s} {'n':>5s} {'landed':>7s} {'finished':>9s}")
    for name, r in counters.items():
        print(f"{name:20s} {r['started']:5d} {r['landed']:7.1%} {r['finished']:9.1%}")
if has_belief:
    b = result['belief']
    print(f"belief: class acc {b['class_acc']:.1%} (by phase guard/strike/retract/guard: "
          f"{' / '.join(f'{a:.0%}' for a in b['class_acc_by_phase'])}) | phase acc {b['phase_acc']:.1%} | "
          f"tti MAE during strike {b['tti_mae_strike_s'] * 1e3:.0f} ms")
print(f'wrote {out}')
