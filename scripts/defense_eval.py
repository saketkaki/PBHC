"""Evaluate a defense checkpoint (env.config.defense, DEFENSE_MANEUVERS.md): attack outcomes per class, episode survival,
guard tracking and the belief head.

Every env runs --episodes / --num-envs episodes of --episode-s seconds with the training terminations (falls, any body
more than the training threshold from the guard reference). Attacks are counted as the env scores them:
  hit / block / evade   the outcome rules of the env (hit_radius, block_radius)
  interrupted           the episode ended by a fall or drift after the attack's onset and before its outcome
Rates are shares of the resolved attacks; interrupted attacks are reported separately, so a policy that falls instead of
getting hit does not look better (check fall_rate and interrupted together).
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
         [--blind {none,freeze,zero}] [--env-from RUN] [key=value ...]   e.g. env.config.defense.block_radius=0 env.config.defense.placement_noise.yaw_deg=30
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
     'env.config.defense.debug_draw=False', f'env.config.defense.blind_actor={args.blind}'] + overrides))
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

# count the outcomes exactly as the env scores and logs them
log_outcomes = env._log_defense_outcomes
def count_outcomes(labels, event):
    resolved = (event > 0) & counted
    outcomes.index_put_((labels[resolved], event[resolved] - 1), torch.ones(int(resolved.sum()), device=dev), accumulate=True)
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
lengths = []
body_err_sum, err_steps = 0., 0
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

        obs, _, dones, _ = env.step({'actions': policy(obs['actor_obs'])})
        dones = dones.bool()
        alive = ~dones & counted  # terminal states are reset already
        if alive.any():
            body_err_sum += (env.ref_body_pos_extend[alive] - env._rigid_body_pos_extend[alive]).norm(dim=-1).mean(dim=-1).sum().item() * 1e3
            err_steps += int(alive.sum())
        by = env.reset_buf_terminate_by
        for i in torch.nonzero(dones & counted).flatten().tolist():
            if 'gravity' in by and bool(by['gravity'][i]):
                episode_counts['fall'] += 1
            elif bool(env.time_out_buf[i]):
                episode_counts['survived'] += 1
            else:
                episode_counts['drift'] += 1
            lengths.append(int(env.last_episode_length_buf[i]) * env.dt)
        episodes_done += (dones & counted).long()

n_ep = sum(episode_counts.values())
per_class = {}
for c in range(C):
    n = outcomes[c].tolist()
    resolved = max(sum(n[:3]), 1)
    per_class[ATTACKS[c]] = {'attacks': int(sum(n)), 'hit': n[0] / resolved, 'block': n[1] / resolved, 'evade': n[2] / resolved,
                             'interrupted': n[3] / max(sum(n), 1)}
tot = outcomes.sum(dim=0).tolist()
resolved = max(sum(tot[:3]), 1)
result = {
    'checkpoint': str(ckpt), 'name': args.name, 'seed': args.seed, 'blind': args.blind, 'env_from': args.env_from, 'overrides': overrides,
    'episodes': n_ep, 'episode_s': args.episode_s,
    'survival_rate': episode_counts['survived'] / n_ep, 'fall_rate': episode_counts['fall'] / n_ep,
    'drift_rate': episode_counts['drift'] / n_ep, 'mean_episode_s': float(np.mean(lengths)),
    'guard_body_err_mm': body_err_sum / max(err_steps, 1),
    'attacks': int(sum(tot)), 'hit_rate': tot[0] / resolved, 'block_rate': tot[1] / resolved, 'evade_rate': tot[2] / resolved,
    'interrupted_rate': tot[3] / max(sum(tot), 1),
    'per_class': per_class,
}
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

print(f"\nRESULT {args.name} seed {args.seed}{'' if args.blind == 'none' else f' (blind: {args.blind})'}: {n_ep} episodes | survived {result['survival_rate']:.1%} | "
      f"fall {result['fall_rate']:.1%} | drift {result['drift_rate']:.1%} | mean length {result['mean_episode_s']:.1f} s | "
      f"guard body err {result['guard_body_err_mm']:.1f} mm")
print(f"{'attack':15s} {'n':>5s} {'hit':>6s} {'block':>6s} {'evade':>6s} {'interr':>7s}")
for name, r in list(per_class.items()) + [('all', {'attacks': result['attacks'], 'hit': result['hit_rate'], 'block': result['block_rate'],
                                                   'evade': result['evade_rate'], 'interrupted': result['interrupted_rate']})]:
    print(f"{name:15s} {r['attacks']:5d} {r['hit']:6.1%} {r['block']:6.1%} {r['evade']:6.1%} {r['interrupted']:7.1%}")
if has_belief:
    b = result['belief']
    print(f"belief: class acc {b['class_acc']:.1%} (by phase guard/strike/retract/guard: "
          f"{' / '.join(f'{a:.0%}' for a in b['class_acc_by_phase'])}) | phase acc {b['phase_acc']:.1%} | "
          f"tti MAE during strike {b['tti_mae_strike_s'] * 1e3:.0f} ms")
print(f'wrote {out}')
