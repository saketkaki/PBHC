"""Evaluate a checkpoint on one terrain: completion, falls and loss of tracking, and tracking errors.

Every episode starts at the motion start (evaluation mode). An episode ends when the motion ends
(completed), when the robot tips over (fall: the gravity termination of the training config) or when any
body is more than --far m from its reference (drift, as in ratio_eps.py). Tracking errors are averaged over
all steps of all episodes:
  root_rel_body_err  mean body position error relative to the root [mm] (like E_mpbpe; ignores drift)
  global_body_err    mean global body position error [mm] (against the terrain-adapted reference if enabled)
  joint_err          mean absolute joint angle error [rad]
Each env contributes its first ceil(episodes / num_envs) episodes, so short (failed) episodes are not over-counted.
On step tiles each episode also records how close the reference feet come to the step edge, where the raised side is
at the start relative to the robot heading, how many reference feet start on it, and how often a shin (knee link)
touches anything (`step_episodes`).

usage (from the repo root, env `pbhc` active):
  python scripts/terrain_eval.py CKPT --name step_d10 [--episodes 200] [--num-envs 100] [--seed 0] [--far 0.7]
         [key=value ...]   e.g. "terrain.terrain_types=[step]" "terrain.terrain_proportions=[1.0]" terrain.fixed_difficulty=1.0
The terrain group is replaced by terrain_kungfu (no curriculum) before the overrides. Results go to
CKPT_DIR/metrics/<--out-dir>/<ckpt name>/<name>_seed<seed>.json.
"""
import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('ckpt')
parser.add_argument('--name', required=True)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--num-envs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--far', type=float, default=0.7)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--out-dir', default='terrain_eval_v2')  # terrain_eval: older results (first episodes to finish)
args, overrides = parser.parse_known_args()

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import isaacgym  # noqa: F401,E402  (must be imported before torch)
import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
import humanoidverse.utils.config_utils  # noqa: F401,E402  (registers the `eval` resolver)
from humanoidverse.utils.helpers import pre_process_config  # noqa: E402

ckpt = Path(args.ckpt)
config = OmegaConf.load(ckpt.parent / 'config.yaml')
if config.get('eval_overrides') is not None:
    config = OmegaConf.merge(config, config.eval_overrides)
terrain = OmegaConf.load('humanoidverse/config/terrain/terrain_kungfu.yaml')
config['terrain'] = terrain['terrain']
config = OmegaConf.merge(config, OmegaConf.from_dotlist(['headless=True', f'num_envs={args.num_envs}'] + overrides))
termination = config.env.config.termination
for key in termination:
    termination[key] = key in ('terminate_when_motion_end', 'terminate_by_gravity', 'terminate_when_motion_far')
config.env.config.termination_scales.termination_motion_far_threshold = args.far
config.env.config.termination_curriculum.terminate_when_motion_far_curriculum = False
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

episodes_per_env = -(-args.episodes // args.num_envs)
episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

# step edge of each env's tile: unit normal and offset from the tile centre (the env origin)
terrain_gen = getattr(env.simulator, 'terrain', None)
step_edges = getattr(terrain_gen, 'step_edges', None)
has_step = step_edges is not None and not np.isnan(step_edges).all()
if has_step:
    origins = env.env_origins[:, :2]
    rows = (origins[:, 0] / terrain_gen.env_length).long().cpu().numpy()
    cols = (origins[:, 1] / terrain_gen.env_width).long().cpu().numpy()
    edge = torch.tensor(step_edges[rows, cols], dtype=torch.float, device=env.device)  # [num_envs, 2]
    edge_normal = torch.stack([torch.cos(edge[:, 0]), torch.sin(edge[:, 0])], dim=-1)
    min_edge_dist = torch.full((env.num_envs, len(env.feet_indices)), float('inf'), device=env.device)
    straddle_steps = torch.zeros(env.num_envs, device=env.device)
    raised_bearing = torch.zeros(env.num_envs, device=env.device)  # [deg] raised side in the start heading frame, 0 = ahead, 90 = left
    start_feet_high = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    shin_ids = [env.simulator.find_rigid_body_indice(f'{side}_knee_link') for side in ('left', 'right')]
    shin_contact_steps = torch.zeros(env.num_envs, device=env.device)
    need_start = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)  # start configuration not recorded yet
    step_episodes = []

counts = {'completed': 0, 'fall': 0, 'drift': 0}
lengths, ratios = [], []
err_sum = {'root_rel_body_err': 0., 'global_body_err': 0., 'joint_err': 0.}
err_steps = 0
with torch.no_grad():
    obs = env.reset_all()
    while (episodes_done < episodes_per_env).any():
        obs, _, dones, _ = env.step({'actions': policy(obs['actor_obs'])})
        dones = dones.bool()
        counted = episodes_done < episodes_per_env
        if has_step:  # before the ended episodes are recorded; the reference of reset envs is the new start
            feet = env.ref_body_pos_extend[:, env.feet_indices, :2] - origins[:, None]
            side = (feet * edge_normal[:, None]).sum(dim=-1) - edge[:, 1:]  # signed distance of each foot [m]
            live = ~dones
            min_edge_dist[live] = torch.minimum(min_edge_dist[live], side[live].abs())
            straddle_steps += (live & ((side > 0).any(dim=-1) & (side < 0).any(dim=-1))).float()
            started = live & need_start  # first step of an episode: record the start configuration
            need_start &= ~started
            if started.any():
                quat = env.simulator.robot_root_states[:, 3:7]
                x, y, z, w = quat.unbind(-1)
                yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                bearing = torch.rad2deg(torch.remainder(edge[:, 0] - yaw + torch.pi, 2 * torch.pi) - torch.pi)
                raised_bearing[started] = bearing[started]
                start_feet_high[started] = (side > 0).sum(dim=-1)[started]
            shin_force = env.simulator.contact_forces[:, shin_ids].norm(dim=-1)
            shin_contact_steps += (live & (shin_force > 1.).any(dim=-1)).float()
        alive = ~dones & counted  # errors of the steps that did not end an episode (terminal states are reset already)
        if alive.any():
            ref, robot = env.ref_body_pos_extend[alive], env._rigid_body_pos_extend[alive]
            rel = (ref - ref[:, :1]) - (robot - robot[:, :1])
            err_sum['root_rel_body_err'] += rel.norm(dim=-1).mean(dim=-1).sum().item() * 1e3
            err_sum['global_body_err'] += (ref - robot).norm(dim=-1).mean(dim=-1).sum().item() * 1e3
            err_sum['joint_err'] += env.dif_joint_angles[alive].abs().mean(dim=-1).sum().item()
            err_steps += int(alive.sum())
        by = env.reset_buf_terminate_by
        for i in torch.nonzero(dones & counted).flatten().tolist():
            if bool(by['motion_end'][i]):
                outcome = 'completed'
            elif 'gravity' in by and bool(by['gravity'][i]):
                outcome = 'fall'
            else:
                outcome = 'drift'
            counts[outcome] += 1
            length = int(env.last_episode_length_buf[i])
            lengths.append(length)
            ratios.append(min(1., length * env.dt / float(env.motion_len[i])))
            if has_step:
                step_episodes.append({'outcome': outcome, 'edge_offset': float(edge[i, 1]),
                                      'min_foot_edge_dist': min_edge_dist[i].tolist(),
                                      'straddle_frac': float(straddle_steps[i]) / max(length, 1),
                                      'raised_bearing_deg': float(raised_bearing[i]),
                                      'start_feet_high': int(start_feet_high[i]),
                                      'shin_contact_frac': float(shin_contact_steps[i]) / max(length, 1)})
        episodes_done += (dones & counted).long()
        if has_step:
            min_edge_dist[dones] = float('inf')
            straddle_steps[dones] = 0
            shin_contact_steps[dones] = 0
            need_start |= dones

n = sum(counts.values())
result = {
    'checkpoint': str(ckpt), 'name': args.name, 'seed': args.seed, 'episodes': n, 'far_threshold': args.far,
    'overrides': overrides,
    'completion_rate': counts['completed'] / n, 'fall_rate': counts['fall'] / n, 'drift_rate': counts['drift'] / n,
    'episode_length_ratio': float(np.mean(ratios)),
    **{k: v / max(err_steps, 1) for k, v in err_sum.items()},
}
if has_step:
    result['step_episodes'] = step_episodes
out = ckpt.parent / 'metrics' / args.out_dir / ckpt.stem / f'{args.name}_seed{args.seed}.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(f"RESULT {args.name} seed {args.seed}: episodes {n} | completed {result['completion_rate']:.1%} | fall {result['fall_rate']:.1%} | "
      f"drift {result['drift_rate']:.1%} | length ratio {result['episode_length_ratio']:.1%} | "
      f"root-rel body err {result['root_rel_body_err']:.1f} mm | global body err {result['global_body_err']:.1f} mm | "
      f"joint err {result['joint_err']:.3f} rad")
print(f'wrote {out}')
