"""Build the PBHC motion-tracking env from a checkpoint config and run terrain-aware checks.

usage (from the repo root, env `pbhc` active):
  python scripts/terrain_aware_check.py MODE CKPT [--merge cfg.yaml ...] [key=value ...]
  MODE = regress   flat ground: terrain-aware code paths must reproduce original PBHC exactly
  MODE = terrain   terrain: compare reference foot clearance, naive vs ground-relative
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf
import isaacgym  # noqa: F401  (must be imported before torch)
import torch
from hydra.utils import instantiate
import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)

mode, ckpt = sys.argv[1], Path(sys.argv[2])
merges, dotlist = [], []
args = sys.argv[3:]
while args:
    a = args.pop(0)
    if a == '--merge':
        merges.append(args.pop(0))
    else:
        dotlist.append(a)

config = OmegaConf.load(ckpt.parent / 'config.yaml')
if config.get('eval_overrides') is not None:
    config = OmegaConf.merge(config, config.eval_overrides)
for path in merges:
    cfg = OmegaConf.load(path)
    cfg.pop('defaults', None)
    for key in cfg:  # replace whole groups (e.g. `terrain`, `obs`), like a hydra group override
        config[key] = cfg[key]
config = OmegaConf.merge(config, OmegaConf.from_dotlist(['headless=True'] + dotlist))

from humanoidverse.utils.helpers import pre_process_config  # noqa: E402
pre_process_config(config)
env = instantiate(config.env, device='cuda:0')
env.set_is_evaluating() if mode == 'regress' else None
torch.manual_seed(0)
env.reset_all()


def kick(flag):
    env._ground_relative_ref = flag
    env._kick_motion_res_counter = -1
    res = env.kick_motion_res()
    env._kick_motion_res_counter = -1
    return {k: v.clone() for k, v in res.items() if torch.is_tensor(v)}


if mode == 'regress':
    env._update_adaptive_sigma = lambda *a, **k: None  # rewards must not mutate state while we compare
    worst = {}
    for step in range(60):
        actions = 0.3 * torch.randn(env.num_envs, env.dim_actions, device=env.device)
        env.step({'actions': actions})
        off, on = kick(False), kick(True)
        for k in off:
            worst[k] = max(worst.get(k, 0.), (off[k].float() - on[k].float()).abs().max().item())
        env._ground_relative_ref = False
        r_off = env._reward_teleop_body_position_feet()
        env._ground_relative_ref = True
        env.ref_support_height[:] = on['ref_support_height']
        env.ref_feet_ground_height[:] = on['ref_feet_ground_height']
        r_on = env._reward_teleop_body_position_feet()
        worst['reward_feet'] = max(worst.get('reward_feet', 0.), (r_off - r_on).abs().max().item())
        env._ground_relative_ref = False
    scan = env._get_obs_height_scan()
    root_z = env.simulator.robot_root_states[:, 2:3]
    worst['height_scan_vs_root_z'] = (scan - torch.clip(root_z - 0.75, -1, 1)).abs().max().item()
    print('\nREGRESSION (max |difference| between original and terrain-aware paths, flat ground):')
    for k, v in worst.items():
        print(f'  {k:24s} {v:.3e}  {"OK" if v == 0 else "DIFFERENT"}')
    print(f'height_scan shape {tuple(scan.shape)}')

elif mode == 'terrain':
    feet = env.feet_indices
    variants = {  # name: (ground_relative_ref, per_foot_ref, ref_velocity_correction)
        'naive (original PBHC offset)': (False, False, False),
        'rigid shift, no vel. corr.': (True, False, False),
        'rigid shift (row C)': (True, False, True),
        'per-foot (row C+)': (True, True, True),
    }

    def kick_variant(gr, pf, vc):
        env._per_foot_ref, env._ref_velocity_correction = pf, vc
        return kick(gr)

    def flat_feet_z():  # reference feet height as authored on flat ground
        times = (env.episode_length_buf + 1) * env.dt + env.motion_start_times
        offset = env._ref_offset().clone()
        offset[:, 2] = 0.
        return env._motion_lib.get_motion_state(env.motion_ids, times, offset=offset)['rg_pos_t'][:, feet, 2]

    stats = {name: {'err': [], 'low': [], 'vel': [], 'jump': []} for name in variants}
    prev = {name: None for name in variants}
    prev_err = {name: None for name in variants}
    for step in range(120):
        prev_len = env.episode_length_buf.clone()
        env.step({'actions': torch.zeros(env.num_envs, env.dim_actions, device=env.device)})
        cont = env.episode_length_buf == prev_len + 1  # envs that were not reset in this step
        env._ground_relative_ref = True
        authored = flat_feet_z()
        for name, flags in variants.items():
            res = kick_variant(*flags)
            ref_feet = res['rg_pos_t'][:, feet]
            clearance = ref_feet[..., 2] - env._terrain_height_at(ref_feet[..., :2])
            stats[name]['err'].append(clearance - authored)
            stats[name]['low'].append(clearance.min(dim=-1)[0])
            if prev[name] is not None and cont.any():  # reference z velocity vs. change of reference z position
                fd = (ref_feet[..., 2] - prev[name]) / env.dt
                stats[name]['vel'].append((fd - res['body_vel_t'][:, feet, 2])[cont].abs())
                stats[name]['jump'].append((ref_feet[..., 2] - authored - prev_err[name])[cont].abs())  # change of the applied terrain offset
            prev[name] = ref_feet[..., 2]
            prev_err[name] = ref_feet[..., 2] - authored
    env._per_foot_ref, env._ref_velocity_correction = False, True

    print('\nREFERENCE FEET ON THE TERRAIN (ankle-link origin). error = clearance above the ground under the foot - clearance as authored on flat ground')
    print(f'  {"variant":30s} {"mean|err|":>9s} {"feet >3cm too high":>19s} {"feet >1cm in ground":>20s} {"lower foot below ground":>24s} {"|vz - dz/dt| [m/s]":>19s} {"p99 / max offset change per step":>33s}')
    for name in variants:
        e = torch.cat(stats[name]['err']).flatten()
        low = torch.cat(stats[name]['low'])
        v = torch.cat(stats[name]['vel']).mean().item() if stats[name]['vel'] else float('nan')
        jump = torch.cat(stats[name]['jump']).flatten()
        j = f"{jump.quantile(0.99).item():.3f} / {jump.max().item():.3f} m"
        print(f'  {name:30s} {e.abs().mean():9.3f} {(e > 0.03).float().mean() * 100:18.1f}% {(e < -0.01).float().mean() * 100:19.1f}% '
              f'{(low < 0).float().mean() * 100:23.1f}% {v:19.3f} {j:>33s}')
        print(f'  {"":30s} lower foot below ground: worst {low.min():+.3f} m')
    scan = env._get_obs_height_scan()
    print(f'height_scan shape {tuple(scan.shape)}, range [{scan.min():.3f}, {scan.max():.3f}]')
