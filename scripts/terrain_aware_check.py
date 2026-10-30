"""Build the PBHC motion-tracking env from a checkpoint config and run terrain-aware checks.

usage (from the repo root, env `pbhc` active):
  python scripts/terrain_aware_check.py MODE CKPT [--merge cfg.yaml ...] [key=value ...]
  MODE = regress   flat ground: terrain-aware code paths must reproduce original PBHC exactly
  MODE = terrain   terrain: compare reference foot clearance, naive vs ground-relative
  MODE = scan_noise  terrain, a main_terrain checkpoint: each height-scan perception error against what it should do,
                   and that the actor gets the perturbed scan and the critic the exact one, e.g.
                   ... --merge humanoidverse/config/obs/motion_tracking/main_terrain_noisy.yaml \
                   "env.config.terrain_aware.height_scan_noise.delay_steps=[0,3]" \
                   obs.noise_scales.height_scan=0 obs.noise_scales.height_scan_noisy=0
  MODE = lookup    terrain: legged_gym (min3) vs exact terrain height under weight-bearing feet (physics as the truth)
                   and the shapes of the perception v2 terms, e.g.
                   ... --merge humanoidverse/config/obs/motion_tracking/main_terrain_v2.yaml \
                   "env.config.terrain_aware.height_scan.y_range=[-0.5,0.5]" env.config.terrain_aware.height_lookup=exact
  MODE = swing     terrain (e.g. flat + step): reference foot soles inside the terrain with and without
                   env.config.terrain_aware.swing_clearance (must not change anything on flat ground)
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf
import isaacgym  # noqa: F401  (must be imported before torch)
import torch
from hydra.utils import instantiate
from isaac_utils.rotations import quaternion_to_matrix, xyzw_to_wxyz
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
    variants = {  # name: (ground_relative_ref, per_foot_ref, ref_velocity_correction, leg_ik_ref)
        'naive (original PBHC offset)': (False, False, False, False),
        'rigid shift, no vel. corr.': (True, False, False, False),
        'rigid shift (row C)': (True, False, True, False),
        'per-foot (row C+)': (True, True, True, False),
        'leg IK': (True, False, True, True),
    }

    def kick_variant(gr, pf, vc, ik):
        env._per_foot_ref, env._ref_velocity_correction, env._leg_ik_ref = pf, vc, ik
        return kick(gr)

    def joint_position_conflict(res):  # |ankle position implied by the reference joints - reference ankle position|
        pelvis_R = quaternion_to_matrix(xyzw_to_wxyz(res['root_rot']))
        rel = (pelvis_R[:, None].transpose(-1, -2) @ (res['rg_pos'][:, feet] - res['root_pos'][:, None])[..., None]).squeeze(-1)
        fk_pos, _ = env._leg_fk(res['dof_pos'][:, env._leg_dof_ids])
        return (rel - fk_pos[:, :, -1]).norm(dim=-1)  # [N, 2]

    def flat_feet_z():  # reference feet height as authored on flat ground
        times = (env.episode_length_buf + 1) * env.dt + env.motion_start_times
        offset = env._ref_offset().clone()
        offset[:, 2] = 0.
        return env._motion_lib.get_motion_state(env.motion_ids, times, offset=offset)['rg_pos_t'][:, feet, 2]

    stats = {name: {'err': [], 'low': [], 'vel': [], 'jump': [], 'conflict': []} for name in variants}
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
            stats[name]['conflict'].append(joint_position_conflict(res))
            if prev[name] is not None and cont.any():  # reference z velocity vs. change of reference z position
                fd = (ref_feet[..., 2] - prev[name]) / env.dt
                stats[name]['vel'].append((fd - res['body_vel_t'][:, feet, 2])[cont].abs())
                stats[name]['jump'].append((ref_feet[..., 2] - authored - prev_err[name])[cont].abs())  # change of the applied terrain offset
            prev[name] = ref_feet[..., 2]
            prev_err[name] = ref_feet[..., 2] - authored
    env._per_foot_ref, env._ref_velocity_correction, env._leg_ik_ref = False, True, False

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
        conflict = torch.cat(stats[name]['conflict']).flatten()
        print(f'  {"":30s} lower foot below ground: worst {low.min():+.3f} m; joint-vs-position conflict (ankle) '
              f'mean {conflict.mean() * 100:.2f} cm, max {conflict.max() * 100:.2f} cm')
    scan = env._get_obs_height_scan()
    print(f'height_scan shape {tuple(scan.shape)}, range [{scan.min():.3f}, {scan.max():.3f}]')

elif mode == 'scan_noise':
    assert env._scan_noise_enabled, 'set a non-zero height_scan_noise option so that the noise buffers exist'
    ids = torch.arange(env.num_envs, device=env.device)
    clip = env._height_scan_clip
    zeros = torch.zeros(env.num_envs, env.dim_actions, device=env.device)

    def configure(point=0., bias=0., drift=0., dropout=0., delay=(0, 0)):
        env._scan_point_std, env._scan_bias_std, env._scan_drift_std, env._scan_dropout = point, bias, drift, dropout
        env._scan_delay_range = list(delay)
        env._sample_height_scan_noise(ids)

    def step():  # returns the envs that were not reset in this step
        prev_len = env.episode_length_buf.clone()
        env.step({'actions': zeros})
        return env.episode_length_buf == prev_len + 1

    def exact(xy_offset=None, bias=0.):
        return torch.clip(env._height_scan(xy_offset) - bias, -clip, clip)

    perceived = lambda: env._get_obs_height_scan_noisy()  # noqa: E731
    rows = []  # (check, value, expected)
    configure()
    for _ in range(5):
        step()
    rows.append(('all options 0: |noisy - exact| max', (perceived() - exact()).abs().max().item(), '0'))

    configure(bias=0.05)
    step()
    rows.append(('bias 0.05: |noisy - (exact - bias)| max', (perceived() - exact(bias=env._scan_bias)).abs().max().item(), '0'))
    rows.append(('bias 0.05: std of sampled bias [m]', env._scan_bias.std().item(), '~0.05'))

    configure(drift=0.05)
    step()
    rows.append(('drift 0.05: |noisy - exact at shifted points| max', (perceived() - exact(env._scan_drift)).abs().max().item(), '0'))
    rows.append(('drift 0.05: mean |noisy - exact| [m]', (perceived() - exact()).abs().mean().item(), '> 0 on steps'))

    configure(point=0.02)
    step()
    ref = exact()
    inside = ref.abs() < clip - 0.1
    rows.append(('point 0.02: std of noisy - exact [m]', (perceived() - ref)[inside].std().item(), '~0.02'))

    configure(dropout=0.3)
    step()
    held = []
    for _ in range(10):
        before = perceived().clone()
        cont = step()
        held.append((perceived() == before)[cont].float().mean())
    rows.append(('dropout 0.3: share of points keeping their value', torch.stack(held).mean().item(), '~0.3'))

    configure(delay=(3, 3))
    history, cont_hist, worst, worst_reset, n_reset = [], [], 0., 0., 0
    for k in range(40):
        cont = step()
        history.append(exact().clone())
        cont_hist.append(cont)
        if k >= 3:
            ok = torch.stack(cont_hist[-3:]).all(dim=0)  # same episode for the last 3 steps
            worst = max(worst, (perceived() - history[-4])[ok].abs().max().item() if ok.any() else 0.)
        if k >= 1 and (~cont).any():  # just reset: the buffer is refilled with the first scan of the new episode
            worst_reset = max(worst_reset, (perceived() - history[-1])[~cont].abs().max().item())
            n_reset += int((~cont).sum())
    rows.append(('delay 3: |noisy(t) - exact(t-3)| max', worst, '0'))
    rows.append((f'delay 3, {n_reset} resets: |noisy - exact| max right after reset', worst_reset, '0'))

    configure(point=0.02, bias=0.05)
    step()
    raw = env.obs_buf_dict_raw
    if 'height_scan_noisy' in raw.get('actor_obs', {}):
        rows.append(('actor term height_scan_noisy == perceived', (raw['actor_obs']['height_scan_noisy'] - perceived()).abs().max().item(), '0'))
        rows.append(('critic term height_scan == exact', (raw['critic_obs']['height_scan'] - exact()).abs().max().item(), '0'))

    print('\nHEIGHT-SCAN NOISE CHECKS')
    for name, value, expected in rows:
        print(f'  {name:55s} {value:9.4f}   expected {expected}')

elif mode == 'lookup':
    assert env._exact_height_map is not None, 'set env.config.terrain_aware.height_lookup=exact (trimesh terrain)'
    zeros = torch.zeros(env.num_envs, env.dim_actions, device=env.device)
    terrain_gen = env.simulator.terrain
    rows = (env.env_origins[:, 0] / terrain_gen.env_length).long().tolist()
    cols = (env.env_origins[:, 1] / terrain_gen.env_width).long().tolist()
    terrain_name = [terrain_gen.tile_types[i, j] for i, j in zip(rows, cols)]
    # ankle height above the ground under the ankle, for feet that carry weight and are nearly level: the same for
    # every foot (sole offset) if the lookup matches the collision mesh
    clearance = {'min3': {}, 'exact': {}}
    for k in range(40):
        env.step({'actions': zeros})
        if k < 5:
            continue
        feet = env.simulator._rigid_body_pos[:, env.feet_indices]
        up_z = quaternion_to_matrix(xyzw_to_wxyz(env.simulator._rigid_body_rot[:, env.feet_indices]))[..., 2, 2]
        loaded = (env.simulator.contact_forces[:, env.feet_indices].norm(dim=-1) > 100.) & (up_z > 0.97)
        for lookup, height in (('min3', env._terrain_height_min3), ('exact', env._terrain_height_exact)):
            c = feet[..., 2] - height(feet[..., :2])
            for i, j in torch.nonzero(loaded).tolist():
                clearance[lookup].setdefault(terrain_name[i], []).append(c[i, j].item())
    print('\nTERRAIN LOOKUP: ankle height above the looked-up ground, weight-bearing level feet [cm];'
          ' >2cm off: share more than 2 cm from the flat-ground median')
    print(f'  {"terrain":8s} {"n":>5s}   {"min3 mean / std / >2cm off":>28s}   {"exact mean / std / >2cm off":>28s}')
    flat_median = torch.tensor(clearance['exact'].get('flat', [0.])).median() * 100  # sole offset of a level foot
    for name in sorted(clearance['exact']):
        cells = []
        for lookup in ('min3', 'exact'):
            v = torch.tensor(clearance[lookup][name]) * 100
            cells.append(f'{v.mean():8.2f} {v.std():8.2f} {((v - flat_median).abs() > 2).float().mean() * 100:7.1f}%')
        print(f'  {name:8s} {len(clearance["exact"][name]):5d}   {cells[0]:>28s}   {cells[1]:>28s}')
    raw = env.obs_buf_dict_raw
    for obs_key in raw:
        for term in ('height_scan', 'foot_scan', 'privileged_height_scan'):
            if term in raw[obs_key]:
                print(f'  {obs_key}.{term}: shape {tuple(raw[obs_key][term].shape)}')

elif mode == 'swing':
    feet = env.feet_indices
    terrain_gen = env.simulator.terrain
    rows = (env.env_origins[:, 0] / terrain_gen.env_length).long().tolist()
    cols = (env.env_origins[:, 1] / terrain_gen.env_width).long().tolist()
    tile = [terrain_gen.tile_types[i, j] for i, j in zip(rows, cols)]
    is_flat = torch.tensor([t == 'flat' for t in tile], device=env.device)
    sole = env._scan_grid([-0.065, 0.14], [-0.036, 0.036], 0.0205 / 2)  # sole of the ankle roll link (URDF spheres)
    stats = {flag: {'pen': [], 'swing_pen': []} for flag in (False, True)}
    flat_diff, lift_max = 0., 0.
    for step in range(200):
        env.step({'actions': torch.zeros(env.num_envs, env.dim_actions, device=env.device)})
        res = {}
        for flag in (False, True):
            env._swing_clearance = flag
            res[flag] = kick(True)
            ref_feet = res[flag]['rg_pos_t'][:, feet]  # [N, 2, 3]
            points = env._yaw_rotate(sole, res[flag]['rg_rot_t'][:, feet]) + ref_feet[:, :, None, :2]
            sole_z = ref_feet[..., 2] - env._foot_scan_base_offset  # [N, 2]
            pen = (env._terrain_height_at(points) - sole_z[..., None]).max(dim=-1)[0]  # deepest sole point [N, 2]
            lift = ref_feet[..., 2] - ref_feet[..., 2].min(dim=-1, keepdim=True)[0]
            swinging = (lift > 0.02) & ~is_flat[:, None]
            stats[flag]['pen'].append(pen[~is_flat].flatten())
            stats[flag]['swing_pen'].append(pen[swinging])
            lift_max = max(lift_max, lift.max().item())
        if is_flat.any():
            flat_diff = max(flat_diff, (res[True]['rg_pos_t'] - res[False]['rg_pos_t'])[is_flat].abs().max().item())
    env._swing_clearance = False
    print(f'\nSWING CLEARANCE: reference sole points inside the terrain (non-flat tiles); max reference foot lift {lift_max * 100:.1f} cm')
    for flag in (False, True):
        pen, swing_pen = torch.cat(stats[flag]['pen']), torch.cat(stats[flag]['swing_pen'])
        print(f'  swing_clearance={str(flag):5s}  all feet: >1 cm in ground {(pen > 0.01).float().mean() * 100:5.1f}%, '
              f'>3 cm {(pen > 0.03).float().mean() * 100:5.1f}%  |  swinging feet (lift > 2 cm, {len(swing_pen)}): '
              f'>1 cm in ground {(swing_pen > 0.01).float().mean() * 100:5.1f}%, >3 cm {(swing_pen > 0.03).float().mean() * 100:5.1f}%, '
              f'worst {swing_pen.max() * 100 if len(swing_pen) else 0:.1f} cm')
    print(f'  flat tiles: max |reference change| with swing clearance {flat_diff:.2e} m (expected 0)')

