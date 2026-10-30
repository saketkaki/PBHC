"""Warm-start a checkpoint for a larger observation set (e.g. PBHC `main` -> `main_terrain` with the height scan).

The first layer of the actor and of the critic is rebuilt for the new observation layout: the weights of every
observation term present in the old layout are copied to that term's new position, and new terms (e.g.
`height_scan`) get zero weights. A height scan whose grid changed (env.config.terrain_aware.height_scan, set with
key=value overrides of the new config) keeps the weights of the points present in both grids; new points get zero
weights. The expanded network therefore computes exactly the same actions and values
as the original until training changes the new weights. The Adam moments of the first layers are expanded the
same way (zero for new terms), so the optimizer state can be loaded too (`algo.config.load_optimizer=True`);
the iteration counter is kept.

usage (from the repo root, env `pbhc` active):
  python scripts/expand_checkpoint.py CKPT NEW_OBS_YAML OUT_CKPT [key=value ...]
  e.g. python scripts/expand_checkpoint.py example/pretrained_horse_stance_pose/model_50000.pt \\
           humanoidverse/config/obs/motion_tracking/main_terrain.yaml logs/warmstart/main_terrain_from_50000.pt
       python scripts/expand_checkpoint.py RUN/model_59500.pt humanoidverse/config/obs/motion_tracking/main_terrain_v2.yaml \\
           logs/warmstart/v2.pt "env.config.terrain_aware.height_scan.y_range=[-0.5,0.5]"
(train the expanded checkpoint with the same overrides, so that the environment builds the same grids)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)
from humanoidverse.utils.helpers import pre_process_config


def obs_slices(config):
    config = config.copy()
    pre_process_config(config)
    return OmegaConf.to_container(config.env.config.obs.post_compute_config.obs_slices), dict(config.robot.algo_obs_dim_dict)


def scan_grid(config):
    """(x, y) of each point of the `height_scan` grid, in the order of the observation (x-major)"""
    hs = config.env.config.terrain_aware.get('height_scan', None) or {}
    (x0, x1), (y0, y1), res = hs.get('x_range', [-0.5, 0.5]), hs.get('y_range', [-0.3, 0.3]), hs.get('resolution', 0.1)
    xs = np.linspace(x0, x1, int(round((x1 - x0) / res)) + 1)
    ys = np.linspace(y0, y1, int(round((y1 - y0) / res)) + 1)
    return [(round(x, 4), round(y, 4)) for x in xs for y in ys]


def expand_first_layer(weight, old_slices, new_slices, new_dim, quiet=False):
    new_weight = torch.zeros(weight.shape[0], new_dim, dtype=weight.dtype)
    for key, (new_start, new_end) in new_slices.items():
        if key in old_slices and old_slices[key][1] - old_slices[key][0] != new_end - new_start:
            assert key in ('height_scan', 'height_scan_noisy'), f'{key}: size changed'
            old_start = old_slices[key][0]
            old_index = {p: i for i, p in enumerate(old_grid)}
            assert len(old_index) == old_slices[key][1] - old_start, f'{key}: grid does not match the old config'
            assert len(new_grid) == new_end - new_start, f'{key}: grid does not match the new config'
            kept = [(j, old_index[p]) for j, p in enumerate(new_grid) if p in old_index]
            for j, i in kept:
                new_weight[:, new_start + j] = weight[:, old_start + i]
            if not quiet:
                print(f'  {key}: {len(old_grid)} -> {len(new_grid)} points, {len(kept)} kept, the rest zero weights')
        elif key in old_slices:
            old_start, old_end = old_slices[key]
            new_weight[:, new_start:new_end] = weight[:, old_start:old_end]
        elif not quiet:
            print(f'  new term {key} [{new_start}:{new_end}] -> zero weights')
    return new_weight


ckpt, new_obs_yaml, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
old_config = OmegaConf.load(ckpt.parent / 'config.yaml')
new_config = old_config.copy()
new_obs = OmegaConf.load(new_obs_yaml)
new_obs.pop('defaults', None)
new_config['obs'] = new_obs['obs']
new_config = OmegaConf.merge(new_config, OmegaConf.from_dotlist(sys.argv[4:]))
old_grid, new_grid = scan_grid(old_config), scan_grid(new_config)
old_slices, old_dims = obs_slices(old_config)
new_slices, new_dims = obs_slices(new_config)

state = torch.load(ckpt, map_location='cpu')
for net, obs_key, model_key, layer in (('actor', 'actor_obs', 'actor_model_state_dict', 'actor_module.module.0.weight'),
                                       ('critic', 'critic_obs', 'critic_model_state_dict', 'critic_module.module.0.weight')):
    weight = state[model_key][layer]
    assert weight.shape[1] == old_dims[obs_key], f'{net}: checkpoint input {weight.shape[1]} != config {old_dims[obs_key]}'
    print(f'{net}: {old_dims[obs_key]} -> {new_dims[obs_key]} inputs')
    state[model_key][layer] = expand_first_layer(weight, old_slices[obs_key], new_slices[obs_key], new_dims[obs_key])
    # optimizer state is indexed by parameter order, which is the order of the model state dict
    optimizer_state = state[f'{net}_optimizer_state_dict']['state'][list(state[model_key]).index(layer)]
    for moment in ('exp_avg', 'exp_avg_sq'):
        assert optimizer_state[moment].shape == weight.shape, f'{net} {moment}: shape {optimizer_state[moment].shape}'
        optimizer_state[moment] = expand_first_layer(optimizer_state[moment], old_slices[obs_key], new_slices[obs_key],
                                                     new_dims[obs_key], quiet=True)

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, out)
print(f'wrote {out}')
