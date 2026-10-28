"""Warm-start a checkpoint for a larger observation set (e.g. PBHC `main` -> `main_terrain` with the height scan).

The first layer of the actor and of the critic is rebuilt for the new observation layout: the weights of every
observation term present in the old layout are copied to that term's new position, and new terms (e.g.
`height_scan`) get zero weights. The expanded network therefore computes exactly the same actions and values
as the original until training changes the new weights. The optimizer state is dropped (fine-tune with
`algo.config.load_optimizer=False`); the iteration counter is kept.

usage (from the repo root, env `pbhc` active):
  python scripts/expand_checkpoint.py CKPT NEW_OBS_YAML OUT_CKPT
  e.g. python scripts/expand_checkpoint.py example/pretrained_horse_stance_pose/model_50000.pt \\
           humanoidverse/config/obs/motion_tracking/main_terrain.yaml logs/warmstart/main_terrain_from_50000.pt
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)
from humanoidverse.utils.helpers import pre_process_config


def obs_slices(config):
    config = config.copy()
    pre_process_config(config)
    return OmegaConf.to_container(config.env.config.obs.post_compute_config.obs_slices), dict(config.robot.algo_obs_dim_dict)


def expand_first_layer(weight, old_slices, new_slices, new_dim):
    new_weight = torch.zeros(weight.shape[0], new_dim, dtype=weight.dtype)
    for key, (new_start, new_end) in new_slices.items():
        if key in old_slices:
            old_start, old_end = old_slices[key]
            assert old_end - old_start == new_end - new_start, f'{key}: size changed'
            new_weight[:, new_start:new_end] = weight[:, old_start:old_end]
        else:
            print(f'  new term {key} [{new_start}:{new_end}] -> zero weights')
    return new_weight


ckpt, new_obs_yaml, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
old_config = OmegaConf.load(ckpt.parent / 'config.yaml')
new_config = old_config.copy()
new_obs = OmegaConf.load(new_obs_yaml)
new_obs.pop('defaults', None)
new_config['obs'] = new_obs['obs']
old_slices, old_dims = obs_slices(old_config)
new_slices, new_dims = obs_slices(new_config)

state = torch.load(ckpt, map_location='cpu')
for net, obs_key, model_key, layer in (('actor', 'actor_obs', 'actor_model_state_dict', 'actor_module.module.0.weight'),
                                       ('critic', 'critic_obs', 'critic_model_state_dict', 'critic_module.module.0.weight')):
    weight = state[model_key][layer]
    assert weight.shape[1] == old_dims[obs_key], f'{net}: checkpoint input {weight.shape[1]} != config {old_dims[obs_key]}'
    print(f'{net}: {old_dims[obs_key]} -> {new_dims[obs_key]} inputs')
    state[model_key][layer] = expand_first_layer(weight, old_slices[obs_key], new_slices[obs_key], new_dims[obs_key])
state.pop('actor_optimizer_state_dict')
state.pop('critic_optimizer_state_dict')

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, out)
print(f'wrote {out}')
