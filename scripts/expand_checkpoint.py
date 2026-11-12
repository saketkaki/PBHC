"""Warm-start a checkpoint for a larger observation set (e.g. PBHC `main` -> `main_terrain` with the height scan).

The first layer of the actor and of the critic is rebuilt for the new observation layout: the weights of every
observation term present in the old layout are copied to that term's new position, and new terms (e.g.
`height_scan`) get zero weights. A height scan whose grid changed (env.config.terrain_aware.height_scan, set with
key=value overrides of the new config) keeps the weights of the points present in both grids; new points get zero
weights. Likewise a `ref_preview` with other preview times (env.config.terrain_aware.ref_preview.times) keeps the weights
of the times present in both; new times get zero weights. The expanded network therefore computes exactly the same actions and values
as the original until training changes the new weights. The Adam moments of the first layers are expanded the
same way (zero for new terms), so the optimizer state can be loaded too (`algo.config.load_optimizer=True`);
the iteration counter is kept. If the overrides add reward terms (rewards.reward_scales.<term>=<scale>), the multi-head
critic gets a zero-initialised value head for each.

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


def terrain_aware(config):
    return config.env.config.get('terrain_aware', None) or {}  # missing in configs older than the terrain options


def scan_grid(config):
    """(x, y) of each point of the `height_scan` grid, in the order of the observation (x-major)"""
    hs = terrain_aware(config).get('height_scan', None) or {}
    (x0, x1), (y0, y1), res = hs.get('x_range', [-0.5, 0.5]), hs.get('y_range', [-0.3, 0.3]), hs.get('resolution', 0.1)
    xs = np.linspace(x0, x1, int(round((x1 - x0) / res)) + 1)
    ys = np.linspace(y0, y1, int(round((y1 - y0) / res)) + 1)
    return [(round(x, 4), round(y, 4)) for x in xs for y in ys]


def preview_times(config):
    rp = terrain_aware(config).get('ref_preview', None) or {}
    return [round(float(t), 4) for t in rp.get('times', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])]


def expand_first_layer(weight, old_slices, new_slices, new_dim, quiet=False):
    new_weight = torch.zeros(weight.shape[0], new_dim, dtype=weight.dtype)
    for key, (new_start, new_end) in new_slices.items():
        if key in old_slices and old_slices[key][1] - old_slices[key][0] != new_end - new_start:
            assert key in ('height_scan', 'height_scan_noisy', 'ref_preview'), f'{key}: size changed'
            # points of the scan grid, or blocks of 11 values per preview time
            (old_items, new_items, size) = (old_times, new_times, 11) if key == 'ref_preview' else (old_grid, new_grid, 1)
            old_start = old_slices[key][0]
            old_index = {p: i for i, p in enumerate(old_items)}
            assert size * len(old_index) == old_slices[key][1] - old_start, f'{key}: does not match the old config'
            assert size * len(new_items) == new_end - new_start, f'{key}: does not match the new config'
            kept = [(j, old_index[p]) for j, p in enumerate(new_items) if p in old_index]
            for j, i in kept:
                new_weight[:, new_start + size * j:new_start + size * (j + 1)] = \
                    weight[:, old_start + size * i:old_start + size * (i + 1)]
            if not quiet:
                print(f'  {key}: {len(old_items)} -> {len(new_items)} points/times, {len(kept)} kept, the rest zero weights')
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
old_times, new_times = preview_times(old_config), preview_times(new_config)
old_slices, old_dims = obs_slices(old_config)
new_slices, new_dims = obs_slices(new_config)

def policy_input_slices(slices, obs_dim, extra):
    """ Input layout of the policy MLP of an attack-conditioned actor (agents/modules/attack_modules.py): actor_obs
    without the `attacker_keypoints` slice, then the latent and belief (`extra` values) the encoder makes from it. """
    start, end = slices['attacker_keypoints']
    layout = {key: (a, b) if b <= start else (a - (end - start), b - (end - start))
              for key, (a, b) in slices.items() if key != 'attacker_keypoints'}
    layout['_attack_latent_and_belief'] = (obs_dim - (end - start), obs_dim - (end - start) + extra)
    return layout, obs_dim - (end - start) + extra


state = torch.load(ckpt, map_location='cpu')
attack_actor = 'actor_module.policy.0.weight' in state['actor_model_state_dict']  # defense: the encoder is unchanged
actor_layer = 'actor_module.policy.0.weight' if attack_actor else 'actor_module.module.0.weight'
for net, obs_key, model_key, layer in (('actor', 'actor_obs', 'actor_model_state_dict', actor_layer),
                                       ('critic', 'critic_obs', 'critic_model_state_dict', 'critic_module.module.0.weight')):
    weight = state[model_key][layer]
    old_layout, new_layout, old_dim, new_dim = old_slices[obs_key], new_slices[obs_key], old_dims[obs_key], new_dims[obs_key]
    if net == 'actor' and attack_actor:
        assert old_layout['attacker_keypoints'][1] - old_layout['attacker_keypoints'][0] == \
            new_layout['attacker_keypoints'][1] - new_layout['attacker_keypoints'][0], 'attacker_keypoints: size changed'
        extra = weight.shape[1] - (old_dim - (old_layout['attacker_keypoints'][1] - old_layout['attacker_keypoints'][0]))
        old_layout, old_dim = policy_input_slices(old_layout, old_dim, extra)
        new_layout, new_dim = policy_input_slices(new_layout, new_dim, extra)
    assert weight.shape[1] == old_dim, f'{net}: checkpoint input {weight.shape[1]} != config {old_dim}'
    print(f'{net}: {old_dim} -> {new_dim} inputs')
    state[model_key][layer] = expand_first_layer(weight, old_layout, new_layout, new_dim)
    # optimizer state is indexed by parameter order, which is the order of the model state dict
    optimizer_state = state[f'{net}_optimizer_state_dict']['state'][list(state[model_key]).index(layer)]
    for moment in ('exp_avg', 'exp_avg_sq'):
        assert optimizer_state[moment].shape == weight.shape, f'{net} {moment}: shape {optimizer_state[moment].shape}'
        optimizer_state[moment] = expand_first_layer(optimizer_state[moment], old_layout, new_layout, new_dim, quiet=True)

# multi-head critic: one value per reward term (the non-zero reward scales in config order, `termination` last). New reward
# terms (e.g. rewards.reward_scales.defense_counter_hit=250 among the overrides) get a zero-initialised value head.
def reward_terms(config):
    scales = config.rewards.reward_scales
    return [k for k in scales if scales[k] != 0 and k != 'termination'] + (['termination'] if scales.get('termination', 0) != 0 else [])


old_terms, new_terms = reward_terms(old_config), reward_terms(new_config)
critic = state['critic_model_state_dict']
last = max(int(k.split('.')[2]) for k in critic if k.startswith('critic_module.module.') and k.endswith('.weight'))
weight_key, bias_key = f'critic_module.module.{last}.weight', f'critic_module.module.{last}.bias'
if old_terms != new_terms and critic[weight_key].shape[0] == len(old_terms):
    assert set(old_terms) <= set(new_terms), f'reward terms removed: {set(old_terms) - set(new_terms)}'
    rows = [old_terms.index(k) if k in old_terms else None for k in new_terms]
    print(f'critic value heads: {len(old_terms)} -> {len(new_terms)}, new (zero): {[k for k in new_terms if k not in old_terms]}')

    def expand_rows(x):
        new_x = torch.zeros(len(rows), *x.shape[1:], dtype=x.dtype)
        for j, i in enumerate(rows):
            if i is not None:
                new_x[j] = x[i]
        return new_x
    optimizer = state['critic_optimizer_state_dict']['state']
    for key in (weight_key, bias_key):
        index = list(critic).index(key)
        critic[key] = expand_rows(critic[key])
        for moment in ('exp_avg', 'exp_avg_sq'):
            optimizer[index][moment] = expand_rows(optimizer[index][moment])

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, out)
print(f'wrote {out}')
