"""Checks of the defense setup (env.config.defense, DEFENSE_MANEUVERS.md).

usage (from the repo root, env `pbhc` active):
  python scripts/defense_check.py MODE [key=value ...]
  MODE = bank    no simulator: the attack bank's forward kinematics against the dataset (striker at the impact frame
                 vs attack_meta impact_point), then the outcome of every clip against the canonical defender holding
                 its guard perfectly, scored with the env's hit / block rules (the do-nothing baseline)
  MODE = policy  no simulator: the attack-conditioned actor on random observations: belief loss and its gradient to the
                 encoder, and the ONNX export against PyTorch
  MODE = env     Isaac Gym, a few envs with zero actions: observation shapes, the attacker placed on the defender
                 at each clip start, outcome events and the logged rates
key=value overrides go to the composed training config (e.g. env.config.defense.block_radius=0).
"""
import sys
from collections import defaultdict

mode, overrides = sys.argv[1], sys.argv[2:]
if mode == 'env':
    import isaacgym  # noqa: F401  (must be imported before torch)
import numpy as np
import torch
from hydra import compose, initialize
import humanoidverse.utils.config_utils  # noqa: F401  (registers the `eval` resolver)
from humanoidverse.utils.helpers import pre_process_config

GUARD = 'g1_attack_dataset/PBHC_attacks/motion_data/attacks/defender_guard.pkl'
with initialize(config_path='../humanoidverse/config', version_base='1.1'):
    config = compose(config_name='base', overrides=[
        '+simulator=isaacgym', '+exp=defense', '+terrain=terrain_locomotion_plane', '+obs=defense/main',
        '+robot=g1/g1_23dof_lock_wrist', '+domain_rand=main', '+rewards=defense/main',
        f'robot.motion.motion_file={GUARD}', 'num_envs=8', 'headless=True'] + overrides)
pre_process_config(config)
dc = config.env.config.defense


def check_bank():
    from humanoidverse.envs.defense.attack_bank import AttackBank, load_joblib_compat, point_segment_distance, sweep_segment_distance
    from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    bank = AttackBank(dc.attack_file, config.robot.motion, 'cpu', attacks=dc.attacks, keypoints=dc.keypoints)
    data = load_joblib_compat(dc.attack_file)
    err = torch.stack([(bank.striker_pos[i, bank.impact_frame[i]] - torch.tensor(data[k]['attack_meta']['impact_point'])).norm()
                       for i, k in enumerate(bank.keys)])
    print(f'{bank.num_clips} clips; striker at impact vs dataset impact_point: mean {err.mean() * 1e3:.2f} mm, max {err.max() * 1e3:.2f} mm')

    # canonical defender: the guard clip's FK, moved onto each clip's defender root and heading
    guard = load_joblib_compat(GUARD)['defender_guard']
    out = Humanoid_Batch(config.robot.motion).fk_batch(torch.as_tensor(guard['pose_aa'][:1])[None].float(),
                                                       torch.as_tensor(guard['root_trans_offset'][:1])[None].float())
    g_pos, g_mat = out.global_translation_extend[0, 0], out.global_rotation_mat_extend[0, 0]
    q = guard['root_rot'][0]
    g_yaw, g_xy = 2 * np.arctan2(q[2], q[3]), torch.as_tensor(guard['root_trans_offset'][0, :2]).float()
    name = bank.body_names.index
    hand = torch.tensor([e['pos'] for e in config.robot.motion.extend_config if e['joint_name'] in ('left_hand_link', 'right_hand_link')])
    outcomes = defaultdict(lambda: np.zeros(4))
    gaps = defaultdict(list)  # before the onset: closest attacker body to a defender body or hand, and striker to forearm
    for i in range(bank.num_clips):
        yaw = float(bank.def_root_yaw[i]) - g_yaw
        R = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=torch.float32)
        pos = (g_pos - torch.cat([g_xy, torch.zeros(1)])) @ R.T + torch.cat([bank.def_root_xy[i], torch.zeros(1)])
        mat = R @ g_mat
        tb = name(bank.target_link[i])
        target = pos[tb] + mat[tb] @ bank.target_local[i]
        elbows = [name('left_elbow_link'), name('right_elbow_link')]
        elbow = pos[elbows][None]
        hands = (elbow[0] + torch.stack([mat[e] @ h for e, h in zip(elbows, hand)]))[None]
        # the attacker in the env: back by the standoff in guard, stepping in for the strike
        back = dc.get('standoff', 0.0) * bank.standoff_weight(torch.full((bank.max_frames,), i), torch.arange(bank.max_frames) * bank.dt)
        striker = bank.striker_pos[i] - back[:, None] * torch.tensor([1.0, 0.0, 0.0])
        onset = int(bank.onset_frame[i])
        attacker = bank.body_pos[i, :onset + 1] - back[:onset + 1, None, None] * torch.tensor([1.0, 0.0, 0.0])
        gaps[bank.attack_names[i]].append(min(float(torch.cdist(attacker.reshape(-1, 3), torch.cat([pos, hands[0]])).min()),
                                              float(point_segment_distance(striker[onset][None], elbow[0], hands[0]).min())))
        status = 0
        for f in range(1, int(bank.length[i])):
            if f < bank.onset_frame[i] or f - 1 > bank.recovered_frame[i]:
                continue
            p0, p1 = striker[f - 1][None], striker[f][None]
            if sweep_segment_distance(p0, p1, elbow, hands)[0] < dc.block_radius:
                status = 2
            elif point_segment_distance(target, p0[0], p1[0]) < dc.hit_radius:
                status = 1
            elif f >= bank.recovered_frame[i]:
                status = 3
            if status:
                break
        outcomes[bank.attack_names[i]][status] += 1
    print(f"\nperfect static guard (hit_radius {dc.hit_radius}, block_radius {dc.block_radius}, standoff {dc.get('standoff', 0.0)}): "
          'share of clips, and the smallest gap between the guards before the onset [m]')
    print(f"{'attack':15s} {'hit':>6s} {'block':>6s} {'evade':>6s} {'gap':>6s}")
    total = np.zeros(4)
    for attack, n in outcomes.items():
        total += n
        print(f'{attack:15s} {n[1] / n.sum():6.2f} {n[2] / n.sum():6.2f} {n[3] / n.sum():6.2f} {min(gaps[attack]):6.3f}')
    print(f"{'all':15s} {total[1] / total.sum():6.2f} {total[2] / total.sum():6.2f} {total[3] / total.sum():6.2f}")


def check_policy():
    import torch.nn.functional as F
    from humanoidverse.agents.modules.attack_modules import AttackAwareActor
    from humanoidverse.envs.defense.attack_bank import NUM_PHASES
    slices = config.env.config.obs.post_compute_config.obs_slices
    dims = config.robot.algo_obs_dim_dict
    C = 8
    info = dict(num_frames=dc.keypoint_history.length, num_classes=C, num_phases=NUM_PHASES,
                label=(0, C), phase=(C, C + NUM_PHASES), time_to_impact=C + NUM_PHASES)
    actor = AttackAwareActor(dims, config.algo.config.module_dict.actor, 23, 0.8, slices.actor_obs, slices.critic_obs, info)
    print(actor)
    B = 64
    actor_obs, critic_obs = torch.randn(B, dims.actor_obs), torch.randn(B, dims.critic_obs)
    p = slices.critic_obs.attack_privileged[0]
    critic_obs[:, p:p + C] = F.one_hot(torch.randint(C, (B,)), C).float()
    critic_obs[:, p + C:p + C + NUM_PHASES] = F.one_hot(torch.randint(NUM_PHASES, (B,)), NUM_PHASES).float()
    actions = actor.act(actor_obs)
    assert actions.shape == (B, 23)
    loss, log = actor.belief_loss(actor_obs, critic_obs)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in actor.actor_module.encoder.parameters())
    print(f'belief loss {loss.item():.3f} {log}; encoder gradient {grad:.3f}')
    assert grad > 0
    # the policy must depend on the attacker slice and on nothing else being silently dropped
    a0, a1 = slices.actor_obs.attacker_keypoints
    other = actor_obs.clone(); other[:, a0:a1] += 1.0
    assert not torch.allclose(actor.act_inference(actor_obs), actor.act_inference(other))

    import os, tempfile
    import onnxruntime as ort
    from humanoidverse.utils.inference_helpers import export_policy_as_onnx
    actor.distribution = None  # as after loading a checkpoint (eval_agent.py exports before acting)
    with tempfile.TemporaryDirectory() as d:
        export_policy_as_onnx({'actor': actor}, d, 'defense.onnx', {'actor_obs': actor_obs[:1]})
        sess = ort.InferenceSession(os.path.join(d, 'defense.onnx'))
        onnx_out = sess.run(None, {sess.get_inputs()[0].name: actor_obs[:1].numpy()})[0]
    diff = np.abs(onnx_out - actor.act_inference(actor_obs[:1]).detach().numpy()).max()
    print(f'ONNX export: max |onnx - torch| = {diff:.2e}')
    assert diff < 1e-4


def check_env():
    from hydra.utils import instantiate
    env = instantiate(config.env, device='cuda:0')
    obs = env.reset_all()
    bank = env._attack_bank
    print({k: tuple(v.shape) for k, v in obs.items()}, dict(config.robot.algo_obs_dim_dict))
    assert all(obs[k].shape[1] == config.robot.algo_obs_dim_dict[k] for k in obs)
    events = defaultdict(int)
    starts = 0
    for step in range(600):
        t_before = env._attack_t.clone()
        obs, rew, done, extras = env.step({'actions': torch.zeros(env.num_envs, env.dim_actions, device=env.device)})
        for e in env._defense_event.tolist():
            events[e] += 1
        # on a clip start (t back to 0) the clip's canonical defender root must sit on the defender root
        started = (env._attack_t < t_before).nonzero().flatten()
        if len(started):
            starts += len(started)
            ids = env._attack_ids[started]
            canon = bank.to_world(torch.cat([bank.def_root_xy[ids], torch.zeros(len(ids), 1, device=env.device)], 1),
                                  ids, env._attack_anchor[started], env._attack_anchor_yaw[started])
            err = (canon[:, :2] - env._attack_anchor[started, :2]).norm(dim=-1).max()
            assert err < 1e-4, err
    print(f'{starts} clip starts in 600 steps x {env.num_envs} envs; events (1 hit, 2 block, 3 evade): {dict(events)}')
    print({k: round(float(v), 3) for k, v in env.log_dict.items() if k.startswith('defense_')})
    print('attack_privileged env 0:', obs['critic_obs'][0, slice(*config.env.config.obs.post_compute_config.obs_slices.critic_obs.attack_privileged)])


{'bank': check_bank, 'policy': check_policy, 'env': check_env}[mode]()
