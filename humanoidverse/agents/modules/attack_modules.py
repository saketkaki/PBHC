from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .ppo_modules import PPOActor


def _mlp(input_dim, hidden_dims, output_dim, activation):
    layers, dim = [], input_dim
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(dim, hidden_dim), getattr(nn, activation)()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class AttackEncoder(nn.Module):
    """ Embedding z of the attacker's recent motion: keypoint frames [B, num_frames, frame_dim] (newest first) -> [B, latent_dim].
    type Conv1D: temporal convolutions over the frames, then a linear layer; type MLP: an MLP on the flattened history. """
    def __init__(self, frame_dim, num_frames, cfg):
        super().__init__()
        self.frame_dim, self.num_frames = frame_dim, num_frames
        if cfg.type == 'Conv1D':
            layers, channels, length = [], frame_dim, num_frames
            for out_channels, kernel, stride in zip(cfg.channels, cfg.kernel_sizes, cfg.strides):
                layers += [nn.Conv1d(channels, out_channels, kernel, stride), getattr(nn, cfg.activation)()]
                channels, length = out_channels, (length - kernel) // stride + 1
            assert length > 0, "attack encoder: too few frames for these kernels and strides"
            self.conv = nn.Sequential(*layers)
            self.head = nn.Linear(channels * length, cfg.latent_dim)
        elif cfg.type == 'MLP':
            self.conv = None
            self.head = _mlp(frame_dim * num_frames, cfg.hidden_dims, cfg.latent_dim, cfg.activation)
        else:
            raise NotImplementedError(f"attack encoder type {cfg.type}")

    def forward(self, frames):
        if self.conv is None:
            return self.head(frames.flatten(1))
        return self.head(self.conv(frames.transpose(1, 2)).flatten(1))


class AttackConditionedPolicy(nn.Module):
    """ Policy conditioned on the attack: actor_obs -> action mean.

    The attacker keypoint history (the `attacker_keypoints` slice of actor_obs) is encoded into z; a belief head on z
    predicts the attack class, its phase and the time to impact, and is trained on privileged labels (belief_loss),
    so its softmax outputs approximate the posterior over the attack given the observed motion. The policy MLP sees the
    rest of actor_obs, z and (condition_on_belief) the belief. Everything is one module on the flat actor_obs, so
    act_inference and the ONNX export work unchanged. """
    def __init__(self, input_dim, attack_slice, num_actions, actor_cfg, encoder_cfg, num_frames, num_classes, num_phases):
        super().__init__()
        self.start, self.end = attack_slice
        attack_dim = self.end - self.start
        assert attack_dim % num_frames == 0
        self.num_frames, self.frame_dim = num_frames, attack_dim // num_frames
        self.num_classes, self.num_phases = num_classes, num_phases
        self.condition_on_belief = encoder_cfg.condition_on_belief
        self.encoder = AttackEncoder(self.frame_dim, num_frames, encoder_cfg)
        self.belief_head = nn.Linear(encoder_cfg.latent_dim, num_classes + num_phases + 1)
        policy_input_dim = input_dim - attack_dim + encoder_cfg.latent_dim + (num_classes + num_phases + 1) * self.condition_on_belief
        self.policy = _mlp(policy_input_dim, actor_cfg.layer_config.hidden_dims, num_actions, actor_cfg.layer_config.activation)

    def encode(self, actor_obs):
        """ -> latent z [B, latent_dim], belief head output [B, num_classes + num_phases + 1] (class logits, phase logits,
        time to impact [s]) """
        frames = actor_obs[:, self.start:self.end].reshape(-1, self.num_frames, self.frame_dim)
        z = self.encoder(frames)
        return z, self.belief_head(z)

    def forward(self, actor_obs):
        z, belief = self.encode(actor_obs)
        features = [actor_obs[:, :self.start], actor_obs[:, self.end:], z]
        if self.condition_on_belief:
            C, P = self.num_classes, self.num_phases
            features += [F.softmax(belief[:, :C], dim=-1), F.softmax(belief[:, C:C + P], dim=-1), belief[:, C + P:]]
        return self.policy(torch.cat(features, dim=-1))


class AttackAwareActor(PPOActor):
    """ PPOActor whose mean is an AttackConditionedPolicy (module_dict.actor.attack_encoder), with the supervised belief
    loss. attack_info is the env's attack_obs_info (frames, classes, phases, layout of attack_privileged). """
    def __init__(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, actor_obs_slices, critic_obs_slices, attack_info):
        nn.Module.__init__(self)
        self.actor_module = AttackConditionedPolicy(
            obs_dim_dict['actor_obs'], actor_obs_slices['attacker_keypoints'], num_actions, module_config_dict,
            module_config_dict.attack_encoder, attack_info['num_frames'], attack_info['num_classes'], attack_info['num_phases'])
        privileged = critic_obs_slices['attack_privileged'][0]
        self.label_slice = slice(privileged + attack_info['label'][0], privileged + attack_info['label'][1])
        self.phase_slice = slice(privileged + attack_info['phase'][0], privileged + attack_info['phase'][1])
        self.tti_index = privileged + attack_info['time_to_impact']

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def belief_loss(self, actor_obs, critic_obs):
        """ Cross-entropy of the attack class and phase plus a smooth-L1 loss on the time to impact, against the critic's
        privileged labels. Returns the loss and logging values (class accuracy over all steps and during strikes). """
        policy = self.actor_module
        C, P = policy.num_classes, policy.num_phases
        _, belief = policy.encode(actor_obs)
        label = critic_obs[:, self.label_slice].argmax(dim=-1)
        phase = critic_obs[:, self.phase_slice].argmax(dim=-1)
        tti = critic_obs[:, self.tti_index]
        class_loss = F.cross_entropy(belief[:, :C], label)
        phase_loss = F.cross_entropy(belief[:, C:C + P], phase)
        tti_loss = F.smooth_l1_loss(belief[:, C + P], tti)
        with torch.no_grad():
            correct = (belief[:, :C].argmax(dim=-1) == label).float()
            striking = (phase == 1) | (phase == 2)
            info = {'Belief_class': class_loss.item(), 'Belief_phase': phase_loss.item(), 'Belief_tti': tti_loss.item(),
                    'Belief_class_acc': correct.mean().item(),
                    'Belief_class_acc_strike': (correct[striking].mean() if striking.any() else correct.new_zeros(())).item(),
                    'Belief_phase_acc': (belief[:, C:C + P].argmax(dim=-1) == phase).float().mean().item()}
        return class_loss + phase_loss + tti_loss, info
