import torch
import torch.nn.functional as F
from loguru import logger

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.envs.defense.attack_bank import ATTACKS, NUM_PHASES, AttackBank, closest_point_on_segment, sweep_segment_closest
from humanoidverse.envs.env_utils.visualization import Point
from isaac_utils.rotations import calc_heading_quat_inv, my_quat_rotate, quat_conjugate, quat_mul

PENDING, HIT, BLOCKED, EVADED = 0, 1, 2, 3
BLIND_SKELETON = [('pelvis', 'head_link'), ('left_elbow_link', 'left_hand_link'), ('right_elbow_link', 'right_hand_link'),
                  ('pelvis', 'left_knee_link'), ('left_knee_link', 'left_ankle_roll_link'),
                  ('pelvis', 'right_knee_link'), ('right_knee_link', 'right_ankle_roll_link')]  # keypoint pairs, for the viewer


def _heading_yaw(quat):  # xyzw
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LeggedRobotDefense(LeggedRobotMotionTracking):
    """ Defend against attacks thrown by a second G1 (env.config.defense).

    The defender tracks a static guard pose (robot.motion.motion_file = defender_guard.pkl, held past its end) as a style
    prior. The attacker is a kinematic ghost replaying clips of the attack dataset back to back (AttackBank): it is not
    simulated and never collides, so outcomes are scored geometrically from the swept path of its striking point between
    control steps:
      hit      the striker passes within hit_radius of the target point on the defender (target_link + offset)
      blocked  the striker passes within block_radius of a defender forearm (elbow to hand) first
      evaded   neither by the attack's recovered frame
    Each clip is placed so that its canonical defender lands on the defender's root and heading when the clip starts;
    it then stays put, so moving the target away works. The attacker's pose blends over blend_time from the end of one
    clip into the next (every clip starts with at least 0.35 s of guard). With a viewer (attacker_actor), it is drawn as a
    G1: one visual-only actor per body (IsaacGym.add_visual_link_actors), posed from the clip's forward kinematics.

    Observations: attacker_keypoints (the attacker's keypoints relative to the defender root in its heading frame, over
    a short history; actor and critic) and attack_privileged (attack class, phase, time to impact, striker to target,
    resolved; critic only, and the targets of the actor's belief head). blind_actor (freeze / zero) hides the attacker from
    the actor's attacker_keypoints only, to test how much a trained policy relies on seeing it.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self._relax_tracking_rewards()

    def _create_envs(self):
        setting = self.config.defense.get('attacker_actor', 'auto')
        self._attacker_actor = (not self.headless) if setting == 'auto' else bool(setting)
        if self._attacker_actor and not hasattr(self.simulator, 'add_visual_link_actors'):
            logger.warning('attacker_actor: the simulator cannot add visual actors; the attacker is drawn as a skeleton')
            self._attacker_actor = False
        if self._attacker_actor:
            self.simulator.add_visual_link_actors(list(self.config.robot.body_names), color=(0.75, 0.2, 0.15))
        super()._create_envs()

    def _init_buffers(self):
        super()._init_buffers()
        self._init_defense()

    def _init_defense(self):
        dc = self.config.defense
        self._attack_bank = bank = AttackBank(dc.attack_file, self.config.robot.motion, self.device,
                                              attacks=dc.get('attacks', None), keypoints=dc.keypoints)
        N, K, B = self.num_envs, len(dc.keypoints), len(bank.body_names)
        self._kp_hist_len, self._kp_hist_stride = int(dc.keypoint_history.length), int(dc.keypoint_history.stride)
        obs_dims = self.config.obs.obs_dims
        if 'attacker_keypoints' in obs_dims:
            assert obs_dims['attacker_keypoints'] == K * 3 * self._kp_hist_len, \
                f"obs_dims.attacker_keypoints must be {K} keypoints x 3 x {self._kp_hist_len} frames"
        self._hit_radius, self._block_radius = dc.hit_radius, dc.block_radius
        self._blend_time = dc.blend_time
        self._threat_lead_frames = dc.threat_lead_time / bank.dt
        self._proximity_sigma = dc.proximity_sigma
        self._tti_clip = dc.tti_clip
        self._outcome_log_decay = dc.outcome_log_decay
        self._yaw_noise = torch.pi / 180 * dc.placement_noise.yaw_deg
        self._range_noise = dc.placement_noise.range
        self._standoff = dc.get('standoff', 0.0)
        self._defense_debug_draw = dc.debug_draw
        self._blind_actor = dc.get('blind_actor', 'none')
        assert self._blind_actor in ('none', 'freeze', 'zero'), f"blind_actor must be none, freeze or zero, not {self._blind_actor}"
        if self._blind_actor != 'none':
            self._actor_kp_slice = slice(*self.config.obs.post_compute_config.obs_slices.actor_obs.attacker_keypoints)
            self._blind_kp = torch.zeros(N, self._actor_kp_slice.stop - self._actor_kp_slice.start, device=self.device)
            logger.info(f"blind_actor={self._blind_actor}: the actor's attacker_keypoints are "
                        + ("held from the last guard frame before each strike" if self._blind_actor == 'freeze' else "zeros"))

        body = self.simulator._body_list.index
        self._clip_target_body = torch.tensor([body(link) for link in bank.target_link], dtype=torch.long, device=self.device)
        self._elbow_ids = torch.tensor([body('left_elbow_link'), body('right_elbow_link')], dtype=torch.long, device=self.device)
        hands = {e['joint_name']: e['pos'] for e in self.config.robot.motion.extend_config}
        self._hand_offset = torch.tensor([hands['left_hand_link'], hands['right_hand_link']], dtype=torch.float, device=self.device)

        self._attack_ids = bank.sample(N)
        self._attack_t = torch.zeros(N, device=self.device)
        self._attack_anchor = torch.zeros(N, 3, device=self.device)
        self._attack_anchor_yaw = torch.zeros(N, device=self.device)
        self._attack_shift = torch.zeros(N, 3, device=self.device)  # clip frame; -x moves the attacker back
        self._attack_blend = torch.zeros(N, B, 3, device=self.device)
        self._attack_status = torch.zeros(N, dtype=torch.long, device=self.device)
        self._defense_event = torch.zeros(N, dtype=torch.long, device=self.device)  # outcome decided this step
        self._attacker_body_pos = torch.zeros(N, B, 3, device=self.device)
        self._striker_pos = torch.zeros(N, 3, device=self.device)
        self._target_pos = torch.zeros(N, 3, device=self.device)
        self._striker_target_dist = torch.ones(N, device=self.device)
        self._strike_active = torch.zeros(N, device=self.device)
        self._attack_threat = torch.zeros(N, device=self.device)
        self._kp_buffer = torch.zeros(N, (self._kp_hist_len - 1) * self._kp_hist_stride + 1, K, 3, device=self.device)
        self._outcome_counts = torch.zeros(bank.num_classes, 3, device=self.device)
        # for the viewer: where each attack is aimed, its closest approach to the target so far, and where the last
        # resolved attack ended (contact on the forearm, hit point, or closest approach), with the target at that moment
        self._aim_pos = torch.zeros(N, 3, device=self.device)
        self._closest_dist = torch.full((N,), float('inf'), device=self.device)
        self._closest_point = torch.zeros(N, 3, device=self.device)
        self._closest_target = torch.zeros(N, 3, device=self.device)  # the target at the closest approach
        self._outcome_status = torch.zeros(N, dtype=torch.long, device=self.device)
        self._outcome_point = torch.zeros(N, 3, device=self.device)
        self._outcome_target = torch.zeros(N, 3, device=self.device)
        self._outcome_arm = torch.zeros(N, 2, 3, device=self.device)  # elbow and hand of the blocking forearm
        self._outcome_age = torch.full((N,), float('inf'), device=self.device)  # [s] since the last outcome
        self._outcome_hold = dc.get('outcome_marker_time', 1.0)
        self._striker_trail = []  # env 0, for the viewer
        if self._attacker_actor:  # the robot's bodies in the attack bank, their rotations and the blend between clips
            names = list(self.config.robot.body_names)
            self._actor_body_ids = torch.tensor([bank.body_names.index(n) for n in names], device=self.device)
            # kinematic chain (parents first) and each body's offset in its parent's frame, to keep blended poses rigid
            self._actor_chain = [(c, names.index(bank.body_names[p])) for c, p in
                                 enumerate(bank.body_parents[i] for i in self._actor_body_ids.tolist()) if p >= 0]
            pos, rot = bank.body_pos[0, 0, self._actor_body_ids], bank.body_rot[0, 0, self._actor_body_ids]
            self._actor_link_offset = torch.zeros(len(names), 3, device=self.device)
            for c, p in self._actor_chain:
                self._actor_link_offset[c] = my_quat_rotate(quat_conjugate(rot[p:p + 1], w_last=True), (pos[c] - pos[p])[None])[0]
            self._attacker_body_rot = torch.zeros(N, len(self._actor_body_ids), 4, device=self.device)
            self._attacker_body_rot[..., 3] = 1.0
            self._attack_blend_rot = self._attacker_body_rot.clone()

        # layout of attack_privileged, for the actor's belief head (agents/modules/attack_modules.py)
        C = bank.num_classes
        self.attack_obs_info = dict(num_frames=self._kp_hist_len, num_classes=C, num_phases=NUM_PHASES,
                                    label=(0, C), phase=(C, C + NUM_PHASES), time_to_impact=C + NUM_PHASES)

    def _relax_tracking_rewards(self):
        """ Scale the guard-tracking rewards (teleop_*) by 1 - tracking_relax while an attack threatens (from
        threat_lead_time before its onset to its recovered frame), so that defending costs less tracking reward. """
        relax = self.config.defense.tracking_relax
        if relax <= 0:
            return
        for i, name in enumerate(self.reward_names):
            if name.startswith('teleop_'):
                self.reward_functions[i] = lambda fn=self.reward_functions[i]: fn() * (1.0 - relax * self._attack_threat)

    ###########################################################################
    #### Attacker playback and outcomes
    ###########################################################################
    def _attacker_world(self, env_ids, blend=True):
        """ World positions of the attacker's bodies [n, B, 3] and striking point [n, 3] at the current clip time. """
        bank = self._attack_bank
        ids, t = self._attack_ids[env_ids], self._attack_t[env_ids]
        anchor, yaw, shift = self._attack_anchor[env_ids], self._attack_anchor_yaw[env_ids], self._attack_shift[env_ids]
        if self._standoff > 0:  # back by the standoff in guard, stepping in for the strike
            shift = shift.clone()
            shift[:, 0] -= self._standoff * bank.standoff_weight(ids, t)
        bodies = bank.to_world(bank.bodies(ids, t) + shift[:, None], ids, anchor, yaw)
        striker = bank.to_world(bank.striker(ids, t) + shift, ids, anchor, yaw)
        if blend and self._blend_time > 0:
            bodies = bodies + (1.0 - t / self._blend_time).clamp(min=0.0)[:, None, None] * self._attack_blend[env_ids]
        return bodies, striker

    def _start_attacks(self, env_ids, blend):
        """ Start a new clip, placed at the defender's current root and heading; blend=True eases the attacker's pose
        from where the previous clip ended. """
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        old_bodies = self._attacker_body_pos[env_ids].clone()
        self._attack_ids[env_ids] = self._attack_bank.sample(n)
        self._attack_t[env_ids] = 0.0
        root = self.simulator.robot_root_states[env_ids]
        self._attack_anchor[env_ids] = torch.cat([root[:, :2], self.env_origins[env_ids, 2:3]], dim=1)
        self._attack_anchor_yaw[env_ids] = _heading_yaw(root[:, 3:7]) + (torch.rand(n, device=self.device) * 2 - 1) * self._yaw_noise
        self._attack_shift[env_ids, 0] = -(torch.rand(n, device=self.device) * 2 - 1) * self._range_noise
        self._attack_status[env_ids] = PENDING
        self._closest_dist[env_ids] = float('inf')
        ids = self._attack_ids[env_ids]
        bank = self._attack_bank
        self._aim_pos[env_ids] = bank.to_world(bank.striker_pos[ids, bank.impact_frame[ids]] + self._attack_shift[env_ids], ids,
                                               self._attack_anchor[env_ids], self._attack_anchor_yaw[env_ids])
        bodies, striker = self._attacker_world(env_ids, blend=False)
        self._attack_blend[env_ids] = old_bodies - bodies if blend else 0.0
        self._attacker_body_pos[env_ids] = bodies
        self._striker_pos[env_ids] = striker
        if self._attacker_actor:
            self._attack_blend_rot[env_ids] = 0.0
            self._attack_blend_rot[env_ids, :, 3] = 1.0
            if blend:  # previous clip's rotation times the inverse of the new clip's at its start
                delta = quat_mul(self._attacker_body_rot[env_ids], quat_conjugate(self._attacker_rots(env_ids), w_last=True), w_last=True)
                self._attack_blend_rot[env_ids] = torch.where(delta[..., 3:] < 0, -delta, delta)
            self._update_attacker_actors()

    def _attacker_rots(self, env_ids):
        """ World rotations [n, num_bodies, 4] (xyzw) of the attacker's robot bodies at the current clip time, unblended. """
        bank = self._attack_bank
        ids = self._attack_ids[env_ids]
        return bank.to_world_rot(bank.body_rots(ids, self._attack_t[env_ids])[:, self._actor_body_ids], ids, self._attack_anchor_yaw[env_ids])

    def _update_attacker_actors(self):
        """ Pose the attacker's visual link actors from its body positions and (blended) rotations. """
        rot = self._attacker_rots(slice(None))
        pos = self._attacker_body_pos[:, self._actor_body_ids]
        if self._blend_time > 0:
            w = (1.0 - self._attack_t / self._blend_time).clamp(min=0.0)[:, None, None]
            identity = torch.zeros_like(rot)
            identity[..., 3] = 1.0
            delta = w * self._attack_blend_rot + (1.0 - w) * identity
            rot = quat_mul(delta / delta.norm(dim=-1, keepdim=True), rot, w_last=True)
            blending = (w[:, 0, 0] > 0).nonzero(as_tuple=False).flatten()
            if len(blending):  # bodies placed down the chain from the blended root and rotations (the same pose otherwise)
                chain_pos = pos[blending].clone()
                for c, p in self._actor_chain:
                    chain_pos[:, c] = chain_pos[:, p] + my_quat_rotate(rot[blending, p], self._actor_link_offset[c].expand(len(blending), 3))
                pos = pos.clone()
                pos[blending] = chain_pos
        self._attacker_body_rot[:] = rot
        states = self.simulator.visual_root_states
        states[..., :3] = pos
        states[..., 3:7] = rot
        states[..., 7:] = 0.0
        self.simulator.set_visual_actor_states()

    def _step_attacks(self):
        bank = self._attack_bank
        N = self.num_envs
        all_ids = torch.arange(N, device=self.device)
        ids = self._attack_ids
        frame_prev = bank.frame(ids, self._attack_t)
        striker_prev = self._striker_pos.clone()
        self._attack_t += self.dt
        frame = bank.frame(ids, self._attack_t)
        self._attacker_body_pos[:], self._striker_pos[:] = self._attacker_world(all_ids)

        # target on the defender and its forearms
        rb_pos, rb_rot = self.simulator._rigid_body_pos, self.simulator._rigid_body_rot
        target_body = self._clip_target_body[ids]
        self._target_pos[:] = rb_pos[all_ids, target_body] + my_quat_rotate(rb_rot[all_ids, target_body], bank.target_local[ids])
        elbow = rb_pos[:, self._elbow_ids]
        hand = elbow + my_quat_rotate(rb_rot[:, self._elbow_ids].reshape(-1, 4), self._hand_offset.repeat(N, 1)).view(N, 2, 3)

        onset, recovered = bank.onset_frame[ids], bank.recovered_frame[ids]
        check = (self._attack_status == PENDING) & (frame >= onset) & (frame_prev <= recovered)
        path_point = closest_point_on_segment(self._target_pos, striker_prev, self._striker_pos)  # nearest the target
        d_target = (path_point - self._target_pos).norm(dim=-1)
        d_block, block_arm, block_point = sweep_segment_closest(striker_prev, self._striker_pos, elbow, hand)
        blocked = check & (d_block < self._block_radius)
        hit = check & ~blocked & (d_target < self._hit_radius)
        closer = check & (d_target < self._closest_dist)
        self._closest_dist[closer], self._closest_point[closer] = d_target[closer], path_point[closer]
        self._closest_target[closer] = self._target_pos[closer]
        evaded = check & ~blocked & ~hit & (frame >= recovered) & (frame_prev < recovered)
        event = torch.zeros_like(self._attack_status)
        event[hit], event[blocked], event[evaded] = HIT, BLOCKED, EVADED
        self._defense_event[:] = event
        self._record_outcomes(event, block_point, block_arm, elbow, hand, path_point)
        self._attack_status[:] = torch.where(event > 0, event, self._attack_status)
        self._strike_active[:] = check.float()
        self._striker_target_dist[:] = (self._striker_pos - self._target_pos).norm(dim=-1)
        self._attack_threat[:] = ((frame >= onset - self._threat_lead_frames) & (frame <= recovered)).float()
        self._log_defense_outcomes(bank.label[ids], event)

        ended = (self._attack_t >= bank.duration(ids)).nonzero(as_tuple=False).flatten()
        self._start_attacks(ended, blend=True)
        if self._attacker_actor and len(ended) == 0:
            self._update_attacker_actors()
        self._kp_buffer[:, 1:] = self._kp_buffer[:, :-1].clone()
        self._kp_buffer[:, 0] = self._local_keypoints(all_ids)

    def _record_outcomes(self, event, block_point, block_arm, elbow, hand, path_point):
        """ Where the attacks resolved this step ended, for the viewer: blocked at the closest point on the forearm, hit
        at the point of the striker's path nearest the target, evaded at the closest approach of the whole attack (with
        the target where it was then). """
        self._outcome_age += self.dt
        resolved = event > 0
        if not resolved.any():
            return
        point = torch.where((event == BLOCKED)[:, None], block_point, torch.where((event == HIT)[:, None], path_point, self._closest_point))
        n = torch.arange(self.num_envs, device=self.device)
        self._outcome_status[resolved] = event[resolved]
        self._outcome_point[resolved] = point[resolved]
        self._outcome_target[resolved] = torch.where((event == EVADED)[:, None], self._closest_target, self._target_pos)[resolved]
        self._outcome_arm[resolved] = torch.stack([elbow[n, block_arm], hand[n, block_arm]], dim=1)[resolved]
        self._outcome_age[resolved] = 0.0
        if self.viewer is not None and resolved[0]:
            bank, i = self._attack_bank, int(self._attack_ids[0])
            miss = float((self._outcome_point[0] - self._outcome_target[0]).norm()) * 100
            how = {HIT: f'HIT on the {bank.target_link[i]}',
                   BLOCKED: f"BLOCKED by the {('left', 'right')[int(block_arm[0])]} forearm, {miss:.0f} cm from its target",
                   EVADED: f'EVADED, closest approach {miss:.0f} cm from its target'}[int(event[0])]
            logger.info(f'{bank.attack_names[i]}: {how}')

    def _local_keypoints(self, env_ids):
        """ Attacker keypoints relative to the defender root, in its heading frame [n, K, 3]. """
        root = self.simulator.robot_root_states[env_ids]
        kp = self._attacker_body_pos[env_ids][:, self._attack_bank.keypoint_ids] - root[:, None, :3]
        n, K = kp.shape[:2]
        heading_inv = calc_heading_quat_inv(root[:, 3:7].clone(), w_last=True)
        return my_quat_rotate(heading_inv[:, None].expand(n, K, 4).reshape(-1, 4), kp.reshape(-1, 3)).view(n, K, 3)

    def _log_defense_outcomes(self, labels, event):
        """ Hit / block / evade rates over recent attacks (counts decay by outcome_log_decay per step), overall and per class. """
        self._outcome_counts *= self._outcome_log_decay
        resolved = event > 0
        self._outcome_counts.index_put_((labels[resolved], event[resolved] - 1), torch.ones_like(labels[resolved], dtype=torch.float), accumulate=True)
        counts = self._outcome_counts
        total = counts.sum().clamp(min=1e-6)
        for outcome, name in enumerate(('hit', 'block', 'evade')):
            self.log_dict[f'defense_{name}_rate'] = counts[:, outcome].sum() / total
        per_class = counts[:, 0] / counts.sum(dim=1).clamp(min=1e-6)
        for c, name in enumerate(ATTACKS):
            self.log_dict[f'defense_hit_rate_{name}'] = per_class[c]

    ###########################################################################
    #### Env hooks
    ###########################################################################
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        self._step_attacks()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states)
        self._start_attacks(env_ids, blend=False)
        self._defense_event[env_ids] = 0
        self._outcome_age[env_ids] = float('inf')
        self._kp_buffer[env_ids] = self._local_keypoints(env_ids)[:, None]

    def _update_reset_buf(self):
        super()._update_reset_buf()
        if self.config.defense.terminate_on_hit:
            self.reset_buf_terminate_by["hit"] = self._defense_event == HIT
            self.reset_buf |= self.reset_buf_terminate_by["hit"]

    ###########################################################################
    #### Observations
    ###########################################################################
    def _post_config_observation_callback(self):
        super()._post_config_observation_callback()
        if self._blind_actor == 'none':
            return
        kp = self.obs_buf_dict['actor_obs'][:, self._actor_kp_slice]  # a view: written in place
        if self._blind_actor == 'freeze':  # hold what the actor saw while the attacker stood in guard before the strike
            bank, ids = self._attack_bank, self._attack_ids
            guard = bank.frame(ids, self._attack_t) < bank.onset_frame[ids] - self._threat_lead_frames
            self._blind_kp[guard] = kp[guard]
        else:
            self._blind_kp[:] = 0.0
        kp[:] = self._blind_kp

    def _get_obs_attacker_keypoints(self):
        return self._kp_buffer[:, ::self._kp_hist_stride].reshape(self.num_envs, -1)

    def _get_obs_attack_privileged(self):
        bank = self._attack_bank
        ids, t = self._attack_ids, self._attack_t
        root = self.simulator.robot_root_states
        time_to_impact = ((bank.impact_frame[ids] - bank.frame(ids, t)) * bank.dt).clamp(-self._tti_clip, self._tti_clip)
        striker_to_target = my_quat_rotate(calc_heading_quat_inv(root[:, 3:7].clone(), w_last=True), self._target_pos - self._striker_pos)
        return torch.cat([F.one_hot(bank.label[ids], bank.num_classes).float(),
                          F.one_hot(bank.phase_at(ids, t), NUM_PHASES).float(),
                          time_to_impact[:, None],
                          striker_to_target,
                          (self._attack_status != PENDING).float()[:, None]], dim=-1)

    ###########################################################################
    #### Rewards
    ###########################################################################
    def _reward_defense_hit(self):
        return (self._defense_event == HIT).float()

    def _reward_defense_block(self):
        return (self._defense_event == BLOCKED).float()

    def _reward_defense_evade(self):
        return (self._defense_event == EVADED).float()

    def _reward_defense_striker_proximity(self):
        # while a strike is live: closeness of the striking point to its target (use a negative scale)
        return torch.exp(-torch.square(self._striker_target_dist / self._proximity_sigma)) * self._strike_active

    ###########################################################################
    #### Visualization
    ###########################################################################
    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        if not self._defense_debug_draw:
            return
        """ Env 0: the striker (orange while pending, then the outcome colour: red hit, blue blocked, green evaded) and
        its trail, the target on the defender (green), where the attack is aimed (yellow, until it resolves), and for
        outcome_marker_time after each outcome where it ended (outcome colour) with a white line to the target. """
        env_id = 0
        colors = {PENDING: (1.0, 0.5, 0.0), HIT: (1.0, 0.0, 0.0), BLOCKED: (0.1, 0.3, 1.0), EVADED: (0.0, 0.8, 0.2)}
        color = colors[int(self._attack_status[env_id])]
        draw_line = lambda a, b, c: self.simulator.draw_line(Point(a), Point(b), Point(c), env_id)
        if not self._attacker_actor:
            bodies = self._attacker_body_pos[env_id].cpu()
            for child, parent in enumerate(self._attack_bank.body_parents):
                if parent >= 0:
                    draw_line(bodies[parent], bodies[child], color)
        self._striker_trail = (self._striker_trail + [self._striker_pos[env_id].cpu()])[-15:]
        for a, b in zip(self._striker_trail[:-1], self._striker_trail[1:]):
            draw_line(a, b, color)
        self.simulator.draw_sphere(self._striker_pos[env_id], 0.04, color, env_id)
        self.simulator.draw_sphere(self._target_pos[env_id], 0.03, (0.0, 1.0, 0.0), env_id)
        if self._attack_status[env_id] == PENDING and self._attack_threat[env_id] > 0:
            self.simulator.draw_sphere(self._aim_pos[env_id], 0.05, (1.0, 0.85, 0.0), env_id)
        if self._outcome_age[env_id] < self._outcome_hold:
            outcome = colors[int(self._outcome_status[env_id])]
            point, target = self._outcome_point[env_id].cpu(), self._outcome_target[env_id].cpu()
            for radius in (0.025, 0.05, 0.07):
                self.simulator.draw_sphere(point, radius, outcome, env_id)
            self.simulator.draw_sphere(target, 0.03, (0.0, 1.0, 0.0), env_id)
            draw_line(point, target, (1.0, 1.0, 1.0))
            if self._outcome_status[env_id] == BLOCKED:  # the blocking forearm, drawn thick
                elbow, hand = self._outcome_arm[env_id].cpu()
                for dz in (-0.006, 0.0, 0.006):
                    offset = torch.tensor([0.0, 0.0, dz])
                    draw_line(elbow + offset, hand + offset, outcome)
        if self._blind_actor == 'freeze':  # the attacker as the blinded actor sees it (newest frame), relative to the defender
            names = list(self.config.defense.keypoints)
            K = len(names)
            root = self.simulator.robot_root_states[env_id:env_id + 1]
            heading = quat_conjugate(calc_heading_quat_inv(root[:, 3:7].clone(), w_last=True), w_last=True)
            local = self._blind_kp[env_id, :3 * K].view(K, 3) / self.config.obs.obs_scales.attacker_keypoints
            seen = (root[:, :3] + my_quat_rotate(heading.expand(K, 4), local)).cpu()
            for i in range(K):
                self.simulator.draw_sphere(seen[i], 0.03, (0.7, 0.3, 1.0), env_id)
            for a, b in BLIND_SKELETON:
                if a in names and b in names:
                    draw_line(seen[names.index(a)], seen[names.index(b)], (0.7, 0.3, 1.0))
