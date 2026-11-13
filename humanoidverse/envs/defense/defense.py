import torch
import torch.nn.functional as F
from loguru import logger

from humanoidverse.envs.motion_tracking.motion_tracking import LeggedRobotMotionTracking
from humanoidverse.envs.defense.attack_bank import ATTACKS, NUM_PHASES, AttackBank, closest_point_on_segment, sweep_segment_closest
from humanoidverse.envs.defense.counter_bank import CounterBank, SHIN, STRIKERS
from humanoidverse.envs.env_utils.visualization import Point
from isaac_utils.rotations import calc_heading_quat_inv, my_quat_rotate, quat_conjugate, quat_mul

PENDING, HIT, BLOCKED, EVADED = 0, 1, 2, 3
BLIND_SKELETON = [('pelvis', 'head_link'), ('left_elbow_link', 'left_hand_link'), ('right_elbow_link', 'right_hand_link'),
                  ('pelvis', 'left_knee_link'), ('left_knee_link', 'left_ankle_roll_link'),
                  ('pelvis', 'right_knee_link'), ('right_knee_link', 'right_ankle_roll_link')]  # keypoint pairs, for the viewer


def _rotation_between(a, b):
    """ Shortest rotations [n, 4] (xyzw) that turn the directions a [n, 3] onto b [n, 3]. """
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    q = torch.cat([torch.cross(a, b, dim=-1), 1.0 + (a * b).sum(dim=-1, keepdim=True)], dim=-1)
    return F.normalize(q, dim=-1)


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
        self._hook_counter_reference()

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
        self._init_counterattacks()

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
        self._attacker_body_pos = torch.zeros(N, B, 3, device=self.device)  # as seen and scored (feet on the ground)
        self._striker_pos = torch.zeros(N, 3, device=self.device)
        self._attacker_nominal = torch.zeros(N, B, 3, device=self.device)  # the clip as placed, on flat ground at z = 0
        self._striker_nominal = torch.zeros(N, 3, device=self.device)
        self._attacker_delta = torch.zeros(N, B, 3, device=self.device)  # _attacker_body_pos - _attacker_nominal
        # the attacker's feet (attacker_feet): planted on the terrain under them while the clip has them in contact
        af = dc.get('attacker_feet', None) or {}
        self._feet_planted = bool(af.get('planted', False))
        self._feet_step_trigger, self._feet_step_time = float(af.get('step_trigger', 0.3)), float(af.get('step_time', 0.3))
        self._feet_step_lift, self._feet_swing_decay = float(af.get('step_lift', 0.05)), float(af.get('swing_decay', 0.08))
        self._attacker_crouch_max, self._attacker_stretch_max = float(af.get('crouch_max', 0.2)), float(af.get('stretch_max', 0.05))
        self._foot_planted_xy = torch.zeros(N, 2, 2, device=self.device)  # [env, left/right, xy]
        self._foot_offset = torch.zeros(N, 2, 2, device=self.device)  # foot xy minus the clip's
        self._foot_step_t = torch.zeros(N, 2, device=self.device)  # > 0 while a foot is being re-planted (time left)
        self._foot_step_from = torch.zeros(N, 2, 2, device=self.device)
        self._foot_lift = torch.zeros(N, 2, device=self.device)
        self._attacker_leg_rot = torch.zeros(N, 2, 2, 4, device=self.device)  # [env, side, thigh/shin] extra rotation for the viewer
        self._attacker_leg_rot[..., 3] = 1.0
        self._target_pos = torch.zeros(N, 3, device=self.device)
        self._striker_target_dist = torch.ones(N, device=self.device)
        self._strike_active = torch.zeros(N, device=self.device)
        self._attack_threat = torch.zeros(N, device=self.device)
        self._kp_buffer = torch.zeros(N, (self._kp_hist_len - 1) * self._kp_hist_stride + 1, K, 3, device=self.device)
        self._kp_refill = torch.zeros(N, dtype=torch.bool, device=self.device)
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
            self._actor_thigh_links = [[i for i, n in enumerate(names) if n.startswith(p) and 'hip' in n] for p in ('left_', 'right_')]
            self._actor_shin_links = [[i for i, n in enumerate(names) if n.startswith(p) and 'knee' in n] for p in ('left_', 'right_')]
            self._attacker_body_rot = torch.zeros(N, len(self._actor_body_ids), 4, device=self.device)
            self._attacker_body_rot[..., 3] = 1.0
            self._attack_blend_rot = self._attacker_body_rot.clone()

        # layout of attack_privileged, for the actor's belief head (agents/modules/attack_modules.py)
        C = bank.num_classes
        self.attack_obs_info = dict(num_frames=self._kp_hist_len, num_classes=C, num_phases=NUM_PHASES,
                                    label=(0, C), phase=(C, C + NUM_PHASES), time_to_impact=C + NUM_PHASES)

    ###########################################################################
    #### Counterattacks
    ###########################################################################
    def _init_counterattacks(self):
        """ Counterattacks (env.config.defense.counter): after an attack that was blocked or evaded, the defender's
        reference switches from the held guard to a kung-fu clip (CounterBank) and back to the guard when the clip ends.
        Meanwhile the attacker holds its guard where it stands. The clip is aimed at it: turned about its start and
        moved along the strike direction so that the strike ends aim_depth short of the attacker's trunk axis, i.e. on
        its surface, with the two bodies apart (the yaw and the shift ease in while the clip blends from the guard into
        the motion and ease out at its end; limits max_aim_yaw_deg and max_aim_shift). The counter lands when the
        defender's striking hand or foot itself passes within counter.hit_radius of the attacker's trunk axis (pelvis to
        head) around the strike time while it is extended (min_extension of the clip's reach from the defender's root). """
        cc = self.config.defense.get('counter', None)
        self._counter_enabled = bool(cc is not None and cc.get('enable', False))
        N = self.num_envs
        self._counter_active = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._counter_event = torch.zeros(N, device=self.device)  # 1 on the step a counter lands
        self._counter_window = torch.zeros(N, device=self.device)  # 1 while a strike can land
        self._counter_dist = torch.ones(N, device=self.device)  # striker to target
        self._counter_depth = torch.zeros(N, device=self.device)  # aim depth of the clip in progress
        self._counter_sigma_now = torch.ones(N, device=self.device)  # proximity sigma of the clip in progress
        self._counter_gain = torch.ones(N, device=self.device)  # reward scale of the clip in progress (leg_reward_scale)
        if not self._counter_enabled:
            return
        names_all = list(cc.clips_all)
        self._counter_bank = bank = CounterBank(cc.clip_dir, names_all, self.config.robot.motion, self.device)
        self._counter_choices = torch.tensor([names_all.index(n) for n in cc.clips], dtype=torch.long, device=self.device)
        self._counter_prob, self._counter_hit_radius = float(cc.prob), float(cc.hit_radius)
        self._counter_hit_window, self._counter_sigma = float(cc.hit_window), float(cc.proximity_sigma)
        self._counter_max_yaw = torch.pi / 180 * float(cc.max_aim_yaw_deg)
        self._counter_max_shift, self._counter_aim_depth = float(cc.max_aim_shift), float(cc.aim_depth)
        self._counter_head_top = float(cc.get('head_top', 0.1))
        self._counter_min_extension = float(cc.get('min_extension', 0.0))
        self._guard_far_threshold = cc.get('guard_far_threshold', None)
        # when a counter ends the guard reference moves to where the robot stands (at most `reanchor` per counter and
        # `reanchor_total` per episode): the robot steps in for the strike and need not step back; the attacker's next
        # clip is placed relative to the robot anyway
        self._counter_reanchor = float(cc.get('reanchor', 0.0)) if self._placement_jitter > 0 else 0.0
        self._counter_reanchor_total = float(cc.get('reanchor_total', 1.0))
        self._counter_reanchor_attacks = bool(cc.get('reanchor_attacks', False))
        self._reanchor_sum = torch.zeros(N, 2, device=self.device)
        bank.hold_shift = self._counter_reanchor > 0  # the reference stays stepped in until the clip ends
        # per clip: a fixed drift cutoff instead of the env's (e.g. the punches keep 0.7 m while the kicks follow the curriculum)
        fixed = cc.get('clip_far_threshold', None) or {}
        self._counter_clip_far = torch.tensor([float(fixed.get(n, float('inf'))) for n in names_all], device=self.device)
        self._counter_clip = torch.zeros(N, dtype=torch.long, device=self.device)
        self._counter_t0 = torch.zeros(N, device=self.device)  # motion time at which the clip started
        self._counter_time = torch.zeros(N, device=self.device)  # time in the clip
        self._counter_landed = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._counter_yaw = torch.zeros(N, device=self.device)  # aim: the clip is turned by this about its start
        self._counter_aim_shift = torch.zeros(N, 2, device=self.device)  # and moved by this (world xy)
        self._counter_target_body = torch.zeros(N, dtype=torch.long, device=self.device)  # attack-bank body index
        self._counter_striker_prev = torch.zeros(N, 3, device=self.device)
        self._counter_land_age = torch.full((N,), 1e3, device=self.device)  # [s] since the last landed counter (viewer)
        self._counter_counts = torch.zeros(bank.num_clips, 2, device=self.device)  # landed, missed (decayed)
        self._counter_totals = torch.zeros(bank.num_clips, 3, device=self.device)  # started, landed, finished (for evals)
        body = self.simulator._body_list.index
        self._ankle_ids = torch.tensor([body('left_ankle_roll_link'), body('right_ankle_roll_link')], dtype=torch.long, device=self.device)
        self._knee_ids = torch.tensor([body('left_knee_link'), body('right_knee_link')], dtype=torch.long, device=self.device)
        self._ankle_pitch_ids = torch.tensor([body('left_ankle_pitch_link'), body('right_ankle_pitch_link')], dtype=torch.long, device=self.device)
        # per clip: how far short of the target's axis the strike is aimed and within which distance of it it lands
        # (a strike at the attacker's leg, the calf kick, has its own: a leg is thinner than the trunk)
        leg = bank.target_leg
        self._counter_clip_depth = torch.where(leg, torch.full_like(bank.duration, float(cc.get('leg_aim_depth', 0.06))),
                                               torch.full_like(bank.duration, self._counter_aim_depth))
        self._counter_clip_sigma = torch.where(leg, torch.full_like(bank.duration, float(cc.get('leg_proximity_sigma', 0.3))),
                                               torch.full_like(bank.duration, self._counter_sigma))
        self._counter_clip_gain = torch.where(leg, torch.full_like(bank.duration, float(cc.get('leg_reward_scale', 1.0))), torch.ones_like(bank.duration))
        self._counter_clip_radius = torch.where(leg, torch.full_like(bank.duration, float(cc.get('leg_hit_radius', 0.12))),
                                                torch.full_like(bank.duration, self._counter_hit_radius))
        # target on the attacker per clip: the body whose height in guard is nearest the strike height
        ab = self._attack_bank
        candidates = [ab.body_names.index(n) for n in ('pelvis', 'torso_link', 'head_link') if n in ab.body_names]
        heights = ab.body_pos[:, 0, candidates, 2].mean(dim=0)  # [C]
        nearest = (heights[None] - bank.impact_point[:, 2:3]).abs().argmin(dim=1)
        self._counter_clip_target = torch.tensor(candidates, dtype=torch.long, device=self.device)[nearest]
        self._trunk_ids = (ab.body_names.index('pelvis'), ab.body_names.index('head_link'))
        for k, name in enumerate(names_all):
            logger.info(f"counter {name}: {bank.duration[k]:.2f} s, strike by {STRIKERS[int(bank.striker[k])]} at "
                        f"{bank.impact_time[k]:.2f} s, {bank.impact_point[k, 2]:.2f} m high, "
                        f"target: attacker {ab.body_names[int(self._counter_clip_target[k])]}")


    def _hook_counter_reference(self):
        """ The reference of robots that are countering comes from their clip (called once the motion library exists). """
        if not self._counter_enabled:
            return
        bank = self._counter_bank
        guard_state = self._motion_lib.get_motion_state
        def get_motion_state(motion_ids, motion_times, offset=None):
            res = guard_state(motion_ids, motion_times, offset=offset)
            active = self._counter_active[motion_ids]
            if active.any():
                env = motion_ids[active]
                bank.write_states(res, active, self._counter_clip[env], motion_times[active] - self._counter_t0[env],
                                  None if offset is None else offset[active], self._counter_yaw[env], self._counter_aim_shift[env])
            return res
        self._motion_lib.get_motion_state = get_motion_state

    def _motion_time_now(self):
        return (self.episode_length_buf + 1) * self.dt + self.motion_start_times  # as kick_motion_res

    def _reanchor(self, ids, shift=None):
        """ Move the guard reference of these robots to where they stand (counter.reanchor); shift [n, 2]: the aim
        shift their reference still carries (at the end of a counter). """
        if self._counter_reanchor <= 0 or len(ids) == 0:
            return
        move = self.simulator.robot_root_states[ids, :2] - self.ref_body_pos_extend[ids, 0, :2]
        if shift is not None:
            move = move + shift
        move = move * (self._counter_reanchor / move.norm(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)
        total = self._reanchor_sum[ids] + move
        total = total * (self._counter_reanchor_total / total.norm(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)
        self._placement_offset[ids] += total - self._reanchor_sum[ids]
        self._reanchor_sum[ids] = total

    def _start_counters(self, env_ids):
        """ Start a counter clip for these robots (their attacker has just started its next clip and holds its guard). """
        n = len(env_ids)
        if n == 0:
            return
        bank = self._counter_bank
        clip = self._counter_choices[torch.randint(len(self._counter_choices), (n,), device=self.device)]
        self._counter_active[env_ids] = True
        self._counter_clip[env_ids] = clip
        self._counter_t0[env_ids] = self._motion_time_now()[env_ids]
        self._counter_time[env_ids] = 0.0
        self._counter_landed[env_ids] = False
        self._counter_totals[:, 0] += torch.bincount(clip, minlength=bank.num_clips)
        # aim the clip at the attacker: its strike goes from the clip's start root towards `v`; turn it to point at the
        # attacker body nearest the strike height, and move it along that direction so that it reaches aim_depth short of
        # that body's vertical
        start = bank.start_xy[clip] + self._ref_offset()[env_ids, :2]
        v = bank.impact_point[clip, :2] - bank.start_xy[clip]
        target_body = self._counter_clip_target[clip]
        self._counter_target_body[env_ids] = target_body
        target_xy = self._attacker_body_pos[env_ids, target_body, :2]
        if bank.target_leg[clip].any():  # a strike at the leg: the attacker's shin nearer the defender
            ab = self._attack_bank
            shins = 0.5 * (self._attacker_body_pos[env_ids][:, ab.knee_ids, :2] + self._attacker_body_pos[env_ids][:, ab.ankle_pitch_ids, :2])
            near = (shins - start[:, None]).norm(dim=-1).argmin(dim=1)
            target_xy = torch.where(bank.target_leg[clip][:, None], shins[torch.arange(n, device=self.device), near], target_xy)
        d = target_xy - start
        yaw = torch.atan2(d[:, 1], d[:, 0]) - torch.atan2(v[:, 1], v[:, 0])
        yaw = (torch.remainder(yaw + torch.pi, 2 * torch.pi) - torch.pi).clamp(-self._counter_max_yaw, self._counter_max_yaw)
        angle = torch.atan2(v[:, 1], v[:, 0]) + yaw
        reach = (d.norm(dim=-1) - self._counter_clip_depth[clip] - v.norm(dim=-1)).clamp(-self._counter_max_shift, self._counter_max_shift)
        self._counter_yaw[env_ids] = yaw
        self._counter_aim_shift[env_ids] = reach[:, None] * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        self._counter_striker_prev[env_ids] = self._defender_strikers()[env_ids, bank.striker[clip]]

    def _attacker_trunk(self):
        """ The attacker's trunk axis: pelvis [N, 3] and the top of the head [N, 3]. """
        head = self._attacker_body_pos[:, self._trunk_ids[1]].clone()
        head[:, 2] += self._counter_head_top
        return self._attacker_body_pos[:, self._trunk_ids[0]], head

    def _counter_target_segments(self):
        """ What each robot's counter clip is aimed at, as segments a, b [N, 4, 3]: the attacker's trunk axis (pelvis to
        the top of the head; repeated), or for a strike at the leg its thighs and shins. """
        ab, pos = self._attack_bank, self._attacker_body_pos
        pelvis, head = self._attacker_trunk()
        a = torch.cat([pos[:, ab.hip_ids], pos[:, ab.knee_ids]], dim=1)
        b = torch.cat([pos[:, ab.knee_ids], pos[:, ab.ankle_pitch_ids]], dim=1)
        leg = self._counter_bank.target_leg[self._counter_clip][:, None, None]
        return torch.where(leg, a, pelvis[:, None].expand_as(a)), torch.where(leg, b, head[:, None].expand_as(b))

    def _defender_strikers(self):
        """ The defender's hands, ankles and shins [N, 6, 3] in the order of counter_bank.STRIKERS. """
        rb_pos, rb_rot = self.simulator._rigid_body_pos, self.simulator._rigid_body_rot
        N = self.num_envs
        elbow = rb_pos[:, self._elbow_ids]
        hand = elbow + my_quat_rotate(rb_rot[:, self._elbow_ids].reshape(-1, 4), self._hand_offset.repeat(N, 1)).view(N, 2, 3)
        knee, ankle = rb_pos[:, self._knee_ids], rb_pos[:, self._ankle_pitch_ids]
        return torch.cat([hand, rb_pos[:, self._ankle_ids], knee + SHIN * (ankle - knee)], dim=1)

    def _step_counters(self):
        """ Advance the counter clips: score strikes, end finished clips (the attacker then starts its next attack). """
        self._counter_event[:] = 0.0
        self._counter_window[:] = 0.0
        if not self._counter_enabled:
            return
        bank, active = self._counter_bank, self._counter_active
        self._counter_time[:] = torch.where(active, self._motion_time_now() - self._counter_t0, torch.zeros_like(self._counter_time))
        clip = self._counter_clip
        all_ids = torch.arange(self.num_envs, device=self.device)
        striker = self._defender_strikers()[all_ids, bank.striker[clip]]
        seg_a, seg_b = self._counter_target_segments()
        window = active & ((self._counter_time - bank.impact_time[clip]).abs() <= self._counter_hit_window)
        d_sweep = sweep_segment_closest(self._counter_striker_prev, striker, seg_a, seg_b)[0]
        # the strike has to be thrown: the striker at least min_extension of the clip's reach away from the defender's own root
        extended = (striker[:, :2] - self.simulator.robot_root_states[:, :2]).norm(dim=-1) >= self._counter_min_extension * bank.extension[clip]
        landed = window & ~self._counter_landed & (d_sweep < self._counter_clip_radius[clip]) & extended
        self._counter_landed |= landed
        self._counter_totals[:, 1] += torch.bincount(clip[landed], minlength=bank.num_clips)
        self._counter_event[:] = landed.float()
        self._counter_gain[:] = self._counter_clip_gain[clip]
        self._counter_land_age = torch.where(landed, torch.zeros_like(self._counter_time), self._counter_land_age + self.dt)
        self._counter_window[:] = window.float()
        self._counter_dist[:] = (striker[:, None] - closest_point_on_segment(striker[:, None], seg_a, seg_b)).norm(dim=-1).min(dim=1)[0]
        self._counter_depth[:] = self._counter_clip_depth[clip]
        self._counter_sigma_now[:] = self._counter_clip_sigma[clip]
        self._counter_striker_prev[:] = striker

        ended = active & (self._counter_time >= bank.duration[clip])
        self._counter_counts *= self._outcome_log_decay
        if ended.any():
            ids = ended.nonzero(as_tuple=False).flatten()
            self._counter_counts.index_put_((clip[ids], (~self._counter_landed[ids]).long()),
                                            torch.ones(len(ids), device=self.device), accumulate=True)
            self._counter_totals[:, 2] += torch.bincount(clip[ids], minlength=bank.num_clips)
            self._counter_active[ids] = False
            self._reanchor(ids, self._counter_aim_shift[ids])
            self._start_attacks(ids, blend=True)  # the attacker's next attack
        counts = self._counter_counts
        self.log_dict['defense_counter_land_rate'] = counts[:, 0].sum() / counts.sum().clamp(min=1e-6)
        self.log_dict['defense_counter_active'] = self._counter_active.float().mean()
        for k in self._counter_choices.tolist():
            self.log_dict[f'defense_counter_land_rate_{bank.names[k]}'] = counts[k, 0] / counts[k].sum().clamp(min=1e-6)

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
        """ The attacker's bodies [n, B, 3] and striking point [n, 3] at the current clip time as the clip places them,
        on flat ground at z = 0 (_ground_attackers puts them on the terrain). """
        bank = self._attack_bank
        ids, t = self._attack_ids[env_ids], self._attack_t[env_ids]
        anchor, yaw, shift = self._attack_anchor[env_ids], self._attack_anchor_yaw[env_ids], self._attack_shift[env_ids]
        anchor = torch.cat([anchor[:, :2], torch.zeros_like(anchor[:, :1])], dim=1)  # heights are added by _ground_attackers
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
        old_bodies = self._attacker_nominal[env_ids].clone()
        self._attack_ids[env_ids] = self._attack_bank.sample(n)
        self._attack_t[env_ids] = 0.0
        root = self.simulator.robot_root_states[env_ids]
        self._attack_anchor[env_ids] = torch.cat([root[:, :2], self._defender_ground_height()[env_ids, None]], dim=1)
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
        self._attacker_nominal[env_ids] = bodies
        self._striker_nominal[env_ids] = striker
        if not blend:  # a new episode: the feet stand where the clip puts them
            self._foot_planted_xy[env_ids] = bodies[:, self._attack_bank.foot_ids, :2]
            self._foot_offset[env_ids] = 0.0
            self._foot_step_t[env_ids] = 0.0
            self._foot_lift[env_ids] = 0.0
        self._ground_attackers()
        if self._attacker_actor:
            self._attack_blend_rot[env_ids] = 0.0
            self._attack_blend_rot[env_ids, :, 3] = 1.0
            if blend:  # previous clip's rotation times the inverse of the new clip's at its start
                delta = quat_mul(self._attacker_body_rot[env_ids], quat_conjugate(self._attacker_rots(env_ids), w_last=True), w_last=True)
                self._attack_blend_rot[env_ids] = torch.where(delta[..., 3:] < 0, -delta, delta)
            self._update_attacker_actors()

    def _defender_ground_height(self):
        """ Height [N] of the ground the defender's guard reference stands on: the env origin on flat ground; with
        terrain_aware.ground_relative_ref the reference's support height (terrain under its feet). """
        return self.ref_support_height if self._ground_relative_ref else self.env_origins[:, 2]

    def _advance_attacker_feet(self):
        """ Once per control step: the attacker's feet stay planted while the clip has them in contact, follow the clip
        when it lifts them (the calf kick), and are re-planted with a step (lifted, over step_time) when the body has
        moved more than step_trigger away from them, e.g. after the defender has moved and the next clip is placed
        relative to it. Without this the feet slide with every change of the clip's placement. """
        if not self._feet_planted:
            return
        bank = self._attack_bank
        nominal = self._attacker_nominal[:, bank.foot_ids, :2]  # [N, 2, 2]
        contact = bank.contact_at(self._attack_ids, self._attack_t)  # [N, 2]
        stepping = self._foot_step_t > 0
        # a planted foot too far from where the clip has it starts a step (one foot at a time, the left first)
        offset = self._foot_planted_xy - nominal
        want = contact & ~stepping & (offset.norm(dim=-1) > self._feet_step_trigger)
        want[:, 1] &= ~(want[:, 0] | stepping[:, 0])
        want[:, 0] &= ~stepping[:, 1]
        self._foot_step_from[want] = offset[want]
        self._foot_step_t[want] = self._feet_step_time
        stepping = self._foot_step_t > 0
        # feet being re-planted
        self._foot_step_t[stepping] -= self.dt
        frac = (1.0 - self._foot_step_t / self._feet_step_time).clamp(0.0, 1.0)
        ease = frac * frac * (3.0 - 2.0 * frac)
        step_offset = self._foot_step_from * (1.0 - ease)[..., None]
        self._foot_lift[:] = torch.where(stepping, self._feet_step_lift * torch.sin(torch.pi * frac), torch.zeros_like(frac))
        # feet the clip has in the air: the offset decays, so the foot goes where the clip aims it
        swing = ~contact & ~stepping
        decayed = self._foot_offset * float(torch.exp(torch.tensor(-self.dt / self._feet_swing_decay)))
        self._foot_offset[:] = torch.where(stepping[..., None], step_offset, torch.where(swing[..., None], decayed, offset))
        moving = stepping | swing
        self._foot_planted_xy[moving] = (nominal + self._foot_offset)[moving]

    def _ground_attackers(self):
        """ Put the attacker on the terrain (_attacker_body_pos, _striker_pos from the nominal clip placement): each foot
        on the ground under it, the pelvis and upper body at the height that keeps the strikes aimed at the defender,
        limited to what the legs can do from the guard (crouch_max down, stretch_max up from the attacker's own ground),
        the knees in between. With attacker_feet.planted the feet also keep their place (see _advance_attacker_feet) and
        the legs shear between the pelvis and the feet (the step-in of `standoff` becomes a lunge). On flat ground
        without planted feet this is the original placement. """
        bank, nominal = self._attack_bank, self._attacker_nominal
        N = self.num_envs
        delta = self._attacker_delta
        delta[:] = 0.0
        if self._feet_planted:  # planted feet stay put whatever the clip's placement has become (e.g. a new clip)
            held = bank.contact_at(self._attack_ids, self._attack_t) & (self._foot_step_t <= 0)
            self._foot_offset[:] = torch.where(held[..., None], self._foot_planted_xy - nominal[:, bank.foot_ids, :2], self._foot_offset)
        foot_offset = self._foot_offset if self._feet_planted else torch.zeros_like(self._foot_offset)
        foot_xy = nominal[:, bank.foot_ids, :2] + foot_offset
        h_def = self._defender_ground_height()
        if self._ground_relative_ref:
            h_foot = self._terrain_height_at(foot_xy, self._ref_height_lookup)  # [N, 2]
            h_own = h_foot.mean(dim=-1)
            h_body = h_own + (h_def - h_own).clamp(-self._attacker_crouch_max, self._attacker_stretch_max)
        else:
            h_foot, h_body = h_def[:, None].expand(N, 2), h_def
        share = bank.leg_share  # [2, B]
        leg = share.sum(dim=0)  # 1 ankle, 0.5 knee, 0 elsewhere
        delta[..., 2] = (1.0 - leg)[None] * h_body[:, None] + ((h_foot + self._foot_lift) @ share)
        delta[..., :2] = torch.einsum('nsx,sb->nbx', foot_offset, share)
        self._attacker_body_pos[:] = nominal + delta
        # striking point: fists move with the upper body, the shin with its leg
        striker = self._striker_nominal.clone()
        striker[:, 2] += h_body
        knee, ankle = self._attacker_body_pos[:, bank.knee_ids[1]], self._attacker_body_pos[:, bank.ankle_pitch_ids[1]]
        shin = bank.striker_is_shin[self._attack_ids]
        striker[shin] = (knee + 0.75 * (ankle - knee))[shin]
        self._striker_pos[:] = striker
        if self._attacker_actor:  # the viewer's thigh and shin links turn with the sheared legs
            for side in range(2):
                hip, knee_id, ankle_id = bank.hip_ids[side], bank.knee_ids[side], bank.ankle_pitch_ids[side]
                for part, (a, b) in enumerate(((hip, knee_id), (knee_id, ankle_id))):
                    self._attacker_leg_rot[:, side, part] = _rotation_between(nominal[:, b] - nominal[:, a],
                                                                               self._attacker_body_pos[:, b] - self._attacker_body_pos[:, a])

    def _attacker_rots(self, env_ids):
        """ World rotations [n, num_bodies, 4] (xyzw) of the attacker's robot bodies at the current clip time, unblended. """
        bank = self._attack_bank
        ids = self._attack_ids[env_ids]
        return bank.to_world_rot(bank.body_rots(ids, self._attack_t[env_ids])[:, self._actor_body_ids], ids, self._attack_anchor_yaw[env_ids])

    def _update_attacker_actors(self):
        """ Pose the attacker's visual link actors from its body positions and (blended) rotations. """
        rot = self._attacker_rots(slice(None))
        pos = self._attacker_nominal[:, self._actor_body_ids]
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
        pos = pos + self._attacker_delta[:, self._actor_body_ids]
        rot = rot.clone()
        for side in range(2):  # thigh links (hip_*) and the shin (knee link) follow the sheared legs
            for links, part in ((self._actor_thigh_links[side], 0), (self._actor_shin_links[side], 1)):
                for link in links:
                    rot[:, link] = quat_mul(self._attacker_leg_rot[:, side, part], rot[:, link], w_last=True)
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
        self._attack_t += self.dt * (~self._counter_active).float()  # the attacker holds its guard during a counter
        frame = bank.frame(ids, self._attack_t)
        self._attacker_nominal[:], self._striker_nominal[:] = self._attacker_world(all_ids)
        self._advance_attacker_feet()
        self._ground_attackers()

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
        defended = (self._attack_status[ended] == BLOCKED) | (self._attack_status[ended] == EVADED)
        self._start_attacks(ended, blend=True)
        if self._counter_enabled and self._counter_reanchor_attacks:  # the guard follows the robot after every attack too
            self._reanchor(ended)
        if self._counter_enabled and len(ended):  # counter some of the attacks that were blocked or evaded
            self._start_counters(ended[defended & (torch.rand(len(ended), device=self.device) < self._counter_prob)])
        if self._attacker_actor and len(ended) == 0:
            self._update_attacker_actors()
        self._kp_buffer[:, 1:] = self._kp_buffer[:, :-1].clone()
        self._kp_buffer[:, 0] = self._local_keypoints(all_ids)
        if self._kp_refill.any():  # first step after a reset: the reset filled the history before the new tile's
            refill = self._kp_refill.nonzero(as_tuple=False).flatten()  # ground height was known (ground_relative_ref)
            self._kp_buffer[refill] = self._kp_buffer[refill, :1]
            self._kp_refill[:] = False

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
        self._step_counters()
        self._step_attacks()

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids):  # before the reference of the new episode is read: back to the guard
            self._counter_active[env_ids] = False
            if self._counter_enabled:
                self._counter_time[env_ids] = 0.0
                self._reanchor_sum[env_ids] = 0.0
        super()._reset_tasks_callback(env_ids)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        super()._reset_robot_states_callback(env_ids, target_states)
        self._start_attacks(env_ids, blend=False)
        self._defense_event[env_ids] = 0
        self._outcome_age[env_ids] = float('inf')
        self._kp_buffer[env_ids] = self._local_keypoints(env_ids)[:, None]
        self._kp_refill[env_ids] = self._ground_relative_ref

    def _update_terrain_curriculum(self, env_ids):
        """ Terrain curriculum for defense. The guard reference is held past its end, so there is no motion to complete:
        robots that survive the whole episode (time-out) move one difficulty row up, robots whose episode ended (fall
        or drift from the guard) before half of it move one row down; solving the hardest row sends a robot to a
        random one of the terrain.curriculum_graduate_rows hardest rows (any row if unset). """
        if not self.init_done or self.common_step_counter == 0 or self._in_reset_all:
            return
        length = self.last_episode_length_buf[env_ids]
        move_up = length >= self.max_episode_length
        move_down = length < 0.5 * self.max_episode_length
        levels = self.terrain_levels[env_ids] + move_up.long() - move_down.long()
        graduate_rows = self.config.terrain.get('curriculum_graduate_rows', None) or self.max_terrain_level
        graduated = self.max_terrain_level - graduate_rows + torch.randint_like(levels, graduate_rows)
        levels = torch.where(levels >= self.max_terrain_level, graduated, torch.clip(levels, min=0))
        self.terrain_levels[env_ids] = levels
        self.env_origins[env_ids] = self.terrain_origins[levels, self.terrain_types[env_ids]]
        self.log_dict["terrain_level"] = self.terrain_levels.float().mean()
        for name, (start, end) in getattr(self.simulator.terrain, 'sub_terrain_cols', {}).items():
            of_type = (self.terrain_types >= start) & (self.terrain_types < end)
            if of_type.any():
                self.log_dict[f"terrain_level_{name}"] = self.terrain_levels[of_type].float().mean()

    def _update_reset_buf(self):
        super()._update_reset_buf()
        if self._counter_enabled and self._guard_far_threshold is not None and self.config.termination.terminate_when_motion_far:
            # the drift cutoff of the env (which may follow PBHC's curriculum) applies to the counter clips; the guard
            # keeps its own fixed cutoff
            dist = torch.norm(self.dif_global_body_pos, dim=-1).max(dim=-1)[0]
            far = torch.where(self._counter_active, dist > self._counter_clip_far[self._counter_clip], dist > self._guard_far_threshold)
            self.reset_buf_terminate_by["motion_far"] = self.reset_buf_terminate_by["motion_far"] | far
            self.reset_buf |= far
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

    def _get_obs_counter_command(self):
        """ Which counter clip the defender is to perform and how far into it: one-hot over counter.clips_all, the
        phase in the clip, and a flag; all 0 while it holds the guard. """
        K = len(self.config.defense.counter.clips_all)
        if not self._counter_enabled:
            return torch.zeros(self.num_envs, K + 2, device=self.device)
        active = self._counter_active.float()[:, None]
        phase = (self._counter_time / self._counter_bank.duration[self._counter_clip]).clamp(0.0, 1.0)[:, None]
        return torch.cat([F.one_hot(self._counter_clip, K).float() * active, phase * active, active], dim=-1)

    def _get_obs_counter_aim(self):
        """ How the counter clip is aimed at the attacker: the yaw [rad] it is turned by and the shift [m] (world xy,
        the frame of the guard reference) it is moved by; zeros while holding the guard. """
        if not self._counter_enabled:
            return torch.zeros(self.num_envs, 3, device=self.device)
        active = self._counter_active.float()[:, None]
        return torch.cat([self._counter_yaw[:, None], self._counter_aim_shift], dim=-1) * active

    ###########################################################################
    #### Rewards
    ###########################################################################
    def _reward_defense_counter_hit(self):
        return self._counter_event * self._counter_gain

    def _reward_defense_counter_proximity(self):
        return self._counter_gain * self._counter_window * torch.exp(-((self._counter_dist - self._counter_depth).clamp(min=0.0) / self._counter_sigma_now) ** 2) if self._counter_enabled \
            else torch.zeros(self.num_envs, device=self.device)

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
        if self._counter_enabled and self._counter_land_age[env_id] < self._outcome_hold:  # a landed counter: rings on the
            fist = self._defender_strikers()[env_id, self._counter_bank.striker[self._counter_clip[env_id]]].cpu()  # defender's fist or foot
            for radius in (0.03, 0.06, 0.09):
                self.simulator.draw_sphere(fist, radius, (1.0, 1.0, 1.0), env_id)
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
