import torch
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    slerp,
    my_quat_rotate,
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    get_euler_xyz_in_tensor,
    calc_yaw_heading_quat_inv,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
# from isaacgym import gymtorch, gymapi, gymutil
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree


from termcolor import colored
from loguru import logger

import joblib

from typing import Dict, Optional

class WholeBodyState:
    # root_pos: torch.Tensor
    # root_vel: torch.Tensor
    # root_quat: torch.Tensor # XYZW
    # root_ang_vel: torch.Tensor
    root_state: torch.Tensor
    # dof_state: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    
    def __init__(self, root_state, dof_pos, dof_vel):
        self.root_state = root_state
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel

    def interpolate(self, other, alpha):
        # alpha in [0, 1]
        # result = self * (1- alpha) + other * (alpha)
        # result is a WholeBodyState object
        result = WholeBodyState(
            root_state=self.root_state * (1- alpha) + other.root_state * (alpha),
            dof_pos=self.dof_pos * (1- alpha) + other.dof_pos * (alpha),
            dof_vel=self.dof_vel * (1- alpha) + other.dof_vel * (alpha)
        )
        
        
        result.root_state[:, 3:7] = slerp(self.root_state[:, 3:7], other.root_state[:, 3:7], 
                                            torch.tensor([alpha], device=self.root_state.device))
        
        
        return result

    def apply(self, simulator):
        assert not torch.isnan(self.root_state).any(), "Root state contains NaN values"
        assert not torch.isnan(self.dof_pos).any(), "DOF position contains NaN values"
        assert not torch.isnan(self.dof_vel).any(), "DOF velocity contains NaN values"
        
        num_envs, device = self.root_state.shape[0], self.root_state.device
        
        simulator.robot_root_states[:, :] = self.root_state[:, :]
        simulator.dof_pos[:, :] = self.dof_pos[:, :]
        simulator.dof_vel[:, :] = self.dof_vel[:, :]
        
        refresh_env_ids = torch.arange(num_envs, device=device)
        simulator.set_actor_root_state_tensor(refresh_env_ids, simulator.all_root_states)
        simulator.set_dof_state_tensor(refresh_env_ids, simulator.dof_state)

def radial_velocity_potential(cur_vel, ref_vel, sigma_theta=0.75, alpha_norm=0.4):
    # assume cur_vel, ref_vel: [N, d]
    cosine_similarity = torch.nn.functional.cosine_similarity(cur_vel, ref_vel, dim=-1)
    cosine_error = 1-cosine_similarity
    potential_cosine = torch.exp(-cosine_error / sigma_theta)
    # print(f'{potential_cosine=}','\t|',f'{cosine_error=}')
    
    
    norm_cur_velocity = cur_vel.norm(dim=-1)
    norm_ref_velocity = ref_vel.norm(dim=-1)
    norm_ratio = norm_cur_velocity / norm_ref_velocity
    
    potential_norm = norm_ratio*torch.exp(alpha_norm*(1-norm_ratio**(1/alpha_norm)))
    
    
    potential_velocity = potential_cosine * potential_norm
    return potential_velocity


class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        
        super().__init__(config, device)
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True

        self._init_save_motion()

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        if self.config.termination.terminate_when_dof_far:
            self.terminate_when_dof_far_threshold = self.config.termination_curriculum.terminate_when_dof_far_curriculum.init
            logger.info(f"Terminate when dof far threshold: {self.terminate_when_dof_far_threshold}")


        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.sdc_alpha = self.config.soft_dynamic_correction.alpha
            logger.info(f"Soft Dynamic Correction alpha: {self.sdc_alpha}")

    def teleop_callback(self, msg):
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def _init_save_motion(self):
        if "save_motion" in self.config:
            print(f"DEBUG: {self.motion_len[0]=}")
            print(f"DEBUG: {self.dt=}")
            print(f"DEBUG: {self.motion_len[0]/self.dt=}")
            self._motion_episode_length = torch.ceil((self.motion_len/self.dt)[0]).int()
            assert (self._motion_episode_length- self.motion_len[0]/self.dt).norm() <1, f"Motion length {self.motion_len} is not divisible by motion dt {self.dt}, {self._motion_episode_length=}, {(self._motion_episode_length- self.motion_len[0]/self.dt).norm()=}"
            assert self._motion_episode_length > 0, f"Motion length {self.motion_len} is not positive"
            
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)

                
                if hasattr(self.config, 'dump_motion_name'):
                    raise NotImplementedError
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[],'contact_mask':[]}
                self.motion_times_buf = []
                self.start_save = False
                
                self._write_to_file = True

        else:
            self.save_motion = False

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        
        assert self.config.robot.motion.motion_lib_type in ['origin', 'WJX'], "motion_lib_type must be either 'origin' or 'WJX'"
        if self.config.robot.motion.motion_lib_type == 'origin':
            from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
            self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        elif self.config.robot.motion.motion_lib_type == 'WJX':
            from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
            self._motion_lib = MotionLibRobotWJX(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)
        else:
            self._motion_lib.load_motions(random_sample=True)
        self._update_motion_start_xy()

        ref_init_state = self.kick_motion_res() ; self._kick_motion_res_counter = -1
        self.ref_init_rpy = get_euler_xyz_in_tensor(ref_init_state['root_rot'][:1]) # [1,3]
        # breakpoint()

        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions
        
        
        if hasattr(self._motion_lib, 'has_contact_mask') and self._motion_lib.has_contact_mask:
            self.ref_contact_mask = torch.zeros(self.num_envs, self._motion_lib._contact_size, # type: ignore
                                                dtype=torch.float, device=self.device, requires_grad=False)
            
        # self.ref_init_yaw = 

    def _init_tracking_config(self):
        
        
        
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
            
        self.end_time_ratio_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs, 
                                         self.num_bodies + self.num_extend_bodies, 
                                         3, 
                                         dtype=torch.float, 
                                         device=self.device, 
                                         requires_grad=False) # extend
            
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        super()._init_buffers()
        self._init_adaptive_sigma()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self._init_terrain_aware()

    ###########################################################################
    #### Terrain-aware motion tracking
    ###########################################################################
    def _init_terrain_aware(self):
        ta = self.config.get('terrain_aware', None) or {}
        self._ground_relative_ref = bool(ta.get('ground_relative_ref', False))
        self._recenter_motion_xy = bool(ta.get('recenter_motion_xy', False))
        self._support_height_tau = float(ta.get('support_height_tau', 0.02))
        self._spawn_clearance = float(ta.get('spawn_clearance', 0.0))
        self._debug_draw_height_scan = bool(ta.get('debug_draw_height_scan', False))
        self._per_foot_ref = bool(ta.get('per_foot_ref', False))
        self._support_lift_tau = float(ta.get('support_lift_tau', 0.03))
        self._ref_velocity_correction = bool(ta.get('ref_velocity_correction', True))
        self._ref_velocity_correction_max = float(ta.get('ref_velocity_correction_max', 1.0))
        self._feet_ground_footprint = float(ta.get('feet_ground_footprint', 0.04))
        self._leg_ik_ref = bool(ta.get('leg_ik_ref', False))
        self._leg_ik_iterations = int(ta.get('leg_ik_iterations', 6))
        self._leg_ik_support_ratio = float(ta.get('leg_ik_support_ratio', 0.0))
        assert not (self._leg_ik_ref and self._per_foot_ref), 'leg_ik_ref replaces per_foot_ref'
        self._custom_ref_offset = self._ground_relative_ref or self._recenter_motion_xy
        if self._ground_relative_ref:
            logger.info(colored(f"Terrain-aware tracking: ground-relative reference (recenter_motion_xy={self._recenter_motion_xy}, "
                                f"per_foot_ref={self._per_foot_ref}, leg_ik_ref={self._leg_ik_ref}, "
                                f"ref_velocity_correction={self._ref_velocity_correction}, "
                                f"feet_ground_footprint={self._feet_ground_footprint})", "green"))

        # per-foot reference: share of the foot's own ground offset given to each body of that leg
        # (ankle 1, knee 0.5, hip and above 0); the 3 extended bodies (hands, head) get 0
        body_names = list(self.config.robot.body_names)
        num_ref_bodies = len(body_names) + self.config.robot.motion.get('nums_extend_bodies', 0)
        self._leg_body_share = torch.zeros(2, num_ref_bodies, dtype=torch.float, device=self.device)
        for leg, side in enumerate(['left', 'right']):  # same order as feet_indices (body order)
            for i, name in enumerate(body_names):
                if name.startswith(side):
                    self._leg_body_share[leg, i] = 1.0 if 'ankle' in name else 0.5 if 'knee' in name else 0.
        self._init_leg_kinematics()
        # swing clearance: a lifted reference foot is raised by how far the highest ground within `radius` of it lies
        # above the ground its target is relative to (0 on flat ground and when stepping down)
        self._swing_clearance = bool(ta.get('swing_clearance', False))
        self._swing_clearance_lift = float(ta.get('swing_clearance_lift', 0.02))
        radius = float(ta.get('swing_clearance_radius', 0.15))
        rings = [(0., 0.)] + [(r * np.cos(a), r * np.sin(a)) for r in (radius / 3, 2 * radius / 3, radius)
                              for a in np.arange(8) * np.pi / 4]
        self._swing_clearance_offsets = torch.tensor(rings, dtype=torch.float, device=self.device)
        # terrain height under each reference foot, relative to which the feet reward measures clearance
        self.ref_feet_ground_height = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        r = self._feet_ground_footprint
        self._footprint_offsets = torch.tensor([[dx, dy] for dx in (-r, 0., r) for dy in (-r / 2, 0., r / 2)],
                                               dtype=torch.float, device=self.device)

        # per-env reference support height (terrain height under the reference feet)
        self.ref_support_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # start xy of each env's motion, used to re-centre motions on their terrain tile
        self._motion_start_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        # down-weighting of leg joints in the joint-position reward (legs must deviate on uneven ground)
        self._leg_joint_pos_weight = float(ta.get('leg_joint_pos_weight', 1.0))
        self._joint_pos_weights = torch.ones(self.num_dofs, dtype=torch.float, device=self.device)
        for i, name in enumerate(self.config.robot.dof_names):
            if any(k in name for k in ['hip', 'knee', 'ankle']):
                self._joint_pos_weights[i] = self._leg_joint_pos_weight

        # height-scan grid in the robot heading frame
        hs = ta.get('height_scan', None) or {}
        x_range, y_range = hs.get('x_range', [-0.5, 0.5]), hs.get('y_range', [-0.3, 0.3])
        res = hs.get('resolution', 0.1)
        xs = torch.linspace(x_range[0], x_range[1], int(round((x_range[1] - x_range[0]) / res)) + 1, device=self.device)
        ys = torch.linspace(y_range[0], y_range[1], int(round((y_range[1] - y_range[0]) / res)) + 1, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        self.num_height_scan_points = grid_x.numel()
        self._height_scan_points = torch.zeros(self.num_envs, self.num_height_scan_points, 3, device=self.device, requires_grad=False)
        self._height_scan_points[:, :, 0] = grid_x.flatten()
        self._height_scan_points[:, :, 1] = grid_y.flatten()
        self._height_scan_base_offset = hs.get('base_offset', 0.75)
        self._height_scan_clip = hs.get('clip', 1.0)

        # terrain height lookup: 'min3' (legged_gym: min of 3 height-field neighbours) or 'exact' (raster of the
        # collision trimesh: planes interpolated, step walls vertical where the trimesh made them vertical)
        self._height_lookup = ta.get('height_lookup', 'min3')
        assert self._height_lookup in ('min3', 'exact'), f'height_lookup: {self._height_lookup}'
        # fine scan under each foot, in the foot's yaw frame (`foot_scan`): ankle_z - base_offset - terrain height
        fs = ta.get('foot_scan', None) or {}
        self._foot_scan_points = self._scan_grid(fs.get('x_range', [-0.08, 0.16]), fs.get('y_range', [-0.06, 0.06]),
                                                 fs.get('resolution', 0.02))
        self._foot_scan_base_offset = fs.get('base_offset', 0.035)
        self._foot_scan_clip = fs.get('clip', 0.3)
        # wider scan for the critic (`privileged_height_scan`), always from the exact lookup and without noise
        ps = ta.get('privileged_height_scan', None) or {}
        self._privileged_scan_points = self._scan_grid(ps.get('x_range', [-0.7, 0.7]), ps.get('y_range', [-0.7, 0.7]),
                                                       ps.get('resolution', 0.1))
        obs_terms = {term for terms in self.config.obs.obs_dict.values() for term in terms}
        self._exact_height_map = None
        if self.config.terrain.mesh_type == 'trimesh' and (self._height_lookup == 'exact' or 'privileged_height_scan' in obs_terms):
            self._init_exact_height_map(float(ta.get('exact_height_resolution', 0.01)))
        obs_dims = self.config.obs.obs_dims  # a dict after pre_process_config, a list of one-item dicts before
        obs_dims = dict(obs_dims) if hasattr(obs_dims, 'items') else {k: v for d in obs_dims for k, v in d.items()}
        for term, num_points in (('height_scan', self.num_height_scan_points), ('height_scan_noisy', self.num_height_scan_points),
                                 ('foot_scan', 2 * len(self._foot_scan_points)),
                                 ('privileged_height_scan', len(self._privileged_scan_points))):
            assert term not in obs_terms or obs_dims[term] == num_points, f'obs_dims.{term} {obs_dims[term]} != {num_points} points'
        logger.info(colored(f"Height scan: {self.num_height_scan_points} points, lookup={self._height_lookup}, "
                            f"foot_scan {len(self._foot_scan_points)} points/foot, "
                            f"privileged_height_scan {len(self._privileged_scan_points)} points", "green"))

        # realistic perception errors for the policy's scan (`height_scan_noisy` term; with perturb_height_scan also the
        # `height_scan` term, to evaluate policies trained on the exact scan). Per episode: map height bias, horizontal
        # map misregistration, latency; per step and point: noise, dropout (the point keeps its previous value).
        hn = ta.get('height_scan_noise', None) or {}
        self._scan_point_std = float(hn.get('point_std', 0.0))
        self._scan_bias_std = float(hn.get('episode_bias_std', 0.0))
        self._scan_drift_std = float(hn.get('episode_drift_xy_std', 0.0))
        self._scan_dropout = float(hn.get('dropout', 0.0))
        self._scan_delay_range = [int(d) for d in hn.get('delay_steps', [0, 0])]
        self._perturb_height_scan = bool(hn.get('perturb_height_scan', False))
        self._scan_blind = hn.get('blind', None) or None
        assert self._scan_blind in (None, 'flat', 'zero'), f'height_scan_noise.blind: {self._scan_blind}'
        self._scan_noise_enabled = (self._scan_point_std > 0 or self._scan_bias_std > 0 or self._scan_drift_std > 0
                                    or self._scan_dropout > 0 or self._scan_delay_range[1] > 0 or self._scan_blind is not None)
        if self._scan_noise_enabled:
            self._scan_bias = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            self._scan_drift = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
            self._scan_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            # past perturbed scans, newest first; envs flagged in _scan_refill start a new episode with a full buffer
            self._scan_history = torch.zeros(self._scan_delay_range[1] + 1, self.num_envs, self.num_height_scan_points,
                                             dtype=torch.float, device=self.device)
            self._scan_refill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._perceived_height_scan = torch.zeros(self.num_envs, self.num_height_scan_points, dtype=torch.float, device=self.device)
            self._sample_height_scan_noise(torch.arange(self.num_envs, device=self.device))
            logger.info(colored(f"Height-scan noise: point_std={self._scan_point_std}, episode_bias_std={self._scan_bias_std}, "
                                f"episode_drift_xy_std={self._scan_drift_std}, dropout={self._scan_dropout}, "
                                f"delay_steps={self._scan_delay_range}, blind={self._scan_blind}, "
                                f"perturb_height_scan={self._perturb_height_scan}", "green"))

    def _scan_grid(self, x_range, y_range, res):
        """ [num_points, 2] grid of scan points (x-major, like the `height_scan` grid). """
        xs = torch.linspace(x_range[0], x_range[1], int(round((x_range[1] - x_range[0]) / res)) + 1, device=self.device)
        ys = torch.linspace(y_range[0], y_range[1], int(round((y_range[1] - y_range[0]) / res)) + 1, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    def _init_exact_height_map(self, res):
        """ Rasterise the collision trimesh (top surfaces; vertical walls have no area) into a height map with
        `res` spacing, looked up by nearest neighbour: exact on planes to res/2 * slope and on step walls to res/2. """
        terrain = self.simulator.terrain
        border = self.config.terrain.border_size
        vertices = torch.tensor(terrain.vertices, dtype=torch.float, device=self.device)  # trimesh frame = world + border
        tris = vertices[torch.tensor(terrain.triangles.astype(np.int64), device=self.device)]  # [T, 3 vertices, 3]
        e1, e2 = tris[:, 1, :2] - tris[:, 0, :2], tris[:, 2, :2] - tris[:, 0, :2]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        keep = det.abs() > 1e-8
        tris, e1, e2, det = tris[keep], e1[keep], e2[keep], det[keep]
        size = (torch.ceil(vertices[:, :2].max(dim=0).values / res) + 1).long()
        heights = torch.full((int(size[0]) * int(size[1]),), -float('inf'), device=self.device)
        lo = torch.ceil(tris[..., :2].min(dim=1).values / res - 1e-4).long()
        hi = torch.floor(tris[..., :2].max(dim=1).values / res + 1e-4).long()
        extent = (hi - lo + 1).clamp(min=0)
        order = torch.argsort(extent.max(dim=-1).values)  # chunks of similar bounding-box size
        for chunk in torch.split(order, 16384):
            n = int(extent[chunk].max())
            if n == 0:
                continue
            offsets = torch.stack(torch.meshgrid(*[torch.arange(n, device=self.device)] * 2, indexing='ij'), dim=-1).view(-1, 2)
            idx = lo[chunk, None] + offsets  # [C, n * n, 2]
            d = idx * res - tris[chunk, None, 0, :2]
            w1 = (d[..., 0] * e2[chunk, None, 1] - d[..., 1] * e2[chunk, None, 0]) / det[chunk, None]
            w2 = (e1[chunk, None, 0] * d[..., 1] - e1[chunk, None, 1] * d[..., 0]) / det[chunk, None]
            w0 = 1 - w1 - w2
            inside = (w0 > -1e-4) & (w1 > -1e-4) & (w2 > -1e-4) & (idx <= hi[chunk, None]).all(dim=-1) \
                & (idx < size).all(dim=-1) & (idx >= 0).all(dim=-1)
            z = w0 * tris[chunk, None, 0, 2] + w1 * tris[chunk, None, 1, 2] + w2 * tris[chunk, None, 2, 2]
            flat = idx[..., 0] * size[1] + idx[..., 1]
            heights.scatter_reduce_(0, flat[inside], z[inside], reduce='amax')
        heights = heights.view(int(size[0]), int(size[1]))
        missing = torch.isinf(heights)
        if missing.any():  # not covered by any triangle (should not happen): legacy lookup
            ij = torch.nonzero(missing).float() * res - border
            heights[missing] = self._terrain_height_min3(ij)
        self._exact_height_map, self._exact_height_res = heights, res
        logger.info(colored(f"Exact terrain height map: {tuple(heights.shape)} at {res} m, "
                            f"{int(missing.sum())} cells filled by the legacy lookup", "green"))

    def _terrain_height_exact(self, xy):
        """ Terrain height [m] at world positions xy [..., 2] from the rasterised collision trimesh. """
        if self._exact_height_map is None:
            return self._terrain_height_at(xy)
        border, res = self.config.terrain.border_size, self._exact_height_res
        px = torch.round((xy[..., 0] + border) / res).long().clamp(0, self._exact_height_map.shape[0] - 1)
        py = torch.round((xy[..., 1] + border) / res).long().clamp(0, self._exact_height_map.shape[1] - 1)
        return self._exact_height_map[px, py]

    def _terrain_height_at(self, xy):
        """ Terrain height [m] at world positions xy [..., 2] (terrain_aware.height_lookup: legged_gym or exact). """
        if getattr(self, '_exact_height_map', None) is not None and self._height_lookup == 'exact':
            return self._terrain_height_exact(xy)
        return self._terrain_height_min3(xy)

    def _terrain_height_min3(self, xy):
        """ Terrain height [m] at world positions xy [..., 2] (same lookup as legged_gym). """
        if self.config.terrain.mesh_type not in ['heightfield', 'trimesh']:
            return torch.zeros(xy.shape[:-1], dtype=torch.float, device=self.device)
        height_samples = self.simulator.height_samples
        scale = self.config.terrain.horizontal_scale
        border = self.config.terrain.border_size
        px = ((xy[..., 0] + border) / scale).long().clamp(0, height_samples.shape[0] - 2)
        py = ((xy[..., 1] + border) / scale).long().clamp(0, height_samples.shape[1] - 2)
        heights = torch.min(torch.min(height_samples[px, py], height_samples[px + 1, py]), height_samples[px, py + 1])
        return heights.float() * self.config.terrain.vertical_scale

    def _feet_ground_height(self, feet_xy):
        """ Terrain height under feet [..., 2] averaged over a foot-sized footprint, so that it changes
        continuously (over ~2 * footprint) when a foot moves across a step edge instead of jumping. """
        if self._feet_ground_footprint <= 0:
            return self._terrain_height_at(feet_xy)
        points = feet_xy[..., None, :] + self._footprint_offsets  # [..., 9, 2]
        return self._terrain_height_at(points).mean(dim=-1)

    def _update_motion_start_xy(self):
        start = self._motion_lib.get_motion_state(self.motion_ids, torch.zeros(self.num_envs, device=self.device))
        self._motion_start_xy[:] = start['root_pos'][:, :2]
        self._kick_motion_res_counter = -1
        self._kick_motion_res_counter_multistep = -1

    def _ref_offset(self):
        """ World offset of the reference motion. Original PBHC: env origin (incl. a fixed terrain height). """
        if not self._custom_ref_offset:
            return self.env_origins
        offset = self.env_origins.clone()
        if self._ground_relative_ref:
            offset[:, 2] = 0.  # the per-frame support height replaces the fixed origin height
        if self._recenter_motion_xy:
            offset[:, :2] -= self._motion_start_xy
        return offset

    def _ground_relative_z_offsets(self, motion_res):
        """ Vertical offsets that move the reference (authored on flat ground at z=0) onto the local terrain.

        Rigid shift (per_foot_ref=False): every body is shifted by one support height, a soft maximum
        of the terrain under the supporting feet, so no reference foot ends up below the ground; the
        foot over lower ground is pulled down by the ground-relative feet reward. Feet lifted in the
        reference are ignored by subtracting their lift.

        Per-foot (per_foot_ref=True): the support height is a lift-weighted mean of the ground under
        the feet (both feet down: the middle; one foot lifted: the stance foot), and each leg is
        additionally shifted towards the ground under its own foot (ankle fully, knee half), so both
        reference feet stand on their own ground.

        Leg IK (leg_ik_ref=True): support height between the ground under the lower supporting foot
        (leg_ik_support_ratio=0) and the per-foot support height (1), every body is shifted by it, and the
        legs are then re-solved by IK (_leg_ik_deltas) so that each reference foot stands on its own
        ground with joint angles that agree with the body positions.

        Both are continuous in time, need no state, and are exactly 0 on flat ground.
        Returns (support height [N], per-body offsets [N, num_ref_bodies], ground under the ref feet [N, 2]).
        """
        ref_feet = motion_res['rg_pos_t'][:, self.feet_indices]  # [N, 2, 3]
        feet_ground = self._feet_ground_height(ref_feet[..., :2])
        feet_lift = ref_feet[..., 2] - ref_feet[..., 2].min(dim=-1, keepdim=True)[0]
        if self._per_foot_ref or self._leg_ik_ref:
            weights = torch.softmax(-feet_lift / self._support_lift_tau, dim=-1)
        else:
            weights = torch.softmax((feet_ground - feet_lift) / self._support_height_tau, dim=-1)
        support_height = (weights * feet_ground).sum(dim=-1)  # [N]
        if self._leg_ik_ref:  # move the pelvis towards the lower supporting foot: legs can bend further than stretch
            low_weights = torch.softmax(-(feet_ground + feet_lift) / self._support_height_tau, dim=-1)
            low = (low_weights * feet_ground).sum(dim=-1)
            support_height = low + self._leg_ik_support_ratio * (support_height - low)

        body_offsets = support_height[:, None].expand(-1, self._leg_body_share.shape[1])
        feet_offsets = support_height[:, None].expand(-1, 2)
        if self._per_foot_ref:
            feet_delta = feet_ground - support_height[:, None]  # [N, 2]
            body_offsets = body_offsets + feet_delta @ self._leg_body_share
            feet_offsets = feet_ground
        elif self._leg_ik_ref:
            feet_offsets = feet_ground
        if self._swing_clearance:  # lift a swinging reference foot over higher ground near it (e.g. a riser ahead)
            near = ref_feet[..., None, :2] + self._swing_clearance_offsets  # [N, 2, P, 2]
            near_ground = self._terrain_height_at(near).max(dim=-1)[0]
            swing = (feet_lift / self._swing_clearance_lift).clamp(0., 1.)
            extra = swing * (near_ground - feet_offsets).clamp(min=0.)  # [N, 2]
            body_offsets = body_offsets + extra @ self._leg_body_share  # ankle fully, knee half
        return support_height, body_offsets, feet_offsets

    def _apply_ground_relative_ref(self, motion_res, prev_motion_res=None):
        """ Shift the reference onto the local terrain (see _ground_relative_z_offsets). If the reference
        of the previous control step is given, the rate of change of the offsets is added to the
        reference z velocities, so that velocity and position targets stay consistent. """
        support_height, body_offsets, feet_offsets = self._ground_relative_z_offsets(motion_res)
        num_bodies = motion_res['rg_pos'].shape[1]
        if self._leg_ik_ref:  # before the shift: the IK works relative to the pelvis
            dq, dp, dquat = self._leg_ik_deltas(motion_res, feet_offsets - support_height[:, None])
            self._apply_leg_deltas(motion_res, dq, dp, dquat)

        motion_res['root_pos'][:, 2] += body_offsets[:, 0]
        motion_res['rg_pos'][..., 2] += body_offsets[:, :num_bodies]
        if motion_res['rg_pos_t'] is not motion_res['rg_pos']:
            motion_res['rg_pos_t'][..., 2] += body_offsets[:, :motion_res['rg_pos_t'].shape[1]]
        if prev_motion_res is not None:
            _, prev_body_offsets, _ = self._ground_relative_z_offsets(prev_motion_res)
            body_vel_z = ((body_offsets - prev_body_offsets) / self.dt).clamp(  # a foot crossing a step edge makes the
                -self._ref_velocity_correction_max, self._ref_velocity_correction_max)  # offset jump; no velocity spike for that
            motion_res['root_vel'][:, 2] += body_vel_z[:, 0]
            motion_res['body_vel'][..., 2] += body_vel_z[:, :num_bodies]
            if motion_res['body_vel_t'] is not motion_res['body_vel']:
                motion_res['body_vel_t'][..., 2] += body_vel_z[:, :motion_res['body_vel_t'].shape[1]]
            if self._leg_ik_ref:  # rate of change of the IK corrections
                prev_support, _, prev_feet = self._ground_relative_z_offsets(prev_motion_res)
                prev_dq, prev_dp, _ = self._leg_ik_deltas(prev_motion_res, prev_feet - prev_support[:, None])
                vmax = self._ref_velocity_correction_max
                dof_ids, body_ids = self._leg_dof_ids.flatten(), self._leg_body_ids.flatten()
                motion_res['dof_vel'][:, dof_ids] += ((dq - prev_dq) / self.dt).clamp(-10 * vmax, 10 * vmax).flatten(1)
                leg_vel = ((dp - prev_dp) / self.dt).clamp(-vmax, vmax).flatten(1, 2)
                motion_res['body_vel'][:, body_ids] += leg_vel
                if motion_res['body_vel_t'] is not motion_res['body_vel']:
                    motion_res['body_vel_t'][:, body_ids] += leg_vel
        motion_res['ref_support_height'] = support_height
        motion_res['ref_feet_ground_height'] = feet_offsets
        return motion_res

    def _init_leg_kinematics(self):
        """ Leg chains pelvis -> ankle roll link (offsets, fixed rotations, joint axes) from the motion MJCF. """
        import xml.etree.ElementTree as ET
        path = Path(self.config.robot.motion.asset.assetRoot) / self.config.robot.motion.asset.assetFileName
        bodies = {b.attrib['name']: b for b in ET.parse(path).getroot().iter('body')}
        body_names, dof_names = list(self.config.robot.body_names), list(self.config.robot.dof_names)
        offsets, rots, axes, body_ids, dof_ids = [], [], [], [], []
        for side in ['left', 'right']:  # same order as feet_indices
            for link in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']:
                body = bodies[f'{side}_{link}_link']
                offsets.append([float(v) for v in body.attrib.get('pos', '0 0 0').split()])
                w, x, y, z = [float(v) for v in body.attrib.get('quat', '1 0 0 0').split()]
                rots.append(sRot.from_quat([x, y, z, w]).as_matrix())
                axes.append([float(v) for v in body.find('joint').attrib['axis'].split()])
                body_ids.append(body_names.index(f'{side}_{link}_link'))
                dof_ids.append(dof_names.index(f'{side}_{link}_joint'))
        as_tensor = lambda v: torch.tensor(np.array(v), dtype=torch.float, device=self.device)  # noqa: E731
        self._leg_offsets = as_tensor(offsets).view(2, 6, 3)
        self._leg_local_rot = as_tensor(rots).view(2, 6, 3, 3)
        axes = as_tensor(axes).view(2, 6, 3)
        self._leg_axes_skew = torch.zeros(2, 6, 3, 3, dtype=torch.float, device=self.device)
        self._leg_axes_skew[..., 0, 1], self._leg_axes_skew[..., 0, 2] = -axes[..., 2], axes[..., 1]
        self._leg_axes_skew[..., 1, 0], self._leg_axes_skew[..., 1, 2] = axes[..., 2], -axes[..., 0]
        self._leg_axes_skew[..., 2, 0], self._leg_axes_skew[..., 2, 1] = -axes[..., 1], axes[..., 0]
        self._leg_body_ids = torch.tensor(body_ids, device=self.device).view(2, 6)
        self._leg_dof_ids = torch.tensor(dof_ids, device=self.device).view(2, 6)
        lower = torch.tensor(list(self.config.robot.dof_pos_lower_limit_list), dtype=torch.float, device=self.device)
        upper = torch.tensor(list(self.config.robot.dof_pos_upper_limit_list), dtype=torch.float, device=self.device)
        self._leg_dof_lower, self._leg_dof_upper = lower[self._leg_dof_ids], upper[self._leg_dof_ids]

    def _leg_fk(self, q):
        """ Positions [..., 2, 6, 3] and rotations [..., 2, 6, 3, 3] of the leg links in the pelvis frame for
        leg joint angles q [..., 2, 6] (same recursion as Humanoid_Batch.forward_kinematics_batch). """
        K = self._leg_axes_skew
        s, c = q.sin()[..., None, None], q.cos()[..., None, None]
        joint_rot = torch.eye(3, device=self.device) + s * K + (1 - c) * (K @ K)  # Rodrigues
        p = torch.zeros(*q.shape[:-1], 3, dtype=q.dtype, device=self.device)
        R = torch.eye(3, device=self.device).expand(*q.shape[:-1], 3, 3)
        positions, rotations = [], []
        for j in range(6):
            p = p + (R @ self._leg_offsets[:, j, :, None]).squeeze(-1)
            R = R @ self._leg_local_rot[:, j] @ joint_rot[..., j, :, :]
            positions.append(p)
            rotations.append(R)
        return torch.stack(positions, dim=-2), torch.stack(rotations, dim=-3)

    def _leg_ik(self, q_ref, feet_shift, rot_weight=0.3, damping=1e-4, max_step=0.2, eps=1e-4):
        """ Leg joint angles [N, 2, 6] that move each ankle roll link by feet_shift [N, 2, 3] (pelvis frame)
        from where q_ref puts it and keep its orientation where the joint limits allow (the ankle roll limit
        often does not, with the legs spread): damped least squares with a finite-difference Jacobian,
        orientation weighted by rot_weight [m/rad] below position, steps clamped to max_step [rad].
        Legs whose result ends farther from the target than q_ref keep q_ref. """
        pos, rot = self._leg_fk(q_ref)
        target_p, target_R = pos[..., -1, :] + feet_shift, rot[..., -1, :, :]
        steps = torch.cat([torch.zeros(1, 6, device=self.device), eps * torch.eye(6, device=self.device)])  # [7, 6]
        weights = torch.tensor([1., 1., 1., rot_weight, rot_weight, rot_weight], device=self.device)

        def pose_error(q):  # weighted [position; orientation] error of the ankle roll link, q [N, (7,) 2, 6] -> [N, (7,) 2, 6]
            pos, rot = self._leg_fk(q)
            extra = (1,) * (q.dim() - 3)
            tp, tR = target_p.view(q.shape[0], *extra, 2, 3), target_R.view(q.shape[0], *extra, 2, 3, 3)
            R_err = tR @ rot[..., -1, :, :].transpose(-1, -2)
            w_err = 0.5 * torch.stack([R_err[..., 2, 1] - R_err[..., 1, 2], R_err[..., 0, 2] - R_err[..., 2, 0],
                                       R_err[..., 1, 0] - R_err[..., 0, 1]], dim=-1)
            return torch.cat([tp - pos[..., -1, :], w_err], dim=-1) * weights

        q = q_ref.clone()
        for _ in range(self._leg_ik_iterations):
            err = pose_error(q[:, None] + steps[:, None])  # q and the 6 perturbed q: [N, 7, 2, 6]
            J = ((err[:, :1] - err[:, 1:]) / eps).permute(0, 2, 3, 1)  # d(pose)/dq: [N, 2, 6 (pose), 6 (q)]
            JJt = J @ J.transpose(-1, -2) + damping * torch.eye(6, device=self.device)
            dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, err[:, 0, ..., None])).squeeze(-1)
            dq = dq * (max_step / dq.abs().amax(dim=-1, keepdim=True).clamp(min=max_step))
            q = torch.max(torch.min(q + dq, self._leg_dof_upper), self._leg_dof_lower)
        worse = pose_error(q).norm(dim=-1) > pose_error(q_ref).norm(dim=-1)
        return torch.where(worse[..., None], q_ref, q)

    def _leg_ik_deltas(self, motion_res, feet_dz):
        """ IK correction of the reference legs that raises each reference foot by feet_dz [N, 2] relative
        to the pelvis. Returns the change of the leg joint angles [N, 2, 6], of the leg link positions
        [N, 2, 6, 3] (world) and the world rotation that maps old to new leg link rotations (xyzw) [N, 2, 6, 4]. """
        q_ref = motion_res['dof_pos'][:, self._leg_dof_ids]  # [N, 2, 6]
        pelvis_R = quaternion_to_matrix(xyzw_to_wxyz(motion_res['root_rot']))  # world from pelvis
        feet_shift = feet_dz[..., None] * pelvis_R[:, None, 2, :]  # world z shift in the pelvis frame
        q = self._leg_ik(q_ref, feet_shift)
        pos_ref, rot_ref = self._leg_fk(q_ref)
        pos, rot = self._leg_fk(q)
        R0 = pelvis_R[:, None, None]
        dp = (R0 @ (pos - pos_ref)[..., None]).squeeze(-1)
        dR = R0 @ rot @ rot_ref.transpose(-1, -2) @ R0.transpose(-1, -2)
        dquat = wxyz_to_xyzw(matrix_to_quaternion(dR))
        identity = torch.tensor([0., 0., 0., 1.], device=self.device)
        dquat = torch.where((feet_dz != 0)[..., None, None], dquat, identity)  # exactly the original reference on flat ground
        return q - q_ref, dp, dquat

    def _apply_leg_deltas(self, motion_res, dq, dp, dquat):
        dof_ids, body_ids = self._leg_dof_ids.flatten(), self._leg_body_ids.flatten()
        motion_res['dof_pos'][:, dof_ids] += dq.flatten(1)
        motion_res['rg_pos'][:, body_ids] += dp.flatten(1, 2)
        if motion_res['rg_pos_t'] is not motion_res['rg_pos']:
            motion_res['rg_pos_t'][:, body_ids] += dp.flatten(1, 2)
        dquat = dquat.flatten(1, 2)
        unchanged = (dquat == torch.tensor([0., 0., 0., 1.], device=self.device)).all(dim=-1, keepdim=True)
        for key in ['rb_rot', 'rg_rot_t']:
            if key == 'rg_rot_t' and motion_res[key] is motion_res['rb_rot']:
                continue
            old = motion_res[key][:, body_ids].contiguous()
            motion_res[key][:, body_ids] = torch.where(unchanged, old, quat_mul(dquat, old, w_last=True))  # quat_mul rounds

    def _reward_termination(self):
        # terminal penalty; terminations other than falls (e.g. drifting away from the reference) are scaled by
        # termination_scales.termination_drift_penalty_scale so that falling is the worst outcome (1 = all alike)
        terminated = self.reset_buf * ~self.time_out_buf
        drift_scale = self.config.termination_scales.get('termination_drift_penalty_scale', 1.0)
        fall = self.reset_buf_terminate_by.get('gravity')
        if drift_scale == 1.0 or fall is None:
            return terminated
        return terminated * torch.where(fall, 1.0, drift_scale)

    def _update_terrain_curriculum(self, env_ids):
        """ Terrain curriculum over the difficulty rows: robots that tracked the motion to its end (over at
        least half of the motion) move one row up, robots whose episode ended before half of the remaining
        motion move one row down, and robots that solve the hardest row go to a random row (legged_gym), or with
        terrain.curriculum_graduate_rows = k to a random one of the k hardest rows, so the hardest terrain stays trained. """
        if not self.init_done or self.common_step_counter == 0:
            return
        elapsed = self.last_episode_length_buf[env_ids] * self.dt
        motion_end = self.reset_buf_terminate_by.get('motion_end')
        completed = motion_end[env_ids] if motion_end is not None else torch.zeros_like(elapsed, dtype=torch.bool)
        remaining = self.motion_len[env_ids] - self.motion_start_times[env_ids]
        move_up = completed & (elapsed >= 0.5 * self.motion_len[env_ids])
        move_down = ~completed & (elapsed < 0.5 * remaining)
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

    def _height_scan(self, xy_offset=None):
        """ Unclipped scan root_z - base_offset - terrain height; xy_offset [num_envs, 2] shifts the sampled points. """
        root_states = self.simulator.robot_root_states
        heading_rot = calc_heading_quat(root_states[:, 3:7].clone(), w_last=True)
        num_points = self.num_height_scan_points
        points = my_quat_rotate(heading_rot.repeat_interleave(num_points, dim=0),
                                self._height_scan_points.view(-1, 3)).view(self.num_envs, num_points, 3)
        points_xy = points[..., :2] + root_states[:, None, :2]
        if xy_offset is not None:
            points_xy = points_xy + xy_offset[:, None, :]
        heights = self._terrain_height_at(points_xy)
        return root_states[:, 2:3] - self._height_scan_base_offset - heights

    def _get_obs_height_scan(self):
        if self._scan_noise_enabled and self._perturb_height_scan:
            return self._perceived_height_scan
        return torch.clip(self._height_scan(), -self._height_scan_clip, self._height_scan_clip)

    def _get_obs_height_scan_noisy(self):
        if not self._scan_noise_enabled:
            return self._get_obs_height_scan()
        return self._perceived_height_scan

    @staticmethod
    def _yaw_rotate(points, quat):
        """ points [P, 2] rotated by the yaw of quaternions quat [..., 4] (xyzw) -> [..., P, 2] """
        x, y, z, w = quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cos, sin = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        return torch.stack([cos * points[:, 0] - sin * points[:, 1], sin * points[:, 0] + cos * points[:, 1]], dim=-1)

    def _get_obs_foot_scan(self):
        """ Fine scan under each foot (left then right), in the foot's yaw frame: ankle_z - base_offset - terrain height. """
        feet_pos = self.simulator._rigid_body_pos[:, self.feet_indices]  # [N, 2, 3]
        feet_rot = self.simulator._rigid_body_rot[:, self.feet_indices]
        points = self._yaw_rotate(self._foot_scan_points, feet_rot) + feet_pos[:, :, None, :2]  # [N, 2, P, 2]
        scan = feet_pos[:, :, None, 2] - self._foot_scan_base_offset - self._terrain_height_at(points)
        return torch.clip(scan, -self._foot_scan_clip, self._foot_scan_clip).view(self.num_envs, -1)

    def _get_obs_privileged_height_scan(self):
        """ Critic only: wider scan around the root in its heading frame from the exact lookup, never perturbed. """
        root_states = self.simulator.robot_root_states
        points = self._yaw_rotate(self._privileged_scan_points, root_states[:, 3:7]) + root_states[:, None, :2]
        scan = root_states[:, 2:3] - self._height_scan_base_offset - self._terrain_height_exact(points)
        return torch.clip(scan, -self._height_scan_clip, self._height_scan_clip)

    def _sample_height_scan_noise(self, env_ids):
        n = len(env_ids)
        self._scan_bias[env_ids] = torch.randn(n, 1, device=self.device) * self._scan_bias_std
        self._scan_drift[env_ids] = torch.randn(n, 2, device=self.device) * self._scan_drift_std
        self._scan_delay[env_ids] = torch.randint(self._scan_delay_range[0], self._scan_delay_range[1] + 1, (n,), device=self.device)
        self._scan_refill[env_ids] = True

    def _update_perceived_height_scan(self):
        """ Once per step, after resets: perturb the scan and push it into the latency buffer. """
        scan = self._height_scan(self._scan_drift) - self._scan_bias
        if self._scan_blind == 'flat':  # ground assumed flat at the height under the root: no terrain shape
            root_states = self.simulator.robot_root_states
            scan = (root_states[:, 2:3] - self._height_scan_base_offset - self._terrain_height_at(root_states[:, None, :2])
                    - self._scan_bias).expand_as(scan)
        elif self._scan_blind == 'zero':  # no terrain information at all (nominal standing height on flat ground)
            scan = torch.zeros_like(scan)
        if self._scan_point_std > 0:
            scan = scan + torch.randn_like(scan) * self._scan_point_std
        if self._scan_dropout > 0:
            dropped = (torch.rand_like(scan) < self._scan_dropout) & ~self._scan_refill[:, None]
            scan = torch.where(dropped, self._scan_history[0], scan)
        self._scan_history = torch.roll(self._scan_history, 1, dims=0)
        self._scan_history[0] = scan
        self._scan_history[:, self._scan_refill] = scan[self._scan_refill]
        self._scan_refill[:] = False
        delayed = self._scan_history[self._scan_delay, torch.arange(self.num_envs, device=self.device)]
        self._perceived_height_scan = torch.clip(delayed, -self._height_scan_clip, self._height_scan_clip)

    def _compute_observations(self):
        if self._scan_noise_enabled:
            self._update_perceived_height_scan()
        super()._compute_observations()

    def _height_scan_world_points(self, env_id):
        root_states = self.simulator.robot_root_states
        heading_rot = calc_heading_quat(root_states[env_id:env_id+1, 3:7].clone(), w_last=True)
        points = my_quat_rotate(heading_rot.repeat(self.num_height_scan_points, 1), self._height_scan_points[env_id])
        points[:, :2] += root_states[env_id, :2]
        points[:, 2] = self._terrain_height_at(points[:, :2])
        return points

    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        
        
        end_time = self.last_episode_length_buf[env_ids]*self.dt + self.motion_start_times[env_ids]
        self.end_time_ratio_buf[env_ids] = (end_time / self.motion_len[env_ids])
        # breakpoint()
        
        self.log_dict["end_time_ratio"] = self.end_time_ratio_buf.mean()
        self.log_dict["end_time_ratio_std"] = self.end_time_ratio_buf.std()

        if self.custom_origins and self.config.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)  # before the motion times are resampled
        if self._scan_noise_enabled:
            self._sample_height_scan_noise(env_ids)

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids) # need to resample before reset root states
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
            
        if self.config.termination.terminate_when_dof_far and self.config.termination_curriculum.terminate_when_dof_far_curriculum.enable:
            self._update_terminate_when_dof_far_curriculum()
            
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable and self.config.soft_dynamic_correction.curriculum.enable:
            self._update_soft_dynamic_correction()
    
    def _update_soft_dynamic_correction(self):
        sdc_curr = self.config.soft_dynamic_correction.curriculum
        
        if self.average_episode_length < sdc_curr.level_down_threshold:
            self.sdc_alpha *= (1 + sdc_curr.degree)
        elif self.average_episode_length > sdc_curr.level_up_threshold:
            self.sdc_alpha *= (1 - sdc_curr.degree)
        self.sdc_alpha = np.clip(self.sdc_alpha, sdc_curr.min_alpha, sdc_curr.max_alpha)

    def _update_terminate_when_dof_far_curriculum(self):
        dof_far_curr = self.config.termination_curriculum.terminate_when_dof_far_curriculum
        
        if self.average_episode_length < dof_far_curr.level_down_threshold:
            self.terminate_when_dof_far_threshold *= (1 + dof_far_curr.degree)
        elif self.average_episode_length > dof_far_curr.level_up_threshold:
            self.terminate_when_dof_far_threshold *= (1 - dof_far_curr.degree)
        self.terminate_when_dof_far_threshold = np.clip(self.terminate_when_dof_far_threshold, 
                                                        dof_far_curr.min, 
                                                        dof_far_curr.max)
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
        

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                # logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _update_reset_buf(self):
        super()._update_reset_buf()
        
        # print(f"DEBUG: motion far = {torch.norm(self.dif_global_body_pos, dim=-1).max()}\t|  threshold={self.terminate_when_motion_far_threshold}")
        
        if self.config.termination.terminate_when_motion_far:
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf_terminate_by["motion_far"] = reset_buf_motion_far
            self.reset_buf |= reset_buf_motion_far
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)
                
                
        if self.config.termination.terminate_when_dof_far:
            reset_buf_dof_far = torch.any(torch.norm(self.dif_joint_angles, dim=-1) > self.terminate_when_dof_far_threshold, dim=-1)
            self.reset_buf_terminate_by["dof_far"] = reset_buf_dof_far
            self.reset_buf |= reset_buf_dof_far
            # log current dof far threshold
            if self.config.termination_curriculum.terminate_when_dof_far_curriculum.enable:
                self.log_dict["terminate_when_dof_far_threshold"] = torch.tensor(self.terminate_when_dof_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.reset_buf_terminate_by["motion_end"] = current_time > self.motion_len
            self.time_out_buf |= self.reset_buf_terminate_by["motion_end"]
            
            

    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self._update_motion_start_xy()
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        self._kick_motion_res_counter = -1
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        self._update_motion_start_xy()

        
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))


    def _compute_reward(self):
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm

        if self._ground_relative_ref:
            # sanity of the terrain-adapted reference: the lower reference foot should sit on the ground (~ankle height)
            ref_feet = self.ref_body_pos_extend[:, self.feet_indices]
            ref_feet_clearance = ref_feet[..., 2] - self._terrain_height_at(ref_feet[..., :2])
            robot_feet = self.simulator._rigid_body_pos[:, self.feet_indices]
            robot_feet_clearance = robot_feet[..., 2] - self._terrain_height_at(robot_feet[..., :2])
            self.log_dict["ref_support_height"] = self.ref_support_height.mean()
            self.log_dict["ref_feet_min_clearance"] = ref_feet_clearance.min(dim=-1)[0].mean()
            self.log_dict["robot_feet_min_clearance"] = robot_feet_clearance.min(dim=-1)[0].mean()
        
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.log_dict["sdc_alpha"] = torch.tensor(self.sdc_alpha, dtype=torch.float, device=self.device)
        if 'adaptive_tracking_sigma' in self.config.rewards and self.config.rewards.adaptive_tracking_sigma.enable:
            for key, value in self._reward_error_ema.items():
                self.log_dict['error_ema_'+key] = torch.tensor(value, dtype=torch.float, device=self.device)
                self.log_dict['adp_sigma_'+key] = torch.tensor(self.config.rewards.reward_tracking_sigma[key], dtype=torch.float, device=self.device)
        

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(1):
        # for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                # draw marker joints
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    # import ipdb; ipdb.set_trace()
                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)

                if self._debug_draw_height_scan:
                    for point in self._height_scan_world_points(env_id):
                        self.simulator.draw_sphere(point, 0.015, (0.1, 0.6, 1.0), env_id)


            else:
                # draw teleop joints
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # breakpoint()
        if self.custom_origins and not self._ground_relative_ref: # trimesh (original PBHC path)
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            motion_res = self.kick_motion_res()

            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            

        else:
            # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            # offset = self.env_origins
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            motion_res = self.kick_motion_res()



            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]

            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                # XYZW
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                # Isaac sim internally uses WXYZ
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                # breakpoint() # Check Quaternion format
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                # breakpoint() # Check Quaternion format
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

        # self.simulator.robot_root_states[env_ids, 2] += 0.05 # in case under the terrain
        if self._spawn_clearance > 0:
            self.simulator.robot_root_states[env_ids, 2] += self._spawn_clearance

    def small_random_quaternions(self, n, max_angle):
            # XYZW
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            # Convert angle-axis to quaternion
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # print("DEBUG: reset", len(self.motions_for_saving['dof']))
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
        # offset = self.env_origins
        # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        motion_res = self.kick_motion_res()

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise


    _kick_motion_res_counter = -1
    _kick_motion_res_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter == self.common_step_counter:
            return self._kick_motion_res_buffer # type: ignore
        
        self._kick_motion_res_counter = self.common_step_counter
        
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        offset = self._ref_offset()
        self._kick_motion_res_buffer = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        if self._ground_relative_ref:
            prev_motion_res = None
            if self._ref_velocity_correction:  # reference one control step earlier, for the offset rate of change
                prev_motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times - self.dt, offset=offset)
            self._apply_ground_relative_ref(self._kick_motion_res_buffer, prev_motion_res)

        return self._kick_motion_res_buffer
    
    _kick_motion_res_counter_multistep = -1
    _kick_motion_res_multistep_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res_multiplestep(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_multistep == self.common_step_counter:
            return self._kick_motion_res_multiplestep_buffer # type: ignore
        
        self._kick_motion_res_counter_multistep = self.common_step_counter
        buffer = {}

        for i in range(self.config.obs.future_ref_steps):
            motion_times = (self.episode_length_buf + 1 + i) * self.dt + self.motion_start_times
            offset = self._ref_offset()
            motion= self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            if self._ground_relative_ref:
                self._apply_ground_relative_ref(motion)
            for k, v in motion.items():
                if k not in buffer:
                    buffer[k] = []
                buffer[k].append(v.unsqueeze(1))

        self._kick_motion_res_multiplestep_buffer = {k: torch.cat(v, dim=1) for k, v in buffer.items()}  # Dict[str, Tensor[B, T, D]]
        return self._kick_motion_res_multiplestep_buffer


    # TimePortion: <12%
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        
        
        # # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        motion_res = self.kick_motion_res()
        if self._ground_relative_ref:
            self.ref_support_height[:] = motion_res['ref_support_height']
            self.ref_feet_ground_height[:] = motion_res['ref_feet_ground_height']
        if "future_ref_steps" in self.config.obs and self.config.obs.future_ref_steps > 0:
            future_motion_res = self.kick_motion_res_multiplestep()
            future_ref_joint_pos = future_motion_res["dof_pos"] # [num_envs, num_future_steps, num_dofs]
            future_ref_joint_vel = future_motion_res["dof_vel"] # [num_envs, num_future_steps, num_dofs]
            self._obs_future_ref_dof_pos = future_ref_joint_pos.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            self._obs_future_ref_dof_vel = future_ref_joint_vel.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            #  (Pdb) motion_res.keys()
            #  dict_keys(['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel', 'motion_aa', 'motion_bodies', 'rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel', 'rg_pos_t', 'rg_rot_t', 'body_vel_t', 'body_ang_vel_t'])
            
                # (Pdb) print( [(k,v.shape) for k,v in motion_res.items()  ]  )
                # [('root_pos', torch.Size([1, 3])), ('root_rot', torch.Size([1, 4])), ('dof_pos', torch.Size([1, 23])), ('root_vel', torch.Size([1, 3])), ('root_ang_vel', torch.Size([1, 3])), ('dof_vel', torch.Size([1, 23])), ('motion_aa', torch.Size([1, 72])), ('motion_bodies', torch.Size([1, 17])), ('rg_pos', torch.Size([1, 24, 3])), ('rb_rot', torch.Size([1, 24, 4])), ('body_vel', torch.Size([1, 24, 3])), ('body_ang_vel', torch.Size([1, 24, 3])), ('rg_pos_t', torch.Size([1, 27, 3])), ('rg_rot_t', torch.Size([1, 27, 4])), ('body_vel_t', torch.Size([1, 27, 3])), ('body_ang_vel_t', torch.Size([1, 27, 3]))]
        if self._motion_lib.has_contact_mask:
            self.ref_contact_mask = motion_res["contact_mask"]
            
            
        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend # for visualization and analysis
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_joint_pos = ref_joint_pos.clone()
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]


        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate( # XYZW  
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

        ################### EXTEND Rigid body Rotation #####################
        extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
        
        ################### EXTEND Rigid Body Angular Velocity #####################
        self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
    
        ################### EXTEND Rigid Body Linear Velocity #####################
        self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

        ################### Compute differences #####################

        ## diff compute - kinematic position
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        # import ipdb; ipdb.set_trace()
        ## diff compute - kinematic rotation
        self.dif_global_body_rot = ref_body_rot_extend - self._rigid_body_rot_extend
        ## diff compute - kinematic velocity
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        ## diff compute - kinematic angular velocity
        
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        # ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()



        
        ## diff compute - kinematic joint position
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel

        # print(self.dif_joint_angles.max(),self.dif_global_body_pos.norm(dim=-1).max())

        # breakpoint() ## Examine the data structure of ref state
        

        # marker_coords for visualization
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)

        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]

        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)

        heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)


        self.relyaw = self.rpy[:,2:3] - self.ref_init_rpy[0,2]
        relyaw_heading_inv_quat = calc_yaw_heading_quat_inv(self.relyaw)
        relyaw_heading_inv_quat_expand = relyaw_heading_inv_quat.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
        # print(self.relyaw[0])

        
        
        ########## Local Diff Rigid Body Pos ##########
            # Meaning: This section computes the local difference rigid body positions relative to the robot's root frame.
            # i.e. what's the diff of each rigid body's position in the local frame?
        dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
        
        self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)


        ########## Local Ref Rigid Body Pos ##########
            # Meaning: This section computes the local reference rigid body positions relative to the robot's root frame.
            # It first calculates the global positions relative to the root by subtracting the root position,
            # then transforms these positions into the local frame using the inverse heading rotation.
            # The result is stored in self._obs_local_ref_rigid_body_pos for observation purposes.
        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)




        global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_global_ref_rigid_body_vel = global_ref_body_vel.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        self._obs_local_ref_rigid_body_pos_relyaw = my_quat_rotate(relyaw_heading_inv_quat_expand.view(-1, 4), 
                                                                   global_ref_body_vel.view(-1, 3)).view(env_batch_size, -1)
        

        ######################VR 3 point ########################
        if not self.config.use_teleop_control:
            ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        
        
        
        
        #################### Deepmimic phase ###################### 

        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
            max_phase = self._ref_motion_phase.max()
            # import ipdb; ipdb.set_trace()
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
        # print(f"ref_motion_phase: {self._ref_motion_phase[0].item():.2f}")
        # print(f"ref_motion_length: {self._ref_motion_length[0].item():.2f}")
        
        self._log_motion_tracking_info()

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        # action noise
        # self.actions += torch.randn_like(self.actions) * 0.01
        # self.actions *= 1 + torch.randn_like(self.actions) * 0.01
        
        if 'only_track_leg' in self.config.robot and self.config.robot.only_track_leg:
            self.actions = torch.cat(self.actions, self.ref_joint_pos[:, 12:], dim=-1)
        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    # @time_prot_fn_dec_
    # TimePortion: 32%
    def _physics_step(self):
        """
            Add Soft Dynamic Correction : alpha

            For each sim step, we modify the current state toward the target state with a ratio of alpha
            alpha is predefined in the config
            
            Im:
                1. fetch ref state of next step at the beginning 
                2. for each substep:
                    a. fetch current state
                    b. compute the difference between current state and ref state
                    c. apply soft dynamic correction
                    d. run simulate
        
        """
        
        def _physics_step_SDC(self):
            assert self.config.simulator.config.name == "isaacgym", "Soft Dynamic Correction is only supported for Isaac Gym"
            
            # if self.simulator.robot_root_states[:, 2]>0.9:
            #     breakpoint()
            
            offset = self.env_origins # [N,3]
            # motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            motion_res = self.kick_motion_res()

            
            zero_ref_state = WholeBodyState(
                root_state=torch.cat([offset, torch.zeros((self.num_envs, 10), device=self.device)], dim=-1)+self.base_init_state,
                dof_pos=torch.tile(self.default_dof_pos, (self.num_envs, 1)),
                dof_vel=torch.zeros((self.num_envs, self.num_dofs), device=self.device)
            )
            
            ref_whole_body_state = WholeBodyState(
                root_state=torch.cat([motion_res['root_pos'], motion_res['root_rot'], motion_res['root_vel'], motion_res['root_ang_vel']], dim=-1),
                dof_pos=motion_res['dof_pos'],
                dof_vel=motion_res['dof_vel']
            )
            # ref_whole_body_state = zero_ref_state
            
            
            decimation = self.config.simulator.config.sim.control_decimation
            substep_alpha = (1-(1-self.sdc_alpha)**(1/decimation))
            
            # print(f"substep_alpha: {substep_alpha}")
            
            self.render()
            for _ in range(decimation):
                
                self.simulator.refresh_sim_tensors()
                current_state = WholeBodyState(
                    root_state=self.simulator.robot_root_states,
                    dof_pos=self.simulator.dof_pos,
                    dof_vel=self.simulator.dof_vel
                )
                
                current_state.interpolate(ref_whole_body_state, substep_alpha).apply(self.simulator)
                
                # breakpoint()
                
                self._apply_force_in_physics_step()
                self.simulator.simulate_at_each_physics_step()

            ...
            
        # root_state: [N,13], 13 = 3(pos) + 4(rot) + 3(vel) + 3(ang_vel)
        # print(f"Use Soft Dynamic Correction: {self.config.soft_dynamic_correction.enable}")
        if (not 'enable' in self.config.soft_dynamic_correction or 
            not self.config.soft_dynamic_correction.enable # clip it, for faster adaptability
            ):
            return super()._physics_step()
        else:
            if (    
                (self.config.soft_dynamic_correction.type == "prob" 
                    and np.random.rand() > np.sqrt(self.sdc_alpha) )
                or self.sdc_alpha < 0.005) :
                # print(f"SDC: use physics step, alpha: {self.sdc_alpha}, prob alpha: {np.sqrt(self.sdc_alpha)}")
                return super()._physics_step()
            # print(f"SDC: use SDC step, alpha: {self.sdc_alpha}")
            return _physics_step_SDC(self)
        
        # breakpoint()


    # @time_prot_fn_dec_
    # TimePortion: 42.5%
    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
            # print("DEBUG: motion_times",motion_times)
            if (len(self.motions_for_saving['dof'])) == self.config.save_total_steps+3:
                self.saved_motion_dict = {}
                for k, v in self.motions_for_saving.items():
                    self.saved_motion_dict[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                    print("DEBUG: ",k,self.saved_motion_dict[k].shape, self.saved_motion_dict[k].dtype)
                    
                if not self._custom_ref_offset: # otherwise the offset was already removed per frame
                    self.saved_motion_dict['root_trans_offset'] -= self.env_origins.cpu().numpy().reshape(-1, 1, 3)
                self.saved_motion_dict['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                num_frames = len(self.saved_motion_dict['dof'][0])
                keys_to_save = self.saved_motion_dict.keys()

                # motion_key = f"all_motion{self.num_envs}" 
                # dump_data[motion_key] = {
                #     key: self.saved_motion_dict[key] for key in keys_to_save
                # }
                # dump_data[motion_key]['fps'] = 1 / self.dt
                save_path = f"{self.save_motion_dir}_{self.num_envs }x{num_frames}-{self._motion_episode_length}.pkl"
                
                print("self._motion_episode_length=", self._motion_episode_length)
                # breakpoint()
                if self._write_to_file:
                    for i in range(num_motions):
                        motion_key = f"motion{i}" 
                        dump_data[motion_key] = {
                            key: self.saved_motion_dict[key][i] for key in keys_to_save
                        }
                        dump_data[motion_key]['fps'] = 1 / self.dt

                    joblib.dump(dump_data, save_path, compress=3)
                    
                    print(colored(f"Saved motion data to {save_path}", 'green'))
                else:
                    print(colored(f"Not saving motion data to {save_path}, because {self._write_to_file=}", 'red'))
                    
                    
                    
                    
            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self._custom_ref_offset:
                # express the root in the reference motion's frame (flat ground at z=0)
                ref_offset = self._ref_offset().clone()
                if self._ground_relative_ref:
                    ref_offset[:, 2] += self.ref_support_height
                root_trans = root_trans - ref_offset.cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw   
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            else:
                raise NotImplementedError
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float() # sRot.from_quat: need xyzw
            dof = self.simulator.dof_pos.cpu()
            # T, num_env, J, 3
            # print(self._motion_lib.mesh_parsers.dof_axis)
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            
            if torch.any(self.reset_buf.cpu()):
                print("DEBUG: Reset at ",len(self.motions_for_saving['dof']))
            # else:
            #     print("DEBUG: No reset at ",len(self.motions_for_saving['dof']),self.reset_buf,motion_times)
            
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            self.motions_for_saving['contact_mask'].append(self.contacts_filt.cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

    # ############################################################
        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos

    def _get_obs_local_ref_rigid_body_pos_relyaw(self):
        # print(self._obs_local_ref_rigid_body_pos_relyaw.mean(),self._obs_local_ref_rigid_body_pos_relyaw.std())
        return self._obs_local_ref_rigid_body_pos_relyaw

    def _get_obs_ref_motion_phase(self):
        # print(self._ref_motion_phase)
        return self._ref_motion_phase
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    def _get_obs_relyaw(self):
        # print(self.relyaw[0])
        return self.relyaw

    def _get_obs_dif_joint_angles(self):
        return self.dif_joint_angles

    def _get_obs_dif_joint_velocities(self):
        return self.dif_joint_velocities
    
    def _get_obs_local_ref_rigid_body_vel(self):
        return self._obs_local_ref_rigid_body_vel
    
    def _get_obs_global_ref_rigid_body_vel(self):
        return self._obs_global_ref_rigid_body_vel

    def _get_obs_indicator_guider(self):
        return torch.ones((self.num_envs, 1), device=self.device)  # Placeholder for indicator 

    def _get_obs_indicator_learner(self):
        return torch.zeros((self.num_envs, 1), device=self.device)
    
    def _get_obs_zero_vector(self):
        return torch.zeros((self.num_envs, self.config.obs.obs_dims.zero_vector), device=self.device) # zero vector for padding
    
    def _get_obs_future_ref_dof_pos(self):
        return self._obs_future_ref_dof_pos

    def _get_obs_future_ref_dof_vel(self):
        return self._obs_future_ref_dof_vel

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################
    
    def _init_adaptive_sigma(self):
        if not 'adaptive_tracking_sigma' in self.config.rewards or not self.config.rewards.adaptive_tracking_sigma.enable:
            self._update_adaptive_sigma = lambda *args, **kwargs: None
            logger.info("Adaptive tracking sigma Disabled")
            return
        logger.info("Adaptive tracking sigma Enabled")
        self._reward_error_ema = dict()
        for key, value in self.config.rewards.reward_tracking_sigma.items():
            self._reward_error_ema[key] = value
        
        ...
    
    def _update_adaptive_sigma(self, error:torch.Tensor, term:str):
        alpha = self.config.rewards.adaptive_tracking_sigma.alpha
        scale = (self.config.rewards.adaptive_tracking_sigma.scale 
                    if 'scale' in self.config.rewards.adaptive_tracking_sigma else 1.0)
        adptype = (self.config.rewards.adaptive_tracking_sigma.type 
                    if 'type' in self.config.rewards.adaptive_tracking_sigma else "origin")
        
        self._reward_error_ema[term] = self._reward_error_ema[term] * (1-alpha) + error.mean().item() * alpha
        
        if adptype == "scale":
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term] * scale, 
                                                                self.config.rewards.reward_tracking_sigma[term])
        elif adptype == "mean":
            self.config.rewards.reward_tracking_sigma[term] = (min(self._reward_error_ema[term], 
                                                                self.config.rewards.reward_tracking_sigma[term]) +
                                                                self._reward_error_ema[term]) / 2
        elif adptype == "origin":
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term], 
                                                                self.config.rewards.reward_tracking_sigma[term])
        
        
        # print("DEBUG: Adaptive sigma update, term: ", term, '\t sigma:', self.config.rewards.reward_tracking_sigma[term])
        ...
    
    _pos_error_list = []
    def _print_tracking(self):
        
        # 
        # velocity_diff = self.dif_global_body_vel.reshape(1, -1)
        # cur_velocity = self._rigid_body_vel_extend.reshape(1, -1)
        # ref_velocity = velocity_diff + cur_velocity
        
        # velocity_diff = self.dif_joint_velocities
        # cur_velocity = self.simulator.dof_vel
        # ref_velocity = velocity_diff + cur_velocity
        # 
        # potential_velocity = radial_velocity_potential(cur_velocity, ref_velocity)
        # print(f'{potential_velocity=}','\t|')
        
        # sphere_distance = cur_velocity.reshape( -1).dot(ref_velocity.reshape( -1))/(ref_velocity.norm(dim=-1)**2)
        
        # cosine_similarity = torch.nn.functional.cosine_similarity(cur_velocity.reshape(1, -1), ref_velocity.reshape(1, -1), dim=-1)
        # 
        # cosine_error = 1-cosine_similarity
        # 
        # potential_cosine = torch.exp(-cosine_error / 0.5).item()
        # # print(f'{potential_cosine=}','\t|',f'{cosine_error=}')
        # 
        # norm_cur_velocity = cur_velocity.norm(dim=-1)
        # norm_ref_velocity = ref_velocity.norm(dim=-1)
        # 
        # norm_error = torch.abs(torch.log(norm_cur_velocity/norm_ref_velocity))
        # 
        # potential_norm = torch.exp(-norm_error / 5.0).item()
        # 
        # 
        # # print(f'{potential_velocity=}','\t|',f'{norm_error=}')        
        # 
        # potential_velocity = potential_cosine * potential_norm
        
        # radial_alpha =  (torch.atan(norm_ref_velocity/10)/torch.pi*2).item()
        
        # print(f'{radial_alpha=:.4f}','\t|',f'{norm_ref_velocity.item()=:.4f}')
        # print(f'{potential_velocity=}','\t|')
        # print(f'{potential_velocity=}','\t|',f'{cosine_error=}','\t|',f'{norm_error=}','\t|',f'{velocity_diff.norm(dim=-1).item()=}')

        # print('cosine_similarity | ',co)
        
        
        # joint_vel_diff = self.dif_joint_velocities
        # diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        # # r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        # # print('r_joint_vel',r_joint_vel.max(dim=-1)[0])
        # print('joint_vel_diff norm',joint_vel_diff.norm(dim=-1).max(dim=-1)[0])
        
        
        # self._pos_error_list.append(self.dif_joint_angles.norm(dim=-1))
        # # print('int pos_error',self._int_pos_error.norm(dim=-1).max(dim=-1)[0])
        # print('mean pos_error',torch.cat(self._pos_error_list, dim=-1).mean(), 'std pos_error',torch.cat(self._pos_error_list, dim=-1).std())
        
        # joint_pos_diff = self.dif_joint_angles
        # joint_pos_diff_dist = (joint_pos_diff**2).mean(dim=-1)
        # self._pos_error_list.append(joint_pos_diff_dist)
        # print('mean pos_error',torch.cat(self._pos_error_list, dim=-1).mean(), 'std pos_error',torch.cat(self._pos_error_list, dim=-1).std())
        
        
        # rotation_diff = self.dif_global_body_rot
        # # diff_body_rot_dist = (rotation_diff**2).mean(dim=-1).mean(dim=-1)
        # print('diff_body_rot_dist',rotation_diff.norm(dim=-1).norm(dim=-1).max(dim=-1)[0])
        # self._pos_error_list.append(diff_body_rot_dist)
        # print('mean pos_error',torch.cat(self._pos_error_list, dim=-1).mean(), 'std pos_error',torch.cat(self._pos_error_list, dim=-1).std())
        
        
        # 
        # velocity_diff = self.dif_global_body_vel    
        # cur_velocity = self._rigid_body_vel_extend
        # ref_velocity = velocity_diff + cur_velocity
        # 
        # cosine_similarity = torch.nn.functional.cosine_similarity(cur_velocity.reshape(1, -1), ref_velocity.reshape(1, -1), dim=-1)
        # print('cosine_similarity | ',cosine_similarity)
        #     # 0.8->0.5
        
        
        
        # diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        # # print(f"{diff_body_vel_dist=}")
        # r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        # 
        # print('velocity diff norm',velocity_diff.norm(dim=-1).norm(dim=-1).max(dim=-1)[0])
        
        # joint_pos_diff:torch.Tensor = self.dif_joint_angles
        # max_diff_joint_pos:torch.Tensor = (joint_pos_diff.abs()).max(dim=-1)[0]
        # r_max_joint_pos = torch.exp(-max_diff_joint_pos / 0.3)
        # 
        # print(f"{max_diff_joint_pos=} \t| {r_max_joint_pos=}")
        
        
        # print('terminate threshold',torch.norm(self.dif_global_body_pos, dim=-1).max(dim=-1)[0])
        # print('terminate',torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1).item())
            # max diff joint pos: 0.2->0.5

        # print('dof pos threshold',torch.norm(self.dif_joint_angles, dim=-1).max(dim=-1)[0])
        ...

    def _reward_teleop_contact_mask(self):
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 1 - error_contact_mask.mean(dim=-1)
        # print(f"{rew.mean()=} | {rew.std()=} | {error_contact_mask.mean()=} | {error_contact_mask.std()=}")
        return rew
    
    
    def _reward_teleop_contact_mask_v2(self):
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 0.5 - error_contact_mask.mean(dim=-1)
        # print(f"{rew.mean()=} | {rew.std()=} | {error_contact_mask.mean()=} | {error_contact_mask.std()=}")
        return rew
    
    
    
    def _reward_teleop_body_position_extend(self):
        # self._print_tracking()
        
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{diff_body_pos_dist_upper=}")
        
        # print(f"{lower_body_diff.norm(dim=-1).norm(dim=-1).item()=:.6f}") 
        
        # max_body_pos = self.dif_global_body_pos.norm(dim=-1).max(dim=-1)[0]
        # print(f"{max_body_pos=}")
        
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
    
        self._update_adaptive_sigma(diff_body_pos_dist_upper, 'teleop_upper_body_pos')
        self._update_adaptive_sigma(diff_body_pos_dist_lower, 'teleop_lower_body_pos')

        return r_body_pos
    
    def _reward_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{vr_3point_dist=}")
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        
        self._update_adaptive_sigma(vr_3point_dist, 'teleop_vr_3point_pos')
        return r_vr_3point

    def _reward_teleop_body_position_feet(self):

        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        if self._ground_relative_ref:
            # compare foot heights above the local ground instead of world z
            robot_feet = self.simulator._rigid_body_pos[:, self.feet_indices]
            robot_clearance = robot_feet[..., 2] - self._feet_ground_height(robot_feet[..., :2])
            ref_clearance = self.ref_body_pos_extend[:, self.feet_indices, 2] - self.ref_feet_ground_height
            feet_diff = feet_diff.clone()
            feet_diff[..., 2] = ref_clearance - robot_clearance
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        # feet_dist_feet = (feet_diff**2).mean(dim=-1)
        # print(f"{feet_dist=} | {feet_dist_feet.max()=} | {feet_dist_feet.min()=}")
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        
        self._update_adaptive_sigma(feet_dist, 'teleop_feet_pos')
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        rotation_diff = self.dif_global_body_rot
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        
        self._update_adaptive_sigma(diff_body_rot_dist, 'teleop_body_rot')
        return r_body_rot

    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{diff_body_vel_dist=}")
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        
        self._update_adaptive_sigma(diff_body_vel_dist, 'teleop_body_vel')
        return r_body_vel
    
    def _reward_teleop_radial_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel.reshape(self.num_envs, -1)
        cur_velocity = self._rigid_body_vel_extend.reshape(self.num_envs, -1)
        ref_velocity = velocity_diff + cur_velocity
        
        potential_velocity = radial_velocity_potential(cur_velocity, ref_velocity)
        return potential_velocity
    
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{diff_body_ang_vel_dist=}")
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        
        self._update_adaptive_sigma(diff_body_ang_vel_dist, 'teleop_body_ang_vel')
        return r_body_ang_vel

    def _reward_teleop_max_joint_position(self):
        joint_pos_diff:torch.Tensor = self.dif_joint_angles
        max_diff_joint_pos:torch.Tensor = (joint_pos_diff.abs()).max(dim=-1)[0]
        r_max_joint_pos = torch.exp(-max_diff_joint_pos / self.config.rewards.reward_tracking_sigma.teleop_max_joint_pos)
        
        self._update_adaptive_sigma(max_diff_joint_pos, 'teleop_max_joint_pos')
        # print(f"{max_diff_joint_pos=} \t| {r_max_joint_pos=}")
            # max diff joint pos: 0.2->0.5
        return r_max_joint_pos

    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        if self._leg_joint_pos_weight == 1.0:
            diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        else:
            diff_joint_pos_dist = ((joint_pos_diff**2) * self._joint_pos_weights).sum(dim=-1) / self._joint_pos_weights.sum()

        # diff_joint = (joint_pos_diff**2)
        # print(f"{diff_joint_pos_dist=} \t| {diff_joint.max()=} \t| {diff_joint.min()=}")
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        
        self._update_adaptive_sigma(diff_joint_pos_dist, 'teleop_joint_pos')
        # self._reward_feet_air_time() #DEBUG:
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        
        self._update_adaptive_sigma(diff_joint_vel_dist, 'teleop_joint_vel')
        # print(f"{diff_joint_vel_dist.item()=} \t | {r_joint_vel=} \t | {self.dif_joint_velocities.abs().max(dim=-1)[0]}")
        return r_joint_vel
    
    def _reward_teleop_radial_joint_velocity(self):
        velocity_diff = self.dif_joint_velocities
        cur_velocity = self.simulator.dof_vel
        ref_velocity = velocity_diff + cur_velocity
        
        potential_velocity = radial_velocity_potential(cur_velocity, ref_velocity)
        return potential_velocity
    
    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
    


    ## exbody2 rewards, found in envs.locomotion
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - self.config.rewards.desired_feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        # print("Rew air time: ", rew_airTime)
        return rew_airTime
    
    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)
    
    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)