"""Whole-body differential IK for the PBHC G1 23-DoF model (MuJoCo, kinematics only).

Each frame is solved as a weighted least-squares problem over the 29-dim qvel space
(6 floating base + 23 joints):

    min_dq  sum_i w_i ||J_i dq - e_i||^2 + lam ||dq||^2 + mu ||q_j + dq_j - q_nom||^2

Tasks: point positions (hands, elbows, shin, ...), body orientations (feet, pelvis,
torso), the whole-body centre of mass, and a weak posture prior. Joint limits are
enforced by clamping after every step.
"""
from __future__ import annotations

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

# PBHC limits from humanoidverse/config/robot/g1/g1_23dof_lock_wrist.yaml
DOF_LOWER = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
                      -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
                      -2.618, -0.52, -0.52,
                      -3.0892, -1.5882, -2.618, -1.0472,
                      -3.0892, -2.2515, -2.618, -1.0472])
DOF_UPPER = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
                      2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
                      2.618, 0.52, 0.52,
                      2.6704, 2.2515, 2.618, 2.0944,
                      2.6704, 1.5882, 2.618, 2.0944])
DEFAULT_DOF = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0,
                        0, 0, 0, 0.2, 0.2, 0, 0.9, 0.2, -0.2, 0, 0.9])

# Matches PBHC's extend_config: hand = elbow_link + [0.25, 0, 0], head = torso_link + [0, 0, 0.42]
HAND_OFFSET = np.array([0.25, 0.0, 0.0])
HEAD_OFFSET = np.array([0.0, 0.0, 0.42])
# Ankle-roll-link origin sits this far above the sole when the foot is flat (measured from the mesh).
ANKLE_HEIGHT = 0.0354


class G1Kinematics:
    def __init__(self, xml_path: str):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        assert self.m.nq == 30 and self.m.nv == 29, "expected the 23-DoF lock-wrist G1"
        names = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        assert names == DOF_NAMES, "joint order differs from PBHC dof_names"
        self.bid = {self.m.body(i).name: i for i in range(self.m.nbody)}
        self.lo = DOF_LOWER + 0.03
        self.hi = DOF_UPPER - 0.03
        # point on the right shin used as the calf-kick striking surface (75% of the way knee->ankle)
        self.shin_local = 0.75 * self.m.body_pos[self.bid["right_ankle_pitch_link"]].copy()
        self._jp = np.zeros((3, self.m.nv))
        self._jr = np.zeros((3, self.m.nv))
        # self-collision avoidance between limbs (pairs and the clearance they must keep)
        self.m.geom_margin[:] = 0.04
        grp = {}
        for side in ("left", "right"):
            for b in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
                grp[self.bid[f"{side}_{b}_link"]] = f"{side}_arm"
            for b in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
                grp[self.bid[f"{side}_{b}_link"]] = f"{side}_leg"
        for b in ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"):
            grp[self.bid[b]] = "trunk"
        forearm = {self.bid[f"{s}_{b}_link"] for s in ("left", "right") for b in ("shoulder_yaw", "elbow")}
        self._pair_clear = {}
        for b1, g1 in grp.items():
            for b2, g2 in grp.items():
                if b1 >= b2 or g1 == g2:
                    continue
                gs = {g1, g2}
                if gs == {"left_arm", "right_arm"}:
                    self._pair_clear[(b1, b2)] = 0.02
                elif gs == {"left_leg", "right_leg"}:
                    self._pair_clear[(b1, b2)] = 0.02
                elif "trunk" in gs and (b1 in forearm or b2 in forearm):
                    self._pair_clear[(b1, b2)] = 0.008
                elif ("trunk" in gs) and (gs & {"left_leg", "right_leg"}) and \
                        self.m.body(b1 if grp[b1] != "trunk" else b2).name.split("_")[1] in ("knee", "ankle"):
                    self._pair_clear[(b1, b2)] = 0.01
                elif (gs & {"left_arm", "right_arm"}) and (gs & {"left_leg", "right_leg"}):
                    self._pair_clear[(b1, b2)] = 0.02

    # ---------- forward kinematics helpers ----------
    def set_q(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def body_point(self, body, local):
        b = self.d.body(body)
        return b.xpos + b.xmat.reshape(3, 3) @ local

    def hand(self, side):
        return self.body_point(f"{side}_elbow_link", HAND_OFFSET)

    def head(self):
        return self.body_point("torso_link", HEAD_OFFSET)

    def shin(self):
        return self.body_point("right_knee_link", self.shin_local)

    def com(self):
        return self.d.subtree_com[1].copy()

    def rot(self, body):
        return self.d.body(body).xmat.reshape(3, 3).copy()

    # ---------- jacobians ----------
    def _point_jac(self, body, world_point):
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr, world_point, self.bid[body])
        return self._jp.copy()

    def _rot_jac(self, body):
        mujoco.mj_jacBody(self.m, self.d, self._jp, self._jr, self.bid[body])
        return self._jr.copy()

    def _com_jac(self):
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jac, 1)
        return jac

    # ---------- solver ----------
    def collision_rows(self, w=200.0):
        """Rows pushing apart limb pairs closer than their clearance (needs mj_kinematics done)."""
        mujoco.mj_collision(self.m, self.d)
        rows, errs = [], []
        worst = 0.0
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            key = (b1, b2) if b1 < b2 else (b2, b1)
            clr = self._pair_clear.get(key)
            if clr is None or c.dist >= clr:
                continue
            n = c.frame[:3]
            J1 = self._point_jac(self.m.body(b1).name, c.pos)
            J2 = self._point_jac(self.m.body(b2).name, c.pos)
            rows.append(np.sqrt(w) * (n @ (J2 - J1))[None])
            errs.append(np.sqrt(w) * np.array([clr - c.dist]))
            worst = max(worst, clr - c.dist)
        return rows, errs, worst

    def min_clearance(self):
        """Smallest (distance - clearance) over watched limb pairs; <0 means too close."""
        mujoco.mj_collision(self.m, self.d)
        best = np.inf
        for c in self.d.contact[: self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            key = (b1, b2) if b1 < b2 else (b2, b1)
            if key in self._pair_clear:
                best = min(best, c.dist)
        return best

    def solve(self, q0, targets, iters=20, q_nom=None, w_post=0.02, lam=1e-3, tol=1e-5, max_step=0.1,
              q_prev=None, w_prev=0.0, avoid=True):
        """targets: list of task dicts
            {'kind':'point','body':..,'local':..,'pos':..,'w':..}
            {'kind':'rot','body':..,'R':3x3,'w':..}
            {'kind':'com','pos':xyz or xy,'w':..}
        """
        q = q0.copy()
        q_nom = DEFAULT_DOF if q_nom is None else q_nom
        nv = self.m.nv
        for _ in range(iters):
            self.set_q(q)
            rows, errs = [], []
            for t in targets:
                k, w = t["kind"], np.sqrt(t["w"])
                if k == "point":
                    p = self.body_point(t["body"], t["local"])
                    rows.append(w * self._point_jac(t["body"], p))
                    errs.append(w * (t["pos"] - p))
                elif k == "rot":
                    Rc = self.rot(t["body"])
                    e = R.from_matrix(t["R"] @ Rc.T).as_rotvec()
                    rows.append(w * self._rot_jac(t["body"]))
                    errs.append(w * e)
                elif k == "pz":
                    p = self.d.body("pelvis").xpos
                    rows.append(w * self._point_jac("pelvis", p)[2:3])
                    errs.append(w * np.array([t["pos"] - p[2]]))
                elif k == "com":
                    J = self._com_jac()
                    c = self.com()
                    n = len(t["pos"])
                    rows.append(w * J[:n])
                    errs.append(w * (t["pos"] - c[:n]))
            # posture prior on joints only
            Jp = np.zeros((23, nv)); Jp[:, 6:] = np.eye(23)
            rows.append(np.sqrt(w_post) * Jp)
            errs.append(np.sqrt(w_post) * (q_nom - q[7:]))
            if avoid:
                cr, ce, _ = self.collision_rows()
                rows += cr; errs += ce
            if q_prev is not None and w_prev > 0:
                # temporal smoothness: stay close to the previous frame's joint angles
                rows.append(np.sqrt(w_prev) * Jp)
                errs.append(np.sqrt(w_prev) * (q_prev[7:] - q[7:]))
            A = np.vstack(rows); b = np.concatenate(errs)
            # active set: joints sitting on a limit and pushed further are frozen
            free = np.ones(nv, bool)
            for _pass in range(3):
                Af = A[:, free]
                H = Af.T @ Af + lam * np.eye(free.sum())
                dq = np.zeros(nv)
                dq[free] = np.linalg.solve(H, Af.T @ b)
                qj = q[7:] + dq[6:]
                viol = ((qj < self.lo - 1e-9) & (dq[6:] < 0)) | ((qj > self.hi + 1e-9) & (dq[6:] > 0))
                if not viol.any():
                    break
                at_lo = (q[7:] <= self.lo + 1e-6) & (dq[6:] < 0)
                at_hi = (q[7:] >= self.hi - 1e-6) & (dq[6:] > 0)
                newly = (at_lo | at_hi) & free[6:]
                if not newly.any():
                    # scale the step so no joint crosses its limit
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r_lo = np.where(dq[6:] < 0, (self.lo - q[7:]) / dq[6:], np.inf)
                        r_hi = np.where(dq[6:] > 0, (self.hi - q[7:]) / dq[6:], np.inf)
                    alpha = np.clip(np.nanmin(np.r_[r_lo, r_hi, 1.0]), 0.0, 1.0)
                    dq *= alpha
                    break
                free[6:][newly] = False
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            mujoco.mj_integratePos(self.m, q, dq, 1.0)
            q[7:] = np.clip(q[7:], self.lo, self.hi)
            if np.linalg.norm(dq) < tol:
                break
        self.set_q(q)
        return q


def yaw_R(yaw, pitch=0.0, roll=0.0):
    return R.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
