"""Procedural choreography for eight attacks thrown by an orthodox (left-foot-forward) G1.

World frame of every clip (same as the PBHC motion file):
  * the attacker starts at the origin, the opponent stands along +x, z is up;
  * the canonical defender is a G1 in the same guard stance placed at (D, 0) facing -x.

Every attack is a sequence of phases built from minimum-jerk Tracks (see tracks.py):
  guard (random onset) -> chamber -> extend (ends at impact) -> brief hold -> retract -> guard.
The task-space targets are then turned into joint trajectories by whole-body IK (ik.py)
with both feet pinned (or the kicking foot free during a calf kick) and the centre of mass
held inside the support polygon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .ik import ANKLE_HEIGHT, HAND_OFFSET, G1Kinematics, yaw_R
from .tracks import Track

DEG = np.pi / 180.0

ATTACKS = [
    "jab", "cross", "lead_hook", "rear_hook",
    "lead_uppercut", "rear_uppercut", "body_shot", "calf_kick",
]
ATTACK_ID = {a: i for i, a in enumerate(ATTACKS)}

# Range (pelvis-to-pelvis distance D, metres) each attack is thrown from.
D_RANGE = {
    "jab": (0.45, 0.51), "cross": (0.42, 0.48),
    "lead_hook": (0.41, 0.46), "rear_hook": (0.39, 0.44),
    "lead_uppercut": (0.39, 0.44), "rear_uppercut": (0.38, 0.43),
    "body_shot": (0.38, 0.43), "calf_kick": (0.48, 0.55),
}

# Closest the defender may be placed (pelvis to pelvis). Below this the two robots' trunks get
# within ~20 cm of each other and their guards overlap.
D_MIN = {a: (0.46 if a == "calf_kick" else 0.36) for a in D_RANGE}

# Which body point lands the blow.
STRIKER = {
    "jab": "left_hand", "cross": "right_hand", "lead_hook": "left_hand", "rear_hook": "right_hand",
    "lead_uppercut": "left_hand", "rear_uppercut": "right_hand", "body_shot": "left_hand",
    "calf_kick": "right_shin",
}

# Target on the *defender*, as (link, offset in that link's frame). The defender is a G1
# facing the attacker, so his local +x points at the attacker and his local -y is on the
# attacker's left (+y world).
TARGET = {
    "jab": ("torso_link", [0.075, 0.0, 0.42]),              # face
    "cross": ("torso_link", [0.075, 0.0, 0.42]),            # face
    "lead_hook": ("torso_link", [0.02, -0.075, 0.42]),      # defender's right temple
    "rear_hook": ("torso_link", [0.02, 0.075, 0.42]),       # defender's left temple
    "lead_uppercut": ("torso_link", [0.065, -0.01, 0.35]),  # chin
    "rear_uppercut": ("torso_link", [0.065, 0.01, 0.35]),   # chin
    "body_shot": ("torso_link", [0.06, -0.085, 0.13]),      # liver (defender's right side)
    "calf_kick": ("left_knee_link", None),                  # outside of the lead calf, set below
}
TARGET_REGION = {
    "jab": "head", "cross": "head", "lead_hook": "head", "rear_hook": "head",
    "lead_uppercut": "head", "rear_uppercut": "head", "body_shot": "body", "calf_kick": "leg",
}


@dataclass
class Stance:
    stance_len: float = 0.34
    stance_wid: float = 0.23
    lead_foot_yaw: float = -12 * DEG
    rear_foot_yaw: float = -45 * DEG
    blade: float = -25 * DEG
    pelvis_z: float = 0.715
    torso_pitch: float = 6 * DEG
    guard_noise: np.ndarray = field(default_factory=lambda: np.zeros(12))
    sway_amp: float = 0.006
    sway_freq: float = 1.2
    sway_phase: float = 0.0

    @staticmethod
    def sample(rng, kick=False):
        if kick:  # a shorter, narrower stance lets the weight get over the lead foot
            st = Stance.sample(rng)
            st.stance_len = rng.uniform(0.25, 0.29)
            st.stance_wid = rng.uniform(0.17, 0.20)
            return st
        return Stance(
            stance_len=rng.uniform(0.31, 0.37),
            stance_wid=rng.uniform(0.21, 0.25),
            lead_foot_yaw=rng.uniform(-18, -6) * DEG,
            rear_foot_yaw=rng.uniform(-52, -38) * DEG,
            blade=rng.uniform(-30, -20) * DEG,
            pelvis_z=rng.uniform(0.70, 0.73),
            torso_pitch=rng.uniform(3, 9) * DEG,
            guard_noise=rng.uniform(-0.015, 0.015, 12),
            sway_amp=rng.uniform(0.003, 0.009),
            sway_freq=rng.uniform(0.9, 1.6),
            sway_phase=rng.uniform(0, 2 * np.pi),
        )


def _body(st: Stance, p):
    """Point given in the bladed body frame -> world."""
    return yaw_R(st.blade) @ np.asarray(p, float)


def guard_targets(st: Stance):
    dz = st.pelvis_z - 0.73
    n = st.guard_noise
    g = {
        "lf": np.array([st.stance_len / 2, st.stance_wid / 2, ANKLE_HEIGHT]),
        "rf": np.array([-st.stance_len / 2, -st.stance_wid / 2, ANKLE_HEIGHT]),
        "lh": _body(st, [0.235, 0.09, 1.11 + dz]) + n[0:3],
        "rh": _body(st, [0.175, -0.06, 1.10 + dz]) + n[3:6],
        "le": _body(st, [0.07, 0.16, 0.91 + dz]) + n[6:9] * 0.5,
        "re": _body(st, [0.02, -0.15, 0.91 + dz]) + n[9:12] * 0.5,
        "com": np.array([-0.01, -0.005]),
        "pz": np.array([st.pelvis_z]),
    }
    return g


class Choreo:
    """All task-space tracks of one clip, initialised to the guard."""

    def __init__(self, st: Stance, g, t0=0.0):
        self.st = st
        T = lambda v: Track(t0, v)
        self.lh, self.rh, self.le, self.re = T(g["lh"]), T(g["rh"]), T(g["le"]), T(g["re"])
        self.com, self.pz = T(g["com"]), T(g["pz"])
        self.pyaw, self.proll = T(st.blade), T(0.0)
        self.tyaw, self.tpitch, self.troll = T(st.blade), T(st.torso_pitch), T(0.0)
        self.lf, self.rf = T(g["lf"]), T(g["rf"])
        self.lfyaw, self.rfyaw = T(st.lead_foot_yaw), T(st.rear_foot_yaw)
        self.rfpitch, self.rfroll = T(0.0), T(0.0)
        self.rknee = T(np.zeros(3))
        self.w_rknee = T(0.0)
        self.w_prot = T(6.0)    # pelvis orientation weight
        self.w_trot = T(6.0)    # torso orientation weight
        self.w_le = T(1.0)
        self.w_re = T(1.0)
        self.rf_contact_off = []  # list of (t_lift, t_touchdown)
        self.g = g
        self.events = {}
        self.cursor = t0
        self.w_com = 400.0

    def tracks(self):
        return [v for v in self.__dict__.values() if isinstance(v, Track)]

    def hold_all(self, t):
        for tr in self.tracks():
            if tr.t_end < t - 1e-9:
                tr.hold(t)

    def at(self, t, start=None, **kv):
        """Move the named tracks to new values, arriving at time t.

        Each named track starts moving at `start` (default: the time of the previous `at`
        call), holding its previous value until then. Tracks that are not named keep
        following their own keyframes, so e.g. the centre of mass can drift smoothly
        across several strike phases instead of stopping at every phase boundary.
        """
        s = self.cursor if start is None else start
        for name, v in kv.items():
            tr = getattr(self, name)
            if tr.t_end < s - 1e-9:
                tr.hold(s)
            if isinstance(v, tuple):
                tr.to(t, v[0], vias=v[1])
            else:
                tr.to(t, v)
        self.cursor = max(self.cursor, t)

    def move(self, name, t_start, t_end, v):
        """Independent keyframe for one track (does not touch the phase cursor)."""
        tr = getattr(self, name)
        if tr.t_end < t_start - 1e-9:
            tr.hold(t_start)
        tr.to(t_end, v)

    def rf_contact(self, t):
        return 0.0 if any(a <= t < b for a, b in self.rf_contact_off) else 1.0

    def sway(self, t):
        st = self.st
        ph = 2 * np.pi * st.sway_freq * t + st.sway_phase
        return st.sway_amp * np.array([np.sin(ph), 0.4 * np.sin(0.5 * ph + 1.0), 0.5 * np.sin(2 * ph)])

    def ik_targets(self, kin: G1Kinematics, t):
        s = self.sway(t)
        tg = []
        lf, rf = self.lf(t), self.rf(t)
        c = self.rf_contact(t)
        wl = 1000 if c else 1500   # planted feet must not slide
        tg.append(dict(kind="point", body="left_ankle_roll_link", local=np.zeros(3), pos=lf, w=wl))
        tg.append(dict(kind="rot", body="left_ankle_roll_link", R=yaw_R(self.lfyaw(t)[0]), w=wl / 2))
        rR = yaw_R(self.rfyaw(t)[0], self.rfpitch(t)[0], self.rfroll(t)[0])
        tg.append(dict(kind="point", body="right_ankle_roll_link", local=np.zeros(3), pos=rf, w=1000 if c else 300))
        tg.append(dict(kind="rot", body="right_ankle_roll_link", R=rR, w=200 if c else 20))
        wk = self.w_rknee(t)[0]
        if wk > 1e-3:
            tg.append(dict(kind="point", body="right_knee_link", local=np.zeros(3), pos=self.rknee(t), w=10 * wk))
        tg.append(dict(kind="com", pos=self.com(t) + s[:2], w=self.w_com if c else 900.0))
        tg.append(dict(kind="pz", pos=self.pz(t)[0] + s[2], w=8))
        tg.append(dict(kind="rot", body="pelvis", R=yaw_R(self.pyaw(t)[0], 0.0, self.proll(t)[0]),
                       w=self.w_prot(t)[0]))
        tg.append(dict(kind="rot", body="torso_link",
                       R=yaw_R(self.tyaw(t)[0], self.tpitch(t)[0], self.troll(t)[0]), w=self.w_trot(t)[0]))
        bob = 0.6 * s
        tg.append(dict(kind="point", body="left_elbow_link", local=HAND_OFFSET, pos=self.lh(t) + bob, w=40))
        tg.append(dict(kind="point", body="right_elbow_link", local=HAND_OFFSET, pos=self.rh(t) + bob, w=40))
        tg.append(dict(kind="point", body="left_elbow_link", local=np.zeros(3), pos=self.le(t) + bob, w=3 * self.w_le(t)[0]))
        tg.append(dict(kind="point", body="right_elbow_link", local=np.zeros(3), pos=self.re(t) + bob, w=3 * self.w_re(t)[0]))
        return tg


# --------------------------------------------------------------------------------------
# defender (target) model
# --------------------------------------------------------------------------------------
class Defender:
    """Canonical defender: G1 in the default guard, placed at (D, 0) facing the attacker."""

    def __init__(self, kin: G1Kinematics, q_guard):
        self.kin = kin
        self.q_local = q_guard.copy()
        kin.set_q(q_guard)
        calf_local = 0.45 * kin.m.body_pos[kin.bid["left_ankle_pitch_link"]] + np.array([0.0, 0.045, 0.0])
        self.target_local = {a: (TARGET[a][0], np.array(TARGET[a][1] if TARGET[a][1] is not None else calf_local))
                             for a in ATTACKS}

    def q_world(self, D):
        """Defender qpos when placed at (D,0) facing -x (rotation by pi about z)."""
        from scipy.spatial.transform import Rotation as R
        q = self.q_local.copy()
        Rz = R.from_euler("z", np.pi)
        q[:3] = Rz.apply(q[:3]) + np.array([D, 0.0, 0.0])
        w, x, y, z = q[3:7]
        qr = (Rz * R.from_quat([x, y, z, w])).as_quat()
        q[3:7] = [qr[3], qr[0], qr[1], qr[2]]
        return q

    def target_world(self, attack, D):
        link, local = self.target_local[attack]
        self.kin.set_q(self.q_world(D))
        return self.kin.body_point(link, local)


# --------------------------------------------------------------------------------------
# attack choreographies
# --------------------------------------------------------------------------------------
def _dirn(a, b):
    v = np.asarray(b, float) - np.asarray(a, float)
    return v / np.linalg.norm(v)


def build_attack(name, st: Stance, g, target, rng, onset, speed, tail):
    """Returns (Choreo, info). `target` is the world point to hit; `speed` scales all strike timings."""
    ch = Choreo(st, g)
    b = st.blade
    k = 1.0 / speed
    t = onset
    ch.cursor = t
    info = {}
    jit = lambda lo, hi: rng.uniform(lo, hi)

    if name in ("jab", "cross"):
        lead = name == "jab"
        H, E = ("lh", "le") if lead else ("rh", "re")
        start = getattr(ch, H).last
        d = _dirn(start, target)
        impact = target + 0.06 * d                          # aim through the target: the fist is still
        elbow = impact - 0.26 * d + np.array([0, 0, -0.03])  # fast when it crosses it
        tc, te, th, tr = t + 0.05 * k, t + (0.20 if lead else 0.25) * k, 0.03, (0.18 if lead else 0.24) * k
        if lead:
            ch.cursor = tc
            ch.at(te, lh=impact, le=elbow, tyaw=b - jit(10, 16) * DEG, pyaw=b - 5 * DEG,
                  tpitch=st.torso_pitch + 8 * DEG)
            dcom = np.array([0.006, 0.002])
        else:
            ch.at(tc, pyaw=b - 4 * DEG, tyaw=b - 6 * DEG)
            ch.at(te, rh=impact, re=elbow, tyaw=b + jit(50, 60) * DEG, pyaw=b + jit(26, 34) * DEG,
                  tpitch=st.torso_pitch + 10 * DEG, troll=-5 * DEG,
                  lh=_body(st, [0.16, 0.13, 1.12 + st.pelvis_z - 0.73]))
            dcom = np.array([0.01, 0.006])
        t_imp = te
        ch.cursor = te + th
        t_end = te + th + tr
        ch.at(t_end, **{H: g[H], E: g[E]}, tyaw=b, pyaw=b, tpitch=st.torso_pitch, troll=0.0,
              **({} if lead else {"lh": g["lh"]}))
        ch.move("com", t, te + 0.04, g["com"] + dcom)
        ch.move("com", te + 0.04, t_end + 0.12, g["com"])

    elif name in ("lead_hook", "rear_hook", "body_shot"):
        lead = name != "rear_hook"
        body = name == "body_shot"
        sgn = 1.0 if lead else -1.0          # side the hand swings in from (+y for the lead hand)
        H, E = ("lh", "le") if lead else ("rh", "re")
        dz = st.pelvis_z - 0.73
        tc = t + (0.16 if body else 0.10) * k
        te = tc + (0.24 if body else 0.22) * k
        tr = (0.26 if body else 0.24) * k
        # chamber: load the hips the opposite way, lift the elbow, (body shot) change levels
        cham_h = getattr(ch, H).last + np.array([-0.02, 0.05 * sgn, -0.12 if body else 0.0])
        cham_e = getattr(ch, E).last + np.array([-0.03, 0.06 * sgn, 0.05 if not body else -0.04])
        ch.at(tc, **{H: cham_h, E: cham_e},
              tyaw=b + sgn * 8 * DEG, pyaw=b + sgn * 5 * DEG,
              pz=g["pz"] - (jit(0.07, 0.09) if body else 0.0),
              tpitch=st.torso_pitch + (12 * DEG if body else 0.0),
              troll=(6 * DEG if body else 0.0))
        impact = target + np.array([0.035, -0.06 * sgn, 0.0])         # follow through across
        via = target + np.array([-0.08, 0.20 * sgn, 0.03 if not body else 0.02])
        if body:
            elbow = impact + np.array([-0.09, 0.17 * sgn, -0.09])
        else:
            elbow = impact + np.array([-0.06, 0.21 * sgn, -0.035])
        swing = -sgn * (jit(30, 38) if lead else jit(38, 48)) * DEG   # lead: turn right (yaw down)
        ch.at(te, **{H: (impact, [via]), E: elbow},
              tyaw=b + swing, pyaw=b + 0.6 * swing)
        t_imp = te
        ch.cursor = te + 0.03
        t_end = te + 0.03 + tr
        ch.at(t_end, **{H: g[H], E: g[E]}, tyaw=b, pyaw=b, pz=g["pz"], tpitch=st.torso_pitch,
              troll=0.0)
        ch.move("com", t, te + 0.04, g["com"] + np.array([0.0, -0.006 * sgn]))
        ch.move("com", te + 0.04, t_end + 0.12, g["com"])

    elif name in ("lead_uppercut", "rear_uppercut"):
        lead = name == "lead_uppercut"
        sgn = 1.0 if lead else -1.0
        H, E = ("lh", "le") if lead else ("rh", "re")
        tc = t + 0.15 * k
        te = tc + 0.20 * k
        tr = 0.24 * k
        dz = st.pelvis_z - 0.73
        drop_h = _body(st, [0.18 if lead else 0.10, 0.11 * sgn, 0.94 + dz - 0.04])
        drop_e = _body(st, [0.02 if lead else -0.04, 0.15 * sgn, 0.84 + dz - 0.04])
        ch.at(tc, **{H: drop_h, E: drop_e}, pz=g["pz"] - jit(0.04, 0.06),
              troll=sgn * 6 * DEG, tyaw=b + sgn * 6 * DEG, pyaw=b + sgn * 4 * DEG,
              tpitch=st.torso_pitch + 4 * DEG)
        impact = target + np.array([0.015, 0.0, 0.06])
        via = target + np.array([-0.10, 0.03 * sgn, -0.15])
        elbow = impact + np.array([-0.12, 0.05 * sgn, -0.19])
        ch.at(te, **{H: (impact, [via]), E: elbow}, pz=g["pz"] + 0.005,
              troll=-sgn * 3 * DEG, tyaw=b - sgn * (jit(8, 14) if lead else jit(32, 40)) * DEG,
              pyaw=b - sgn * (6 if lead else 22) * DEG, tpitch=st.torso_pitch - 2 * DEG)
        t_imp = te
        ch.cursor = te + 0.03
        t_end = te + 0.03 + tr
        ch.at(t_end, **{H: g[H], E: g[E]}, tyaw=b, pyaw=b, pz=g["pz"], tpitch=st.torso_pitch,
              troll=0.0)
        ch.move("com", t, te + 0.04, g["com"] + np.array([0.006, -0.004 * sgn]))
        ch.move("com", te + 0.04, t_end + 0.12, g["com"])

    elif name == "calf_kick":
        lf, rf = g["lf"], g["rf"]
        # 1. shift weight onto the lead foot (CoM to the inner half of the lead sole)
        ts = t + 0.42 * k
        foot_c = lf[:2] + yaw_R(st.lead_foot_yaw)[:2, :2] @ np.array([0.03, -0.018])
        ch.at(ts, com=foot_c, pz=g["pz"] - 0.035, pyaw=b + 8 * DEG, tyaw=b + 4 * DEG,
              proll=-3 * DEG, troll=-5 * DEG, w_prot=0.8, w_trot=1.5)
        # 2. lift and chamber the rear leg
        tl = ts + 0.14 * k
        ch.rf_contact_off.append((ts, None))
        lift = rf + np.array([0.10, -0.06, 0.16])
        ch.at(tl, rf=lift, rfpitch=25 * DEG, rfyaw=st.rear_foot_yaw + 10 * DEG,
              rknee=rf + np.array([0.22, -0.10, 0.38]), w_rknee=1.0,
              pyaw=b + 25 * DEG, tyaw=b + 10 * DEG, proll=-6 * DEG, troll=-7 * DEG)
        # 3. swing the shin through the outside of the opponent's lead calf
        te = tl + 0.24 * k
        imp_ankle = target + np.array([0.04, 0.10, -0.11])
        via = target + np.array([-0.04, -0.12, -0.055])
        ch.at(te, rf=(imp_ankle, [via]), rfyaw=st.rear_foot_yaw + 75 * DEG, rfpitch=35 * DEG,
              rknee=target + np.array([-0.17, -0.06, 0.12]),
              pyaw=b + jit(55, 65) * DEG, tyaw=b + 35 * DEG, proll=-8 * DEG, troll=-10 * DEG,
              tpitch=st.torso_pitch - 6 * DEG,
              lh=_body(st, [0.24, 0.13, 1.12]), rh=_body(st, [0.05, -0.20, 1.02]))
        t_imp = te
        # 4. retract and put the foot back down
        tb = te + 0.26 * k
        ch.at(tb, rf=(rf + np.array([0.0, 0.0, 0.09]), [lift + np.array([0.0, -0.04, 0.0])]),
              rfyaw=st.rear_foot_yaw, rfpitch=5 * DEG,
              rknee=rf + np.array([0.18, -0.06, 0.40]),
              pyaw=b + 15 * DEG, tyaw=b + 8 * DEG, proll=-4 * DEG, troll=-5 * DEG,
              tpitch=st.torso_pitch, lh=g["lh"], rh=g["rh"])
        td = tb + 0.14 * k
        ch.at(td, rf=rf, rfpitch=0.0, rfroll=0.0, w_rknee=0.0, proll=0.0, troll=0.0)
        ch.rf_contact_off[-1] = (ts + 0.04 * k, td + 0.02)
        # 5. weight back to centre
        t_end = td + 0.30 * k
        ch.at(t_end, com=g["com"], pyaw=b, tyaw=b, pz=g["pz"], w_prot=6.0, w_trot=6.0)
        info["swing_window"] = (tl, te)
    else:
        raise ValueError(name)

    ch.events = dict(onset=onset, impact=t_imp, recovered=t_end)
    ch.hold_all(t_end + tail)
    return ch, info
