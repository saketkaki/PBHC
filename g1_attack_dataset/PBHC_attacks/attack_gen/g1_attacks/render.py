"""Render attack clips (attacker + canonical defender) with MuJoCo.

Needs an OpenGL context; on a headless Linux box run under `xvfb-run -a` with MUJOCO_GL=glfw,
or set MUJOCO_GL=egl on a machine with a GPU.
"""
from __future__ import annotations

import mujoco


def build_scene(xml_path):
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    w = spec.worldbody
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.86, 0.87, 0.89, 1])
    w.add_light(pos=[0.3, -1.0, 3.0], dir=[0, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    w.add_light(pos=[0.3, 1.5, 2.5], dir=[0, -0.5, -1], diffuse=[0.4, 0.4, 0.4])
    for prefix, rgba in (("atk_", [0.85, 0.35, 0.2, 1]), ("def_", [0.25, 0.45, 0.8, 1])):
        child = mujoco.MjSpec.from_file(xml_path)
        for g in child.geoms:
            g.rgba = rgba
            g.contype = 0
            g.conaffinity = 0
        # drop the child's own floor/lights if it has any
        for g in list(child.worldbody.geoms):
            child.delete(g)
        for l in list(child.worldbody.lights):
            child.delete(l)
        frame = w.add_frame()
        frame.attach_body(child.body("pelvis"), prefix, "")
    target = w.add_body(name="target", mocap=True)
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[0.1, 0.8, 0.2, 0.8],
                    contype=0, conaffinity=0)
    m = spec.compile()
    return m


class Renderer:
    def __init__(self, xml_path, width=480, height=360):
        self.m = build_scene(xml_path)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

    def frame(self, q_atk, q_def, target=None, view="side"):
        self.d.qpos[:30] = q_atk
        self.d.qpos[30:60] = q_def
        if target is not None:
            self.d.mocap_pos[0] = target
        mujoco.mj_forward(self.m, self.d)
        mid = 0.5 * (q_atk[:3] + q_def[:3])
        self.cam.lookat[:] = [mid[0], mid[1], 0.75]
        self.cam.distance = 2.3
        views = {"side": (90.0, -8.0), "oblique": (135.0, -22.0), "rear": (180.0 + 25.0, -15.0),
                 "top": (90.0, -80.0), "other_side": (-90.0, -8.0)}
        self.cam.azimuth, self.cam.elevation = views[view]
        self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render().copy()
