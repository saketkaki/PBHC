"""Keyframe tracks with minimum-jerk timing.

A Track holds keyframes (t_i, v_i) and optional via-points per segment. Between two
keyframes the progress variable follows a minimum-jerk profile
    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,
so velocity and acceleration are zero at every keyframe. With via-points the value
moves along a smooth curve (chord-length cubic spline) through the vias at progress s,
which gives curved strike paths (hooks, uppercuts, kicks) with a single smooth speed
profile instead of stopping at each via.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


class Track:
    def __init__(self, t0, v0):
        self.times = [float(t0)]
        self.vals = [np.atleast_1d(np.asarray(v0, dtype=float))]
        self.vias = []  # one entry per segment
        self._splines = []

    def to(self, t, v, vias=None):
        """Add a keyframe at absolute time t (must be increasing)."""
        assert t > self.times[-1] + 1e-9, (t, self.times[-1])
        self.times.append(float(t))
        self.vals.append(np.atleast_1d(np.asarray(v, dtype=float)))
        self.vias.append([np.atleast_1d(np.asarray(p, dtype=float)) for p in (vias or [])])
        self._splines.append(None)
        return self

    def hold(self, t):
        return self.to(t, self.vals[-1])

    @property
    def last(self):
        return self.vals[-1].copy()

    @property
    def t_end(self):
        return self.times[-1]

    def _spline(self, i):
        if self._splines[i] is None:
            pts = np.stack([self.vals[i], *self.vias[i], self.vals[i + 1]])
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            d = d / max(d[-1], 1e-9)
            self._splines[i] = CubicSpline(d, pts, axis=0, bc_type="natural")
        return self._splines[i]

    def __call__(self, t):
        ts = self.times
        if t <= ts[0]:
            return self.vals[0].copy()
        if t >= ts[-1]:
            return self.vals[-1].copy()
        i = int(np.searchsorted(ts, t, side="right") - 1)
        s = minjerk((t - ts[i]) / (ts[i + 1] - ts[i]))
        if self.vias[i]:
            return self._spline(i)(s)
        return (1 - s) * self.vals[i] + s * self.vals[i + 1]
