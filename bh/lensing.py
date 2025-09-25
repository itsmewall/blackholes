from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import time
from bh.metrics.base import Metric
from bh.geodesic import GeodesicSolver, GeoState, GeoOptions
from bh.camera import PinholeCamera

@dataclass
class RenderOptions:
    max_steps: int = 24000
    h0: float = 0.05
    atol: float = 1e-8
    rtol: float = 1e-7
    hmin: float = 1e-5
    hmax: float = 0.2
    r_escape: float = 400.0
    spp: int = 1
    verbose: bool = True
    log_every: int = 8
    max_wall_s: float = 600.0

class ShadowRenderer:
    def __init__(self, metric: Metric, camera: PinholeCamera, M: float):
        self.metric = metric
        self.cam = camera
        self.M = float(M)

    @staticmethod
    def energy_factor(metric: Metric, x: np.ndarray, v: np.ndarray, u_obs: np.ndarray) -> float:
        g = metric.g(x)
        p = v
        p_cov = g @ p
        Eobs = -float(p_cov @ u_obs)
        return max(0.0, Eobs)

    def integrate_ray(self, p0: np.ndarray, ropt: RenderOptions) -> tuple[bool, float]:
        x0 = self.cam.position_vec()
        u_obs = self.cam.observer_u()
        v0 = -p0.copy()
        state0 = GeoState(x=x0, v=v0)
        opts = GeoOptions(step=ropt.h0 * self.M, nsteps=ropt.max_steps, stop_at_horizon=True, r_min_clip=1e-9 * self.M)
        solver = GeodesicSolver(self.metric, adaptive=True, atol=ropt.atol, rtol=ropt.rtol, hmin=ropt.hmin * self.M, hmax=ropt.hmax * self.M)
        traj, vel = solver.integrate(state0, opts)
        r_h = self.metric.horizon_radius()
        r_last = traj[-1, 1]
        g = self.energy_factor(self.metric, traj[0], vel[0], u_obs)
        if r_last <= r_h * (1 + 1e-6):
            return False, g
        if r_last >= ropt.r_escape * self.M:
            return True, g
        return (r_last > 2 * r_h), g

    def render(self, ropt: RenderOptions) -> np.ndarray:
        H, W = self.cam.height, self.cam.width
        img = np.zeros((H, W), dtype=float)
        rng = np.random.default_rng(42)
        t0 = time.time()
        for j in range(H):
            for i in range(W):
                s = 0.0
                for _ in range(max(1, ropt.spp)):
                    dx = (rng.random() - 0.5) / W
                    dy = (rng.random() - 0.5) / H
                    alpha, beta = self.cam.pixel_angles(i + dx * self.cam.width, j + dy * self.cam.height)
                    p0 = self.cam.ray_pcoord(alpha, beta)
                    esc, g = self.integrate_ray(p0, ropt)
                    s += g if esc else 0.0
                img[j, i] = s / max(1, ropt.spp)
            if ropt.verbose and (j % max(1, ropt.log_every) == 0 or j == H - 1):
                dt = time.time() - t0
                frac = (j + 1) / H
                eta = (dt / frac - dt) if frac > 0 else 0.0
                print(f"[render] linha {j+1}/{H}  {frac*100:5.1f}%  dt={dt:6.1f}s  eta={eta:6.1f}s")
            if time.time() - t0 > ropt.max_wall_s:
                if ropt.verbose:
                    print("[render] timeout")
                break
        m = img.max() if img.max() > 0 else 1.0
        img = img / m
        return img
