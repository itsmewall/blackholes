from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from bh.metrics.base import Metric
from bh.metrics.schwarzschild import Schwarzschild
from bh.integrators import integrate_adaptive
from bh.curvature import tetrad_from_u, riemann_numeric

@dataclass
class PartState:
    x: np.ndarray
    u: np.ndarray
    J: np.ndarray
    K: np.ndarray

@dataclass
class PartOptions:
    h0: float
    nmax: int
    atol: float
    rtol: float
    hmin: float
    hmax: float
    stop_at_horizon: bool = True
    r_min_clip: float = 1e-6

class ParticleJacobi:
    def __init__(self, metric: Metric):
        self.metric = metric

    def _ruju_schw(self, x: np.ndarray, u: np.ndarray, J: np.ndarray) -> np.ndarray:
        M = float(getattr(self.metric, "M"))
        r = float(x[1])
        lam_r = -2.0 * M / (r**3)
        lam_t = 1.0 * M / (r**3)
        E = tetrad_from_u(self.metric, x, u)
        g = self.metric.g(x)
        E_cov = E @ g.T
        J_hat = E_cov @ J
        a_hat = np.zeros(4)
        a_hat[1] = -lam_r * J_hat[1]
        a_hat[2] = -lam_t * J_hat[2]
        a_hat[3] = -lam_t * J_hat[3]
        a = E.T @ a_hat
        return a

    def _ruju_generic(self, x: np.ndarray, u: np.ndarray, J: np.ndarray) -> np.ndarray:
        R = riemann_numeric(self.metric, x)
        return -np.einsum('m n a b, n, a, b -> m', R, u, J, u)

    def _rhs(self, y: np.ndarray) -> np.ndarray:
        x = y[0:4]
        u = y[4:8]
        J = y[8:12]
        K = y[12:16]
        G = self.metric.christoffel(x)
        a = -np.einsum('mab,a,b->m', G, u, u)
        if isinstance(self.metric, Schwarzschild):
            RuJu = self._ruju_schw(x, u, J)
        else:
            RuJu = self._ruju_generic(x, u, J)
        dJ = K - np.einsum('mab,a,b->m', G, u, J)
        dK = RuJu - np.einsum('mab,a,b->m', G, u, K)
        dy = np.zeros_like(y)
        dy[0:4] = u
        dy[4:8] = a
        dy[8:12] = dJ
        dy[12:16] = dK
        return dy

    def integrate(self, s0: PartState, opt: PartOptions) -> np.ndarray:
        y0 = np.hstack([s0.x, s0.u, s0.J, s0.K])
        r_h = self.metric.horizon_radius()
        def clipx(z):
            z = z.copy()
            z[1] = max(z[1], max(r_h*(1.0+1e-12), opt.r_min_clip))
            z[2] = np.clip(z[2], 1e-12, np.pi-1e-12)
            return z
        y0[:4] = clipx(y0[:4])
        hist = integrate_adaptive(y0, self._rhs, opt.h0, opt.nmax, opt.atol, opt.rtol, opt.hmin, opt.hmax)
        out = []
        for z in hist:
            if not np.all(np.isfinite(z)): break
            z[:4] = clipx(z[:4])
            out.append(z.copy())
            if opt.stop_at_horizon and z[1] <= max(r_h*(1.0+1e-12), opt.r_min_clip):
                break
        return np.array(out)
