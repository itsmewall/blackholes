from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .metrics.base import Metric, Vec4
from .integrators import rk4_step

@dataclass
class GeoState:
    x: np.ndarray   # (t, r, theta, phi)
    v: np.ndarray   # dx^μ/dλ (4,)

@dataclass
class GeoOptions:
    step: float = 0.01       # passo em λ (geométrico)
    nsteps: int = 10000
    stop_at_horizon: bool = True
    r_min_clip: float = 1e-6

class GeodesicSolver:
    """
    Integra geodésicas com sistema de 1ª ordem:
      d x^μ/dλ = v^μ
      d v^μ/dλ = - Γ^μ_{αβ}(x) v^α v^β
    """
    def __init__(self, metric: Metric):
        self.metric = metric

    def _rhs(self, y: np.ndarray) -> np.ndarray:
        x = y[:4]
        v = y[4:]
        Gamma = self.metric.christoffel(x)
        # a^μ = - Γ^μ_{αβ} v^α v^β
        a = np.zeros(4)
        # contração eficiente: a^μ = -Σ_{α,β} Γ^μ_{αβ} v^α v^β
        # usamos einsum para clareza
        a = -np.einsum('mab,a,b->m', Gamma, v, v)
        dy = np.zeros_like(y)
        dy[:4] = v
        dy[4:] = a
        return dy

    def integrate(self, state0: GeoState, opt: GeoOptions) -> Tuple[np.ndarray, np.ndarray]:
        y = np.hstack([state0.x.copy(), state0.v.copy()])
        traj = [state0.x.copy()]
        vel  = [state0.v.copy()]
        step = opt.step

        r_h = self.metric.horizon_radius()

        for _ in range(opt.nsteps):
            # stopping criteria
            r = y[1]
            if opt.stop_at_horizon and r <= max(r_h*(1+1e-6), opt.r_min_clip):
                break

            y = rk4_step(y, step, self._rhs)
            traj.append(y[:4].copy())
            vel.append(y[4:].copy())

        return np.array(traj), np.array(vel)

    @staticmethod
    def normalize_timelike(metric: Metric, x: Vec4, v_guess: Vec4) -> Vec4:
        """
        Ajusta v^t para timelike: g_{μν} v^μ v^ν = -1, dado v^r,v^θ,v^φ.
        Assume v^t >= 0. Resolve quadraticamente em v^t.
        """
        g = metric.g(x)
        # Escreve: A (v^t)^2 + 2B v^t + C = -1  ->  A (v^t)^2 + 2B v^t + (C+1) = 0
        A = g[0,0]
        B = g[0,1]*v_guess[1] + g[0,2]*v_guess[2] + g[0,3]*v_guess[3]
        C = (
            g[1,1]*v_guess[1]**2 +
            g[2,2]*v_guess[2]**2 +
            g[3,3]*v_guess[3]**2 +
            2*(g[1,2]*v_guess[1]*v_guess[2] + g[1,3]*v_guess[1]*v_guess[3] + g[2,3]*v_guess[2]*v_guess[3])
        )
        # Quadrática: A x^2 + 2B x + (C+1) = 0
        a = A
        b = 2*B
        c = (C + 1.0)
        disc = b*b - 4*a*c
        if disc < 0:
            raise ValueError("Não foi possível normalizar vetor timelike (discriminante < 0)")
        # v^t >= 0
        vt1 = (-b + np.sqrt(disc)) / (2*a)
        vt2 = (-b - np.sqrt(disc)) / (2*a)
        vt = max(vt1, vt2)
        v = v_guess.copy()
        v[0] = vt
        return v
