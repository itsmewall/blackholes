from __future__ import annotations
import numpy as np
from bh.metrics.base import Metric

def _rel_step(v: float) -> float:
    m = max(1.0, abs(v))
    return 1e-5 * m

def _sanitize_x(metric: Metric, x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    x[1] = max(x[1], metric.horizon_radius() * (1.0 + 1e-10))
    x[2] = np.clip(x[2], 1e-10, np.pi - 1e-10)
    return x

def riemann(metric: Metric, x: np.ndarray) -> np.ndarray:
    x = _sanitize_x(metric, x)
    G0 = metric.christoffel(x)
    dG = np.zeros((4,4,4,4))
    for mu in range(4):
        h = _rel_step(x[mu])
        ok = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            hh = h * scale
            xp = _sanitize_x(metric, np.array([x[0], x[1], x[2], x[3]]))
            xm = _sanitize_x(metric, np.array([x[0], x[1], x[2], x[3]]))
            xp[mu] += hh; xm[mu] -= hh
            Gp = metric.christoffel(xp); Gm = metric.christoffel(xm)
            diff = (Gp - Gm) / (2.0 * hh)
            if np.all(np.isfinite(diff)):
                dG[mu] = diff
                ok = True
                break
        if not ok:
            dG[mu] = 0.0
    R = np.zeros((4,4,4,4))
    for rho in range(4):
        for sig in range(4):
            for mu in range(4):
                for nu in range(4):
                    t = dG[mu,rho,sig,nu] - dG[nu,rho,sig,mu]
                    for a in range(4):
                        t += G0[rho,a,mu]*G0[a,sig,nu] - G0[rho,a,nu]*G0[a,sig,mu]
                    R[rho,sig,mu,nu] = t
    return R

def kretschmann(metric: Metric, x: np.ndarray) -> float:
    x = _sanitize_x(metric, x)
    R = riemann(metric, x)
    g = metric.g(x); gi = np.linalg.inv(g)
    K = 0.0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        for f in range(4):
                            for h in range(4):
                                for k in range(4):
                                    K += R[a,b,c,d]*R[e,f,h,k]*gi[a,e]*gi[b,f]*gi[c,h]*gi[d,k]
    return float(K)

def _mip(g: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    return float(u.T @ g @ v)

def tetrad_from_u(metric: Metric, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = _sanitize_x(metric, x)
    g = metric.g(x)
    e0 = u.astype(float).copy()
    n0 = -_mip(g, e0, e0)
    if not np.isfinite(n0) or n0 <= 0:
        e0 = np.array([1.0,0.0,0.0,0.0]); n0 = -_mip(g, e0, e0)
    e0 = e0 / np.sqrt(n0)

    seeds = [np.array([0.0,1.0,0.0,0.0]),
             np.array([0.0,0.0,1.0,0.0]),
             np.array([0.0,0.0,0.0,1.0])]
    E = [e0]
    for s in seeds:
        v = s.copy()
        for k in E:
            v = v - _mip(g, v, k) * k
        n = _mip(g, v, v)
        if np.isfinite(n) and n > 1e-14:
            E.append(v / np.sqrt(n))
        if len(E) == 4:
            break
    while len(E) < 4:
        z = np.random.default_rng(123).normal(size=4)
        z[0] = 0.0
        v = z
        for k in E:
            v = v - _mip(g, v, k) * k
        n = _mip(g, v, v)
        if np.isfinite(n) and n > 1e-14:
            E.append(v / np.sqrt(n))
    return np.vstack(E[:4])

def tidal_tensor(metric: Metric, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = _sanitize_x(metric, x)
    R = riemann(metric, x)
    E = tetrad_from_u(metric, x, u)
    Rhat = np.einsum('am,bn,cr,ds,mnrs->abcd', E, E, E, E, R, optimize=True)
    T = Rhat[1:4,0,1:4,0]
    T = 0.5 * (T + T.T)
    if not np.all(np.isfinite(T)):
        T[:] = np.nan
    return T