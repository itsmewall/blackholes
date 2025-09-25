from __future__ import annotations
import numpy as np
from typing import Callable

def rkf45_step(y: np.ndarray, h: float, f: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray,float]:
    a2=1/4;a3=3/8;a4=12/13;a5=1;a6=1/2
    b21=1/4
    b31=3/32;b32=9/32
    b41=1932/2197;b42=-7200/2197;b43=7296/2197
    b51=439/216;b52=-8;b53=3680/513;b54=-845/4104
    b61=-8/27;b62=2;b63=-3544/2565;b64=1859/4104;b65=-11/40
    c1=16/135;c3=6656/12825;c4=28561/56430;c5=-9/50;c6=2/55
    ch1=25/216;ch3=1408/2565;ch4=2197/4104;ch5=-1/5
    k1=f(y)
    k2=f(y+h*b21*k1)
    k3=f(y+h*(b31*k1+b32*k2))
    k4=f(y+h*(b41*k1+b42*k2+b43*k3))
    k5=f(y+h*(b51*k1+b52*k2+b53*k3+b54*k4))
    k6=f(y+h*(b61*k1+b62*k2+b63*k3+b64*k4+b65*k5))
    y5=y+h*(c1*k1+c3*k3+c4*k4+c5*k5+c6*k6)
    y4=y+h*(ch1*k1+ch3*k3+ch4*k4+ch5*k5)
    err=np.linalg.norm(y5-y4, ord=np.inf)
    return y5, err

def integrate_adaptive(y0: np.ndarray, f: Callable[[np.ndarray], np.ndarray], h0: float, nmax: int, atol: float, rtol: float, hmin: float, hmax: float) -> list[np.ndarray]:
    y=y0.copy()
    h=h0
    out=[y.copy()]
    for _ in range(nmax):
        y1,err=rkf45_step(y,h,f)
        tol=atol+rtol*max(np.linalg.norm(y,ord=np.inf),np.linalg.norm(y1,ord=np.inf))
        s=2.0
        if err>0:
            s=0.9*(tol/err)**0.25
        if err<=tol or h<=hmin*1.01:
            y=y1
            out.append(y.copy())
        h=np.clip(s*h, hmin, hmax)
    return out