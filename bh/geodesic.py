from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from bh.metrics.base import Metric, Vec4
from bh.integrators import integrate_adaptive

@dataclass
class GeoState:
    x: np.ndarray
    v: np.ndarray

@dataclass
class GeoOptions:
    step: float = 0.01
    nsteps: int = 10000
    stop_at_horizon: bool = True
    r_min_clip: float = 1e-6

class GeodesicSolver:
    def __init__(self, metric: Metric, adaptive: bool=False, atol: float=1e-8, rtol: float=1e-7, hmin: float=1e-6, hmax: float=0.1):
        self.metric=metric
        self.adaptive=adaptive
        self.atol=atol
        self.rtol=rtol
        self.hmin=hmin
        self.hmax=hmax

    def _rhs(self, y: np.ndarray) -> np.ndarray:
        x=y[:4]; v=y[4:]
        x1=x.copy()
        x1[1]=max(x1[1],self.metric.horizon_radius()*(1.0+1e-12))
        x1[2]=np.clip(x1[2],1e-12,np.pi-1e-12)
        G=self.metric.christoffel(x1)
        a=-np.einsum('mab,a,b->m',G,v,v)
        dy=np.zeros_like(y)
        dy[:4]=v
        dy[4:]=a
        return dy

    def integrate(self, state0: GeoState, opt: GeoOptions) -> Tuple[np.ndarray, np.ndarray]:
        y=np.hstack([state0.x.copy(),state0.v.copy()])
        r_h=self.metric.horizon_radius()
        def _finite_clip(z):
            z=np.asarray(z)
            z[1]=max(z[1],max(r_h*(1.0+1e-12),opt.r_min_clip))
            z[2]=np.clip(z[2],1e-12,np.pi-1e-12)
            return z
        y[:4]=_finite_clip(y[:4])
        if self.adaptive:
            def f(z): return self._rhs(z)
            hist=integrate_adaptive(y,f,opt.step,opt.nsteps,self.atol,self.rtol,self.hmin,self.hmax)
            out=[]
            for z in hist:
                if not np.all(np.isfinite(z)): break
                z[:4]=_finite_clip(z[:4])
                out.append(z.copy())
                if opt.stop_at_horizon and z[1]<=max(r_h*(1.0+1e-12),opt.r_min_clip):
                    break
            arr=np.array(out)
            return arr[:,:4],arr[:,4:]
        else:
            h=opt.step
            traj=[y[:4].copy()]; vel=[y[4:].copy()]
            for _ in range(opt.nsteps):
                if not np.all(np.isfinite(y)): break
                if opt.stop_at_horizon and y[1]<=max(r_h*(1.0+1e-12),opt.r_min_clip): break
                k1=self._rhs(y)
                k2=self._rhs(y+0.5*h*k1)
                k3=self._rhs(y+0.5*h*k2)
                k4=self._rhs(y+h*k3)
                y=y+(h/6.0)*(k1+2*k2+2*k3+k4)
                y[:4]=_finite_clip(y[:4])
                traj.append(y[:4].copy()); vel.append(y[4:].copy())
            return np.array(traj),np.array(vel)

    @staticmethod
    def normalize_timelike(metric: Metric, x: Vec4, v_guess: Vec4) -> Vec4:
        g=metric.g(x)
        A=g[0,0]
        B=g[0,1]*v_guess[1]+g[0,2]*v_guess[2]+g[0,3]*v_guess[3]
        C=(g[1,1]*v_guess[1]**2+g[2,2]*v_guess[2]**2+g[3,3]*v_guess[3]**2
            +2*(g[1,2]*v_guess[1]*v_guess[2]+g[1,3]*v_guess[1]*v_guess[3]+g[2,3]*v_guess[2]*v_guess[3]))
        a=A;b=2*B;c=C+1.0
        disc=b*b-4*a*c
        if disc<0: disc=0.0
        vt=max(( -b+np.sqrt(disc) )/(2*a),( -b-np.sqrt(disc) )/(2*a))
        v=v_guess.copy()
        v[0]=vt
        return v