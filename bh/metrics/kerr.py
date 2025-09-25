from __future__ import annotations
import numpy as np
from .base import Metric, Vec4, Mat4

class Kerr(Metric):
    def __init__(self, M_geom: float, a_geom: float):
        self.M=float(M_geom)
        self.a=float(a_geom)

    def _Sigma(self,r,th):
        return r*r+(self.a*np.cos(th))**2

    def _Delta(self,r):
        return r*r-2*self.M*r+self.a*self.a

    def g(self, x: Vec4) -> Mat4:
        t,r,th,ph=x
        s=np.sin(th)
        c=np.cos(th)
        Sig=self._Sigma(r,th)
        Del=self._Delta(r)
        g=np.zeros((4,4),dtype=float)
        g[0,0]=-(1-2*self.M*r/Sig)
        g[0,3]=-2*self.M*r*self.a*(s**2)/Sig
        g[3,0]=g[0,3]
        g[1,1]=Sig/Del
        g[2,2]=Sig
        g[3,3]=(r*r+self.a*self.a+(2*self.M*r*self.a*self.a*(s**2)/Sig))*(s**2)
        return g

    def horizon_radius(self)->float:
        return self.M+np.sqrt(self.M*self.M-self.a*self.a)

    def christoffel(self, x: Vec4) -> np.ndarray:
        h=1e-5*max(1.0,abs(x[1]))
        g0=self.g(x)
        gi=np.linalg.inv(g0)
        dgdq=np.zeros((4,4,4))
        for mu in range(4):
            xp=x.copy(); xm=x.copy()
            xp[mu]+=h; xm[mu]-=h
            gp=self.g(xp); gm=self.g(xm)
            dgdq[mu]=(gp-gm)/(2*h)
        Gamma=np.zeros((4,4,4))
        for mu in range(4):
            for a in range(4):
                for b in range(4):
                    s=0.0
                    for nu in range(4):
                        s+=gi[mu,nu]*(dgdq[a,nu,b]+dgdq[b,nu,a]-dgdq[nu,a,b])
                    Gamma[mu,a,b]=0.5*s
        return Gamma