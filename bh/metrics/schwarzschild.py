import numpy as np
from .base import Metric, Vec4, Mat4

class Schwarzschild(Metric):
    def __init__(self, M_geom: float):
        self.M=float(M_geom)

    def g(self, x: Vec4) -> Mat4:
        _, r, th, _ = x
        r=max(r,2.0*self.M*(1.0+1e-12))
        th=np.clip(th,1e-12,np.pi-1e-12)
        f=1.0-2.0*self.M/r
        g=np.zeros((4,4),dtype=float)
        g[0,0]=-f
        g[1,1]=1.0/f
        g[2,2]=r*r
        g[3,3]=(r*r)*(np.sin(th)**2)
        return g

    def horizon_radius(self)->float:
        return 2.0*self.M

    def christoffel(self, x: Vec4) -> np.ndarray:
        _, r, th, _ = x
        r=max(r,2.0*self.M*(1.0+1e-12))
        th=np.clip(th,1e-12,np.pi-1e-12)
        M=self.M
        s=np.sin(th); c=np.cos(th)
        f=1.0-2.0*M/r
        df_dr=2.0*M/(r*r)
        Gamma=np.zeros((4,4,4),dtype=float)
        Gamma[0,0,1]=Gamma[0,1,0]=df_dr/(2.0*f)
        Gamma[1,0,0]=f*df_dr/2.0
        Gamma[1,1,1]=df_dr/(2.0*f)
        Gamma[1,2,2]=-f*r
        Gamma[1,3,3]=-f*r*(s**2)
        Gamma[2,1,2]=Gamma[2,2,1]=1.0/r
        Gamma[2,3,3]=-s*c
        Gamma[3,1,3]=Gamma[3,3,1]=1.0/r
        if abs(s)<1e-12:
            Gamma[3,2,3]=Gamma[3,3,2]=0.0
        else:
            Gamma[3,2,3]=Gamma[3,3,2]=c/s
        return Gamma