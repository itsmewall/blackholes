from __future__ import annotations
import numpy as np
from bh.metrics.base import Metric

def metric_dot(g,u,v):
    return float(u.T@g@v)

def gram_schmidt_tetrad(metric: Metric, x: np.ndarray) -> np.ndarray:
    g=metric.g(x)
    e0=np.array([1.0,0.0,0.0,0.0],dtype=float)
    n0=-metric_dot(g,e0,e0)
    e0=e0/np.sqrt(n0)
    e1=np.array([0.0,1.0,0.0,0.0],dtype=float)
    e1=e1-(metric_dot(g,e1,e0))*e0
    e1=e1/np.sqrt(metric_dot(g,e1,e1))
    e2=np.array([0.0,0.0,1.0,0.0],dtype=float)
    for k in [e0,e1]:
        e2=e2-(metric_dot(g,e2,k))*k
    e2=e2/np.sqrt(metric_dot(g,e2,e2))
    e3=np.array([0.0,0.0,0.0,1.0],dtype=float)
    for k in [e0,e1,e2]:
        e3=e3-(metric_dot(g,e3,k))*k
    e3=e3/np.sqrt(metric_dot(g,e3,e3))
    E=np.vstack([e0,e1,e2,e3])
    return E

class PinholeCamera:
    def __init__(self, metric: Metric, x_obs: np.ndarray, fov_deg: float=50.0, width: int=400, height: int=300):
        self.metric=metric
        self.x_obs=x_obs.astype(float)
        self.fov=float(np.deg2rad(fov_deg))
        self.width=int(width)
        self.height=int(height)

    def position_vec(self)->np.ndarray:
        return self.x_obs

    def tetrad(self)->np.ndarray:
        return gram_schmidt_tetrad(self.metric,self.x_obs)

    def pixel_angles(self,i:int,j:int)->tuple[float,float]:
        x=((i+0.5)/self.width)*2.0-1.0
        y=((j+0.5)/self.height)*2.0-1.0
        aspect=self.width/self.height
        alpha=x*(self.fov/2.0)
        beta=y*(self.fov/2.0)/aspect
        return alpha,beta

    def ray_pcoord(self,alpha:float,beta:float)->np.ndarray:
        E=self.tetrad()
        p_loc=np.array([1.0, -np.cos(alpha)*np.cos(beta), np.sin(beta), np.sin(alpha)*np.cos(beta)],dtype=float)
        return E.T@p_loc

    def observer_u(self)->np.ndarray:
        E=self.tetrad()
        return E[0].copy()