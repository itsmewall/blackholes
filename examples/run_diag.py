import numpy as np
import matplotlib.pyplot as plt
from bh.units import mass_kg_to_geom_length
from bh.constants import Msun
from bh.metrics.schwarzschild import Schwarzschild
from bh.camera import PinholeCamera
from bh.lensing import ShadowRenderer, RenderOptions
from bh.geodesic import GeodesicSolver, GeoState, GeoOptions

def smoke_ray():
    M=mass_kg_to_geom_length(Msun)
    metric=Schwarzschild(M)
    x_obs=np.array([0.0,30.0*M,np.pi/2,0.0])
    cam=PinholeCamera(metric,x_obs,fov_deg=40.0,width=64,height=48)
    alpha,beta=cam.pixel_angles(32,24)
    p0=cam.ray_pcoord(alpha,beta)
    state=GeoState(x=x_obs,v=p0)
    solver=GeodesicSolver(metric,adaptive=True,atol=1e-8,rtol=1e-7,hmin=1e-5*M,hmax=0.2*M)
    traj,vel=solver.integrate(state,GeoOptions(step=0.05*M,nsteps=8000,stop_at_horizon=True))
    r=traj[-1,1]
    print(f"[smoke] pontos={len(traj)} r_final={r/M:.3f}M")

def render_small():
    M=mass_kg_to_geom_length(Msun)
    metric=Schwarzschild(M)
    x_obs=np.array([0.0,30.0*M,np.pi/2,0.0])
    cam=PinholeCamera(metric,x_obs,fov_deg=50.0,width=96,height=72)
    rnd=ShadowRenderer(metric,cam,M)
    ropt=RenderOptions(max_steps=8000,h0=0.05,atol=1e-8,rtol=1e-7,hmin=1e-5,hmax=0.2,r_escape=200.0,spp=1,verbose=True,log_every=6,max_wall_s=120.0)
    img=rnd.render(ropt)
    plt.imshow(img,cmap="gray",origin="lower",extent=[-1,1,-1,1])
    plt.title("diag 96x72")
    plt.tight_layout()
    plt.savefig("diag_shadow.png",dpi=120)
    print("[diag] salvo: diag_shadow.png")
    try:
        plt.show()
    except:
        pass

def main():
    smoke_ray()
    render_small()

if __name__=="__main__":
    main()