import numpy as np
import matplotlib.pyplot as plt
from bh.units import mass_kg_to_geom_length
from bh.constants import Msun
from bh.metrics.schwarzschild import Schwarzschild
from bh.camera import PinholeCamera
from bh.lensing import ShadowRenderer, RenderOptions

def main():
    M=mass_kg_to_geom_length(Msun)
    metric=Schwarzschild(M)
    x_obs=np.array([0.0,30.0*M,np.pi/2,0.0])
    cam=PinholeCamera(metric,x_obs,fov_deg=50.0,width=480,height=360)
    ropt=RenderOptions(max_steps=26000,h0=0.03,atol=1e-9,rtol=1e-8,hmin=5e-6,hmax=0.15,r_escape=400.0,spp=2)
    rnd=ShadowRenderer(metric,cam,M)
    img=rnd.render(ropt)
    plt.figure(figsize=(6,4.8))
    plt.imshow(img,cmap="gray",origin="lower",extent=[-1,1,-1,1])
    plt.title("Sombra – Schwarzschild (adaptive, spp=2)")
    plt.tight_layout()
    plt.savefig("shadow_schw_heavy.png",dpi=150)
    plt.show()

if __name__=="__main__":
    main()