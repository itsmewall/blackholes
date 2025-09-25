import numpy as np
import matplotlib.pyplot as plt
from bh.units import mass_kg_to_geom_length
from bh.constants import Msun
from bh.metrics.kerr import Kerr
from bh.camera import PinholeCamera
from bh.lensing import ShadowRenderer, RenderOptions

def main():
    M=mass_kg_to_geom_length(Msun)
    a=0.95*M
    metric=Kerr(M,a)
    x_obs=np.array([0.0,30.0*M,np.pi/2,0.0])
    cam=PinholeCamera(metric,x_obs,fov_deg=50.0,width=360,height=270)
    ropt=RenderOptions(max_steps=30000,h0=0.04,atol=5e-9,rtol=5e-8,hmin=1e-5,hmax=0.2,r_escape=500.0,spp=1)
    rnd=ShadowRenderer(metric,cam,M)
    img=rnd.render(ropt)
    plt.figure(figsize=(6,4.8))
    plt.imshow(img,cmap="inferno",origin="lower",extent=[-1,1,-1,1])
    plt.title("Sombra – Kerr a/M=0.95, r=30M")
    plt.tight_layout()
    plt.savefig("shadow_kerr.png",dpi=150)
    plt.show()

if __name__=="__main__":
    main()