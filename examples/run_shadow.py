# examples/run_shadow.py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

from bh.units import mass_kg_to_geom_length
from bh.constants import Msun
from bh.metrics.schwarzschild import Schwarzschild
from bh.camera import PinholeCameraEquatorial
from bh.lensing import ShadowRenderer, RenderOptions

def main():
    print("[shadow] iniciando...")
    t0 = time.time()

    M_geom = mass_kg_to_geom_length(Msun)
    metric = Schwarzschild(M_geom)

    cam = PinholeCameraEquatorial(
        r_obs = 30.0 * M_geom,
        fov_deg = 50.0,
        width = 220,    
        height = 160
    )
    rnd = ShadowRenderer(metric, cam, M_geom)
    ropt = RenderOptions(max_steps=8000, step=0.04, r_escape=200.0)

    H, W = cam.height, cam.width
    img = np.zeros((H, W), dtype=float)
    print(f"[shadow] render {W}x{H}, step={ropt.step}, max_steps={ropt.max_steps}")
    for j in range(H):
        if j % 10 == 0:
            print(f"[shadow] linha {j}/{H}")
        for i in range(W):
            alpha, beta = cam.pixel_angles(i, j)
            p0 = cam.ray_momentum_from_angles(M_geom, alpha, beta)
            escaped = rnd.integrate_ray(p0, ropt)
            img[j, i] = 1.0 if escaped else 0.0

    escaped_ratio = img.mean()
    print(f"[shadow] escaped={escaped_ratio:.3f}, captured={(1-escaped_ratio):.3f}")

    out = "shadow_schwarzschild.png"
    plt.figure(figsize=(6,4.8))
    plt.imshow(img, cmap="gray", origin="lower", extent=[-1,1,-1,1])
    plt.title("Sombra – Schwarzschild (r=30M)")
    plt.xlabel("x (FOV)"); plt.ylabel("y (FOV)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[shadow] salvo: {out}")

    try:
        plt.show(block=True)
    except Exception as e:
        print(f"[shadow] janela não abriu ({e}). Abra o PNG salvo.")

    print(f"[shadow] fim em {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()