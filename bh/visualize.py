import numpy as np
import matplotlib.pyplot as plt

def plot_orbit_xy(traj: np.ndarray, title: str = "Órbita no plano equatorial"):
    # traj: (N,4) com (t,r,θ,φ)
    r = traj[:,1]
    th = traj[:,2]
    ph = traj[:,3]
    # projeta no plano XY assumindo θ≈π/2
    x = r * np.cos(ph)
    y = r * np.sin(ph)
    plt.figure(figsize=(6,6))
    plt.plot(x, y, lw=1.2)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x [geom]')
    plt.ylabel('y [geom]')
    plt.title(title)
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.show()

def plot_r_lambda(traj: np.ndarray, step: float, title: str = "r(λ)"):
    lam = np.arange(traj.shape[0]) * step
    r = traj[:,1]
    plt.figure(figsize=(7,4))
    plt.plot(lam, r, lw=1.2)
    plt.xlabel('λ [geom]')
    plt.ylabel('r [geom]')
    plt.title(title)
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.show()
