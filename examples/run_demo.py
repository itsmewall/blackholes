import numpy as np
from bh.constants import Msun, Rsun, schwarzschild_radius, photon_sphere_radius_geom, weak_deflection_angle_rad
from bh.units import mass_kg_to_geom_length
from bh.metrics.schwarzschild import Schwarzschild
from bh.geodesic import GeodesicSolver, GeoState, GeoOptions
from bh.visualize import plot_orbit_xy, plot_r_lambda

def main():
    # ===== 1) Números "poderosos" (Sol) =====
    Rs_sun = schwarzschild_radius(Msun)
    print(f"R_s (Sol)         = {Rs_sun/1000:.2f} km")
    print(f"Photon sphere (3M)= {1.5*Rs_sun/1000:.2f} km")

    alpha = weak_deflection_angle_rad(Msun, Rsun)  # no limbo solar
    print(f"Deflexão fraca no limbo do Sol ≈ {alpha*206265:.3f} arcsec (~1.75 arcsec esperado)")

    # ===== 2) Geometrized mass M = GM/c^2 =====
    M_geom = mass_kg_to_geom_length(Msun)  # em metros (geom)
    metric = Schwarzschild(M_geom)

    # ===== 3) Órbita circular timelike em r = 8M, θ=π/2 =====
    r0   = 8.0 * M_geom           # raio da órbita
    th0  = np.pi/2
    phi0 = 0.0
    t0   = 0.0

    # Velocidade angular de órbita circular (coordenada t): Ω = sqrt(M/r^3)
    Omega = np.sqrt(M_geom / (r0**3))
    v_guess = np.array([0.0, 0.0, 0.0, 0.0])  # chute inicial para v^μ
    # Queremos v^r = 0, v^θ = 0, v^φ = Ω * v^t  (vamos normalizar)
    # Primeiro define v^φ proporcional a Ω, depois normaliza v^t
    v_guess[3] = 1.0   # temporário; normalização cuidará de v^t correto relativo
    x0 = np.array([t0, r0, th0, phi0])

    # Normaliza para timelike g_{μν} v^μ v^ν = -1
    # Impomos v^φ = Ω * v^t, então vamos iterar simples:
    # 1) chute v^t=1, v^φ=Ω*1, normaliza; 2) reimpoe relação v^φ=Ω v^t e normaliza de novo.
    vt = 1.0
    for _ in range(3):
        v_guess = np.array([vt, 0.0, 0.0, Omega*vt])
        v = GeodesicSolver.normalize_timelike(metric, x0, v_guess)
        vt = v[0]

    state0 = GeoState(x=x0, v=v)
    opts   = GeoOptions(step=0.02*M_geom, nsteps=30000, stop_at_horizon=True)

    solver = GeodesicSolver(metric)
    traj, vel = solver.integrate(state0, opts)

    # ===== 4) Plots =====
    plot_orbit_xy(traj, title="Órbita circular ~ r=8M (Schwarzschild)")
    plot_r_lambda(traj, step=opts.step, title="r(λ) para órbita (esperado ~ constante)")

    # ===== 5) Mais números =====
    r_ph = photon_sphere_radius_geom(M_geom)
    print(f"Raio da photon sphere (geom): {r_ph:.3e} m ; em R_s: {(r_ph)/(0.5*schwarzschild_radius(Msun)):.2f} * (R_s/2)")
    print("OK.")

if __name__ == "__main__":
    main()
