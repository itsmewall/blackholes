import numpy as np
import matplotlib.pyplot as plt
import csv
from bh.constants import Msun
from bh.units import mass_kg_to_geom_length
from bh.metrics.schwarzschild import Schwarzschild
from bh.dynamics import ParticleJacobi, PartState, PartOptions
from bh.curvature import kretschmann, tidal_tensor

def main():
    M=mass_kg_to_geom_length(Msun)
    metric=Schwarzschild(M)
    r0=10.0*M; th0=np.pi/2; ph0=0.0; t0=0.0
    x0=np.array([t0,r0,th0,ph0])
    L=np.sqrt(M*r0)/(1.0) 
    Omega=L/(r0*r0)
    u_guess=np.array([1.0,0.0,0.0,Omega])
    from bh.geodesic import GeodesicSolver
    u0=GeodesicSolver.normalize_timelike(metric,x0,u_guess)
    J0=np.array([0.0,1e-6,0.0,0.0])
    K0=np.zeros(4)
    s0=PartState(x=x0,u=u0,J=J0,K=K0)
    opt=PartOptions(h0=0.02*M,nmax=40000,atol=1e-9,rtol=1e-8,hmin=5e-6*M,hmax=0.1*M,stop_at_horizon=True)
    solver=ParticleJacobi(metric)
    hist=solver.integrate(s0,opt)
    x=hist[:,0:4]; u=hist[:,4:8]; J=hist[:,8:12]; K=hist[:,12:16]
    r=x[:,1]
    Ksc=np.array([kretschmann(metric,xi) for xi in x[::50]])
    Tvals=[]
    for xi,ui in zip(x[::50],u[::50]):
        E = tidal_tensor(metric, xi, ui)
        if not np.all(np.isfinite(E)):
            continue
        try:
            w = np.linalg.eigvalsh(0.5*(E+E.T))
            Tvals.append(w)
        except np.linalg.LinAlgError:
            continue
    Tvals = np.array(Tvals)
    with open("particle_series.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["idx","t","r","theta","phi","u_t","u_r","u_th","u_ph","J_t","J_r","J_th","J_ph","K_t","K_r","K_th","K_ph"])
        for i,zi in enumerate(hist):
            w.writerow([i]+list(zi[0:4])+list(zi[4:8])+list(zi[8:12])+list(zi[12:16]))
    plt.figure(figsize=(7,4)); plt.plot(r/M); plt.ylabel("r/M"); plt.xlabel("passos"); plt.title("r/M vs passos"); plt.grid(True,ls=":"); plt.tight_layout(); plt.savefig("r_series.png",dpi=140)
    plt.figure(figsize=(7,4)); plt.semilogy(Ksc); plt.ylabel("K"); plt.xlabel("amostras (*/50)"); plt.title("Kretschmann"); plt.grid(True,ls=":"); plt.tight_layout(); plt.savefig("kretschmann_series.png",dpi=140)
    plt.figure(figsize=(7,4)); 
    if Tvals.size>0:
        for k in range(Tvals.shape[1]):
            plt.plot(Tvals[:,k],label=f"λ{k}")
        plt.legend()
    plt.ylabel("eig(E)"); plt.xlabel("amostras (*/50)"); plt.title("Marés (autovalores)"); plt.grid(True,ls=":"); plt.tight_layout(); plt.savefig("tidal_eigs.png",dpi=140)
    print("OK: particle_series.csv, r_series.png, kretschmann_series.png, tidal_eigs.png")

if __name__=="__main__":
    main()
