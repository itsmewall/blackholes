# -*- coding: utf-8 -*-

# Este script gera um arquivo README.md com conteúdo predefinido.

# O conteúdo do Markdown é armazenado em uma string multilinha (triple-quoted string).
# O 'r' antes da string (r''') indica uma string bruta, o que ajuda a evitar problemas
# com barras invertidas que podem ser interpretadas como sequências de escape.
markdown_content = r'''
# BlackHole Lab

> A minimal Python project for simulating black holes with basic General Relativity tools.

---

## What it does

* Geodesics (massive and null)
* Black hole shadow (lensing)
* Curvature diagnostics (Riemann/Kretschmann) and tidal tensor

---

## Requirements

* Python 3.10+
* `pip install -r requirements.txt`

---

## How to run

```shell
python -m examples.run_demo
python -m examples.run_shadow
python -m examples.run_shadow_schwarzschild_heavy
python -m examples.run_shadow_kerr
python -m examples.run_particle_heavy
```

---

## Theory overview

### Spacetime, metrics, and units

We model gravity via spacetime curvature. The metric `$g_{\mu\nu}$` sets intervals `$ds^2=g_{\mu\nu}dx^\mu dx^\nu$`.

* **Schwarzschild** (mass `$M$`, non-rotating): in spherical coordinates `$(t,r,\theta,\phi)$`,
    $$
    ds^2 = -\Big(1-\frac{2M}{r}\Big)dt^2 + \Big(1-\frac{2M}{r}\Big)^{-1}dr^2 + r^2 d\theta^2 + r^2\sin^2\theta\, d\phi^2.
    $$

* **Kerr** (mass `$M$`, spin `$a$`) uses Boyer–Lindquist form with `$\Delta=r^2-2Mr+a^2$` and `$\Sigma=r^2+a^2\cos^2\theta$`.

We use geometric units `$G=c=1$`. In these units, `$M$` has length dimension; the horizon of Schwarzschild sits at `$r_h=2M$`.

### Geodesics

Test particles and light follow geodesics:
$$\frac{d^2 x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta}\frac{dx^\alpha}{d\lambda}\frac{dx^\beta}{d\lambda}=0,$$
with Christoffel symbols `$\Gamma^\mu_{\alpha\beta}$` computed from `$g_{\mu\nu}$`.

* **Timelike (massive):** 4-velocity `$u^\mu=dx^\mu/d\tau$` normalized by `$g_{\mu\nu}u^\mu u^\nu=-1$`.
* **Null (photons):** 4-momentum `$k^\mu$` obeys `$g_{\mu\nu}k^\mu k^\nu=0$`. The affine parameter is arbitrary up to scale.

Conserved quantities in stationary, axisymmetric metrics (like Kerr) include energy `$E=-\xi_{(t)\mu}u^\mu$`, axial angular momentum `$L_z=\xi_{(\phi)\mu}u^\mu$`, and, for Kerr, Carter’s constant `$Q$`.

### Circular orbits, photon sphere, ISCO

In Schwarzschild:

* **Photon sphere** at `$r=3M$`: unstable circular null orbits; it sets the sharp rim of the shadow.
* **ISCO** (innermost stable circular orbit) at `$r=6M$` for timelike equatorial motion. Inside the ISCO, circular timelike orbits are unstable or impossible.

### Black hole shadow and lensing

An observer at `$r_{\text{obs}}$` defines an orthonormal tetrad (local inertial frame). Each image pixel corresponds to a local direction `$(\alpha,\beta)$`. We map it to an initial photon 4-momentum `$k^{(\hat a)}=(1,\hat{\mathbf{n}})$` in the tetrad and convert to coordinates `$k^\mu$`.

Tracing backwards in time: if the ray crosses the horizon, the pixel is “captured” (dark); if it escapes to large `$r$`, it is “sky” (bright). The shadow’s angular size is set by the capture cross-section; in Kerr it becomes asymmetric due to frame dragging.
Redshift and beaming can be included via the observed energy `$E_{\text{obs}}=-k_\mu u^\mu_{\text{obs}}$`, used as a weight or color.

### Curvature and tidal effects

The Riemann tensor `$R^\rho{}_{\sigma\mu\nu}$` encodes curvature. A scalar diagnostic is Kretschmann
$$K=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma},$$
which grows toward the singularity.

Tidal forces along a worldline with 4-velocity `$u^\mu$` are captured by the electric part of Riemann,
`$E_{ij}=R_{\hat{i}\hat{0}\hat{j}\hat{0}}$`,
defined in the particle’s orthonormal frame `$\{\hat{0},\hat{1},\hat{2},\hat{3}\}$`. Eigenvalues of `$E_{ij}$` quantify stretching/compression.

Relative separation `$\eta^\mu$` of nearby geodesics obeys the geodesic deviation (Jacobi) equation:
$$\frac{D^2\eta^\mu}{d\tau^2}+R^\mu{}_{\nu\alpha\beta}u^\nu \eta^\alpha u^\beta=0.$$

---

## Numerical approach and caveats

We use an adaptive Runge–Kutta–Fehlberg (RKF45) integrator for stability near strong curvature, finite-difference Christoffels (for Kerr) or analytic ones (Schwarzschild), and horizon/angle clamping to avoid coordinate singularities.

Shadows are computed by backward ray tracing; particle runs use proper-time stepping and periodic normalization. Finite-difference curvature is sensitive to step size; we use relative steps and guard against non-finite values.

---

## Notes

* Internal units use `$G=c=1$`. Helpers convert SI ↔ geometric.
* For faster runs, reduce image size or integration limits; for higher fidelity, increase resolution and samples-per-pixel.
'''

# Define o nome do arquivo a ser criado
file_name = "readme.md"

# Abre o arquivo em modo de escrita ('w'). Se o arquivo já existir, será sobrescrito.
# 'encoding="utf-8"' garante que caracteres especiais sejam salvos corretamente.
with open(file_name, "w", encoding="utf-8") as f:
    f.write(markdown_content)

# Imprime uma mensagem de confirmação no console
print(f"Arquivo '{file_name}' gerado com sucesso!")
