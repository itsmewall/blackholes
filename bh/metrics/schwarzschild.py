import numpy as np
from .base import Metric, Vec4, Mat4

class Schwarzschild(Metric):
    """
    Métrica de Schwarzschild em coordenadas (t, r, θ, φ).
    Assumimos unidades geométricas (G=c=1) e parâmetro M em metros (M = GM_SI/c^2).
    g_tt = -(1 - 2M/r), g_rr = (1 - 2M/r)^(-1), g_thth = r^2, g_phph = r^2 sin^2 θ.
    """

    def __init__(self, M_geom: float):
        self.M = float(M_geom)

    def g(self, x: Vec4) -> Mat4:
        _, r, th, _ = x
        f = 1.0 - 2.0*self.M/r
        g = np.zeros((4,4), dtype=float)
        g[0,0] = -f
        g[1,1] = 1.0/f
        g[2,2] = r*r
        g[3,3] = (r*r)*(np.sin(th)**2)
        return g

    def horizon_radius(self) -> float:
        return 2.0*self.M

    def christoffel(self, x: Vec4) -> np.ndarray:
        """
        Símbolos de Christoffel Γ^μ_{αβ} analíticos para Schwarzschild.
        Coordenadas: (t=0, r=1, θ=2, φ=3)
        Só dependem de r e θ.
        """
        _, r, th, _ = x
        M = self.M
        s, c = np.sin(th), np.cos(th)
        f = 1.0 - 2.0*M/r

        # Inicializa
        Gamma = np.zeros((4,4,4), dtype=float)

        # Convenções: Γ^μ_{αβ}
        # Derivadas úteis:
        df_dr = 2.0*M/(r*r)

        # Componentes não-nulas clássicas:

        # Γ^t_{tr} = Γ^t_{rt} = (1/2) g^{tt} ∂_r g_{tt} = (1/2)(-1/f)*( -df ) = (df)/(2f)
        Gamma[0,0,1] = Gamma[0,1,0] = df_dr / (2.0*f)

        # Γ^r_{tt} = -(1/2) g^{rr} ∂_r g_{tt} = -(1/2) f * (-df) = (f*df)/2
        Gamma[1,0,0] = f * df_dr / 2.0

        # Γ^r_{rr} = -(1/2) g^{rr} ∂_r g_{rr} = -(1/2) f * ( - df / f^2 ) = (df)/(2f)
        Gamma[1,1,1] = df_dr / (2.0 * f)

        # Γ^r_{θθ} = -(1/2) g^{rr} ∂_r g_{θθ} = -(1/2) f * (2r) = -f*r
        Gamma[1,2,2] = -f * r

        # Γ^r_{φφ} = -(1/2) g^{rr} ∂_r g_{φφ} = -(1/2) f * (2r sin^2θ) = -f*r sin^2θ
        Gamma[1,3,3] = -f * r * (s**2)

        # Γ^θ_{rθ} = Γ^θ_{θr} = (1/2) g^{θθ} ∂_r g_{θθ} = (1/2)(1/r^2)*(2r) = 1/r
        Gamma[2,1,2] = Gamma[2,2,1] = 1.0/r

        # Γ^θ_{φφ} = -(1/2) g^{θθ} ∂_θ g_{φφ} = -(1/2)(1/r^2)*(2 r^2 sinθ cosθ) = -sinθ cosθ
        Gamma[2,3,3] = -s*c

        # Γ^φ_{rφ} = Γ^φ_{φr} = (1/2) g^{φφ} ∂_r g_{φφ} = (1/2)(1/(r^2 s^2))*(2 r s^2) = 1/r
        Gamma[3,1,3] = Gamma[3,3,1] = 1.0/r

        # Γ^φ_{θφ} = Γ^φ_{φθ} = (1/2) g^{φφ} ∂_θ g_{φφ} = (1/2)(1/(r^2 s^2))*(2 r^2 s c) = cotθ
        if abs(s) > 1e-15:
            Gamma[3,2,3] = Gamma[3,3,2] = c/s
        else:
            # Evita singularidade no polo
            Gamma[3,2,3] = Gamma[3,3,2] = 0.0

        return Gamma
