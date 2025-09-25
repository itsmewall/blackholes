from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

Vec4 = np.ndarray   # shape (4,)
Mat4 = np.ndarray   # shape (4,4,)

class Metric(ABC):
    """
    Interface de uma métrica em coordenadas (t, r, theta, phi) no contínuo 4D.
    Unidades: geométricas (G=c=1). Parâmetros em metros se você usa M=GM/c^2.
    """

    @abstractmethod
    def g(self, x: Vec4) -> Mat4:
        """Matriz métrica g_{μν} na posição x=(t,r,θ,φ)."""
        ...

    def g_inv(self, x: Vec4) -> Mat4:
        return np.linalg.inv(self.g(x))

    @abstractmethod
    def christoffel(self, x: Vec4) -> np.ndarray:
        """Γ^μ_{αβ} como array shape (4,4,4) em x."""
        ...

    @abstractmethod
    def horizon_radius(self) -> float:
        ...
