from __future__ import annotations
import numpy as np
from typing import Callable

Array = np.ndarray

def rk4_step(y: Array, h: float, f: Callable[[Array], Array]) -> Array:
    k1 = f(y)
    k2 = f(y + 0.5*h*k1)
    k3 = f(y + 0.5*h*k2)
    k4 = f(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
