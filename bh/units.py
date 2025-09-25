from .constants import G, c

def mass_kg_to_geom_length(M_kg: float) -> float:
    """Converte massa (kg) para unidade geométrica de comprimento M = GM/c^2 (em metros)."""
    return G * M_kg / (c*c)

def length_geom_to_seconds(L_m: float) -> float:
    """Converte comprimento (m) em unidades geométricas para tempo (s) (c=1 => t=L/c)."""
    return L_m / c

def seconds_to_length_geom(t_s: float) -> float:
    return t_s * c