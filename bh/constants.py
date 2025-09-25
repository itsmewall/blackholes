# Constantes físicas (SI)
G  = 6.67430e-11            # m^3 kg^-1 s^-2
c  = 299_792_458.0          # m/s
Msun   = 1.98847e30         # kg
Mearth = 5.9722e24          # kg
Rsun   = 6.9634e8           # m

def schwarzschild_radius(M_kg: float) -> float:
    """R_s = 2GM/c^2 (em metros)"""
    return 2.0 * G * M_kg / (c*c)

def photon_sphere_radius_geom(M_geom: float) -> float:
    """r_photon = 3M (em unidades geométricas; M_geom = GM/c^2)"""
    return 3.0 * M_geom

def weak_deflection_angle_rad(M_kg: float, impact_parameter_m: float) -> float:
    """Aproximação fraca: alpha ≈ 4GM/(c^2 b) (em radianos)"""
    return 4.0 * G * M_kg / (c*c * impact_parameter_m)
