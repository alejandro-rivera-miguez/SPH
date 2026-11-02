"""
PASO 3 - Empaquetamiento hexagonal en 2D
=========================================
VARIANTE = "triangular": hexagonal compacto (cada particula con 6 vecinos a
           la misma distancia). Es el empaquetado mas denso en 2D y el que usa
           Negi (paper 6) como estado inicial por ser estable.
VARIANTE = "panal": red de panal de abeja (3 vecinos). Colagrossi (paper 5,
           tabla 1) la llama "hexagonal" y demuestra que es SIEMPRE inestable;
           se incluye solo para poder verlo.

MISMO_VOLUMEN = True -> se ajusta el paso para que cada particula ocupe dx^2
(mismo N y misma masa que la cubica).
"""
from sph_common import crear_geometria, DistribucionSPH, procesar, red_hexagonal

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR = 1.2
VARIANTE = "triangular"     # "triangular" o "panal"
MISMO_VOLUMEN = True
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    x = red_hexagonal(geom, DX, mismo_volumen=MISMO_VOLUMEN, variante=VARIANTE)
    nombre = "03_hexagonal" if VARIANTE == "triangular" else "03_panal"
    dist = DistribucionSPH(x, geom, nombre, DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, mostrar=MOSTRAR)
