"""
PASO 2 - "Face centered cubic" en 2D (como el panel SPH de HyperMesh)
====================================================================
Celda cuadrada con una particula en cada esquina y otra en el centro.

OJO, detalle importante: en 2D el FCC es geometricamente una red cuadrada
girada 45 grados (con paso a/sqrt(2)). La diferencia real con la cubica
aparece en 3D (12 vecinos en FCC frente a 6 en cubica). En 2D sirve para
ver como cambia el contorno "en escalera" al girar la red.

MISMO_VOLUMEN = True  -> lado de celda a = sqrt(2) dx: mismo numero de
                         particulas (y misma masa) que la cubica. Comparacion justa.
MISMO_VOLUMEN = False -> lado de celda a = dx (el "pitch" de HyperMesh):
                         el doble de particulas.
"""
from sph_common import crear_geometria, DistribucionSPH, procesar, red_fcc

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR = 1.2
MISMO_VOLUMEN = True
DESFASE = (0.5, 0.5)
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    x = red_fcc(geom, DX, mismo_volumen=MISMO_VOLUMEN, desfase=DESFASE)
    dist = DistribucionSPH(x, geom, "02_fcc", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, mostrar=MOSTRAR)
