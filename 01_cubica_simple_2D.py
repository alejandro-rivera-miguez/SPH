"""
PASO 1 - Red cubica simple (cuadrada) en 2D
============================================
Particulas a la misma distancia dx formando cuadrados, recortadas a la
geometria. Es la referencia "clasica" (lattice) de los papers.

Ejecutar con F5 en VS Code (o: python 01_cubica_simple_2D.py).
Resultados en la carpeta resultados/: figura, metricas (.json),
particulas (.npz), include de PAM-CRASH (.inc) y CSV.
"""
from sph_common import crear_geometria, DistribucionSPH, procesar, red_cubica

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # circulo, cuadrado, rectangulo, triangulo, elipse,
                          # hexagono, estrella, capsula (pajaro 2D), poligono
PARAMS_GEOM = {}          # p.ej. {"R": 53.5} para capsula/circulo, {"L": 100} cuadrado
                          # poligono: {"vertices": [(0,0), (100,0), (60,80)]}
DX = 4.0                  # distancia entre particulas [mm]
RHO0 = 9.5e-7             # densidad [kg/mm^3] (gelatina ~ 950 kg/m^3)
H_FACTOR = 1.2            # h = H_FACTOR * dx  (smoothing length sugerido)
DESFASE = (0.5, 0.5)      # posicion de la red respecto al origen (fraccion de dx)
MOSTRAR = True
# --- figuras (lista completa de paneles en sph_common/figuras.py) ---
FIGURAS = ["resumen", "superficie", "convergencia"]   # quita las que no quieras
PANELES = ["particulas", "lambda", "densidad", "dnn",
           "hist_dnn", "hist_lambda", "gradgamma", "tabla"]
#   disponibles: particulas, lambda, densidad, gradgamma, gradgamma_sup, dnn,
#                hist_dnn, hist_lambda, hist_densidad, hist_gradgamma,
#                perfil_superficie, perfil_dnn, tabla
SECCION = ("y", 0.0)      # solo 3D
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    x = red_cubica(geom, DX, desfase=DESFASE)
    dist = DistribucionSPH(x, geom, "01_cubica", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, mostrar=MOSTRAR)
