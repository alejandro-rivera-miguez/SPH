"""
PASO 5 (3D) - Algoritmo hibrido de Negi & Ramachandran (2021)  [paper 6]
=======================================================================
Particulas libres en red HCP + particulas de contorno que deslizan sobre la
superficie desplazada dx/2 hacia dentro + repulsion + amortiguamiento.

Diferencias respecto al paper, para que sea viable en un PC:
  * CONTENEDOR = "forma": las libres solo ocupan el cuerpo + una banda de
    MARGEN*dx, y las congeladas forman una capa que sigue la forma (en el
    paper es una caja: 4-5 veces mas particulas fuera del pajaro).
  * KERNEL = "wendland" con h = 1.3 dx (el paper usa quintic con h = 1.2 dx,
    ~2.6 veces mas vecinos). Puedes poner KERNEL = "quintic" y HDX = 1.2.
Tiempo orientativo con DX = 6.65: 3-6 min.
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar
from sph_common.packing import empaquetar_negi

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # o "stl" para usar tu propio fichero (ver abajo)
PARAMS_GEOM = {}
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
DX = 6.65
RHO0 = 1.1317e-6
H_FACTOR_SIM = 1.2
# --- empaquetado ---
KERNEL = "wendland"         # "wendland" (rapido) o "quintic" (como el paper)
HDX = 1.3                   # 1.3 con wendland, 1.2 con quintic
CONTENEDOR = "forma"        # "forma" o "caja"
MARGEN = 2                  # banda de particulas libres fuera del cuerpo (en dx)
TOL = 1e-4
MAX_ITER = 4000
MIN_ITER = 1000
MOSTRAR = True
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x, hist, info = empaquetar_negi(geom, DX, hdx=HDX, kernel=KERNEL, contenedor=CONTENEDOR,
                                    margen=MARGEN, tol=TOL, max_iter=MAX_ITER,
                                    min_iter=MIN_ITER)
    dist = DistribucionSPH(x, geom, "15_negi", DX, rho0=RHO0, h_factor=H_FACTOR_SIM)
    procesar(dist, hist=hist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO,
             abrir_html=ABRIR_HTML)
