"""
PASO 2 (3D) - Face centered cubic (FCC), como el panel SPH de HyperMesh
=======================================================================
Celda cubica con una particula en cada vertice y otra en el centro de cada
cara (4 particulas por celda, 12 vecinos a la misma distancia).

MISMO_VOLUMEN = True  -> lado de celda a = 4^(1/3) dx: cada particula ocupa
                         dx^3 (mismo N y misma masa que la cubica).
MISMO_VOLUMEN = False -> lado de celda a = dx (el "pitch" de HyperMesh):
                         4 veces mas particulas.
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar, red_fcc3d

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # o "stl" para usar tu propio fichero (ver abajo)
PARAMS_GEOM = {}
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
DX = 6.65
RHO0 = 1.1317e-6
H_FACTOR = 1.2
MISMO_VOLUMEN = True
MOSTRAR = True
# --- figuras (lista completa de paneles en sph_common/figuras.py) ---
FIGURAS = ["resumen", "superficie", "convergencia"]   # quita las que no quieras
PANELES = ["vista3d", "dnn@y", "lambda@z", "densidad@x",
           "hist_dnn", "hist_lambda", "perfil_dnn", "tabla"]
#   disponibles: particulas, lambda, densidad, gradgamma, gradgamma_sup, dnn,
#                hist_dnn, hist_lambda, hist_densidad, hist_gradgamma,
#                perfil_superficie, perfil_dnn, tabla, vista3d
#   los mapas son secciones: "lambda@z" corta por z = 0, "densidad@x=-50" por x = -50
SECCION = ("y", 0.0)      # plano por defecto de los mapas sin @
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x = red_fcc3d(geom, DX, mismo_volumen=MISMO_VOLUMEN)
    dist = DistribucionSPH(x, geom, "12_fcc", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO, abrir_html=ABRIR_HTML)
