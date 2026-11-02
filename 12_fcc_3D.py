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
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x = red_fcc3d(geom, DX, mismo_volumen=MISMO_VOLUMEN)
    dist = DistribucionSPH(x, geom, "12_fcc", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO, abrir_html=ABRIR_HTML)
