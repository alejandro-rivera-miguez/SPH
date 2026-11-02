"""
PASO 6 (3D) - Extended Weighted Voronoi Tessellation  [paper 4, DLR]
===================================================================
El metodo que usan Siemann & Ritt para el pajaro SPH en PAM-CRASH/VPS.
Red cubica agitada (50 % dx) -> desplazamientos repulsivos con intensidad xi
decreciente -> contacto rigido con la superficie a dx/2.
N = Volumen / dx^3 exacto: con DX = 6.65 salen ~5440 particulas, el modelo
"grueso" del paper (el fino tiene 40824 -> DX ~ 3.4).
Tiempo orientativo con DX = 6.65: 1-3 min.
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar
from sph_common.packing import empaquetar_wvt

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # o "stl" para usar tu propio fichero (ver abajo)
PARAMS_GEOM = {}
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
DX = 6.65
RHO0 = 1.1317e-6
H_FACTOR_SIM = 1.2
# --- WVT (tabla 1 del paper) ---
H_FACTOR_WVT = 1.0
XI = 4e-2
KAPPA = 4e-4
CHI = 1e-2
CR = 1e-8
MAX_ITER = 6000
AGITAR = 0.5
SEMILLA = 0
MOSTRAR = True
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x, hist, info = empaquetar_wvt(geom, DX, h_factor=H_FACTOR_WVT, xi=XI, kappa=KAPPA,
                                   chi=CHI, CR=CR, max_iter=MAX_ITER, agitar=AGITAR,
                                   semilla=SEMILLA)
    dist = DistribucionSPH(x, geom, "16_wvt", DX, rho0=RHO0, h_factor=H_FACTOR_SIM)
    procesar(dist, hist=hist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO,
             abrir_html=ABRIR_HTML)
