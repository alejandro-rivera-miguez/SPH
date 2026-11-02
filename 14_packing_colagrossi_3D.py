"""
PASO 4 (3D) - Particle packing de Colagrossi et al. (2012)  [paper 5]
====================================================================
Igual que en 2D: du/dt = -beta grad(Gamma) - zeta u, con capas de
particulas FIJAS sobre superficies paralelas al contorno (dx/2, 3dx/2, ...)
que hacen de pared, para que la superficie quede lisa.

H_FACTOR: h/dx del kernel durante el empaquetado. En 2D el paper recomienda
2.0; en 3D se usa 1.4 por coste (con 2.0 cada particula tiene ~270 vecinos).
Tiempo orientativo con DX = 6.65 (~5600 particulas): 4-8 min.
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar
from sph_common.packing import empaquetar_colagrossi

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
RED_INICIAL = "cubica"      # "cubica", "hexagonal" (hcp), "fcc"
H_FACTOR = 1.4
ALPHA = 5e-3
MAX_ITER = 3000
TOL_EK = 1e-7
MOSTRAR = True
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x, hist, info = empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR, alpha=ALPHA,
                                          red_inicial_tipo=RED_INICIAL, max_iter=MAX_ITER,
                                          tol_ek=TOL_EK)
    dist = DistribucionSPH(x, geom, "14_colagrossi", DX, rho0=RHO0, h_factor=H_FACTOR_SIM)
    procesar(dist, hist=hist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO,
             abrir_html=ABRIR_HTML)
