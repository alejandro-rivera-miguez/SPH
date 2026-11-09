"""
PASO 6 - Weighted Voronoi Tessellation extendida (EWVT)  [paper 4, DLR]
=====================================================================
Metodo de Siemann & Ritt (2019) para modelos de pajaro SPH (bird-strike),
basado en Diehl et al. (2015):

    dr_i = xi h sum_j f(r_ij) e_ij ,   f = 1/(r/h + chi)^2 - 1/(2 + chi)^2
    xi_{n+1} = (1 - kappa) xi_n

  * Se parte de una red cubica "agitada" aleatoriamente hasta un 50 % de dx.
  * El contorno actua como contacto rigido a una distancia = radio de
    particula (dx/2), como la "contact height" del paper.
  * N = Area/dx^2 exacto -> todas las particulas con el mismo volumen y masa.

Parametros del paper (tabla 1): xi = 4e-2, kappa = 4e-4, chi = 1e-2.
En el paper la WVT da 0.83 <= Lambda <= 1 y espaciado +-5 %.
Este metodo admite de forma natural resolucion variable h(x) -> sera la
base de la fase de "inflation".
"""
from sph_common import crear_geometria, DistribucionSPH, procesar
from sph_common.packing import empaquetar_wvt

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2
# --- WVT ---
H_FACTOR_WVT = 1.0          # h del WVT relativo a dx (soporte 2h)
XI = 4e-2
KAPPA = 4e-4
CHI = 1e-2
CR = 1e-8                   # criterio de convergencia dR/dRmax
MAX_ITER = 10000
AGITAR = 0.5                # agitado inicial (fraccion de dx)
SEMILLA = 0                 # cambia la semilla para otra distribucion aleatoria
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
    x, hist, info = empaquetar_wvt(geom, DX, h_factor=H_FACTOR_WVT, xi=XI, kappa=KAPPA,
                                   chi=CHI, CR=CR, max_iter=MAX_ITER, agitar=AGITAR,
                                   semilla=SEMILLA)
    dist = DistribucionSPH(x, geom, "06_wvt", DX, rho0=RHO0, h_factor=H_FACTOR_SIM,
                           info=info)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, hist=hist, mostrar=MOSTRAR)
