"""
PASO 4 - Colagrossi et al. (2012) - VERSION 1: solo Wendland  [paper 5]
======================================================================
Se parte de una red y se dejan evolucionar las particulas con
    du/dt = -beta grad(Gamma) - zeta u
hasta el equilibrio, con el kernel Wendland C2. El contorno se modela con
capas de particulas FIJAS paralelas a la superficie (dx/2, 3dx/2, ...).

  H_FACTOR  h/dx del Wendland durante el empaquetado (2.0 (paper, tabla 1)).
  ALPHA     amortiguamiento adimensional (paper: 1e-3 .. 5e-3).
  TOL_EK    parada cuando la energia cinetica especifica < TOL_EK.

Las metricas se calculan con el kernel W4 y H_FACTOR_SIM (los de PAM-CRASH).
La version 2 (04_..._V2) anade un afinado con W4 despues de esto.
"""
from sph_common import crear_geometria, DistribucionSPH, procesar
from sph_common.packing import empaquetar_colagrossi

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2          # h/dx de la SIMULACION (el de PAM-CRASH)
# --- empaquetado (Wendland) ---
RED_INICIAL = "cubica"      # "cubica" o "hexagonal"
H_FACTOR = 2.0
ALPHA = 5e-3
MAX_ITER = 4000
TOL_EK = 1e-7
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
    x, hist, info = empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR, alpha=ALPHA,
                                          kernel="wendland", red_inicial_tipo=RED_INICIAL,
                                          max_iter=MAX_ITER, tol_ek=TOL_EK)
    dist = DistribucionSPH(x, geom, "04_colagrossi_V1", DX, rho0=RHO0, h_factor=H_FACTOR_SIM)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, hist=hist, mostrar=MOSTRAR)
