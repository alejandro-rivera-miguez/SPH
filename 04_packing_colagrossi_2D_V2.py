"""
PASO 4 - Colagrossi et al. (2012) - VERSION 2: Wendland + afinado W4  [paper 5]
======================================================================
Se parte de una red y se dejan evolucionar las particulas con
    du/dt = -beta grad(Gamma) - zeta u
hasta el equilibrio, con el kernel Wendland C2. El contorno se modela con
capas de particulas FIJAS paralelas a la superficie (dx/2, 3dx/2, ...).

  H_FACTOR  h/dx del Wendland durante el empaquetado (2.0 (paper, tabla 1)).
  ALPHA     amortiguamiento adimensional (paper: 1e-3 .. 5e-3).
  TOL_EK    parada cuando la energia cinetica especifica < TOL_EK.

FASE 2 (afinado con W4, el kernel de PAM-CRASH):
  Parte del resultado de la fase 1 y lo lleva al equilibrio del W4 con
  h = H_FACTOR_SIM * dx. Se activa la repulsion de Negi porque el W4 sufre
  "pairing instability" (las particulas se emparejan, se ponen una encima
  de otra) y se congela la capa exterior (FIJAR_CAPA) para conservar la
  superficie regular que deja Wendland.
  Resultado tipico (2D, capsula): |grad Gamma|*h interior ~3 veces menor que
  en la V1, con el mismo o mejor CV de superficie.

Las metricas se calculan con el kernel W4 y H_FACTOR_SIM (los de PAM-CRASH).
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
# --- fase 2: afinado con W4 ---
ITER_W4 = 1000              # iteraciones maximas de la fase 2
ALPHA_W4 = 5e-2             # amortiguamiento de la fase 2 (mas alto: converge sin oscilar)
FIJAR_CAPA = 1.0            # congela las particulas a menos de FIJAR_CAPA*dx de la superficie (0 = no)
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
    # Fase 1: Wendland (estable, sin emparejamiento de particulas)
    x, hist, info = empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR, alpha=ALPHA,
                                          kernel="wendland", red_inicial_tipo=RED_INICIAL,
                                          max_iter=MAX_ITER, tol_ek=TOL_EK)
    # Fase 2: afinado con W4 y la h de la simulacion + repulsion anti-emparejamiento
    x, hist, info = empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR_SIM, alpha=ALPHA_W4,
                                          kernel="w4", x_inicial=x, repulsion=True,
                                          fijar_capa=FIJAR_CAPA, max_iter=ITER_W4,
                                          min_iter=200, tol_ek=TOL_EK)
    dist = DistribucionSPH(x, geom, "04_colagrossi_V2", DX, rho0=RHO0, h_factor=H_FACTOR_SIM)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, hist=hist, mostrar=MOSTRAR)
