"""
PASO 4 - Particle packing de Colagrossi et al. (2012)  [paper 5]
================================================================
Se parte de una red (cubica u hexagonal) y se dejan evolucionar las
particulas con   du/dt = -beta grad(Gamma) - zeta u
hasta que grad(Gamma) ~ 0 (equilibrio). El contorno del cuerpo se modela con
capas de particulas FIJAS paralelas al contorno (a dx/2, 3dx/2, ...).

Parametros clave (paper):
  H_FACTOR  h/dx del kernel Wendland durante el empaquetado. Tabla 1 del paper:
            con h/dx = 2.0 la red cartesiana y la triangular son estables.
  ALPHA     amortiguamiento adimensional, rango recomendado 1e-3 .. 5e-3.
  TOL_EK    parada cuando la energia cinetica especifica < TOL_EK * beta.

El smoothing length para la SIMULACION (H_FACTOR_SIM) es independiente.
"""
from sph_common import crear_geometria, DistribucionSPH, procesar
from sph_common.packing import empaquetar_colagrossi

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2          # h para la simulacion / exportacion
# --- empaquetado ---
RED_INICIAL = "cubica"      # "cubica" o "hexagonal"
H_FACTOR = 2.0
ALPHA = 5e-3
MAX_ITER = 4000
TOL_EK = 1e-7
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    x, hist, info = empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR, alpha=ALPHA,
                                          red_inicial_tipo=RED_INICIAL, max_iter=MAX_ITER,
                                          tol_ek=TOL_EK)
    dist = DistribucionSPH(x, geom, "04_colagrossi", DX, rho0=RHO0, h_factor=H_FACTOR_SIM,
                           info=info)
    procesar(dist, hist=hist, mostrar=MOSTRAR)
