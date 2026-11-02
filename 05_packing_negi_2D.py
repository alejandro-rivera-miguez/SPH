"""
PASO 5 - Algoritmo hibrido de Negi & Ramachandran (2021)  [paper 6]
==================================================================
(El codigo del repositorio sph_geom reescrito sin PySPH.)

  * Red hexagonal de particulas LIBRES en una caja que envuelve al cuerpo,
    rodeada de particulas CONGELADAS.
  * El contorno se desplaza dx/2 hacia dentro (ec. 11) y las libres que se
    acercan se convierten en particulas de CONTORNO que solo deslizan sobre el.
  * Fuerzas: gradiente de densidad numerica (pb) + repulsion (ec. 15-16) +
    amortiguamiento. Esquinas convexas: particula fija en la esquina.
  * Al final se separan las particulas interiores (el cuerpo).

Constantes por defecto = las del codigo de referencia (pb=1, nu=0.2/dx,
k=0.004 dx, corte 0.95, quintic spline h=1.2 dx, proyeccion cada 50 it,
tolerancia 1e-4 en max|u|dt/h).
"""
import matplotlib.pyplot as plt
import numpy as np

from sph_common import crear_geometria, DistribucionSPH, procesar
from sph_common.packing import empaquetar_negi
from sph_common.visor import _contorno

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2
# --- empaquetado ---
HDX = 1.2
TOL = 1e-4
MAX_ITER = 8000
FREQ_PROY = 50
VER_CAJA_COMPLETA = True    # dibuja tambien las particulas exteriores/congeladas
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    x, hist, info = empaquetar_negi(geom, DX, hdx=HDX, tol=TOL, max_iter=MAX_ITER,
                                    freq_proy=FREQ_PROY)
    if VER_CAJA_COMPLETA:
        fig, ax = plt.subplots(figsize=(10, 6))
        xa, t, xf = info["x_todas"], info["tipo"], info["x_congeladas"]
        ax.scatter(xf[:, 0], xf[:, 1], s=6, c="#55aa55", label="congeladas")
        ax.scatter(xa[t == 0, 0], xa[t == 0, 1], s=6, c="#4477aa", label="libres")
        ax.scatter(xa[t == 1, 0], xa[t == 1, 1], s=10, c="#cc3311", label="contorno")
        ax.scatter(xa[t == 2, 0], xa[t == 2, 1], s=30, c="k", marker="s", label="esquinas")
        _contorno(ax, geom, color="k", lw=0.8)
        ax.set_aspect("equal"); ax.legend(loc="upper right")
        ax.set_title("Negi: todas las particulas al final del empaquetado")
        fig.tight_layout()
    dist = DistribucionSPH(x, geom, "05_negi", DX, rho0=RHO0, h_factor=H_FACTOR_SIM,
                           info={k: v for k, v in info.items() if np.isscalar(v)})
    procesar(dist, hist=hist, mostrar=MOSTRAR)
