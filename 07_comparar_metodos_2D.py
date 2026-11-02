"""
PASO 7 - Postprocesado y comparacion de metodos (2D)
=====================================================
Llama a todos los metodos (o carga los resultados ya guardados por los
scripts 01-06 en resultados/) para la MISMA geometria y el MISMO dx, calcula
las metricas de calidad y las compara:

  * Lambda = lmin/lmax del tensor de inercia SPH (Siemann, paper 4)
  * |grad Gamma| * h (Colagrossi, paper 5): "fuerza residual" en t = 0.
    Es lo que provoca el reacomodo espurio de particulas al empezar el
    calculo -> ruido en la presion.
  * Error de densidad SPH (Negi, paper 6)
  * Distancias entre vecinos, volumen de Voronoi (Siemann, figs. 5 y 6)
  * Hueco y regularidad de la capa exterior: la superficie que golpea primero.

Salidas en resultados/: comparacion_<geom>.csv, comparacion_<geom>_*.png
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sph_common import (crear_geometria, DistribucionSPH, calcular_metricas,
                        red_cubica, red_fcc, red_hexagonal, METRICAS_TABLA)
from sph_common.packing import empaquetar_colagrossi, empaquetar_negi, empaquetar_wvt
from sph_common.visor import panel_comparacion, barras_comparacion
from sph_common.metricas import DESCRIPCION

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2
METODOS = ["01_cubica", "02_fcc", "03_hexagonal", "04_colagrossi", "05_negi", "06_wvt"]
USAR_GUARDADOS = True     # True: usa resultados/<geom>_<metodo>.npz si existe
GUARDAR_NUEVOS = True     # guarda los que haya que calcular
CARPETA = "resultados"
MOSTRAR = True
# ---------------------------------------------------------------------------

# Para cada metrica: True si "mas alto es mejor", False si "mas bajo es mejor",
# o un valor objetivo (float) si lo ideal es acercarse a ese valor.
SENTIDO = {
    "Lambda_min": True, "Lambda_media": True, "Lambda_P5": True, "%_Lambda>0.9": True,
    "gradGamma_h_max": False, "gradGamma_h_media": False,
    "err_rho_Linf_%": False, "err_rho_std_%": False, "err_rho_media_%": 0.0, "dnn_min": True,
    "dnn_CV_%": False, "Vvor_max/min": 1.0, "Vvor_CV_%": False,
    "hueco_sup_media": 0.5, "hueco_sup_max": 0.5, "capa_ext_CV_%": False,
}


def generar(metodo, geom):
    if metodo == "01_cubica":
        return red_cubica(geom, DX)
    if metodo == "02_fcc":
        return red_fcc(geom, DX, mismo_volumen=True)
    if metodo == "03_hexagonal":
        return red_hexagonal(geom, DX, mismo_volumen=True)
    if metodo == "04_colagrossi":
        return empaquetar_colagrossi(geom, DX)[0]
    if metodo == "05_negi":
        return empaquetar_negi(geom, DX)[0]
    if metodo == "06_wvt":
        return empaquetar_wvt(geom, DX)[0]
    raise ValueError(metodo)


def obtener(metodo, geom):
    ruta = os.path.join(CARPETA, f"{geom.nombre}_{metodo}.npz")
    if USAR_GUARDADOS and os.path.exists(ruta):
        d = DistribucionSPH.cargar(ruta, geom)
        if abs(d.dx_nominal - DX) < 1e-9:
            print(f"  {metodo}: cargado de {ruta}")
            d.h_factor = H_FACTOR_SIM
            return d
        print(f"  {metodo}: el guardado tiene otro dx ({d.dx_nominal}), se recalcula")
    print(f"  {metodo}: calculando...")
    d = DistribucionSPH(generar(metodo, geom), geom, metodo, DX, rho0=RHO0,
                        h_factor=H_FACTOR_SIM)
    if GUARDAR_NUEVOS:
        d.guardar(CARPETA)
    return d


def mejor(clave, valores):
    s = SENTIDO.get(clave)
    v = np.array(valores, dtype=float)
    if s is None or np.all(~np.isfinite(v)):
        return None
    if s is True:
        return int(np.nanargmax(v))
    if s is False:
        return int(np.nanargmin(v))
    return int(np.nanargmin(np.abs(v - s)))


if __name__ == "__main__":
    os.makedirs(CARPETA, exist_ok=True)
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    lista, res = [], []
    for m in METODOS:
        d = obtener(m, geom)
        r, c = calcular_metricas(d)
        lista.append((d, r, c))
        res.append(r)

    # --- tabla en consola (* = mejor valor de cada fila) --------------------
    ancho = 15
    print("\n" + "metrica".ljust(20) + "".join(m[:ancho - 1].rjust(ancho) for m in METODOS))
    for k in METRICAS_TABLA:
        vals = [r[k] for r in res]
        b = mejor(k, vals)
        fila = ""
        for n, v in enumerate(vals):
            t = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"
            fila += (t + ("*" if n == b else " ")).rjust(ancho)
        print(k.ljust(20) + fila)
    print("\n(*) mejor valor de la fila. Significado de cada metrica:")
    for k in METRICAS_TABLA:
        print(f"   {k:<18s} {DESCRIPCION.get(k, '')}")

    # --- CSV ---------------------------------------------------------------
    ruta_csv = os.path.join(CARPETA, f"comparacion_{geom.nombre}.csv")
    with open(ruta_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["metrica"] + METODOS + ["descripcion"])
        for k in METRICAS_TABLA:
            w.writerow([k] + [f"{r[k]:.6g}" for r in res] + [DESCRIPCION.get(k, "")])
    print("\nCSV ->", ruta_csv)

    # --- figuras -------------------------------------------------------------
    base = os.path.join(CARPETA, f"comparacion_{geom.nombre}")
    panel_comparacion(lista, guardar=base + "_particulas.png", mostrar=False)
    barras_comparacion(res, guardar=base + "_barras.png", mostrar=False)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for d, r, c in lista:
        axs[0].hist(c["dnn"], bins=50, range=(0.6, 1.3), histtype="step", lw=1.5,
                    label=d.metodo, density=True)
        axs[1].hist(c["Lambda"][c["interior"]], bins=50, range=(0.7, 1.0),
                    histtype="step", lw=1.5, label=d.metodo, density=True)
        err = 100 * (c["rho_rel"][c["interior"]] - 1)
        axs[2].hist(err, bins=50, range=(-6, 6), histtype="step", lw=1.5,
                    label=d.metodo, density=True)
    axs[0].set_xlabel("distancia al vecino mas cercano / dx")
    axs[1].set_xlabel("Lambda (interior)")
    axs[2].set_xlabel("error de densidad SPH interior (%)")
    for ax in axs:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base + "_histogramas.png", dpi=120)
    print("Figuras ->", base + "_*.png")
    if MOSTRAR:
        plt.show()
