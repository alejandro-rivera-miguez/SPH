"""
PASO 7 - Postprocesado y comparacion de metodos
=======================================================
Carga los resultados guardados por los scripts 01-06 en resultados/ (o los
calcula si no existen) para la MISMA geometria y el MISMO dx, calcula las
metricas con el kernel W4 y H_FACTOR_SIM (los de PAM-CRASH) y los compara.

Figuras que genera (en resultados/, prefijo comparacion2D_<geom>):
  _empaquetados.png          rejilla 2x3 con los empaquetados de FIG_EMPAQUETADOS
  _comparar_<metrica>.png    una por cada metrica de COMPARAR_METRICAS:
                             fila superior = los empaquetados de COMPARAR_METODOS
                             (2 o 3), fila inferior = la metrica con la misma escala
                             de color, % frente al mejor y el mejor en VERDE
  _tabla_diferencias.png     tabla: valor de cada metrica, % frente al mejor,
                             mejor de cada fila en verde
  _superficie.png            |grad Gamma|*h en la capa exterior de todos
  _particulas.png, _barras.png, _histogramas.png   (las de siempre)
Ademas: comparacion2D_<geom>.csv con todas las metricas y su % frente al mejor.

Metricas para COMPARAR_METRICAS:
  "densidad"       error de densidad SPH (criterio: desviacion tipica interior)
  "gradgamma"      |grad Gamma|*h interior (criterio: maximo)
  "gradgamma_sup"  |grad Gamma|*h en la capa exterior (criterio: CV)
  "lambda"         Lambda interior (criterio: media, mayor es mejor)
  "dnn"            distancia al vecino mas cercano (criterio: dispersion)
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sph_common import (crear_geometria, DistribucionSPH, calcular_metricas,
                        red_cubica, red_fcc, red_hexagonal, METRICAS_TABLA)
from sph_common.packing import empaquetar_colagrossi, empaquetar_negi, empaquetar_wvt
from sph_common.visor import panel_comparacion, barras_comparacion, comparacion_superficie
from sph_common.metricas import DESCRIPCION
from sph_common.figuras import (figura_empaquetados, figura_comparar, figura_tabla_diferencias,
                                imprimir_diferencias, mejor_indice, diferencia_frente_al_mejor)

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"
PARAMS_GEOM = {}
DX = 4.0
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2          # h/dx de PAM-CRASH (las metricas se calculan con el)
METODOS = ['01_cubica', '02_fcc', '03_hexagonal', '04_colagrossi_V1', '04_colagrossi_V2', '05_negi', '06_wvt']
USAR_GUARDADOS = True     # True: usa resultados/<geom>_<metodo>.npz si existe
GUARDAR_NUEVOS = True     # guarda los que haya que calcular
CARPETA = "resultados"
MOSTRAR = True
# --- figuras nuevas ---
FIG_EMPAQUETADOS = ['01_cubica', '02_fcc', '03_hexagonal', '04_colagrossi_V2', '05_negi', '06_wvt']
COMPARAR_METODOS = ['01_cubica', '04_colagrossi_V2', '06_wvt']   # 2 o 3
COMPARAR_METRICAS = ["densidad", "gradgamma", "gradgamma_sup", "lambda"]   # una figura por metrica
SECCION = ("y", 0.0)        # 3D: plano de corte (eje, posicion) de las figuras nuevas
# ---------------------------------------------------------------------------


def generar(metodo, geom):
    if metodo == "01_cubica":
        return red_cubica(geom, DX)
    if metodo == "02_fcc":
        return red_fcc(geom, DX, mismo_volumen=True)
    if metodo == "03_hexagonal":
        return red_hexagonal(geom, DX, mismo_volumen=True)
    if metodo == "04_colagrossi_V1":
        return empaquetar_colagrossi(geom, DX, kernel="wendland", max_iter=4000)[0]
    if metodo == "04_colagrossi_V2":
        x = empaquetar_colagrossi(geom, DX, kernel="wendland", max_iter=4000)[0]
        return empaquetar_colagrossi(geom, DX, h_factor=H_FACTOR_SIM, alpha=5e-2, kernel="w4",
                                     x_inicial=x, repulsion=True, fijar_capa=1.0,
                                     max_iter=1000, min_iter=200)[0]
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


if __name__ == "__main__":
    os.makedirs(CARPETA, exist_ok=True)
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    lista, res = [], []
    for met in METODOS:
        d = obtener(met, geom)
        r, c = calcular_metricas(d)
        lista.append((d, r, c))
        res.append(r)

    # --- tabla en consola (* = mejor valor de cada fila) --------------------
    ancho = 17
    print("\n" + "metrica".ljust(22) + "".join(x[:ancho - 1].rjust(ancho) for x in METODOS))
    for k in METRICAS_TABLA:
        vals = [r[k] for r in res]
        b = mejor_indice(k, vals)
        fila = ""
        for j, v in enumerate(vals):
            t = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"
            fila += (t + ("*" if j == b else " ")).rjust(ancho)
        print(k.ljust(22) + fila)
    print("\n(*) mejor valor de la fila. Significado de cada metrica:")
    for k in METRICAS_TABLA:
        print(f"   {k:<22s} {DESCRIPCION.get(k, '')}")
    imprimir_diferencias(lista)

    # --- CSV (valores y % frente al mejor) -----------------------------------
    ruta_csv = os.path.join(CARPETA, f"comparacion2D_{geom.nombre}.csv")
    with open(ruta_csv, "w", newline="") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["metrica"] + METODOS + ["mejor"] + [f"% {x}" for x in METODOS] + ["descripcion"])
        for k in METRICAS_TABLA:
            vals = [r[k] for r in res]
            b, textos, pct = diferencia_frente_al_mejor(k, vals)
            wr.writerow([k] + [f"{v:.6g}" for v in vals]
                        + [METODOS[b] if b is not None else ""] + textos
                        + [DESCRIPCION.get(k, "")])
    print("\nCSV ->", ruta_csv)

    # --- figuras -------------------------------------------------------------
    base = os.path.join(CARPETA, f"comparacion2D_{geom.nombre}")
    figura_empaquetados(lista, FIG_EMPAQUETADOS, seccion=SECCION,
                        guardar=base + "_empaquetados.png", mostrar=False)
    for met in COMPARAR_METRICAS:
        figura_comparar(lista, COMPARAR_METODOS, met, seccion=SECCION,
                        guardar=base + f"_comparar_{met}.png", mostrar=False)
    figura_tabla_diferencias(lista, guardar=base + "_tabla_diferencias.png", mostrar=False)
    panel_comparacion(lista, guardar=base + "_particulas.png", mostrar=False)
    barras_comparacion(res, guardar=base + "_barras.png", mostrar=False)
    comparacion_superficie(lista, guardar=base + "_superficie.png", mostrar=False)

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
