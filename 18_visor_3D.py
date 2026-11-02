"""
VISOR 3D - abrir cualquier resultado guardado sin recalcular
=============================================================
Carga resultados/<GEOMETRIA>_<METODO>.npz (lo guardan los scripts 11-16),
calcula las metricas y abre:
  1) la ventana interactiva de matplotlib (deslizador de corte, campo de color)
  2) el visor HTML en el navegador (WebGL: girar, zoom, cortes, lonchas)
"""
import os
import webbrowser
import matplotlib.pyplot as plt

from sph_common import crear_geometria_3d, DistribucionSPH, calcular_metricas
from sph_common.visor3d import visor_interactivo_3d
from sph_common.visor_html import exportar_html

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # o "stl" para usar tu propio fichero (ver abajo)
PARAMS_GEOM = {}            # los mismos que usaste al generar
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
METODO = "16_wvt"           # 11_cubica, 12_fcc, 13_hcp, 14_colagrossi, 15_negi, 16_wvt
H_FACTOR_SIM = 1.2
CARPETA = "resultados"
ABRIR_HTML = True
VENTANA_MATPLOTLIB = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    ruta = os.path.join(CARPETA, f"{geom.nombre}_{METODO}.npz")
    dist = DistribucionSPH.cargar(ruta, geom)
    dist.h_factor = H_FACTOR_SIM
    print(dist.resumen())
    r, campos = calcular_metricas(dist)
    html = exportar_html(dist, campos, os.path.join(CARPETA, f"{geom.nombre}_{METODO}_visor3D.html"), r)
    if ABRIR_HTML:
        webbrowser.open("file://" + os.path.abspath(html))
    if VENTANA_MATPLOTLIB:
        visor_interactivo_3d(dist, campos)
        plt.show()
