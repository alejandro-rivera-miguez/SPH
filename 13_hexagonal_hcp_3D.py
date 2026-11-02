"""
PASO 3 (3D) - Empaquetamiento hexagonal compacto (HCP)
======================================================
Capas triangulares (hexagonales) apiladas ABAB a lo largo del eje X del
pajaro. 12 vecinos a la misma distancia; es tan compacto como el FCC
(ABCABC), solo cambia el apilamiento. Negi (paper 6) usa empaquetado
hexagonal como estado inicial porque en 3D es estable y la red cubica no.

RED = "hcp" (por defecto) o "bcc" (cubica centrada en el cuerpo, 8 vecinos)
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar, red_regular_3d

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # o "stl" para usar tu propio fichero (ver abajo)
PARAMS_GEOM = {}
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
DX = 6.65
RHO0 = 1.1317e-6
H_FACTOR = 1.2
RED = "hcp"
MISMO_VOLUMEN = True
MOSTRAR = True
VISOR_INTERACTIVO = True
ABRIR_HTML = False
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x = red_regular_3d(geom, DX, RED, mismo_volumen=MISMO_VOLUMEN)
    dist = DistribucionSPH(x, geom, f"13_{RED}", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO, abrir_html=ABRIR_HTML)
