"""
PASO 1 (3D) - Red cubica simple
================================
Particulas a distancia dx formando cubos, recortadas a la geometria.

Ejecutar con F5. Se abre la figura de resumen (vista 3D + secciones) y, si
VISOR_INTERACTIVO = True, una ventana con un deslizador para cortar el cuerpo.
Ademas se genera resultados/<geom>_11_cubica_visor3D.html: abrelo con doble
clic para girar y cortar el pajaro en el navegador.
"""
from sph_common import crear_geometria_3d, DistribucionSPH, procesar, red_sc

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"     # esfera, cubo, caja, cilindro, capsula (pajaro), elipsoide, stl
PARAMS_GEOM = {}          # p.ej. {"R": 53.5}; esfera {"R": 50}; cilindro {"R": 50, "L": 150}
                          # caja {"Lx": 150, "Ly": 100, "Lz": 60}; elipsoide {"a":80,"b":50,"c":50}
# Con STL:  GEOMETRIA = "stl"
#           PARAMS_GEOM = {"archivo": "stl/mi_pajaro.stl", "escala": 1.0, "centrar": True}
#           (pon el .stl en la carpeta stl; escala=1000 si esta en metros)
DX = 6.65                 # [mm] 6.65 -> ~5440 particulas en el pajaro de 1.814 kg (Siemann)
RHO0 = 1.1317e-6          # [kg/mm^3] densidad del pajaro de Siemann (masa total 1.814 kg)
H_FACTOR = 1.2            # h = H_FACTOR * dx para la simulacion
MOSTRAR = True
# --- figuras (lista completa de paneles en sph_common/figuras.py) ---
FIGURAS = ["resumen", "superficie", "convergencia"]   # quita las que no quieras
PANELES = ["vista3d", "dnn@y", "lambda@z", "densidad@x",
           "hist_dnn", "hist_lambda", "perfil_dnn", "tabla"]
#   disponibles: particulas, lambda, densidad, gradgamma, gradgamma_sup, dnn,
#                hist_dnn, hist_lambda, hist_densidad, hist_gradgamma,
#                perfil_superficie, perfil_dnn, tabla, vista3d
#   los mapas son secciones: "lambda@z" corta por z = 0, "densidad@x=-50" por x = -50
SECCION = ("y", 0.0)      # plano por defecto de los mapas sin @
VISOR_INTERACTIVO = True  # ventana con deslizador de corte
ABRIR_HTML = False        # abrir automaticamente el visor HTML en el navegador
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x = red_sc(geom, DX)
    dist = DistribucionSPH(x, geom, "11_cubica", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, paneles=PANELES, seccion=SECCION, figuras=FIGURAS, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO, abrir_html=ABRIR_HTML)
