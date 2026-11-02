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
VISOR_INTERACTIVO = True  # ventana con deslizador de corte
ABRIR_HTML = False        # abrir automaticamente el visor HTML en el navegador
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    x = red_sc(geom, DX)
    dist = DistribucionSPH(x, geom, "11_cubica", DX, rho0=RHO0, h_factor=H_FACTOR)
    procesar(dist, mostrar=MOSTRAR, visor_interactivo=VISOR_INTERACTIVO, abrir_html=ABRIR_HTML)
