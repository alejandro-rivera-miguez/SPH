"""
PASO 31 - INFLATION 3D: resolucion variable (refinar una zona), tambien con .stl
================================================================================
Igual que el 30 pero en 3D. Para tu STL:
    GEOMETRIA = "stl"
    PARAMS_GEOM = {"archivo": "stl/pajaro.stl", "escala": 1.0, "centrar": False}
  centrar=False mantiene las coordenadas del STL (las de tu modelo), asi las
  zonas se definen en las mismas coordenadas que en HyperMesh/PAM-CRASH.

Zonas:
  dict(tipo="esfera", centro=(x, y, z), radio=R)          cabeza del pajaro
  dict(tipo="caja",   pmin=(x0, y0, z0), pmax=(x1, y1, z1))
  dict(tipo="losa",   punto=(x, y, z), normal=(0, 0, 1), espesor=e)
                      -> banda alrededor de un plano: superficie libre de la piscina
  dict(tipo="punto",  centro=(x, y, z))
  Cada zona admite dx=... propio (si no, DX_FINO).

Rendimiento: usa todos los nucleos (variable de entorno SPH_HILOS para
limitarlos). Referencia en un portatil de 2 nucleos: ~55 000 particulas en
~4 min. Mira LEEME.md para casos de millones (piscina).
"""
from sph_common import crear_geometria_3d
from sph_common.inflation import CampoResolucion, ejecutar_inflation

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"          # esfera, cubo, caja, cilindro, capsula, elipsoide, stl
PARAMS_GEOM = {}               # stl: {"archivo": "stl/pajaro.stl", "escala": 1.0, "centrar": False}
DX_FINO = 3.0                  # mm
DX_GRUESO = 6.65               # mm
TASA = 0.08                    # crecimiento por capa
ZONAS = [
    dict(tipo="esfera", centro=(-107.0, 0.0, 0.0), radio=25.0),    # morro del pajaro
]
RHO0 = 9.5e-7
H_FACTOR_SIM = 1.2
ITER_WVT = 100
ITER_DENSIDAD = 500
CALIBRAR = True
CARPETA = "resultados"
EXPORTAR = True
SECCION = ("y", 0.0)           # plano de corte de la figura
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria_3d(GEOMETRIA, **PARAMS_GEOM)
    print(f"Geometria: {geom.nombre}, caja envolvente {geom.bbox[0].round(2)} - {geom.bbox[1].round(2)}")
    campo = CampoResolucion(DX_FINO, DX_GRUESO, TASA, ZONAS)
    ejecutar_inflation(geom, campo, RHO0, H_FACTOR_SIM, ITER_WVT, ITER_DENSIDAD, CALIBRAR,
                       CARPETA, exportar=EXPORTAR, mostrar=MOSTRAR, seccion=SECCION)
