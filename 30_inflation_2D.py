"""
PASO 30 - INFLATION 2D: resolucion variable (refinar una zona)
==============================================================
Script independiente de los demas. Genera particulas con dx = DX_FINO dentro de
las ZONAS, crecimiento suave (TASA por capa) y DX_GRUESO en el resto.

  dx(x) = min( DX_GRUESO , dx_zona + TASA * distancia_a_la_zona )

  * TASA = 0.05 -> cada capa de particulas es un 5 % mayor que la anterior.
    Longitud de la transicion = (DX_GRUESO - DX_FINO) / TASA.
  * Masa por particula m_i = rho0 * V_i, con V_i ~ dx_i^2 calibrado con el
    kernel del solver (W4, h = H_FACTOR_SIM * dx). Masa total EXACTA.
  * En 2D la masa es por mm de espesor (modelo de prueba).

Algoritmo: celdas adaptativas -> WVT con h variable (Siemann) -> correccion de
densidad (Gamma con volumenes variables, Colagrossi) -> calibracion de volumenes.

Zonas (las coordenadas son las de la geometria; mira el eje en la figura):
  dict(tipo="esfera", centro=(x, y), radio=R)            circulo
  dict(tipo="caja",   pmin=(x0, y0), pmax=(x1, y1))       rectangulo
  dict(tipo="losa",   punto=(x, y), normal=(nx, ny), espesor=e)  banda recta
  dict(tipo="punto",  centro=(x, y))
  Cada zona admite dx=... propio (si no, DX_FINO).
"""
from sph_common import crear_geometria
from sph_common.inflation import CampoResolucion, ejecutar_inflation

# ------------------------------- CONFIGURACION -----------------------------
GEOMETRIA = "capsula"          # circulo, cuadrado, rectangulo, elipse, capsula...
PARAMS_GEOM = {}               # p.ej. {"R": 53.5}
DX_FINO = 1.5                  # mm, dentro de las zonas
DX_GRUESO = 4.0                # mm, lejos de las zonas
TASA = 0.08                    # crecimiento por capa (0.03 - 0.10 recomendado)
ZONAS = [
    dict(tipo="esfera", centro=(-107.0, 0.0), radio=15.0),     # morro del pajaro
]
RHO0 = 9.5e-7                  # kg/mm^3
H_FACTOR_SIM = 1.2             # h/dx del solver (metricas y calibracion)
ITER_WVT = 100                 # ordena las particulas (isotropia)
ITER_DENSIDAD = 500            # corrige la densidad (lo importante)
CALIBRAR = True                # calibrar volumenes con el kernel del solver
CARPETA = "resultados"
EXPORTAR = True                # PAM-CRASH (.inc), CSV y VTK
MOSTRAR = True
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geom = crear_geometria(GEOMETRIA, **PARAMS_GEOM)
    campo = CampoResolucion(DX_FINO, DX_GRUESO, TASA, ZONAS)
    ejecutar_inflation(geom, campo, RHO0, H_FACTOR_SIM, ITER_WVT, ITER_DENSIDAD, CALIBRAR,
                       CARPETA, exportar=EXPORTAR, mostrar=MOSTRAR)
