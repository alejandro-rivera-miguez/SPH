"""
PASO 0 (3D) - Ver las geometrias 3D disponibles
================================================
Dibuja cada geometria con su malla y los puntos de la superficie desplazada
dx/2 hacia dentro (donde va la primera capa de particulas en Negi).
Todas estan centradas en el origen; el eje de los cuerpos de revolucion
(cilindro, capsula = pajaro) es el X, que es la direccion de impacto.
"""
import matplotlib.pyplot as plt
from sph_common import crear_geometria_3d
from sph_common.visor3d import _ejes_iguales

DX = 6.65
GEOMETRIAS = {"esfera": {}, "cubo": {}, "caja": {}, "cilindro": {}, "capsula": {},
              "elipsoide": {}}

if __name__ == "__main__":
    fig = plt.figure(figsize=(15, 9))
    for k, (nombre, p) in enumerate(GEOMETRIAS.items()):
        g = crear_geometria_3d(nombre, **p)
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        V, F = g.malla(32)
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="lightgrey",
                        alpha=0.25, linewidth=0.2, edgecolor="grey")
        pts, area = g.muestrear_offset(-0.5 * DX, DX, relajar=20)
        ax.scatter(*pts.T, s=3, c="r")
        _ejes_iguales(ax, g)
        ax.set_title(f"{nombre}: V = {g.volumen:.3g} mm3\n"
                     f"{len(pts)} puntos a dx/2 bajo la superficie", fontsize=9)
    fig.tight_layout()
    plt.show()
