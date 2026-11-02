"""
PASO 0 - Ver las geometrias 2D disponibles
==========================================
Dibuja todas las geometrias con sus esquinas detectadas y el contorno
desplazado dx/2 hacia dentro (donde se colocara la primera capa de
particulas en el metodo de Negi).
"""
import matplotlib.pyplot as plt
from sph_common import crear_geometria
from sph_common.visor import _contorno

DX = 4.0
GEOMETRIAS = {
    "circulo": {}, "cuadrado": {}, "rectangulo": {}, "triangulo": {},
    "elipse": {}, "hexagono": {}, "estrella": {}, "capsula": {},
    "poligono": {"vertices": [(-60, -40), (70, -40), (70, 10), (10, 50), (-60, 30)]},
}

if __name__ == "__main__":
    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    for ax, (nombre, p) in zip(axs.ravel(), GEOMETRIAS.items()):
        g = crear_geometria(nombre, **p)
        _contorno(ax, g, color="k", lw=1.2)
        for c in g.curva_offset(-0.5 * DX, DX):
            v = c.v
            ax.plot(list(v[:, 0]) + [v[0, 0]], list(v[:, 1]) + [v[0, 1]], "r--", lw=0.8)
        e = g.esquinas_offset(0.5 * DX)
        if len(e):
            ax.plot(e[:, 0], e[:, 1], "bs", ms=4)
        ax.set_title(f"{nombre}  A = {g.area:.0f} mm2", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle("Negro: contorno | rojo: contorno - dx/2 | azul: esquinas fijas")
    fig.tight_layout()
    plt.show()
