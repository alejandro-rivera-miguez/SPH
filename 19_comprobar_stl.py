"""
COMPROBAR UN .STL antes de generar particulas
==============================================
1) Copia tu fichero en la carpeta  stl/  de este proyecto.
2) Pon su nombre en ARCHIVO y ejecuta este script (F5).

Te dice si la malla es valida (cerrada), su volumen, su tamano, la masa que
saldra con RHO0 y cuantas particulas tendras con cada DX. Ademas dibuja la
malla, la capa de particulas de contorno (a dx/2 bajo la superficie) y tres
secciones con el dentro/fuera calculado, para comprobar que se lee bien.

Si todo esta bien, en cualquier script 3D (11-18) pon:
    GEOMETRIA = "stl"
    PARAMS_GEOM = {"archivo": "stl/<tu_fichero>.stl", "escala": 1.0, "centrar": True}
"""
import numpy as np
import matplotlib.pyplot as plt

from sph_common import crear_geometria_3d
from sph_common.visor3d import _ejes_iguales

# ------------------------------- CONFIGURACION -----------------------------
ARCHIVO = "stl/ejemplo_pajaro_capsula.stl"   # tu fichero dentro de la carpeta stl
ESCALA = 1.0          # 1000 si el STL esta en metros y quieres mm
CENTRAR = True        # mueve el centro de masas al origen
DX = 6.65             # [mm]
RHO0 = 1.1317e-6      # [kg/mm^3]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g = crear_geometria_3d("stl", archivo=ARCHIVO, escala=ESCALA, centrar=CENTRAR)
    lo, hi = g.bbox
    print("\n=== Comprobacion del STL ===")
    print(f"  fichero              {ARCHIVO}")
    print(f"  triangulos           {len(g.F)}   vertices {len(g.V)}")
    print(f"  malla cerrada        {'SI' if g.aristas_abiertas == 0 else 'NO'} "
          f"(aristas abiertas: {g.aristas_abiertas}, no-manifold: {g.aristas_no_manifold})")
    print(f"  tamano (x, y, z)     {np.round(hi - lo, 2).tolist()} (unidades del STL x escala)")
    print(f"  volumen              {g.volumen:.6g}")
    print(f"  area superficie      {g.area:.6g}")
    print(f"  masa con RHO0        {RHO0 * g.volumen:.5g}")
    print(f"  desplazamiento       {np.round(g.desplazamiento, 3).tolist()} (aplicado al centrar)")
    for d in (DX * 2, DX * 1.5, DX, DX / 1.5, DX / 2):
        print(f"  DX = {d:7.3f}  ->  ~{int(round(g.volumen / d ** 3)):8d} particulas")
    if (hi - lo).max() < 1.0 or (hi - lo).max() > 1e5:
        print("  OJO: el tamano parece raro para mm. Revisa ESCALA (metros -> 1000).")

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    V, F = g.malla()
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="lightgrey", alpha=0.5,
                    linewidth=0.1, edgecolor="grey")
    _ejes_iguales(ax, g)
    ax.set_title(f"{g.nombre}: malla ({len(g.F)} triangulos)", fontsize=9)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    pts, area = g.muestrear_offset(-0.5 * DX, DX, relajar=20)
    ax.scatter(*pts.T, s=3, c="r")
    _ejes_iguales(ax, g)
    ax.set_title(f"capa de contorno a dx/2 ({len(pts)} puntos)", fontsize=9)

    # secciones: puntos de una malla regular coloreados dentro (azul) / fuera (gris)
    for k, (eje, a, b) in enumerate((("z", 0, 1), ("y", 0, 2), ("x", 1, 2))):
        ax = fig.add_subplot(2, 3, 4 + k)
        u = np.linspace(lo[a] - DX, hi[a] + DX, 160)
        v = np.linspace(lo[b] - DX, hi[b] + DX, 160)
        U, W = np.meshgrid(u, v)
        P = np.zeros((U.size, 3))
        P[:, a], P[:, b] = U.ravel(), W.ravel()
        d = g.distancia_firmada(P).reshape(U.shape)
        im = ax.contourf(U, W, d, levels=np.linspace(-0.5 * (hi - lo).max(), 0.5 * (hi - lo).max(), 21),
                         cmap="RdBu_r")
        ax.contour(U, W, d, levels=[0], colors="k", linewidths=1.2)
        ax.set_aspect("equal")
        ax.set_title(f"seccion {eje}=0: distancia firmada (azul = dentro)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
    ax = fig.add_subplot(2, 3, 3)
    ax.axis("off")
    ax.text(0, 1, f"volumen   {g.volumen:.5g}\narea      {g.area:.5g}\nmasa      {RHO0 * g.volumen:.4g}\n"
                  f"N (DX={DX})  ~{int(round(g.volumen / DX ** 3))}\n"
                  f"cerrada   {'SI' if g.aristas_abiertas == 0 else 'NO'}",
            family="monospace", va="top", fontsize=11)
    fig.tight_layout()
    plt.show()
