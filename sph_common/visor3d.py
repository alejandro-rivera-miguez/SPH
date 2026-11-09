"""
Visualizacion 3D con matplotlib (no necesita nada extra).

  panel_resumen_3d(dist, r, campos)   figura de resumen: vista 3D + cortes
                                      (secciones) en los tres planos +
                                      histogramas + metricas.
  visor_interactivo_3d(dist, campos)  ventana interactiva: deslizador para
                                      cortar el pajaro por un plano y ver el
                                      interior, eleccion del campo de color y
                                      la seccion 2D a tamano real.

Para ver el volumen con mas detalle esta tambien el visor HTML
(sph_common/visor_html.py), que se abre en el navegador.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.widgets import Slider, RadioButtons

from .metricas import METRICAS_TABLA

_EJES = {"x": 0, "y": 1, "z": 2}


def campos_para_color(dist, campos):
    """Campos escalares disponibles para colorear (nombre -> (valores, cmap, vmin, vmax))."""
    inter = campos["interior"]
    err = 100 * (campos["rho_rel"] - 1)
    lim = max(1.0, np.nanpercentile(np.abs(err[inter]), 99)) if inter.any() else 5.0
    gg = campos["gradGamma_h"]
    return {
        "Lambda (interior)": (np.where(inter, campos["Lambda"], np.nan), "RdYlGn", 0.6, 1.0),
        "Lambda (todas)": (campos["Lambda"], "RdYlGn", 0.0, 1.0),
        "dnn / dx": (campos["dnn"], "plasma", 0.7, 1.2),
        "error densidad %": (err, "coolwarm", -lim, lim),
        "|grad Gamma| h": (gg, "magma_r", 0.0, max(1e-3, float(np.nanpercentile(gg, 99)))),
        "distancia a superficie / dx": (-campos["dist_borde"] / dist.dx_eq, "viridis", 0.0, None),
    }


# --------------------------------------------------------------------------
# Secciones (cortes)
# --------------------------------------------------------------------------
def _seccion_geometria(ax, geom, eje, pos, color="r"):
    """Dibuja el contorno de la geometria en el plano eje = pos."""
    otros = [k for k in range(3) if k != _EJES[eje]]
    lo, hi = geom.bbox
    m = 0.05 * geom.escala
    a = np.linspace(lo[otros[0]] - m, hi[otros[0]] + m, 220)
    b = np.linspace(lo[otros[1]] - m, hi[otros[1]] + m, 220)
    A, B = np.meshgrid(a, b)
    P = np.zeros((A.size, 3))
    P[:, _EJES[eje]] = pos
    P[:, otros[0]], P[:, otros[1]] = A.ravel(), B.ravel()
    D = geom.distancia_firmada(P).reshape(A.shape)
    if D.min() < 0 < D.max():
        ax.contour(A, B, D, levels=[0], colors=color, linewidths=1.0)


def dibujar_corte(ax, dist, eje="z", pos=0.0, espesor=None, valores=None, cmap="viridis",
                  vmin=None, vmax=None, titulo=None, barra=True, etiqueta=""):
    """Particulas cuyo centro esta a menos de espesor/2 del plano eje=pos,
    dibujadas a tamano real (circulos de diametro dx)."""
    k = _EJES[eje]
    otros = [c for c in range(3) if c != k]
    esp = espesor if espesor is not None else dist.dx_eq
    # el plano se centra en la particula mas cercana: asi en las redes regulares
    # la loncha siempre contiene una fila completa de particulas
    pos = float(dist.x[np.argmin(np.abs(dist.x[:, k] - pos)), k])
    # intervalo semiabierto: si el plano coincide con una fila de la red, la fila entra
    sel = (dist.x[:, k] >= pos - 0.5 * esp - 1e-9 * esp) & (dist.x[:, k] < pos + 0.5 * esp - 1e-9 * esp)
    pts = dist.x[sel][:, otros]
    d = dist.dx_eq
    col = EllipseCollection(np.full(len(pts), d), np.full(len(pts), d), np.zeros(len(pts)),
                            units="xy", offsets=pts, offset_transform=ax.transData,
                            edgecolors="k", linewidths=0.2)
    if valores is not None:
        col.set_array(np.asarray(valores, dtype=float)[sel])
        col.set_cmap(cmap)
        col.set_clim(vmin, vmax)
    else:
        col.set_facecolor("#2b7bba")
    ax.add_collection(col)
    _seccion_geometria(ax, dist.geom, eje, pos)
    lo, hi = dist.geom.bbox
    m = 0.05 * dist.geom.escala
    ax.set_xlim(lo[otros[0]] - m, hi[otros[0]] + m)
    ax.set_ylim(lo[otros[1]] - m, hi[otros[1]] + m)
    ax.set_aspect("equal")
    nombres = "xyz"
    ax.set_xlabel(nombres[otros[0]])
    ax.set_ylabel(nombres[otros[1]])
    ax.set_title(titulo or f"Corte {eje} = {pos:.3g} (+-{esp / 2:.3g})  n={sel.sum()}",
                 fontsize=9)
    if valores is not None and barra:
        ax.figure.colorbar(col, ax=ax, shrink=0.75, label=etiqueta)
    return col


def _dibujar_malla(ax3, geom, alpha=0.08):
    try:
        V, F = geom.malla(32)
    except NotImplementedError:
        return
    ax3.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color="grey",
                     alpha=alpha, linewidth=0.1, edgecolor="grey")


def _ejes_iguales(ax3, geom):
    lo, hi = geom.bbox
    c = 0.5 * (lo + hi)
    r = 0.5 * np.max(hi - lo)
    ax3.set_xlim(c[0] - r, c[0] + r)
    ax3.set_ylim(c[1] - r, c[1] + r)
    ax3.set_zlim(c[2] - r, c[2] + r)
    try:
        ax3.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    return f"{v:.4g}"


# --------------------------------------------------------------------------
# Panel de resumen
# --------------------------------------------------------------------------
def panel_resumen_3d(dist, r, campos, guardar=None, mostrar=True):
    cc = campos_para_color(dist, campos)
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(dist.resumen(), fontsize=10)
    gs = fig.add_gridspec(2, 4)

    # vista 3D: medio pajaro (y <= 0) coloreado por distancia a la superficie
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    mitad = dist.x[:, 1] <= 0
    v, cmap, a, b = cc["distancia a superficie / dx"]
    ax3.scatter(*dist.x[mitad].T, c=v[mitad], cmap=cmap, s=4, depthshade=False)
    _dibujar_malla(ax3, dist.geom)
    _ejes_iguales(ax3, dist.geom)
    ax3.set_title(f"{dist.metodo}: medio cuerpo (y<=0)\ncolor = profundidad", fontsize=9)

    v, cmap, a, b = cc["dnn / dx"]
    dibujar_corte(fig.add_subplot(gs[0, 1]), dist, "y", 0.0, valores=v, cmap=cmap, vmin=a,
                  vmax=b, etiqueta="dnn/dx",
                  titulo="Seccion longitudinal y=0: distancia vecino / dx")
    v, cmap, a, b = cc["Lambda (interior)"]
    dibujar_corte(fig.add_subplot(gs[0, 2]), dist, "z", 0.0, valores=v, cmap=cmap, vmin=a,
                  vmax=b, etiqueta="Lambda", titulo="Seccion z=0: Lambda (interior)")
    v, cmap, a, b = cc["error densidad %"]
    dibujar_corte(fig.add_subplot(gs[0, 3]), dist, "x", 0.0, valores=v, cmap=cmap, vmin=a,
                  vmax=b, etiqueta="%", titulo="Seccion transversal x=0: error densidad %")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(campos["dnn"], bins=50, range=(0.6, 1.4), color="#4477aa")
    ax.set_xlabel("dnn / dx"); ax.set_title("Distancia al vecino mas cercano", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(campos["Lambda"][campos["interior"]], bins=50, range=(0.5, 1.0), color="#228833")
    ax.set_xlabel("Lambda (interior)"); ax.set_title("Histograma Lambda", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    xi = dist.x[:, 0]
    nb = 30
    bins = np.linspace(dist.geom.bbox[0][0], dist.geom.bbox[1][0], nb + 1)
    idx = np.clip(np.digitize(xi, bins) - 1, 0, nb - 1)
    med = [np.mean(campos["dnn"][idx == k]) if np.any(idx == k) else np.nan for k in range(nb)]
    mn = [np.min(campos["dnn"][idx == k]) if np.any(idx == k) else np.nan for k in range(nb)]
    xc = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(xc, med, "o-", ms=3, label="media")
    ax.plot(xc, mn, "s-", ms=3, label="minimo")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_xlabel("x (eje del pajaro)"); ax.set_ylabel("dnn / dx"); ax.legend(fontsize=8)
    ax.set_title("Espaciado a lo largo del eje (cf. Siemann fig. 6)", fontsize=9)
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    txt = "\n".join(f"{k:<18s} {_fmt(r[k])}" for k in METRICAS_TABLA)
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=8.5, va="top")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Visor interactivo
# --------------------------------------------------------------------------
def visor_interactivo_3d(dist, campos, campo_inicial="Lambda (todas)", eje="x"):
    """Ventana con:
      - izquierda: vista 3D de las particulas con coordenada <= corte
        (se ve el interior del cuerpo cortado)
      - derecha: la seccion en el plano del corte, a tamano real
      - deslizador 'corte' para mover el plano, botones para el campo y el eje.
    Gira la vista 3D con el raton."""
    cc = campos_para_color(dist, campos)
    estado = {"campo": campo_inicial if campo_inicial in cc else list(cc)[0], "eje": eje}
    fig = plt.figure(figsize=(15, 8))
    ax3 = fig.add_axes([0.02, 0.18, 0.48, 0.78], projection="3d")
    ax2 = fig.add_axes([0.53, 0.18, 0.30, 0.78])
    axr = fig.add_axes([0.85, 0.55, 0.14, 0.35])
    axe = fig.add_axes([0.85, 0.30, 0.14, 0.18])
    axs = fig.add_axes([0.10, 0.06, 0.70, 0.04])
    axg = fig.add_axes([0.10, 0.01, 0.70, 0.04])
    k = _EJES[eje]
    lo, hi = dist.geom.bbox
    sl = Slider(axs, "corte", lo[k], hi[k], valinit=0.5 * (lo[k] + hi[k]))
    sg = Slider(axg, "espesor/dx", 0.5, 6.0, valinit=1.0)
    rb = RadioButtons(axr, list(cc), active=list(cc).index(estado["campo"]))
    re_ = RadioButtons(axe, ["x", "y", "z"], active="xyz".index(eje))
    axr.set_title("campo", fontsize=9)
    axe.set_title("eje de corte", fontsize=9)
    cbar = {"obj": None}

    def redibujar(_=None):
        v, cmap, a, b = cc[estado["campo"]]
        kk = _EJES[estado["eje"]]
        pos = sl.val
        elev, azim = ax3.elev, ax3.azim
        ax3.cla()
        sel = dist.x[:, kk] <= pos
        sc = ax3.scatter(*dist.x[sel].T, c=v[sel], cmap=cmap, vmin=a,
                         vmax=b if b is not None else np.nanmax(v), s=6, depthshade=False)
        _ejes_iguales(ax3, dist.geom)
        ax3.view_init(elev, azim)
        ax3.set_title(f"{dist.metodo}: particulas con {estado['eje']} <= {pos:.3g}", fontsize=9)
        ax2.cla()
        dibujar_corte(ax2, dist, estado["eje"], pos, espesor=sg.val * dist.dx_eq, valores=v,
                      cmap=cmap, vmin=a, vmax=b if b is not None else np.nanmax(v), barra=False)
        if cbar["obj"] is None:
            cbar["obj"] = fig.colorbar(sc, ax=[ax2], shrink=0.6, location="bottom")
        else:
            cbar["obj"].update_normal(sc)
        cbar["obj"].set_label(estado["campo"])
        fig.canvas.draw_idle()

    def cambiar_campo(lbl):
        estado["campo"] = lbl
        redibujar()

    def cambiar_eje(lbl):
        estado["eje"] = lbl
        kk = _EJES[lbl]
        sl.valmin, sl.valmax = lo[kk], hi[kk]
        sl.ax.set_xlim(lo[kk], hi[kk])
        sl.set_val(0.5 * (lo[kk] + hi[kk]))

    sl.on_changed(redibujar)
    sg.on_changed(redibujar)
    rb.on_clicked(cambiar_campo)
    re_.on_clicked(cambiar_eje)
    ax3.view_init(20, -60)
    redibujar()
    fig._widgets = (sl, sg, rb, re_)          # evitar que se borren
    return fig


def panel_comparacion_3d(lista, guardar=None, mostrar=True):
    """lista = [(dist, r, campos), ...]. Fila 1: seccion longitudinal y=0.
    Fila 2: seccion transversal x=0 coloreada con Lambda (todas las particulas)."""
    n = len(lista)
    fig, axs = plt.subplots(2, n, figsize=(3.8 * n, 7.5), squeeze=False)
    for k, (dist, r, c) in enumerate(lista):
        dibujar_corte(axs[0, k], dist, "y", 0.0, titulo=f"{dist.metodo}  N={dist.N}\ncorte y=0")
        dibujar_corte(axs[1, k], dist, "x", 0.0, valores=c["Lambda"], cmap="RdYlGn",
                      vmin=0.6, vmax=1.0, barra=(k == n - 1), etiqueta="Lambda",
                      titulo=f"corte x=0 | Lambda medio int. {r['Lambda_media']:.3f}")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=110)
    if mostrar:
        plt.show()
    return fig
