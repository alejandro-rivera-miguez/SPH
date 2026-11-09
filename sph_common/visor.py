"""
Visualizador 2D con matplotlib.

  dibujar_particulas(ax, dist, campo, ...)  particulas a tamano real (circulos)
  panel_resumen(dist, r, campos)            figura completa de una distribucion
  dibujar_historial(hist)                   convergencia de los metodos iterativos
  panel_comparacion(lista)                  varias distribuciones lado a lado
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from .metricas import METRICAS_TABLA


def _contorno(ax, geom, **kw):
    v = geom.contorno.v
    v = np.vstack([v, v[:1]])
    ax.plot(v[:, 0], v[:, 1], kw.pop("color", "k"), lw=kw.pop("lw", 1.0), **kw)


def dibujar_particulas(ax, dist, valores=None, cmap="viridis", vmin=None, vmax=None,
                       titulo=None, barra=True, etiqueta="", diametro=None,
                       mascara=None, color="#2b7bba"):
    x = dist.x if mascara is None else dist.x[mascara]
    d = diametro if diametro is not None else dist.dx_eq
    col = EllipseCollection(np.full(len(x), d), np.full(len(x), d), np.zeros(len(x)),
                            units="xy", offsets=x, offset_transform=ax.transData,
                            edgecolors="k", linewidths=0.2)
    if valores is not None:
        v = valores if mascara is None else valores[mascara]
        col.set_array(np.asarray(v, dtype=float))
        col.set_cmap(cmap)
        col.set_clim(vmin, vmax)
    else:
        col.set_facecolor(color)
    ax.add_collection(col)
    _contorno(ax, dist.geom, color="r", lw=1.0)
    (x0, y0), (x1, y1) = dist.geom.bbox
    m = 0.05 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    if valores is not None and barra:
        ax.figure.colorbar(col, ax=ax, shrink=0.8, label=etiqueta)
    return col


def panel_resumen(dist, r, campos, guardar=None, mostrar=True):
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(dist.resumen(), fontsize=10)
    gs = fig.add_gridspec(2, 4)
    inter = campos["interior"]

    ax = fig.add_subplot(gs[0, 0])
    dibujar_particulas(ax, dist, titulo=f"{dist.metodo}  (N={dist.N})", barra=False)

    ax = fig.add_subplot(gs[0, 1])
    lam = np.where(inter, campos["Lambda"], np.nan)
    dibujar_particulas(ax, dist, lam, cmap="RdYlGn", vmin=0.6, vmax=1.0,
                       titulo="Lambda = lmin/lmax (interior)", etiqueta="Lambda")

    ax = fig.add_subplot(gs[0, 2])
    err = 100 * (campos["rho_rel"] - 1)
    lim = max(1.0, np.nanpercentile(np.abs(err[inter]), 99)) if inter.any() else 5
    dibujar_particulas(ax, dist, err, cmap="coolwarm", vmin=-lim, vmax=lim,
                       titulo="Error densidad SPH (%)", etiqueta="%")

    ax = fig.add_subplot(gs[0, 3])
    dibujar_particulas(ax, dist, campos["dnn"], cmap="plasma", vmin=0.7, vmax=1.2,
                       titulo="Distancia al vecino mas cercano / dx", etiqueta="dnn/dx")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(campos["dnn"], bins=40, range=(0.6, 1.3), color="#4477aa")
    ax.set_xlabel("dnn / dx"); ax.set_title("Histograma distancia vecino", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(campos["Lambda"][inter], bins=40, range=(0.5, 1.0), color="#228833")
    ax.set_xlabel("Lambda (interior)"); ax.set_title("Histograma Lambda", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    gg = np.where(inter, campos["gradGamma_h"], np.nan)
    dibujar_particulas(ax, dist, gg, cmap="magma_r", vmin=0,
                       vmax=max(1e-3, np.nanpercentile(gg, 99)) if inter.any() else 1,
                       titulo="|grad Gamma|*h (interior)", etiqueta="")

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    txt = "\n".join(f"{k:<18s} {_fmt(r[k])}" for k in METRICAS_TABLA)
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=8.5, va="top")

    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    return f"{v:.4g}"


def dibujar_historial(hist, titulo="Convergencia", guardar=None, mostrar=True):
    claves = [k for k in hist if k != "iter" and len(hist[k]) == len(hist["iter"])]
    if not claves:
        return None
    fig, axs = plt.subplots(1, len(claves), figsize=(4.2 * len(claves), 3.4))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, claves):
        v = np.asarray(hist[k], dtype=float)
        ax.plot(hist["iter"], v)
        if np.all(v[np.isfinite(v)] > 0):
            ax.set_yscale("log")
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("iteracion")
        ax.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig


def panel_comparacion(lista, campo="Lambda", guardar=None, mostrar=True):
    """lista = [(dist, r, campos), ...]"""
    n = len(lista)
    fig, axs = plt.subplots(2, n, figsize=(3.6 * n, 7.5), squeeze=False)
    for k, (dist, r, c) in enumerate(lista):
        dibujar_particulas(axs[0, k], dist, titulo=f"{dist.metodo}\nN={dist.N}", barra=False)
        lam = np.where(c["interior"], c["Lambda"], np.nan)
        dibujar_particulas(axs[1, k], dist, lam, cmap="RdYlGn", vmin=0.6, vmax=1.0,
                           titulo=f"Lambda medio {r['Lambda_media']:.3f}",
                           barra=(k == n - 1), etiqueta="Lambda")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig


def barras_comparacion(resultados, claves=None, guardar=None, mostrar=True):
    """resultados = lista de diccionarios r de calcular_metricas."""
    claves = claves or ["Lambda_media", "Lambda_P5", "gradGamma_h_max",
                        "err_rho_std_%", "dnn_CV_%", "Vvor_CV_%",
                        "hueco_sup_max", "capa_ext_CV_%"]
    nombres = [r["metodo"] for r in resultados]
    fig, axs = plt.subplots(2, (len(claves) + 1) // 2, figsize=(16, 7))
    for ax, k in zip(axs.ravel(), claves):
        v = [r[k] for r in resultados]
        ax.bar(range(len(v)), v, color=plt.cm.tab10(np.arange(len(v)) % 10))
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8)
        ax.set_title(k, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for ax in axs.ravel()[len(claves):]:
        ax.axis("off")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig


# ==========================================================================
# |grad Gamma|*h en la capa exterior (superficie de impacto)
# ==========================================================================
def _capa_exterior(dist, campos):
    """Particulas de la primera capa (a menos de 1 dx de la superficie)."""
    return campos["dist_borde"] > -1.0 * dist.dx_eq


def _posicion_superficie(dist, mascara):
    """Coordenada para recorrer la superficie: en 2D la longitud de arco sobre
    el contorno (0 = punto mas a la izquierda); en 3D la coordenada x (eje)."""
    x = dist.x[mascara]
    if dist.dim == 3:
        return x[:, 0], "x (eje del pajaro)"
    v = dist.geom.contorno.v
    seg = np.linalg.norm(np.roll(v, -1, 0) - v, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    from scipy.spatial import cKDTree
    _, k = cKDTree(v).query(x)
    s = s[k]
    s0 = s[np.argmin(x[:, 0])]
    return (s - s0) % seg.sum(), "longitud de arco sobre el contorno"


def panel_superficie(dist, r, campos, guardar=None, mostrar=True, vmax=None):
    """Figura para justificar la calidad de la superficie:
       (1) particulas de la capa exterior coloreadas por |grad Gamma|*h
       (2) |grad Gamma|*h a lo largo de la superficie (perfil)
       (3) histograma de |grad Gamma|*h en la capa exterior
    En una superficie libre |grad Gamma|*h nunca es 0 (faltan vecinos fuera);
    lo que importa es que sea UNIFORME: picos = superficie en escalera."""
    capa = _capa_exterior(dist, campos)
    g = campos["gradGamma_h"]
    vmax = vmax or float(np.percentile(g[capa], 99))
    cv = r.get("gradGamma_h_sup_CV_%", 100 * g[capa].std() / g[capa].mean())
    fig = plt.figure(figsize=(16, 4.8))
    fig.suptitle(f"{dist.metodo}: |grad Gamma|*h en la capa exterior   "
                 f"media = {g[capa].mean():.3f}   CV = {cv:.1f} %", fontsize=11)
    if dist.dim == 2:
        ax = fig.add_subplot(1, 3, 1)
        dibujar_particulas(ax, dist, color="#dddddd", barra=False, mascara=~capa)
        dibujar_particulas(ax, dist, g, cmap="inferno", vmin=0, vmax=vmax, mascara=capa,
                           etiqueta="|grad Gamma|*h", titulo="Capa exterior")
    else:
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        xs = dist.x[capa]
        sc = ax.scatter(*xs.T, c=g[capa], cmap="inferno", vmin=0, vmax=vmax, s=6,
                        depthshade=False)
        ax.figure.colorbar(sc, ax=ax, shrink=0.7, label="|grad Gamma|*h")
        lo, hi = dist.geom.bbox
        c, rr = 0.5 * (lo + hi), 0.5 * np.max(hi - lo)
        ax.set_xlim(c[0] - rr, c[0] + rr); ax.set_ylim(c[1] - rr, c[1] + rr)
        ax.set_zlim(c[2] - rr, c[2] + rr)
        ax.set_title("Capa exterior", fontsize=10)
    ax = fig.add_subplot(1, 3, 2)
    s, etiqueta = _posicion_superficie(dist, capa)
    o = np.argsort(s)
    ax.plot(s[o], g[capa][o], ".", ms=3, color="#cc3311")
    ax.axhline(g[capa].mean(), color="k", lw=0.8, ls="--", label="media")
    ax.set_ylim(0, vmax * 1.15)
    ax.set_xlabel(etiqueta); ax.set_ylabel("|grad Gamma|*h")
    ax.set_title("Perfil a lo largo de la superficie (plano = uniforme)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(g[capa], bins=40, range=(0, vmax * 1.15), color="#cc3311")
    ax.set_xlabel("|grad Gamma|*h (capa exterior)")
    ax.set_title("Distribucion (estrecha = uniforme)", fontsize=10)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig


def comparacion_superficie(lista, guardar=None, mostrar=True):
    """lista = [(dist, r, campos), ...]. Una fila de mapas (misma escala de
    color para todos) + perfiles superpuestos + barras del CV."""
    n = len(lista)
    g_all = np.concatenate([c["gradGamma_h"][_capa_exterior(d, c)] for d, r, c in lista])
    vmax = float(np.percentile(g_all, 99))
    fig = plt.figure(figsize=(3.6 * n, 8.5))
    gs = fig.add_gridspec(2, n)
    for k, (d, r, c) in enumerate(lista):
        capa = _capa_exterior(d, c)
        g = c["gradGamma_h"]
        cv = 100 * g[capa].std() / g[capa].mean()
        if d.dim == 2:
            ax = fig.add_subplot(gs[0, k])
            dibujar_particulas(ax, d, color="#dddddd", barra=False, mascara=~capa)
            dibujar_particulas(ax, d, g, cmap="inferno", vmin=0, vmax=vmax, mascara=capa,
                               barra=(k == n - 1), etiqueta="|grad Gamma|*h",
                               titulo=f"{d.metodo}\nCV superficie = {cv:.1f} %")
        else:
            ax = fig.add_subplot(gs[0, k], projection="3d")
            ax.scatter(*d.x[capa].T, c=g[capa], cmap="inferno", vmin=0, vmax=vmax, s=3,
                       depthshade=False)
            ax.set_title(f"{d.metodo}\nCV superficie = {cv:.1f} %", fontsize=10)
            ax.set_axis_off()
    nmit = max(1, n // 2)
    ax = fig.add_subplot(gs[1, :nmit])
    for d, r, c in lista:
        capa = _capa_exterior(d, c)
        s, etiqueta = _posicion_superficie(d, capa)
        o = np.argsort(s)
        ax.plot(s[o], c["gradGamma_h"][capa][o], ".", ms=2.5, label=d.metodo)
    ax.set_xlabel(etiqueta); ax.set_ylabel("|grad Gamma|*h")
    ax.set_ylim(0, vmax * 1.15)
    ax.set_title("|grad Gamma|*h a lo largo de la superficie", fontsize=10)
    ax.legend(fontsize=8, markerscale=4); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, nmit:])
    nombres = [d.metodo for d, r, c in lista]
    cvs = [100 * c["gradGamma_h"][_capa_exterior(d, c)].std() /
           c["gradGamma_h"][_capa_exterior(d, c)].mean() for d, r, c in lista]
    ax.bar(range(n), cvs, color=plt.cm.tab10(np.arange(n) % 10))
    for k, v in enumerate(cvs):
        ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels(nombres, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CV de |grad Gamma|*h en superficie (%)")
    ax.set_title("Irregularidad de la fuerza espuria en la superficie (menor = mejor)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=120)
    if mostrar:
        plt.show()
    return fig
