"""
Figuras del inflation (resolucion variable), 2D y 3D.

  dibujar_campo(ax, geom, campo, seccion)      mapa del dx pedido (vista previa)
  dibujar_particulas(ax, dist, seccion, valores) particulas como circulos de diametro dx
  figura_inflation(dist, res, campos, seccion)   figura resumen 2x3
En 3D todo se dibuja en una SECCION: seccion = ("y", 0.0) -> plano y = 0.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Circle, Rectangle

EJES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
NOMBRE = "xyz"


def ejes_seccion(dim, seccion):
    if dim == 2:
        return 0, 1, None
    a, b = EJES[seccion[0].lower()]
    return a, b, "xyz".index(seccion[0].lower())


def rejilla(geom, seccion, n=260):
    """Puntos de una rejilla en el plano de corte (o en el plano 2D)."""
    d = geom.dim
    a, b, k = ejes_seccion(d, seccion)
    lo, hi = np.asarray(geom.bbox[0], float), np.asarray(geom.bbox[1], float)
    m = 0.04 * (hi - lo).max()
    na = n
    nb = max(20, int(n * (hi[b] - lo[b]) / max(hi[a] - lo[a], 1e-9)))
    u = np.linspace(lo[a] - m, hi[a] + m, na)
    v = np.linspace(lo[b] - m, hi[b] + m, nb)
    U, V = np.meshgrid(u, v)
    P = np.zeros((U.size, d))
    P[:, a], P[:, b] = U.ravel(), V.ravel()
    if k is not None:
        P[:, k] = seccion[1]
    return U, V, P


def contorno_geometria(ax, geom, seccion, U=None, V=None, sd=None, **kw):
    kw = {"colors": "k", "linewidths": 1.2, **kw}
    if geom.dim == 2 and hasattr(geom, "contorno"):
        v = np.vstack([geom.contorno.v, geom.contorno.v[:1]])
        ax.plot(v[:, 0], v[:, 1], color=kw["colors"], lw=kw["linewidths"])
        return
    if U is None:
        U, V, P = rejilla(geom, seccion)
        sd = geom.distancia_firmada(P).reshape(U.shape)
    if np.nanmin(sd) < 0 < np.nanmax(sd):
        ax.contour(U, V, sd, levels=[0.0], **kw)


def dibujar_zonas(ax, campo, seccion, dim, color="tab:red"):
    a, b, k = ejes_seccion(dim, seccion)
    for z in campo.zonas:
        if z.tipo in ("esfera", "punto"):
            dk = 0.0 if k is None else z.c[k] - seccion[1]
            r2 = z.R ** 2 - dk ** 2
            if z.tipo == "punto" or r2 <= 0:
                ax.plot(z.c[a], z.c[b], "x", color=color, ms=9, mew=2)
            if r2 > 0:
                ax.add_patch(Circle((z.c[a], z.c[b]), np.sqrt(r2), fill=False, ec=color, lw=1.6))
        elif z.tipo == "caja":
            if k is None or z.lo[k] <= seccion[1] <= z.hi[k]:
                ax.add_patch(Rectangle((z.lo[a], z.lo[b]), z.hi[a] - z.lo[a], z.hi[b] - z.lo[b],
                                       fill=False, ec=color, lw=1.6))
        else:  # losa: dos rectas paralelas al plano (si la normal esta en el plano de corte)
            n2 = np.array([z.n[a], z.n[b]])
            if np.linalg.norm(n2) > 1e-6:
                n2 = n2 / np.linalg.norm(n2)
                t = np.array([-n2[1], n2[0]])
                c = np.array([z.c[a], z.c[b]])
                L = 1e4
                for s in (-0.5, 0.5):
                    p = c + s * z.e * n2
                    ax.plot([p[0] - L * t[0], p[0] + L * t[0]], [p[1] - L * t[1], p[1] + L * t[1]],
                            color=color, lw=1.4, ls="-")


def dibujar_campo(ax, geom, campo, seccion=("y", 0.0), n=260, barra=True, titulo=True):
    """Mapa del dx PEDIDO dentro del cuerpo + contorno + zonas + fin de la transicion."""
    U, V, P = rejilla(geom, seccion, n)
    sd = geom.distancia_firmada(P).reshape(U.shape)
    dxm = campo.dx(P).reshape(U.shape)
    im = ax.pcolormesh(U, V, np.where(sd < 0, dxm, np.nan), cmap="viridis_r", shading="auto",
                       vmin=campo.dx_min, vmax=campo.dx_grueso)
    contorno_geometria(ax, geom, seccion, U, V, sd)
    if campo.zonas and campo.dx_grueso > campo.dx_min:
        ax.contour(U, V, dxm, levels=[campo.dx_grueso * 0.999], colors="w", linestyles="--", linewidths=1)
    dibujar_zonas(ax, campo, seccion, geom.dim)
    a, b, k = ejes_seccion(geom.dim, seccion)
    ax.set_xlabel(NOMBRE[a]); ax.set_ylabel(NOMBRE[b])
    ax.set_aspect("equal")
    ax.set_xlim(U.min(), U.max()); ax.set_ylim(V.min(), V.max())
    if titulo:
        t = "dx pedido" + ("" if k is None else f"  (corte {NOMBRE[k]} = {seccion[1]:.4g})")
        ax.set_title(t + "\n(rojo: zonas; blanco discontinuo: fin de la transicion)", fontsize=9)
    if barra:
        ax.figure.colorbar(im, ax=ax, shrink=0.8, label="dx")
    return im


def en_seccion(dist, seccion, espesor=0.5):
    """Mascara de particulas a menos de espesor*dx_i del plano de corte."""
    if dist.dim == 2:
        return np.ones(dist.N, bool)
    k = "xyz".index(seccion[0].lower())
    return np.abs(dist.x[:, k] - seccion[1]) < espesor * dist.dx


def dibujar_particulas(ax, dist, seccion=("y", 0.0), valores=None, cmap="viridis_r",
                       vmin=None, vmax=None, maximo=80000, espesor=0.5, etiqueta=None,
                       barra=True, contorno=True, gris_nan=True):
    """Particulas de la seccion como circulos de diametro 0.9*dx (tamano real)."""
    a, b, k = ejes_seccion(dist.dim, seccion)
    msk = en_seccion(dist, seccion, espesor)
    idx = np.where(msk)[0]
    if len(idx) > maximo:
        idx = np.random.default_rng(0).choice(idx, maximo, replace=False)
    val = dist.dx if valores is None else np.asarray(valores, float)
    v = val[idx]
    D = 0.9 * dist.dx[idx]
    col = EllipseCollection(D, D, np.zeros(len(idx)), units="xy",
                            offsets=np.column_stack([dist.x[idx, a], dist.x[idx, b]]),
                            offset_transform=ax.transData, cmap=cmap, edgecolors="none")
    finitos = np.isfinite(v)
    col.set_array(np.where(finitos, v, np.nan))
    if gris_nan:
        col.cmap = plt.get_cmap(cmap).copy()
        col.cmap.set_bad("0.82")
    vv = v[finitos]
    col.set_clim(vmin if vmin is not None else (vv.min() if len(vv) else 0),
                 vmax if vmax is not None else (vv.max() if len(vv) else 1))
    ax.add_collection(col)
    if contorno:
        contorno_geometria(ax, dist.geom, seccion)
    dibujar_zonas(ax, dist.campo, seccion, dist.dim)
    lo, hi = dist.geom.bbox
    m = 0.04 * (np.asarray(hi) - np.asarray(lo)).max()
    ax.set_xlim(lo[a] - m, hi[a] + m); ax.set_ylim(lo[b] - m, hi[b] + m)
    ax.set_aspect("equal")
    ax.set_xlabel(NOMBRE[a]); ax.set_ylabel(NOMBRE[b])
    if barra:
        ax.figure.colorbar(col, ax=ax, shrink=0.8, label=etiqueta or ("dx" if valores is None else ""))
    return col


def figura_inflation(dist, res, campos, seccion=("y", 0.0), guardar=None, mostrar=True):
    """Resumen 2x3: particulas (dx), error de densidad, Lambda, perfil dx frente
    a la distancia a la zona, histograma de densidad por region, tabla."""
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    a, b, k = ejes_seccion(dist.dim, seccion)
    sec = "" if k is None else f"  -  corte {NOMBRE[k]} = {seccion[1]:.4g}"
    fig.suptitle(f"Inflation - {dist.geom.nombre}: N = {dist.N},  dx {dist.dx.min():.3g} - "
                 f"{dist.dx.max():.3g},  tasa {100 * dist.campo.tasa:.3g} %/capa{sec}", fontsize=12)
    I = campos["interior"]
    dibujar_particulas(axs[0, 0], dist, seccion, etiqueta="dx")
    axs[0, 0].set_title("particulas (diametro = dx) y zonas", fontsize=10)
    err = np.where(I, campos["err_rho"], np.nan)
    lim = max(0.5, np.nanpercentile(np.abs(err), 99)) if np.any(I) else 1
    dibujar_particulas(axs[0, 1], dist, seccion, err, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       etiqueta="error de densidad SPH (%)")
    axs[0, 1].set_title(f"error de densidad interior: desv. {res['err_rho_std_%']:.2f} %, "
                        f"max {res['err_rho_Linf_%']:.2f} %", fontsize=10)
    lam = np.where(I, campos["Lambda"], np.nan)
    dibujar_particulas(axs[0, 2], dist, seccion, lam, cmap="viridis", vmin=0.8, vmax=1.0,
                       etiqueta="Lambda")
    axs[0, 2].set_title(f"Lambda interior: media {res['Lambda_media']:.3f}, min {res['Lambda_min']:.3f}",
                        fontsize=10)

    ax = axs[1, 0]
    dz = campos["dist_zona"]
    if np.all(np.isinf(dz)):
        ax.text(0.5, 0.5, "sin zonas de refinado", ha="center", transform=ax.transAxes)
    else:
        rng = np.random.default_rng(1)
        sel = np.where(I)[0]
        if len(sel) > 20000:
            sel = rng.choice(sel, 20000, replace=False)
        ax.plot(dz[sel], campos["dx_real"][sel], ".", ms=2, alpha=0.4, color="tab:blue",
                label="dx real (interior)")
        dd = np.linspace(0, max(dz[np.isfinite(dz)].max(), 1e-9), 300)
        for z in dist.campo.zonas:
            dz0 = z.dx or dist.campo.dx_fino
            ax.plot(dd, np.minimum(dist.campo.dx_grueso, dz0 + dist.campo.tasa * dd), "k-", lw=2,
                    label="dx pedido" if z is dist.campo.zonas[0] else None)
        ax.axvline(dist.campo.longitud_transicion, color="0.5", ls="--", lw=1)
        ax.set_xlabel("distancia a la zona de refinado"); ax.set_ylabel("dx")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"espaciado real frente al pedido: {res['dx_real/dx_media']:.3f} "
                 f"(CV {res['dx_real/dx_CV_%']:.2f} %)", fontsize=10)

    ax = axs[1, 1]
    L = dist.campo.longitud_transicion
    regiones = [("zona fina", dz <= 0), ("transicion", (dz > 0) & (dz < L)), ("zona gruesa", dz >= L)]
    for nom, m in regiones:
        mm = I & m
        if mm.sum() > 5:
            e = campos["err_rho"][mm]
            ax.hist(e, bins=40, histtype="step", lw=1.6, density=True,
                    label=f"{nom}: {e.mean():+.2f} +/- {e.std():.2f} %")
    ax.set_xlabel("error de densidad SPH interior (%)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("densidad por region (misma media = transicion limpia)", fontsize=10)

    ax = axs[1, 2]
    ax.axis("off")
    from .inflation import DESCRIPCION_INF
    filas = []
    for kk in ["N", "N_zona_fina", "masa_total", "error_masa_%", "masa_max/min", "err_rho_std_%",
               "err_rho_media_%", "err_rho_Linf_%", "dx_real/dx_media", "salto_vecinas_max",
               "Lambda_media", "gradGamma_h_max", "gradGamma_h_sup_CV_%", "dnn_min"]:
        v = res.get(kk)
        filas.append([kk, f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"])
    t = ax.table(cellText=filas, colLabels=["metrica", "valor"], loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.25)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig
