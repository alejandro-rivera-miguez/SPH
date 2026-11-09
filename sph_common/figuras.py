"""
Figuras configurables (2D y 3D) para los scripts y los comparadores.

1) figura_paneles(dist, r, campos, paneles=[...])
   Figura de resumen de UNA distribucion con los paneles que tu elijas.
   Paneles disponibles (en 3D los mapas son una SECCION del cuerpo):

     "particulas"      particulas a tamano real
     "lambda"          mapa de Lambda (interior)
     "densidad"        mapa del error de densidad SPH (%) (interior; la capa
                       exterior en gris porque su deficit es propio de la superficie libre)
     "gradgamma"       mapa de |grad Gamma|*h (interior)
     "gradgamma_sup"   mapa de |grad Gamma|*h solo en la capa exterior
     "dnn"             mapa de la distancia al vecino mas cercano / dx
     "hist_dnn", "hist_lambda", "hist_densidad", "hist_gradgamma"   histogramas
     "perfil_superficie"  |grad Gamma|*h a lo largo de la superficie
     "perfil_dnn"      espaciado medio y minimo a lo largo del eje x
     "vista3d"         (solo 3D) medio cuerpo en 3D coloreado por profundidad
     "tabla"           tabla de metricas

   En 3D cada mapa se puede cortar por un plano distinto con "nombre@eje" o
   "nombre@eje=pos", p.ej. "lambda@z", "densidad@x=-50". Sin @ se usa SECCION.

2) figura_empaquetados(lista, metodos)          rejilla 2x3 con las particulas
3) figura_comparar(lista, metodos, metrica)     fila superior: empaquetados;
                                                fila inferior: la metrica elegida
                                                (misma escala), % frente al mejor
                                                y el mejor resaltado en verde
4) figura_tabla_diferencias(lista, claves)      tabla de valores + % frente al
                                                mejor, mejor de cada fila en verde
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .metricas import METRICAS_TABLA, DESCRIPCION
from .visor import dibujar_particulas, _capa_exterior, _posicion_superficie

# --------------------------------------------------------------------------
# Sentido de cada metrica: True = mas alto es mejor, False = mas bajo es mejor,
# float = lo ideal es acercarse a ese valor
# --------------------------------------------------------------------------
SENTIDO = {
    "Lambda_min": True, "Lambda_media": True, "Lambda_P5": True, "%_Lambda>0.9": True,
    "gradGamma_h_max": False, "gradGamma_h_media": False,
    "gradGamma_h_sup_media": None, "gradGamma_h_sup_CV_%": False,
    "err_rho_Linf_%": False, "err_rho_std_%": False, "err_rho_media_%": 0.0, "dnn_min": True,
    "dnn_CV_%": False, "Vvor_max/min": 1.0, "Vvor_CV_%": False,
    "hueco_sup_media": 0.5, "hueco_sup_max": 0.5, "capa_ext_CV_%": False,
}

# Metricas para figura_comparar: campo que se pinta y numero que las resume
METRICAS_COMPARACION = {
    "densidad": dict(titulo="Error de densidad SPH (%)", escalar="err_rho_std_%",
                     texto="desv. tipica densidad interior"),
    "gradgamma": dict(titulo="|grad Gamma|*h (interior)", escalar="gradGamma_h_max",
                      texto="max |grad Gamma|*h interior"),
    "gradgamma_sup": dict(titulo="|grad Gamma|*h en la capa exterior",
                          escalar="gradGamma_h_sup_CV_%", texto="CV superficie"),
    "lambda": dict(titulo="Lambda (interior)", escalar="Lambda_media",
                   texto="Lambda medio interior"),
    "dnn": dict(titulo="Distancia al vecino mas cercano / dx", escalar="dnn_CV_%",
                texto="dispersion de dnn"),
}

_EJES = {"x": 0, "y": 1, "z": 2}
VERDE = "#1a9850"


# ==========================================================================
# Utilidades
# ==========================================================================
def mejor_indice(clave, valores):
    s = SENTIDO.get(clave)
    v = np.array(valores, dtype=float)
    if s is None or np.all(~np.isfinite(v)):
        return None
    if s is True:
        return int(np.nanargmax(v))
    if s is False:
        return int(np.nanargmin(v))
    return int(np.nanargmin(np.abs(v - s)))


def diferencia_frente_al_mejor(clave, valores, corto=False):
    """Para cada valor, cuanto peor es que el mejor de la fila.
    Devuelve (indice del mejor, textos, porcentajes).
      * metricas "mayor/menor es mejor": porcentaje respecto al mejor
      * metricas con valor objetivo (p.ej. hueco ideal 0.5): diferencia
        absoluta en la distancia al objetivo
      * si el mejor es practicamente 0 (p.ej. |grad Gamma| de una red perfecta)
        el porcentaje no tiene sentido y se da la diferencia absoluta"""
    b = mejor_indice(clave, valores)
    v = np.array(valores, dtype=float)
    if b is None:
        return None, ["" for _ in valores], [np.nan for _ in valores]
    s = SENTIDO.get(clave)
    objetivo = isinstance(s, float) and not isinstance(s, bool)
    if objetivo:
        dist = np.abs(v - s)
        peor = dist - dist[b]
        ref = 0.0
    elif s is True:
        ref = v[b]
        peor = v[b] - v
    else:
        ref = v[b]
        peor = v - v[b]
    absoluta = objetivo or abs(ref) < 1e-3 * max(np.nanmax(np.abs(v)), 1e-30)
    textos, pct = [], []
    for k in range(len(v)):
        if k == b:
            textos.append("MEJOR")
            pct.append(0.0)
        elif absoluta:
            textos.append(f"+{peor[k]:.3g}" if corto else f"+{peor[k]:.3g} peor (abs.)")
            pct.append(np.nan)
        else:
            p = 100 * peor[k] / abs(ref)
            textos.append(f"+{p:.0f}%" if corto else f"{p:+.1f} % peor")
            pct.append(p)
    return b, textos, pct


def _parse_panel(nombre, seccion):
    """'lambda@z=10' -> ('lambda', ('z', 10.0))."""
    eje, pos = seccion
    if "@" in nombre:
        nombre, s = nombre.split("@", 1)
        if "=" in s:
            eje, p = s.split("=", 1)
            pos = float(p)
        else:
            eje = s
    return nombre.strip(), (eje.strip().lower(), float(pos))


def _campo(nombre, dist, campos):
    """(valores, cmap, vmin, vmax, etiqueta, mascara) para un mapa."""
    inter = campos["interior"]
    capa = _capa_exterior(dist, campos)
    if nombre == "lambda":
        return np.where(inter, campos["Lambda"], np.nan), "RdYlGn", 0.6, 1.0, "Lambda", None
    if nombre == "densidad":
        err = 100 * (campos["rho_rel"] - 1)
        vals = np.where(~capa, err, np.nan)
        lim = max(0.5, float(np.nanpercentile(np.abs(vals), 99))) if np.isfinite(vals).any() else 5
        return vals, "coolwarm", -lim, lim, "error densidad %", None
    if nombre == "gradgamma":
        g = np.where(inter, campos["gradGamma_h"], np.nan)
        vmax = max(1e-3, float(np.nanpercentile(g, 99))) if np.isfinite(g).any() else 1
        return g, "magma_r", 0.0, vmax, "|grad Gamma|*h", None
    if nombre == "gradgamma_sup":
        g = campos["gradGamma_h"]
        return g, "inferno", 0.0, float(np.percentile(g[capa], 99)), "|grad Gamma|*h", capa
    if nombre == "dnn":
        return campos["dnn"], "plasma", 0.7, 1.2, "dnn / dx", None
    raise ValueError(nombre)


def _mapa(ax, dist, campos, nombre, seccion, vmin=None, vmax=None, barra=True, titulo=None,
          espesor=None):
    """Dibuja un mapa (2D: todas las particulas; 3D: seccion)."""
    if nombre == "particulas":
        vals, cmap, a, b, et, masc = None, None, None, None, "", None
    else:
        vals, cmap, a, b, et, masc = _campo(nombre, dist, campos)
        a = a if vmin is None else vmin
        b = b if vmax is None else vmax
    if dist.dim == 2:
        if masc is not None:
            dibujar_particulas(ax, dist, color="#dddddd", barra=False, mascara=~masc)
            return dibujar_particulas(ax, dist, vals, cmap=cmap, vmin=a, vmax=b, mascara=masc,
                                      barra=barra, etiqueta=et, titulo=titulo)
        if vals is not None:
            # particulas sin valor (NaN) en gris claro debajo
            dibujar_particulas(ax, dist, color="#dddddd", barra=False,
                               mascara=~np.isfinite(vals))
            return dibujar_particulas(ax, dist, vals, cmap=cmap, vmin=a, vmax=b,
                                      mascara=np.isfinite(vals), barra=barra, etiqueta=et,
                                      titulo=titulo)
        return dibujar_particulas(ax, dist, titulo=titulo, barra=False)
    from .visor3d import dibujar_corte
    eje, pos = seccion
    if vals is not None and masc is not None:
        vals = np.where(masc, vals, np.nan)
    col = dibujar_corte(ax, dist, eje, pos, espesor=espesor, valores=vals, cmap=cmap, vmin=a, vmax=b,
                        barra=barra, etiqueta=et)
    if vals is not None:
        col.cmap.set_bad("#dddddd")
    ax.set_title((titulo + "\n" if titulo else "") + f"seccion {eje} = {pos:.3g}", fontsize=9)
    return col


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    return f"{v:.4g}"


# ==========================================================================
# 1) Figura de una distribucion con paneles a elegir
# ==========================================================================
PANELES_2D = ["particulas", "lambda", "densidad", "dnn",
              "hist_dnn", "hist_lambda", "gradgamma", "tabla"]
PANELES_3D = ["vista3d", "dnn@y", "lambda@z", "densidad@x",
              "hist_dnn", "hist_lambda", "perfil_dnn", "tabla"]
TITULOS = {
    "particulas": "Particulas", "lambda": "Lambda = lmin/lmax (interior)",
    "densidad": "Error de densidad SPH (%)", "gradgamma": "|grad Gamma|*h (interior)",
    "gradgamma_sup": "|grad Gamma|*h (capa exterior)", "dnn": "Distancia vecino / dx",
}


def figura_paneles(dist, r, campos, paneles=None, seccion=("y", 0.0), columnas=4,
                   guardar=None, mostrar=True):
    paneles = list(paneles or (PANELES_2D if dist.dim == 2 else PANELES_3D))
    n = len(paneles)
    ncol = min(columnas, n)
    nfil = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 4.0 * nfil + 0.6))
    fig.suptitle(dist.resumen(), fontsize=10)
    inter = campos["interior"]
    capa = _capa_exterior(dist, campos)
    for k, nombre_bruto in enumerate(paneles):
        nombre, sec = _parse_panel(nombre_bruto, seccion)
        if nombre == "vista3d" and dist.dim == 3:
            from .visor3d import _dibujar_malla, _ejes_iguales
            ax = fig.add_subplot(nfil, ncol, k + 1, projection="3d")
            mitad = dist.x[:, 1] <= 0
            ax.scatter(*dist.x[mitad].T, c=-campos["dist_borde"][mitad] / dist.dx_eq,
                       cmap="viridis", s=4, depthshade=False)
            _dibujar_malla(ax, dist.geom)
            _ejes_iguales(ax, dist.geom)
            ax.set_title(f"{dist.metodo}: medio cuerpo (y<=0)\ncolor = profundidad", fontsize=9)
            continue
        ax = fig.add_subplot(nfil, ncol, k + 1)
        if nombre in ("particulas", "lambda", "densidad", "gradgamma", "gradgamma_sup", "dnn"):
            t = TITULOS[nombre] + (f"  (N={dist.N})" if nombre == "particulas" else "")
            _mapa(ax, dist, campos, nombre, sec, titulo=t)
        elif nombre == "hist_dnn":
            ax.hist(campos["dnn"], bins=50, range=(0.6, 1.4), color="#4477aa")
            ax.set_xlabel("dnn / dx"); ax.set_title("Histograma distancia al vecino", fontsize=9)
        elif nombre == "hist_lambda":
            ax.hist(campos["Lambda"][inter], bins=50, range=(0.5, 1.0), color="#228833")
            ax.set_xlabel("Lambda (interior)"); ax.set_title("Histograma Lambda", fontsize=9)
        elif nombre == "hist_densidad":
            e = 100 * (campos["rho_rel"][inter] - 1)
            ax.hist(e, bins=50, color="#aa3377")
            ax.set_xlabel("error densidad interior (%)")
            ax.set_title("Histograma error de densidad", fontsize=9)
        elif nombre == "hist_gradgamma":
            ax.hist(campos["gradGamma_h"][inter], bins=50, color="#ee7733", label="interior")
            ax.set_xlabel("|grad Gamma|*h"); ax.set_title("Histograma |grad Gamma|*h", fontsize=9)
        elif nombre == "perfil_superficie":
            g = campos["gradGamma_h"]
            s, et = _posicion_superficie(dist, capa)
            o = np.argsort(s)
            ax.plot(s[o], g[capa][o], ".", ms=3, color="#cc3311")
            ax.axhline(g[capa].mean(), color="k", lw=0.8, ls="--")
            ax.set_ylim(0, 1.15 * np.percentile(g[capa], 99.5))
            ax.set_xlabel(et); ax.set_ylabel("|grad Gamma|*h")
            ax.set_title(f"|grad Gamma|*h en la superficie (CV = "
                         f"{r.get('gradGamma_h_sup_CV_%', np.nan):.1f} %)", fontsize=9)
            ax.grid(alpha=0.3)
        elif nombre == "perfil_dnn":
            xi = dist.x[:, 0]
            nb = 30
            bins = np.linspace(dist.geom.bbox[0][0], dist.geom.bbox[1][0], nb + 1)
            idx = np.clip(np.digitize(xi, bins) - 1, 0, nb - 1)
            med = [np.mean(campos["dnn"][idx == j]) if np.any(idx == j) else np.nan for j in range(nb)]
            mn = [np.min(campos["dnn"][idx == j]) if np.any(idx == j) else np.nan for j in range(nb)]
            xc = 0.5 * (bins[1:] + bins[:-1])
            ax.plot(xc, med, "o-", ms=3, label="media")
            ax.plot(xc, mn, "s-", ms=3, label="minimo")
            ax.axhline(1.0, color="k", lw=0.5)
            ax.set_xlabel("x"); ax.set_ylabel("dnn / dx"); ax.legend(fontsize=8)
            ax.set_title("Espaciado a lo largo del eje x (cf. Siemann fig. 6)", fontsize=9)
        elif nombre == "tabla":
            ax.axis("off")
            txt = "\n".join(f"{c:<20s} {_fmt(r[c])}" for c in METRICAS_TABLA if c in r)
            ax.text(0.0, 1.0, txt, family="monospace", fontsize=8, va="top")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"panel desconocido:\n{nombre}", ha="center", va="center")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig


# ==========================================================================
# 2) Rejilla con los empaquetados
# ==========================================================================
def _seleccionar(lista, metodos):
    if not metodos:
        return lista
    por_nombre = {d.metodo: (d, r, c) for d, r, c in lista}
    fuera = [m for m in metodos if m not in por_nombre]
    if fuera:
        raise ValueError(f"Metodos no calculados: {fuera}. Disponibles: {list(por_nombre)}")
    return [por_nombre[m] for m in metodos]


def figura_empaquetados(lista, metodos=None, seccion=("y", 0.0), columnas=3,
                        guardar=None, mostrar=True):
    sel = _seleccionar(lista, metodos)
    n = len(sel)
    ncol = min(columnas, n)
    nfil = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nfil, ncol, figsize=(5.2 * ncol, 3.6 * nfil + 0.4), squeeze=False)
    for k, (d, r, c) in enumerate(sel):
        ax = axs.ravel()[k]
        _mapa(ax, d, c, "particulas", seccion, titulo=f"{d.metodo}   (N = {d.N})")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig


# ==========================================================================
# 3) Comparar 2-3 empaquetados con una metrica
# ==========================================================================
def figura_comparar(lista, metodos, metrica="gradgamma", seccion=("y", 0.0),
                    guardar=None, mostrar=True):
    if metrica not in METRICAS_COMPARACION:
        raise ValueError(f"METRICA debe ser una de {list(METRICAS_COMPARACION)}")
    info = METRICAS_COMPARACION[metrica]
    sel = _seleccionar(lista, metodos)
    n = len(sel)
    clave = info["escalar"]
    valores = [r[clave] for d, r, c in sel]
    b, textos, _ = diferencia_frente_al_mejor(clave, valores)
    # escala de color comun para todos
    vmins, vmaxs = [], []
    for d, r, c in sel:
        vals, cmap, a, bb, et, masc = _campo(metrica, d, c)
        vmins.append(a); vmaxs.append(bb)
    if metrica == "densidad":
        lim = max(abs(min(vmins)), abs(max(vmaxs)))
        vmin, vmax = -lim, lim
    else:
        vmin, vmax = min(vmins), max(vmaxs)
    fig, axs = plt.subplots(2, n, figsize=(5.2 * n + 1.0, 8.2), squeeze=False,
                            layout="constrained")
    for k, (d, r, c) in enumerate(sel):
        _mapa(axs[0, k], d, c, "particulas", seccion, titulo=f"{d.metodo}  (N = {d.N})")
        t = f"{info['texto']} = {_fmt(r[clave])}\n{textos[k]}"
        col = _mapa(axs[1, k], d, c, metrica, seccion, vmin=vmin, vmax=vmax, barra=False,
                    titulo=t)
        if k == n - 1:
            fig.colorbar(col, ax=list(axs[1, :]), shrink=0.85, label=info["titulo"])
        if b is not None and k == b:
            for ax in (axs[0, k], axs[1, k]):
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       lw=4, ec=VERDE, clip_on=False))
            axs[1, k].title.set_color(VERDE)
            axs[1, k].title.set_fontweight("bold")
    fig.suptitle(f"{info['titulo']}  -  criterio: {info['texto']} "
                 f"({'mayor' if SENTIDO.get(clave) is True else 'menor'} es mejor; "
                 f"verde = mejor)", fontsize=11)
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig


# ==========================================================================
# 4) Tabla de diferencias porcentuales
# ==========================================================================
CLAVES_TABLA = ["gradGamma_h_max", "gradGamma_h_sup_CV_%", "err_rho_std_%",
                "err_rho_media_%", "Lambda_media", "Lambda_min", "dnn_min", "dnn_CV_%",
                "Vvor_CV_%", "hueco_sup_max", "capa_ext_CV_%"]


def tabla_diferencias(lista, claves=None):
    """Lista de filas: (clave, valores, indice_mejor, textos)."""
    claves = claves or CLAVES_TABLA
    filas = []
    for k in claves:
        vals = [r[k] for d, r, c in lista]
        b, textos, pct = diferencia_frente_al_mejor(k, vals)
        filas.append((k, vals, b, textos))
    return filas


def imprimir_diferencias(lista, claves=None):
    nombres = [d.metodo for d, r, c in lista]
    ancho = 21
    print("\n=== Diferencias frente al mejor de cada metrica "
          "(% = cuanto peor que el mejor; abs. si el mejor es ~0 o hay valor objetivo) ===")
    print("metrica".ljust(22) + "".join(m[:ancho - 1].rjust(ancho) for m in nombres))
    for k in (claves or CLAVES_TABLA):
        vals = [r[k] for d, r, c in lista]
        b, textos, pct = diferencia_frente_al_mejor(k, vals, corto=True)
        fila = ""
        for j, v in enumerate(vals):
            t = f"{v:.3g} {'*' if j == b else '(' + textos[j] + ')'}"
            fila += " " + t.rjust(ancho - 1)
        print(k.ljust(22) + fila)
    print("(*) = mejor de la fila")


def figura_tabla_diferencias(lista, claves=None, guardar=None, mostrar=True):
    filas = tabla_diferencias(lista, claves)
    nombres = [d.metodo for d, r, c in lista]
    celdas, colores = [], []
    for k, vals, b, textos in filas:
        fila, col = [], []
        for j, v in enumerate(vals):
            fila.append(f"{_fmt(v)}\n{textos[j]}" if textos[j] else _fmt(v))
            col.append("#c7e9c0" if j == b else "white")
        celdas.append(fila)
        colores.append(col)
    etiquetas = [f"{k}\n{DESCRIPCION.get(k, '')[:46]}" for k, *_ in filas]
    fig, ax = plt.subplots(figsize=(2.3 * len(nombres) + 4.5, 0.62 * len(filas) + 1.4))
    ax.axis("off")
    t = ax.table(cellText=celdas, cellColours=colores, rowLabels=etiquetas, colLabels=nombres,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 2.3)
    ax.set_title("Valor de cada metrica y diferencia frente al mejor (verde = mejor)",
                 fontsize=11)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130, bbox_inches="tight")
    if mostrar:
        plt.show()
    return fig
