"""
Algoritmos de empaquetado de particulas (pasos 4, 5 y 6), sin PySPH.

  empaquetar_colagrossi : Colagrossi et al. 2012 (paper 5)
  empaquetar_negi       : Negi & Ramachandran 2021, algoritmo hibrido (paper 6)
  empaquetar_wvt        : Weighted Voronoi Tessellation extendida (EWVT),
                          Diehl et al. 2015 / Siemann & Ritt 2019 (paper 4)

Todas devuelven (x, historial, info) con x = posiciones finales del cuerpo.
Funcionan igual en 2D y en 3D (la dimension se toma de geom.dim).
"""
import time
import numpy as np

from .kernels import crear_kernel
from .vecinos import pares, sumar_por_particula, ListaVerlet
from .redes2d import red_regular
from .redes3d import red_regular_3d


# ==========================================================================
# Utilidades comunes
# ==========================================================================
class Superficie:
    """Conjunto de curvas cerradas (p.ej. el contorno desplazado dx/2 hacia
    dentro) con consulta de punto mas cercano."""

    def __init__(self, curvas):
        self.curvas = curvas
        self.longitud = sum(c.perimetro for c in curvas)

    def punto_cercano(self, p):
        best_q, best_d, best_n = None, None, None
        for c in self.curvas:
            q, d, n = c.punto_cercano(p)
            if best_d is None:
                best_q, best_d, best_n = q, d, n
            else:
                m = d < best_d
                best_q[m], best_d[m], best_n[m] = q[m], d[m], n[m]
        return best_q, best_d, best_n


def red_inicial(geom, dx, tipo="cubica", **kw):
    """Red regular en 2D o 3D. tipo: 'cubica' | 'hexagonal' (hcp en 3D) | 'fcc' | ..."""
    if geom.dim == 2:
        return red_regular(geom, dx, tipo, **kw)
    mapa = {"cubica": "sc", "hexagonal": "hcp"}
    return red_regular_3d(geom, dx, mapa.get(tipo, tipo), **kw)


def capa_paralela(geom, c, ds):
    """Puntos con espaciado ~ds sobre la superficie paralela a distancia c
    (c>0 fuera, c<0 dentro)."""
    if geom.dim == 2:
        return np.vstack([cv.remuestrear(ds)[0] for cv in geom.curva_offset(c, ds)])
    return geom.muestrear_offset(c, ds)[0]


def superficie_paralela(geom, c, ds):
    """Objeto con punto_cercano(p) para la superficie paralela, y su medida
    (longitud en 2D, area en 3D)."""
    if geom.dim == 2:
        sup = Superficie(geom.curva_offset(c, ds))
        return sup, sup.longitud
    _, area = geom.puntos_banda(c, ds / 3.0)
    return geom.superficie_offset(c), area


def _direcciones_aleatorias(rng, n, dim):
    v = rng.normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _log(verbose, *a):
    if verbose:
        print(*a, flush=True)


def _empujar_dentro(geom, x, u, distancia, idx=None):
    """Contacto con la pared: toda particula a menos de 'distancia' del contorno
    (o fuera) se recoloca a 'distancia' hacia dentro y se anula su velocidad
    normal. Devuelve el numero de particulas corregidas."""
    idx = np.arange(len(x)) if idx is None else idx
    if len(idx) == 0:
        return 0
    sd = geom.distancia_firmada(x[idx])
    malas = idx[sd > -distancia]
    if len(malas):
        q, _, n = geom.punto_cercano(x[malas])
        x[malas] = q - n * distancia
        if u is not None:
            un = np.einsum("ij,ij->i", u[malas], n)
            u[malas] -= np.maximum(un, 0)[:, None] * n
    return int(len(malas))


class Pared:
    """Aplica _empujar_dentro solo a las particulas de una banda cercana al
    contorno (la banda se recalcula cada 'cada' llamadas). Mucho mas rapido."""

    def __init__(self, geom, distancia, banda, cada=20):
        self.geom, self.distancia, self.banda, self.cada = geom, distancia, banda, cada
        self.n = 0
        self.idx = None

    def aplicar(self, x, u=None):
        if self.idx is None or self.n % self.cada == 0:
            sd = self.geom.distancia_firmada(x)
            self.idx = np.where(sd > -self.banda)[0]
        self.n += 1
        return _empujar_dentro(self.geom, x, u, self.distancia, self.idx)


# ==========================================================================
# Paso 4: Colagrossi et al. (2012)
# ==========================================================================
def empaquetar_colagrossi(geom, dx, h_factor=None, alpha=5e-3, cfl=1.0,
                          red_inicial_tipo="cubica", kernel="wendland",
                          max_iter=4000, min_iter=300, tol_ek=1e-7,
                          cada=25, verbose=True):
    """Sistema de la ec. (2.16) del paper:
         du/dt = -beta grad(Gamma) - zeta u ,   dr/dt = u
    con particulas solidas FIJAS fuera del contorno (el contorno libre del
    pajaro se trata como pared). beta = 2 p0/rho0 (=1 aqui, adimensional),
    zeta = alpha sqrt(beta)/V0^(1/d), dt = CFL V0^(1/d)/sqrt(beta).

    Las capas fijas se colocan sobre curvas/superficies paralelas al contorno a
    dx/2, 3dx/2, 5dx/2... para que el borde quede liso (no en escalera).
    Parada: energia cinetica especifica < tol_ek * beta (paper: 1e-7..1e-8).
    """
    t0 = time.time()
    dim = geom.dim
    # h/dx: 2.0 en 2D (tabla 1 del paper); 1.4 en 3D (coste: ~3 veces menos vecinos)
    h_factor = h_factor or (2.0 if dim == 2 else 1.4)
    x = red_inicial(geom, dx, red_inicial_tipo).copy()
    N = len(x)
    V0 = geom.volumen / N
    dxe = V0 ** (1.0 / dim)
    ker = crear_kernel(kernel, dim)
    h = h_factor * dxe
    rc = ker.radio(h)

    # --- particulas fijas (fantasma) en capas paralelas al contorno --------
    fijas = []
    k = 0
    while (k + 0.5) * dxe < rc + dxe:
        fijas.append(capa_paralela(geom, (k + 0.5) * dxe, dxe))
        k += 1
    xf = np.vstack(fijas)
    _log(verbose, f"[Colagrossi] N libres = {N}, N fijas = {len(xf)}, "
                  f"h = {h:.4g} ({h_factor} dx), soporte = {rc:.4g}")

    beta = 1.0
    zeta = alpha * np.sqrt(beta) / dxe
    dt = cfl * dxe / np.sqrt(beta)
    u = np.zeros_like(x)
    lv = ListaVerlet(rc, 0.5 * dxe, n_dest=N)
    pared = Pared(geom, 0.2 * dxe, 3.0 * dxe)
    hist = {"iter": [], "Ek_especifica": [], "gradGamma_h_max": [], "corregidas_pared": []}

    for it in range(max_iter + 1):
        todo = np.vstack([x, xf])
        i, j, xij, rij = lv.pares(todo)
        m = i < N
        i, j, xij, rij = i[m], j[m], xij[m], rij[m]
        dW = ker.dWdr(rij, h)
        grad = V0 * sumar_por_particula(i, (dW / np.maximum(rij, 1e-12))[:, None] * xij, N)
        a = -beta * grad - zeta * u
        u += dt * a
        x += dt * u
        nc = pared.aplicar(x, u)

        if it % cada == 0:
            ek = 0.5 * np.mean(np.einsum("ij,ij->i", u, u))
            gmax = np.linalg.norm(grad, axis=1).max() * h
            hist["iter"].append(it)
            hist["Ek_especifica"].append(ek / beta)
            hist["gradGamma_h_max"].append(gmax)
            hist["corregidas_pared"].append(nc)
            if it % (cada * 20) == 0:
                _log(verbose, f"  it {it:6d}  Ek/beta = {ek / beta:.3e}  "
                              f"max|gradGamma|h = {gmax:.3e}")
            if it >= min_iter and ek / beta < tol_ek:
                _log(verbose, f"  Convergido en {it} iteraciones (Ek/beta < {tol_ek})")
                break
    info = dict(iteraciones=it, tiempo_s=time.time() - t0, x_fijas=xf, h=h)
    _log(verbose, f"[Colagrossi] terminado en {info['tiempo_s']:.1f} s")
    return x, hist, info


# ==========================================================================
# Paso 5: Negi & Ramachandran (2021), algoritmo hibrido
# ==========================================================================
def empaquetar_negi(geom, dx, hdx=None, pb=1.0, nu=None, k=None, corte=0.95,
                    freq_proy=50, tol=1e-4, max_iter=8000, min_iter=1000, margen=None,
                    kernel=None, contenedor=None, sep_contorno=0.5, limitar_Ns=False,
                    verbose=True):
    """Ec. (18) del paper:  du/dt = -grad(pb)/rho - grad(phi_RF) - zeta u

      * particulas LIBRES en red hexagonal llenando una caja que contiene al
        cuerpo (dentro y fuera) + particulas CONGELADAS alrededor de la caja.
      * NODOS de contorno: el contorno desplazado dx/2 hacia dentro (ec. 11).
      * cada freq_proy iteraciones las libres a < 0.5 dx (0.65 dx tras 4
        pasadas vacias) de la superficie se convierten en particulas de
        CONTORNO, que solo se mueven tangencialmente sobre la superficie.
      * esquinas convexas: se fija una particula en la esquina ("hard point").
      * parada: max|u| dt / h < tol una vez terminada la proyeccion (ec. 23).
    Constantes por defecto del codigo de referencia: pb=1, nu=0.2/dx (2D) o
    0.5/dx (3D), k=0.004 dx (2D) o 0.006 dx (3D), corte=0.95, quintic spline
    con h = 1.2 dx.

    contenedor = "caja"  : como el paper (caja alrededor del cuerpo).
                 "forma" : las libres ocupan el cuerpo + una banda de 'margen'
                           dx alrededor, y las congeladas forman una capa que
                           sigue la forma. Mismo resultado en el cuerpo y
                           muchas menos particulas (por defecto en 3D).
    """
    t0 = time.time()
    dim = geom.dim
    # 2D: quintic spline h=1.2dx (paper). 3D: Wendland h=1.3dx por coste
    # (soporte 2.6dx frente a 3.6dx: ~2.6 veces menos vecinos y mismo resultado)
    kernel = kernel or ("quintic" if dim == 2 else "wendland")
    hdx = hdx or (1.2 if kernel.startswith("quin") else 1.3)
    nu = (0.2 if dim == 2 else 0.5) / dx if nu is None else nu
    k = (0.004 if dim == 2 else 0.006) * dx if k is None else k
    ker = crear_kernel(kernel, dim)
    h = hdx * dx
    rc = ker.radio(h)
    mpart = dx ** dim                        # rho0 = 1, volumen por particula dx^d

    contenedor = contenedor or ("caja" if dim == 2 else "forma")
    margen = margen if margen is not None else (4 if contenedor == "caja" else 2)

    # --- caja con particulas libres (hexagonal/hcp, mismo volumen) ----------
    lo = geom.bbox[0] - margen * dx
    hi = geom.bbox[1] + margen * dx

    class _Caja:
        bbox = np.array([lo, hi])
        dim = geom.dim

        @staticmethod
        def contiene(p):
            return np.all((p > lo) & (p < hi), axis=1)

        @staticmethod
        def distancia_firmada(p):
            return np.max(np.maximum(lo - p, p - hi), axis=1)

    ncap = int(np.ceil(rc / dx)) + 1
    ejes = [np.arange(lo[d] - (ncap - 0.5) * dx, hi[d] + ncap * dx, dx) for d in range(dim)]
    g = np.stack([a.ravel() for a in np.meshgrid(*ejes, indexing="ij")], 1)
    if contenedor == "caja":
        x = red_inicial(_Caja, dx, "hexagonal", margen=0.45 * dx).copy()
        # congeladas: red cubica dx alrededor de la caja
        xf = g[~_Caja.contiene(g)]
    else:
        # libres: hexagonal/hcp dentro de {sdf < margen*dx}
        xl = red_inicial(_Caja, dx, "hexagonal", margen=0.0)
        x = xl[geom.distancia_firmada(xl) < margen * dx - 0.45 * dx].copy()
        # congeladas: capa de red cubica que sigue la forma
        sdg = geom.distancia_firmada(g)
        xf = g[(sdg >= margen * dx) & (sdg < margen * dx + rc + dx)]

    # --- superficie de contorno desplazada dx/2 hacia dentro ---------------
    sup, medida = superficie_paralela(geom, -0.5 * dx, dx)
    Ns = int(round(medida / dx ** (dim - 1)))
    esquinas = geom.esquinas_offset(0.5 * dx)
    N = len(x)
    tipo = np.zeros(N, dtype=int)            # 0 libre, 1 contorno, 2 fija (esquina)
    u = np.zeros_like(x)
    _log(verbose, f"[Negi] libres = {N}, congeladas = {len(xf)}, Ns estimado = {Ns}, "
                  f"esquinas = {len(esquinas)}, h = {h:.4g}")

    hist = {"iter": [], "max_u_dt_h": [], "n_contorno": [], "L_inf_rho": []}
    vacias = 0
    convergido = False
    dt = 0.1 * h / np.sqrt(pb)
    lv = ListaVerlet(rc, 0.3 * dx, n_dest=N)

    for it in range(max_iter + 1):
        todo = np.vstack([x, xf])
        nt = len(todo)
        i, j, xij, rij = lv.pares(todo)
        W = ker.W(rij, h)
        rho = np.ones(nt)                    # congeladas: red regular, rho = 1
        rho[:N] = mpart * (ker.W(0.0, h) + sumar_por_particula(i, W, N))
        V = mpart / rho
        dW = ker.dWdr(rij, h)
        eij = xij / np.maximum(rij, 1e-12)[:, None]
        # gradiente de densidad numerica (ec. 13)
        a = sumar_por_particula(i, (-pb * V[i] * V[j] / mpart * dW)[:, None] * eij, N)
        # fuerza de repulsion (ec. 16)
        cerca = rij < corte * dx
        if np.any(cerca):
            r_ = np.maximum(rij[cerca], 0.5 * dx)
            c = 2.0 * corte * dx / 3.0
            f = 12.0 * k * (3 * c ** 2 / r_ ** 4 - 2 * c / r_ ** 3)
            a += sumar_por_particula(i[cerca], np.maximum(f, 0)[:, None] * eij[cerca], N)
        a -= nu * u                              # amortiguamiento (ec. 17)

        # paso de tiempo (como en el codigo de referencia)
        vmax = np.sqrt(np.max(np.einsum("ij,ij->i", u + dt * a, u + dt * a)))
        dt = 0.1 * h / np.sqrt(pb)
        if vmax > 1e-14:
            dt = min(dt, np.sqrt(0.1 * h / nu / vmax))

        u += dt * a
        u[tipo == 2] = 0.0
        cont = tipo == 1
        if np.any(cont):                          # solo movimiento tangencial
            _, _, nrm = sup.punto_cercano(x[cont])
            un = np.einsum("ij,ij->i", u[cont], nrm)
            u[cont] -= un[:, None] * nrm
        x += dt * u
        if np.any(cont):                          # mantener sobre la superficie
            q, _, _ = sup.punto_cercano(x[cont])
            x[cont] = q

        # --- conversion/proyeccion de libres al contorno (Algoritmo 1) ----
        if it % freq_proy == 0:
            if it == 0 and len(esquinas):
                for e in esquinas:
                    libres = np.where(tipo == 0)[0]
                    kk = libres[np.argmin(np.linalg.norm(x[libres] - e, axis=1))]
                    x[kk], u[kk], tipo[kk] = e, 0.0, 2
            libres = np.where(tipo == 0)[0]
            q, d, nrm = sup.punto_cercano(x[libres])
            umbral = 0.5 * dx if vacias < 4 else 0.65 * dx
            cand = libres[d < umbral]
            qcand = q[d < umbral]
            # no convertir si quedaria pegada (<0.5 dx) a otra de contorno
            ocupadas = x[tipo >= 1]
            nuevas = 0
            for kk, qq in zip(cand, qcand):
                if limitar_Ns and len(ocupadas) >= Ns:
                    break
                if len(ocupadas) and np.min(np.linalg.norm(ocupadas - qq, axis=1)) < sep_contorno * dx:
                    continue
                x[kk], tipo[kk] = qq, 1
                u[kk] = 0.0
                ocupadas = np.vstack([ocupadas, qq]) if len(ocupadas) else qq[None]
                nuevas += 1
            if nuevas > 0 and vacias < 4:
                vacias = 0
            else:
                vacias += 1

            movil = tipo < 2
            crit = vmax * dt / h if np.any(movil) else 0.0
            dentro = geom.distancia_firmada(x) < 0
            err = np.abs(rho[:N][dentro] - 1.0).max() if dentro.any() else np.nan
            hist["iter"].append(it)
            hist["max_u_dt_h"].append(crit)
            hist["n_contorno"].append(int((tipo >= 1).sum()))
            hist["L_inf_rho"].append(err)
            if it % (freq_proy * 10) == 0:
                _log(verbose, f"  it {it:6d}  max|u|dt/h = {crit:.3e}  "
                              f"contorno = {(tipo >= 1).sum()}/{Ns}  Linf(rho-1) = {err:.3e}")
            if vacias > 4 and crit < tol and it >= min_iter:
                convergido = True
                _log(verbose, f"  Convergido en {it} iteraciones (max|u|dt/h < {tol})")
                break

    # --- separar interior / exterior (seccion 3.1.7) -----------------------
    sd = geom.distancia_firmada(x)
    cuerpo = (tipo >= 1) | ((tipo == 0) & (sd < -0.5 * dx))
    info = dict(iteraciones=it, tiempo_s=time.time() - t0, convergido=convergido,
                x_todas=x.copy(), tipo=tipo.copy(), x_congeladas=xf,
                n_contorno=int((tipo >= 1).sum()), Ns=Ns, h=h)
    _log(verbose, f"[Negi] terminado en {info['tiempo_s']:.1f} s; particulas del cuerpo = "
                  f"{cuerpo.sum()} (contorno {info['n_contorno']})")
    return x[cuerpo].copy(), hist, info


# ==========================================================================
# Paso 6: (Extended) Weighted Voronoi Tessellation - Siemann & Ritt (2019)
# ==========================================================================
def empaquetar_wvt(geom, dx, h_factor=1.0, xi=4e-2, kappa=4e-4, chi=1e-2,
                   CR=1e-8, max_iter=10000, agitar=0.5, contacto=0.5,
                   semilla=0, cada=25, verbose=True, h_func=None):
    """Ec. (7)-(10) del paper 4:
         dr_i = xi h_i sum_j f(h_ij, r_ij) r_ij/|r_ij|
         f    = 1/(r/h + chi)^2 - 1/(2 + chi)^2     (r < 2h)
         xi_{n+1} = (1 - kappa) xi_n
    Contorno: contacto (penalizacion rigida) con la pared a 'contacto'*dx
    (el paper usa una altura de contacto = radio de particula).
    Inicio: red cubica agitada aleatoriamente hasta un 50 % de dx (paper).
    N = V/dx^d exacto.
    Parada: Delta R_n / Delta R_max <= CR.
    h_func (opcional): funcion h(x) para resolucion variable (WVT "ponderada";
    se usara en la fase de 'inflation').
    """
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    dim = geom.dim
    Nobj = int(round(geom.volumen / dx ** dim))
    x = red_inicial(geom, dx, "cubica").copy()
    # ajustar al numero objetivo de particulas N = V/dx^d
    if len(x) > Nobj:
        x = x[rng.choice(len(x), Nobj, replace=False)]
    while len(x) < Nobj:
        p = rng.uniform(geom.bbox[0], geom.bbox[1], size=(4 * (Nobj - len(x)) + 10, dim))
        p = p[geom.distancia_firmada(p) < -contacto * dx]
        x = np.vstack([x, p[: Nobj - len(x)]])
    # agitado ("node shaking")
    mag = rng.uniform(0, agitar, len(x)) * dx
    x += _direcciones_aleatorias(rng, len(x), dim) * mag[:, None]
    N = len(x)
    dxe = (geom.volumen / N) ** (1.0 / dim)
    _empujar_dentro(geom, x, None, contacto * dxe)

    hist = {"iter": [], "dR_rel": [], "xi": []}
    dR_max = 0.0
    xi_n = xi
    _log(verbose, f"[WVT] N = {N}, h = {h_factor} dx, xi0 = {xi}, kappa = {kappa}")
    hmax = h_factor * dxe * (1.0 if h_func is None else float(np.max(h_func(x))) * 1.2)
    lv = ListaVerlet(2.0 * hmax, 0.3 * dxe)
    pared = Pared(geom, contacto * dxe, 3.0 * dxe)
    for it in range(max_iter + 1):
        hi = h_factor * dxe * (np.ones(N) if h_func is None else h_func(x))
        i, j, xij, rij = lv.pares(x)
        hij = 0.5 * (hi[i] + hi[j])
        ok = rij < 2.0 * hij
        i, j, xij, rij, hij = i[ok], j[ok], xij[ok], rij[ok], hij[ok]
        f = 1.0 / (rij / hij + chi) ** 2 - 1.0 / (2.0 + chi) ** 2
        eij = xij / np.maximum(rij, 1e-12)[:, None]
        sup = rij < 1e-6 * dxe                   # particulas superpuestas:
        if np.any(sup):                          # se separan en direccion aleatoria
            eij[sup] = _direcciones_aleatorias(rng, int(sup.sum()), dim)
        dr = xi_n * hi[:, None] * sumar_por_particula(i, f[:, None] * eij, N)
        x_ant = x.copy()
        x += dr
        pared.aplicar(x)
        dR = np.sum((x - x_ant) ** 2)
        dR_max = max(dR_max, dR)
        rel = dR / dR_max if dR_max > 0 else 1.0
        xi_n *= (1.0 - kappa)
        if it % cada == 0:
            hist["iter"].append(it)
            hist["dR_rel"].append(rel)
            hist["xi"].append(xi_n)
            if it % (cada * 40) == 0:
                _log(verbose, f"  it {it:6d}  dR/dRmax = {rel:.3e}  xi = {xi_n:.3e}")
        if it > 50 and rel <= CR:
            _log(verbose, f"  Convergido en {it} iteraciones (dR/dRmax <= {CR})")
            break
    info = dict(iteraciones=it, tiempo_s=time.time() - t0)
    _log(verbose, f"[WVT] terminado en {info['tiempo_s']:.1f} s")
    return x, hist, info
