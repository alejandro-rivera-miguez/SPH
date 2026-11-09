"""
INFLATION: distribucion de particulas con resolucion variable (2D y 3D)
=======================================================================
Idea: se define un campo de tamano dx(x). Dentro de las ZONAS de refinado
dx = dx_fino; fuera crece suavemente con la distancia a la zona y se queda
en dx_grueso:

        dx(x) = min( dx_grueso ,  min_k [ dx_k + tasa * d_k(x) ] )

  d_k(x) = distancia del punto a la zona k (0 dentro de la zona)
  tasa   = crecimiento por capa de particulas: si una capa tiene espaciado s,
           la siguiente tiene s*(1 + tasa). tasa = 0.05 -> +5 % por capa.
           Longitud de la transicion = (dx_grueso - dx_fino) / tasa.

Pasos:
  1. distribucion_inicial(): celdas adaptativas (quadtree/octree) con el numero
     de particulas que pide dx(x) en cada celda (N = integral de dV/dx^d).
  2. relajar_wvt(): WVT de Siemann & Ritt (paper 4) con h_i = dx(x_i), que es
     la version "ponderada" del propio paper. Busqueda de vecinos por bandas
     de tamano (no usa el radio grueso en la zona fina), escalable a millones.
  3. masas(): V_i proporcional a dx_i^d, normalizado para que sum V_i = V del
     cuerpo -> masa total EXACTA = rho0 * V. m_i = rho0 * V_i.
  4. metricas_inflation(): densidad SPH con h variable (W4, h_ij medio), Lambda,
     |grad Gamma|*h, espaciado real frente al pedido, saltos entre vecinas.

Las zonas son independientes de la geometria (una esfera en el morro del
pajaro, una caja en el punto de impacto, una "losa" alrededor de la superficie
libre de la piscina...).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

from .kernels import crear_kernel

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy >= 2 / < 2


def _log(verbose, *a):
    if verbose:
        print(*a, flush=True)


# ==========================================================================
# Zonas de refinado
# ==========================================================================
class Zona:
    """Zona de refinado.

    tipo = "esfera" : centro, radio            (en 2D es un circulo)
           "caja"   : pmin, pmax               (en 2D un rectangulo)
           "losa"   : punto, normal, espesor   (banda alrededor de un plano:
                                                superficie libre de una piscina)
           "punto"  : centro                   (= esfera de radio 0)
    dx : tamano dentro de la zona (None -> el dx_fino del campo)
    """
    TIPOS = ("esfera", "caja", "losa", "punto")

    def __init__(self, tipo, dx=None, nombre=None, **p):
        tipo = tipo.lower()
        if tipo not in self.TIPOS:
            raise ValueError(f"tipo de zona desconocido: {tipo}  (usa {self.TIPOS})")
        self.tipo, self.dx, self.p = tipo, dx, dict(p)
        self.nombre = nombre or tipo
        if tipo in ("esfera", "punto"):
            self.c = np.asarray(p["centro"], float)
            self.R = float(p.get("radio", 0.0)) if tipo == "esfera" else 0.0
        elif tipo == "caja":
            a, b = np.asarray(p["pmin"], float), np.asarray(p["pmax"], float)
            self.lo, self.hi = np.minimum(a, b), np.maximum(a, b)
        else:
            self.c = np.asarray(p["punto"], float)
            n = np.asarray(p["normal"], float)
            self.n = n / np.linalg.norm(n)
            self.e = float(p.get("espesor", 0.0))

    @property
    def dim(self):
        return len(self.c) if hasattr(self, "c") else len(self.lo)

    def distancia(self, x):
        """Distancia (>= 0) de cada punto a la zona; 0 dentro."""
        x = np.atleast_2d(x)
        if self.tipo in ("esfera", "punto"):
            return np.maximum(np.linalg.norm(x - self.c, axis=1) - self.R, 0.0)
        if self.tipo == "caja":
            d = np.maximum(np.maximum(self.lo - x, x - self.hi), 0.0)
            return np.linalg.norm(d, axis=1)
        return np.maximum(np.abs((x - self.c) @ self.n) - 0.5 * self.e, 0.0)

    def describir(self):
        f = lambda v: "(" + ", ".join(f"{a:g}" for a in v) + ")"
        if self.tipo == "esfera":
            s = f"esfera c={f(self.c)} R={self.R:g}"
        elif self.tipo == "punto":
            s = f"punto c={f(self.c)}"
        elif self.tipo == "caja":
            s = f"caja {f(self.lo)} - {f(self.hi)}"
        else:
            s = f"losa p={f(self.c)} n={f(self.n)} e={self.e:g}"
        return s + (f"  dx={self.dx:g}" if self.dx else "")

    def a_dict(self):
        return dict(tipo=self.tipo, dx=self.dx, nombre=self.nombre, **self.p)

    @staticmethod
    def de_dict(d):
        d = dict(d)
        return Zona(d.pop("tipo"), dx=d.pop("dx", None), nombre=d.pop("nombre", None), **d)


def crear_zona(d):
    return d if isinstance(d, Zona) else Zona.de_dict(d)


# ==========================================================================
# Campo de tamano dx(x)
# ==========================================================================
class CampoResolucion:
    def __init__(self, dx_fino, dx_grueso, tasa=0.05, zonas=()):
        if dx_fino > dx_grueso:
            raise ValueError("dx_fino debe ser <= dx_grueso")
        if not 0.0 < tasa < 0.6:
            raise ValueError("tasa debe estar entre 0 y 0.6 (0.02-0.10 recomendado)")
        self.dx_fino, self.dx_grueso, self.tasa = float(dx_fino), float(dx_grueso), float(tasa)
        self.zonas = [crear_zona(z) for z in zonas]

    def dx(self, x):
        x = np.atleast_2d(x)
        out = np.full(len(x), self.dx_grueso)
        for z in self.zonas:
            dz = z.dx if z.dx else self.dx_fino
            np.minimum(out, dz + self.tasa * z.distancia(x), out=out)
        return out

    def distancia_zonas(self, x):
        x = np.atleast_2d(x)
        if not self.zonas:
            return np.full(len(x), np.inf)
        return np.min([z.distancia(x) for z in self.zonas], axis=0)

    @property
    def dx_min(self):
        if not self.zonas:
            return self.dx_grueso
        return min([z.dx or self.dx_fino for z in self.zonas])

    @property
    def longitud_transicion(self):
        return (self.dx_grueso - self.dx_min) / self.tasa

    def describir(self):
        L = [f"dx fino = {self.dx_fino:g}, dx grueso = {self.dx_grueso:g}, "
             f"tasa = {100 * self.tasa:.1f} % por capa, transicion = {self.longitud_transicion:.4g}"]
        L += [f"  zona {k + 1}: {z.describir()}" for k, z in enumerate(self.zonas)]
        return "\n".join(L)


# ==========================================================================
# 1. Distribucion inicial adaptativa
# ==========================================================================
def _hojas(geom, campo, celda=3.0, max_niveles=24):
    """Celdas hoja (esquina, lado) de un quadtree/octree cuyo lado es
    <= celda * dx en toda la celda. Se descartan las que estan fuera del cuerpo."""
    d = geom.dim
    lo, hi = np.asarray(geom.bbox[0], float), np.asarray(geom.bbox[1], float)
    s = celda * campo.dx_grueso
    n = np.maximum(np.ceil((hi - lo) / s).astype(int), 1)
    centro = 0.5 * (lo + hi)
    lo = centro - 0.5 * n * s
    grid = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, d)
    esq = lo + grid * s
    hijos = np.stack(np.meshgrid(*[[0, 1]] * d, indexing="ij"), -1).reshape(-1, d)
    hojas_e, hojas_s = [], []
    sqd = np.sqrt(d)
    for _ in range(max_niveles):
        if len(esq) == 0:
            break
        c = esq + 0.5 * s
        fuera = np.zeros(len(c), bool)
        for a in range(0, len(c), 500000):
            fuera[a:a + 500000] = geom.distancia_firmada(c[a:a + 500000]) > 0.5 * s * sqd
        esq, c = esq[~fuera], c[~fuera]
        cota = np.maximum(campo.dx(c) - campo.tasa * 0.5 * s * sqd, campo.dx_min)
        partir = s > celda * cota
        hojas_e.append(esq[~partir])
        hojas_s.append(np.full((~partir).sum(), s))
        esq = (esq[partir][:, None, :] + hijos[None] * (0.5 * s)).reshape(-1, d)
        s *= 0.5
    return np.vstack(hojas_e), np.concatenate(hojas_s)


def estimar_n(geom, campo, celda=3.0):
    """Numero de particulas aproximado (sin generarlas)."""
    esq, s = _hojas(geom, campo, celda)
    c = esq + 0.5 * s[:, None]
    dentro = geom.distancia_firmada(c) < 0
    return int(np.sum((s[dentro] / campo.dx(c[dentro])) ** geom.dim))


def distribucion_inicial(geom, campo, semilla=0, celda=3.0, agitar=0.3, verbose=True):
    """Particulas iniciales: en cada celda hoja de lado s se colocan
    n ~ (s/dx)^d particulas en una sub-red agitada; se quedan las de dentro."""
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    d = geom.dim
    esq, s = _hojas(geom, campo, celda)
    c = esq + 0.5 * s[:, None]
    nexp = (s / campo.dx(c)) ** d
    n = np.floor(nexp + rng.random(len(nexp))).astype(int)
    m = np.ceil(n ** (1.0 / d) - 1e-9).astype(int)
    pts = []
    for mm in np.unique(m[n > 0]):
        sel = np.where((m == mm) & (n > 0))[0]
        g = (np.stack(np.meshgrid(*[np.arange(mm)] * d, indexing="ij"), -1).reshape(-1, d) + 0.5) / mm
        for a in range(0, len(sel), max(1, 2_000_000 // len(g))):
            b = sel[a:a + max(1, 2_000_000 // len(g))]
            P = esq[b][:, None, :] + g[None] * s[b][:, None, None]
            P += (rng.random(P.shape) - 0.5) * (2 * agitar / mm) * s[b][:, None, None]
            orden = np.argsort(rng.random((len(b), len(g))), axis=1)
            rango = np.empty_like(orden)
            np.put_along_axis(rango, orden, np.arange(len(g))[None].repeat(len(b), 0), axis=1)
            pts.append(P[rango < n[b][:, None]])
    x = np.vstack(pts) if pts else np.zeros((0, d))
    dentro = np.zeros(len(x), bool)
    for a in range(0, len(x), 500000):
        dentro[a:a + 500000] = geom.distancia_firmada(x[a:a + 500000]) < 0
    x = x[dentro]
    _log(verbose, f"[inflation] distribucion inicial: N = {len(x)} "
                  f"({len(s)} celdas, {time.time() - t0:.1f} s)")
    return x


# ==========================================================================
# 2. Vecinos con radio variable (escalable)
# ==========================================================================
def pares_radio_variable(x, R, factor_banda=1.35):
    """Pares NO dirigidos (i, j) con |xi - xj| < (R_i + R_j)/2.
    Las particulas se agrupan en bandas de R; cada banda busca dentro de si
    misma y contra las bandas de R menor con su propio radio maximo, asi la
    zona fina nunca se busca con el radio de la zona gruesa."""
    N = len(x)
    lr = np.log(R)
    lmin = lr.min()
    nb = int((lr.max() - lmin) / np.log(factor_banda)) + 1
    banda = np.minimum(((lr - lmin) / np.log(factor_banda)).astype(int), nb - 1)
    I, J = [], []
    tipo = np.int32 if N < 2**31 - 1 else np.int64
    menores = None                           # indices de bandas < k
    for k in range(nb):
        idx = np.where(banda == k)[0]
        if len(idx) == 0:
            continue
        Rk = R[idx].max()
        tk = cKDTree(x[idx])
        p = tk.query_pairs(Rk, output_type="ndarray")
        if len(p):
            a, b = idx[p[:, 0]], idx[p[:, 1]]
            r = np.linalg.norm(x[a] - x[b], axis=1)
            ok = r < 0.5 * (R[a] + R[b])
            I.append(a[ok].astype(tipo)); J.append(b[ok].astype(tipo))
        if menores is not None and len(menores):
            tm = cKDTree(x[menores])
            sdm = tk.sparse_distance_matrix(tm, Rk, output_type="ndarray")
            if len(sdm):
                a, b = idx[sdm["i"]], menores[sdm["j"]]
                ok = sdm["v"] < 0.5 * (R[a] + R[b])
                I.append(a[ok].astype(tipo)); J.append(b[ok].astype(tipo))
        menores = idx if menores is None else np.concatenate([menores, idx])
    if not I:
        return np.zeros(0, tipo), np.zeros(0, tipo)
    return np.concatenate(I), np.concatenate(J)


# numero de hilos para el calculo por pares (variable de entorno SPH_HILOS)
N_HILOS = max(1, int(os.environ.get("SPH_HILOS", os.cpu_count() or 1)))


def en_bloques(fun, n_pares, bloque=2_000_000):
    """Aplica fun(a, b) a bloques de pares [a, b) repartidos entre N_HILOS
    hilos y suma los resultados (tuplas de arrays). Limita tambien la memoria."""
    bloque = int(min(bloque, max(100_000, -(-n_pares // (2 * N_HILOS)))))
    rangos = [(a, min(a + bloque, n_pares)) for a in range(0, n_pares, bloque)]
    if not rangos:
        return None
    if N_HILOS > 1 and len(rangos) > 1:
        with ThreadPoolExecutor(N_HILOS) as ex:
            partes = list(ex.map(lambda r: fun(*r), rangos))
    else:
        partes = [fun(a, b) for a, b in rangos]
    total = list(partes[0])
    for p in partes[1:]:
        for k, v in enumerate(p):
            total[k] = total[k] + v
    return total


def _w_y_dw(ker, r, h):
    """W y (dW/dr)/r de una sola vez."""
    q = r / h
    sig = ker.sigma(h)
    return sig * ker._w(q), sig * ker._dwdq(q) / (h * np.maximum(r, 1e-12 * h))


def _sumar(i, j, v, N):
    """sum_j v_ij en i y -v_ij en j (v escalar o vector por par)."""
    if v.ndim == 1:
        return np.bincount(i, v, N) - np.bincount(j, v, N)
    return np.stack([np.bincount(i, v[:, k], N) - np.bincount(j, v[:, k], N)
                     for k in range(v.shape[1])], axis=1)


# ==========================================================================
# 3. Relajacion WVT con h variable
# ==========================================================================
def _pared(geom, x, idx, dist):
    """Las particulas idx a menos de dist_i del contorno se recolocan a dist_i."""
    if len(idx) == 0:
        return 0
    sd = geom.distancia_firmada(x[idx])
    malas = sd > -dist[idx]
    if not np.any(malas):
        return 0
    k = idx[malas]
    q, _, n = geom.punto_cercano(x[k])
    x[k] = q - n * dist[k][:, None]
    return len(k)


def relajar_wvt(geom, x, campo, h_factor=1.0, xi=0.04, kappa=2e-3, chi=1e-2,
                max_iter=600, CR=1e-5, contacto=0.5, piel=0.15, bloque=2_000_000,
                semilla=0, cada=25, verbose=True, progreso=None):
    """WVT (Siemann & Ritt 2019, ec. 7-10) con h_i = h_factor * dx(x_i):
         dr_i = xi * h_i * sum_j f(r_ij / h_ij) e_ij ,  h_ij = (h_i + h_j)/2
         f(q) = 1/(q + chi)^2 - 1/(2 + chi)^2        (q < 2)
         xi_{n+1} = (1 - kappa) xi_n
    Contorno: contacto rigido a 'contacto' * dx_i de la pared.
    Lista de vecinos con 'piel' (se rehace solo cuando alguna particula se ha
    movido mas de piel/2 * h). 'bloque' limita la memoria por iteracion.
    progreso(it, max_iter, texto): callback opcional (interfaz grafica)."""
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    x = np.array(x, float)
    N, d = x.shape
    hist = {"iter": [], "dR_rel": [], "xi": [], "reconstrucciones": 0}
    xi_n, dR_max = xi, 0.0
    lista_x = None
    _log(verbose, f"[WVT variable] N = {N}, h = {h_factor} dx(x), xi0 = {xi}, "
                  f"kappa = {kappa}, max_iter = {max_iter}, hilos = {N_HILOS}")
    for it in range(max_iter + 1):
        dxi = campo.dx(x)
        hi = h_factor * dxi
        if lista_x is None or np.max(np.linalg.norm(x - lista_x, axis=1) / hi) > 0.5 * piel:
            I, J = pares_radio_variable(x, 2.0 * hi * (1.0 + piel))
            lista_x = x.copy()
            banda = np.where(geom.distancia_firmada(x) > -(contacto + 2 * piel + 1.0) * dxi)[0]
            hist["reconstrucciones"] += 1

        def bloque_wvt(a, b):
            i, j = I[a:b], J[a:b]
            xij = x[i] - x[j]
            r = np.sqrt(np.einsum("ij,ij->i", xij, xij))
            hij = 0.5 * (hi[i] + hi[j])
            q = r / hij
            f = np.where(q < 2.0, 1.0 / (q + chi) ** 2 - 1.0 / (2.0 + chi) ** 2, 0.0)
            e = xij / np.maximum(r, 1e-12 * hij)[:, None]
            sup = r < 1e-6 * hij
            if np.any(sup):
                ang = rng.normal(size=(int(sup.sum()), d))
                e[sup] = ang / np.linalg.norm(ang, axis=1)[:, None]
            return (_sumar(i, j, f[:, None] * e, N),)

        res_b = en_bloques(bloque_wvt, len(I), bloque)
        dr = res_b[0] if res_b else np.zeros_like(x)
        dr *= xi_n * hi[:, None]
        paso = np.linalg.norm(dr, axis=1)
        tope = 0.25 * hi
        dr *= np.minimum(1.0, tope / np.maximum(paso, 1e-30))[:, None]
        x += dr
        _pared(geom, x, banda, contacto * dxi)
        dR = float(np.sum(dr * dr))
        dR_max = max(dR_max, dR)
        rel = dR / dR_max if dR_max > 0 else 1.0
        xi_n *= (1.0 - kappa)
        if it % cada == 0:
            hist["iter"].append(it); hist["dR_rel"].append(rel); hist["xi"].append(xi_n)
            texto = f"it {it:5d}  dR/dRmax = {rel:.2e}  pares = {len(I)}"
            if it % (cada * 4) == 0:
                _log(verbose, "  " + texto + f"  ({time.time() - t0:.0f} s)")
            if progreso:
                progreso(it, max_iter, texto)
        if it > 50 and rel <= CR:
            _log(verbose, f"  Convergido en {it} iteraciones (dR/dRmax <= {CR})")
            break
    info = dict(iteraciones=it, tiempo_s=time.time() - t0, pares=len(I))
    _log(verbose, f"[WVT variable] terminado: {it} iteraciones, {info['tiempo_s']:.1f} s, "
                  f"{hist['reconstrucciones']} listas de vecinos")
    return x, hist, info


# ==========================================================================
# 3b. Correccion de densidad: Gamma con volumenes variables (Colagrossi)
# ==========================================================================
_TABLAS_PARED = {}


def _tabla_pared(ker, d, n=401):
    """phi(q) = parte del kernel (h = 1) que cae al otro lado de un plano a
    distancia q = s/h. Es lo que le 'falta' a Gamma a una particula a distancia
    s de la superficie libre (pared semianalitica, sin particulas fantasma).
    Devuelve q, phi(q) y dphi/dq."""
    clave = (type(ker).__name__, d)
    if clave in _TABLAS_PARED:
        return _TABLAS_PARED[clave]
    S = ker.radio(1.0)
    z = np.linspace(0, S, n)
    t = np.linspace(0, S, 801)
    # A(z) = integral del kernel sobre el plano a cota z
    if d == 2:
        y = t
        A = np.array([2 * _trapz(ker.W(np.sqrt(y ** 2 + zz ** 2), 1.0), y) for zz in z])
    else:
        A = np.array([_trapz(2 * np.pi * t * ker.W(np.sqrt(t ** 2 + zz ** 2), 1.0), t) for zz in z])
    phi = np.array([_trapz(A[k:], z[k:]) for k in range(n)])
    _TABLAS_PARED[clave] = (z, phi, -A)
    return _TABLAS_PARED[clave]


def corregir_densidad(geom, x, campo, kernel="wendland", h_factor=1.3, A=0.5,
                      max_iter=500, tol=1e-3, contacto=0.4, piel=0.15, bloque=2_000_000,
                      repulsion=0.8, k_rep=5.0, cada=25, verbose=True, progreso=None):
    """Desplazamiento de particulas (particle shifting) que lleva
           Gamma_i = sum_j V_j W(r_ij, h_ij) + phi_pared(s_i/h_i)   ->   1
    con V_j = volumen OBJETIVO de cada particula (dx(x_j)^d). Si Gamma = 1 en
    todas, la densidad SPH con m_j = rho0 V_j es rho0: es exactamente lo que
    se quiere. dr_i = -A h_i^2 grad Gamma_i (sobreamortiguado).
    phi_pared: correccion semianalitica de la superficie libre (plano local).
    Kernel Wendland por defecto (sin 'pairing instability').
    repulsion: distancia (en dx_ij) por debajo de la cual dos particulas se
    repelen (como Negi, ec. 16); evita que se amontonen en esquinas y aristas."""
    t0 = time.time()
    x = np.array(x, float)
    N, d = x.shape
    ker = crear_kernel(kernel, d)
    S = ker.radio(1.0)
    qt, phit, dphit = _tabla_pared(ker, d)
    lista_x = None
    hist = {"iter": [], "err_gamma": []}
    _log(verbose, f"[densidad] correccion de Gamma: kernel {kernel}, h = {h_factor} dx(x), "
                  f"A = {A}, max_iter = {max_iter}, hilos = {N_HILOS}")
    for it in range(max_iter + 1):
        dxi = campo.dx(x)
        hi = h_factor * dxi
        Vi = dxi ** d
        if lista_x is None or np.max(np.linalg.norm(x - lista_x, axis=1) / hi) > 0.5 * piel:
            I, J = pares_radio_variable(x, S * hi * (1.0 + piel))
            lista_x = x.copy()
            banda = np.where(geom.distancia_firmada(x) > -(S * (1 + piel) + 0.5) * hi)[0]

        def bloque_gamma(a, b):
            i, j = I[a:b], J[a:b]
            xij = x[i] - x[j]
            r = np.sqrt(np.einsum("ij,ij->i", xij, xij))
            W, dWr = _w_y_dw(ker, r, 0.5 * (hi[i] + hi[j]))
            g = np.bincount(i, Vi[j] * W, N) + np.bincount(j, Vi[i] * W, N)
            v = dWr[:, None] * xij
            gr = (np.stack([np.bincount(i, Vi[j] * v[:, k], N) - np.bincount(j, Vi[i] * v[:, k], N)
                            for k in range(d)], axis=1))
            # repulsion de corto alcance (solo si r < repulsion * dx_ij): se suma a
            # grad Gamma como -k_rep (1 - q)^2 / dx_ij * e_ij  (empuja i lejos de j)
            dxij = 0.5 * (dxi[i] + dxi[j])
            q = r / (repulsion * dxij)
            cerca = q < 1.0
            if np.any(cerca):
                ic, jc = i[cerca], j[cerca]
                fr = k_rep * (1.0 - q[cerca]) ** 2 / dxij[cerca]
                vr = (fr / np.maximum(r[cerca], 1e-12 * dxij[cerca]))[:, None] * xij[cerca]
                gr = gr - np.stack([np.bincount(ic, vr[:, k], N) - np.bincount(jc, vr[:, k], N)
                                    for k in range(d)], axis=1)
            return g, gr

        res_b = en_bloques(bloque_gamma, len(I), bloque)
        gam, grad = res_b if res_b else (np.zeros(N), np.zeros_like(x))
        gam = gam + Vi * ker.W(0.0, hi)
        # pared semianalitica en la banda cercana al contorno
        if len(banda):
            q_, sd, nrm = geom.punto_cercano(x[banda])
            sdf = geom.distancia_firmada(x[banda])
            s = np.maximum(-sdf, 0.0) / hi[banda]
            cerca = s < S
            b, s, nrm = banda[cerca], s[cerca], nrm[cerca]
            gam[b] += np.interp(s, qt, phit)
            # d phi/dx = dphi/dq * dq/dx = dphi/dq * (-n)/h
            grad[b] += -(np.interp(s, qt, dphit) / hi[b])[:, None] * nrm
        err = gam - 1.0
        dr = -A * (hi ** 2)[:, None] * grad
        paso = np.linalg.norm(dr, axis=1)
        dr *= np.minimum(1.0, 0.1 * hi / np.maximum(paso, 1e-30))[:, None]
        x += dr
        if len(banda):
            _pared(geom, x, banda, contacto * dxi)
        e = float(np.sqrt(np.mean(err ** 2)))
        if it % cada == 0:
            hist["iter"].append(it); hist["err_gamma"].append(e)
            texto = f"it {it:5d}  error Gamma (rms) = {100 * e:.3f} %"
            if it % (cada * 4) == 0:
                _log(verbose, "  " + texto + f"  ({time.time() - t0:.0f} s)")
            if progreso:
                progreso(it, max_iter, texto)
        if e < tol:
            _log(verbose, f"  Convergido en {it} iteraciones (error Gamma rms < {100 * tol:.2g} %)")
            break
    info = dict(iteraciones=it, tiempo_s=time.time() - t0, err_gamma=e)
    _log(verbose, f"[densidad] terminado: {it} iteraciones, {info['tiempo_s']:.1f} s, "
                  f"error Gamma rms = {100 * e:.3f} %")
    return x, hist, info


# ==========================================================================
# 4. Masas y resultado
# ==========================================================================
def masas(x, campo, volumen, rho0, espesor=1.0):
    """V_i proporcional a dx(x_i)^d con sum V_i = volumen (exacto).
    En 2D la masa es por 'espesor' (por defecto, por mm de espesor)."""
    d = x.shape[1]
    w = campo.dx(x) ** d
    V = w * (volumen / w.sum())
    return rho0 * V * (espesor if d == 2 else 1.0), V


def calibrar_volumenes(geom, x, h, V, kernel="w4", pasadas=2, bloque=2_000_000, verbose=True):
    """Ajusta los volumenes para que la suma SPH sea la unidad con el kernel del
    SOLVER:  Gamma_i = sum_j V_j W(r_ij, h_ij) (+ pared semianalitica) -> 1.
      c_i  = Gamma_i suavizado (media SPH sobre las vecinas, 'pasadas' veces)
      V_i <- V_i / c_i  y se reescala para que sum V_i = V del cuerpo.
    Corrige los desajustes de gran escala (p.ej. la zona fina con un pelo menos
    de particulas) sin meter ruido en las masas. La masa total sigue exacta."""
    t0 = time.time()
    N, d = x.shape
    ker = crear_kernel(kernel, d)
    S = ker.radio(1.0)
    qt, phit, _ = _tabla_pared(ker, d)
    I, J = pares_radio_variable(x, S * h)
    sd = geom.distancia_firmada(x)
    s = np.maximum(-sd, 0.0) / h
    pared = np.where(s < S, np.interp(np.minimum(s, S), qt, phit), 0.0)
    W0 = ker.W(0.0, h)

    def sumas(vals, pesos):
        """sum_j pesos_j vals_j W_ij (incluye j = i)."""
        def blo(a, b):
            i, j = I[a:b], J[a:b]
            r = np.linalg.norm(x[i] - x[j], axis=1)
            W = ker.W(r, 0.5 * (h[i] + h[j]))
            return (np.bincount(i, pesos[j] * vals[j] * W, N) + np.bincount(j, pesos[i] * vals[i] * W, N),)
        rb = en_bloques(blo, len(I), bloque)
        return (rb[0] if rb else 0.0) + pesos * vals * W0

    V = np.array(V, float)
    Vtot = V.sum()
    gam = sumas(np.ones(N), V) + pared
    c = gam.copy()
    norm = sumas(np.ones(N), V) + 1e-300
    for _ in range(pasadas):
        c = sumas(c, V) / norm
    V2 = V / c
    V2 *= Vtot / V2.sum()
    _log(verbose, f"[calibracion] volumenes ajustados al kernel {kernel}: correccion "
                  f"{100 * (1 / c).min() - 100:+.2f} % .. {100 * (1 / c).max() - 100:+.2f} % "
                  f"({time.time() - t0:.1f} s)")
    return V2


class DistribucionVariable:
    """Resultado del inflation: posiciones, dx objetivo, volumen y masa por particula."""

    def __init__(self, x, geom, campo, rho0=9.5e-7, h_factor_sim=1.2, espesor=1.0, info=None,
                 calibrar=True, kernel_sim="w4", V=None, verbose=False):
        self.x = np.asarray(x, float)
        self.geom, self.campo, self.rho0 = geom, campo, rho0
        self.h_factor_sim, self.espesor = h_factor_sim, espesor
        self.dim = self.x.shape[1]
        self.dx = campo.dx(self.x)
        self.h = h_factor_sim * self.dx
        _, V_obj = masas(self.x, campo, geom.volumen, rho0, espesor)
        if V is not None:
            self.V = np.asarray(V, float)
        elif calibrar:
            self.V = calibrar_volumenes(geom, self.x, self.h, V_obj, kernel_sim, verbose=verbose)
        else:
            self.V = V_obj
        self.calibrado = calibrar or V is not None
        self.masa = rho0 * self.V * (espesor if self.dim == 2 else 1.0)
        self.info = dict(info or {})
        self.metodo = "inflation"

    @property
    def N(self):
        return len(self.x)

    @property
    def masa_total(self):
        return float(self.masa.sum())

    def resumen(self):
        mt = self.rho0 * self.geom.volumen * (self.espesor if self.dim == 2 else 1.0)
        return (f"[inflation] {self.geom.nombre}: N = {self.N}, dx = {self.dx.min():.4g}"
                f"..{self.dx.max():.4g}, masa = {self.masa.min():.4e}..{self.masa.max():.4e}, "
                f"masa total = {self.masa_total:.6e} (rho0*V = {mt:.6e})")

    def guardar(self, ruta):
        np.savez_compressed(ruta, x=self.x, dx=self.dx, masa=self.masa, V=self.V, h=self.h,
                            rho0=self.rho0, h_factor_sim=self.h_factor_sim, espesor=self.espesor,
                            campo=np.array([self.campo.dx_fino, self.campo.dx_grueso, self.campo.tasa]),
                            zonas=np.array([repr(z.a_dict()) for z in self.campo.zonas]))
        return ruta


def cargar_distribucion(ruta, geom):
    import ast
    z = np.load(ruta, allow_pickle=False)
    f, g, t = z["campo"]
    campo = CampoResolucion(f, g, t, [ast.literal_eval(s) for s in z["zonas"]])
    return DistribucionVariable(z["x"], geom, campo, float(z["rho0"]),
                                float(z["h_factor_sim"]), float(z["espesor"]), V=z["V"])


def generar_inflation(geom, campo, rho0=9.5e-7, h_factor_sim=1.2, espesor=1.0,
                      iter_wvt=100, iter_densidad=500, semilla=0, verbose=True,
                      progreso=None, kw_wvt=None, kw_densidad=None, calibrar=True):
    """Pasos 1-3 seguidos:
         distribucion inicial -> WVT (ordena, isotropia) -> correccion de densidad
    Devuelve (DistribucionVariable, historial)."""
    _log(verbose, campo.describir())
    t0 = time.time()
    x0 = distribucion_inicial(geom, campo, semilla=semilla, verbose=verbose)
    pw = (lambda it, n, t: progreso(it, n + iter_densidad, "WVT " + t)) if progreso else None
    x, hist, info = relajar_wvt(geom, x0, campo, max_iter=iter_wvt, semilla=semilla,
                                verbose=verbose, progreso=pw, **(kw_wvt or {}))
    if iter_densidad > 0:
        pd = (lambda it, n, t: progreso(iter_wvt + it, iter_wvt + n, "densidad " + t)) if progreso else None
        x, h2, i2 = corregir_densidad(geom, x, campo, max_iter=iter_densidad, verbose=verbose,
                                      progreso=pd, **(kw_densidad or {}))
        hist["densidad"] = h2
        info["densidad"] = i2
    info["tiempo_total_s"] = time.time() - t0
    dist = DistribucionVariable(x, geom, campo, rho0, h_factor_sim, espesor, info,
                                calibrar=calibrar, verbose=verbose)
    info["tiempo_total_s"] = time.time() - t0
    _log(verbose, dist.resumen() + f"  [{info['tiempo_total_s']:.1f} s]")
    return dist, hist


# ==========================================================================
# 5. Metricas con h variable
# ==========================================================================
def metricas_inflation(dist, kernel="w4", bloque=2_000_000, verbose=True):
    """Metricas con el kernel del solver y h_ij = h_factor_sim*(dx_i+dx_j)/2.
    Devuelve (resumen, campos por particula)."""
    t0 = time.time()
    x, N, d = dist.x, dist.N, dist.dim
    ker = crear_kernel(kernel, d)
    hs = dist.h
    R = ker.radio(hs)
    I, J = pares_radio_variable(x, R)
    m = dist.masa / (dist.espesor if d == 2 else 1.0)
    V = dist.V
    rho = m * ker.W(0.0, hs)
    gam = V * ker.W(0.0, hs)
    ngam = 1.0 * ker.W(0.0, hs)                       # densidad numerica sum W
    gg = np.zeros((N, d))
    Iner = np.zeros((N, d, d))
    salto = np.ones(N)
    for a in range(0, len(I), bloque):
        i, j = I[a:a + bloque], J[a:a + bloque]
        xij = x[i] - x[j]
        r = np.linalg.norm(xij, axis=1)
        hij = 0.5 * (hs[i] + hs[j])
        dentro = r < ker.radio(hij)
        i, j, xij, r, hij = i[dentro], j[dentro], xij[dentro], r[dentro], hij[dentro]
        W = ker.W(r, hij)
        rho += np.bincount(i, m[j] * W, N) + np.bincount(j, m[i] * W, N)
        gam += np.bincount(i, V[j] * W, N) + np.bincount(j, V[i] * W, N)
        ngam += np.bincount(i, W, N) + np.bincount(j, W, N)
        dW = ker.dWdr(r, hij)
        e = xij / np.maximum(r, 1e-12)[:, None]
        for k in range(d):
            gg[:, k] += np.bincount(i, V[j] * dW * e[:, k], N) - np.bincount(j, V[i] * dW * e[:, k], N)
            for l in range(k, d):
                v = xij[:, k] * xij[:, l] * W
                Iner[:, k, l] += np.bincount(i, V[j] * v, N) + np.bincount(j, V[i] * v, N)
        q = dist.dx[j] / dist.dx[i]
        np.maximum.at(salto, i, np.maximum(q, 1 / q))
        np.maximum.at(salto, j, np.maximum(q, 1 / q))
    for k in range(d):
        for l in range(k + 1, d):
            Iner[:, l, k] = Iner[:, k, l]
    ev = np.linalg.eigvalsh(Iner)
    lam = np.where(ev[:, -1] > 0, ev[:, 0] / np.maximum(ev[:, -1], 1e-300), 0.0)
    rho_rel = rho / dist.rho0
    ggh = np.linalg.norm(gg, axis=1) * hs
    dx_real = ngam ** (-1.0 / d)                      # espaciado real (densidad numerica)
    # el espaciado de una red con 1 particula por dx^d da sum W = 1/dx^d -> dx_real = dx

    arbol = cKDTree(x)
    dnn = arbol.query(x, k=2)[0][:, 1] / dist.dx
    sd = dist.geom.distancia_firmada(x)
    interior = sd < -ker.radio(hs)
    capa = sd > -1.0 * dist.dx
    dz = dist.campo.distancia_zonas(x)
    fino = dz <= 0
    err = 100 * (rho_rel - 1)

    def est(v, msk):
        v = v[msk & np.isfinite(v)]
        return (np.nan, np.nan, np.nan, np.nan) if len(v) == 0 else (v.mean(), v.std(), v.min(), v.max())

    e_m, e_s, e_lo, e_hi = est(err, interior)
    l_m, _, l_min, _ = est(lam, interior)
    r_m, r_s, _, _ = est(dx_real / dist.dx, interior)
    gs = ggh[capa]
    mt = dist.rho0 * dist.geom.volumen * (dist.espesor if d == 2 else 1.0)
    res = {
        "N": N,
        "N_zona_fina": int(fino.sum()),
        "dx_min": float(dist.dx.min()),
        "dx_max": float(dist.dx.max()),
        "masa_total": dist.masa_total,
        "error_masa_%": 100 * (dist.masa_total - mt) / mt,
        "masa_max/min": float(dist.masa.max() / dist.masa.min()),
        "err_rho_std_%": e_s,
        "err_rho_media_%": e_m,
        "err_rho_Linf_%": float(np.nanmax(np.abs(err[interior]))) if interior.any() else np.nan,
        "dx_real/dx_media": r_m,
        "dx_real/dx_CV_%": 100 * r_s / r_m if r_m else np.nan,
        "salto_vecinas_max": float(salto.max()),
        "salto_vecinas_P99": float(np.percentile(salto, 99)),
        "Lambda_media": l_m,
        "Lambda_min": l_min,
        "gradGamma_h_max": float(np.nanmax(ggh[interior])) if interior.any() else np.nan,
        "gradGamma_h_sup_CV_%": 100 * gs.std() / gs.mean() if len(gs) else np.nan,
        "dnn_min": float(dnn.min()),
        "tiempo_metricas_s": time.time() - t0,
    }
    campos = dict(dx=dist.dx, dx_real=dx_real, ratio=dx_real / dist.dx, rho_rel=rho_rel,
                  err_rho=err, Lambda=lam, gradGamma_h=ggh, dnn=dnn, salto=salto,
                  masa=dist.masa, interior=interior, capa_ext=capa, dist_zona=dz, sdf=sd)
    if verbose:
        imprimir_metricas_inflation(res)
    return res, campos


DESCRIPCION_INF = {
    "N": "numero de particulas",
    "N_zona_fina": "particulas dentro de las zonas de refinado",
    "dx_min": "dx minimo pedido",
    "dx_max": "dx maximo pedido",
    "masa_total": "masa total",
    "error_masa_%": "masa total frente a rho0*V (0 = exacta)",
    "masa_max/min": "relacion de masas mayor/menor",
    "err_rho_std_%": "dispersion de la densidad SPH interior (W4, h variable)",
    "err_rho_media_%": "sesgo medio de la densidad interior",
    "err_rho_Linf_%": "error maximo de densidad interior",
    "dx_real/dx_media": "espaciado real / pedido (1 = cumple el campo)",
    "dx_real/dx_CV_%": "dispersion del espaciado real / pedido",
    "salto_vecinas_max": "max dx_j/dx_i entre vecinas (suavidad de la transicion)",
    "salto_vecinas_P99": "percentil 99 del salto entre vecinas",
    "Lambda_media": "Lambda medio interior (1 = isotropo)",
    "Lambda_min": "Lambda minimo interior",
    "gradGamma_h_max": "max |grad Gamma|*h interior",
    "gradGamma_h_sup_CV_%": "irregularidad de |grad Gamma|*h en la capa exterior",
    "dnn_min": "distancia minima al vecino / dx local",
}


def imprimir_metricas_inflation(res):
    print("\n  Metricas (kernel W4, h variable)")
    for k, txt in DESCRIPCION_INF.items():
        v = res.get(k)
        if v is None:
            continue
        s = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.5g}"
        print(f"  {k:<22s} {s:>12s}   {txt}")


# ==========================================================================
# 6. Exportacion (masa variable por particula; rapida para millones)
# ==========================================================================
def _f16(v):
    """Real en 16 caracteres, siempre con punto decimal (formato fijo PAM-CRASH)."""
    t = f"{v:.9g}"
    if "." not in t and "e" not in t:
        t += "."
    return t.rjust(16)


def _f8_vector(v):
    """Numeros positivos en 8 caracteres (p.ej. 2.550E-5), vectorizado."""
    v = np.asarray(v, float)
    e = np.floor(np.log10(v)).astype(int)
    m = v / 10.0 ** e
    out = []
    for mi, ei in zip(m, e):
        se = str(ei)
        dec = 5 - len(se)
        s = f"{mi:.{dec}f}"
        if s.startswith("10"):
            ei += 1; se = str(ei); dec = 5 - len(se); s = f"{mi / 10:.{dec}f}"
        out.append(f"{s}E{se}".rjust(8))
    return out


def exportar_pamcrash_variable(dist, ruta, id_part=1, id_nodo_inicial=1, id_elem_inicial=1,
                               unidades="mm, kg, ms"):
    """Include de PAM-CRASH (NODE + SPHEL) con la MASA DE CADA PARTICULA en el
    campo 33-40 de SPHEL. En 2D se extruye en z = 0 (solo para pruebas)."""
    x = dist.x if dist.dim == 3 else np.column_stack([dist.x, np.zeros(dist.N)])
    m8 = _f8_vector(dist.masa)
    with open(ruta, "w") as f:
        f.write("$" + "=" * 79 + "\n")
        f.write(f"$ Particulas SPH con resolucion variable (inflation) - Python SPH\n")
        f.write(f"$ Geometria: {dist.geom.nombre}   N = {dist.N}   unidades: {unidades}\n")
        f.write(f"$ dx = {dist.dx.min():.5g} .. {dist.dx.max():.5g}   tasa = {100 * dist.campo.tasa:.3g} %/capa\n")
        f.write(f"$ rho0 = {dist.rho0:.6g}   masa total = {dist.masa_total:.8g}   "
                f"masa = {dist.masa.min():.5g} .. {dist.masa.max():.5g}\n")
        f.write(f"$ h sugerido por particula = {dist.h_factor_sim} * dx (columna h del CSV)\n")
        for z in dist.campo.zonas:
            f.write(f"$ zona: {z.describir()}\n")
        f.write("$" + "=" * 79 + "\n")
        f.write("$#        IDNOD               X               Y               Z\n")
        bloque = 200_000
        for a in range(0, len(x), bloque):
            f.write("".join(f"NODE  /{id_nodo_inicial + a + k:>8d}{_f16(p[0])}{_f16(p[1])}{_f16(p[2])}\n"
                            for k, p in enumerate(x[a:a + bloque])))
        f.write("$#         IDEL   IDPRT   IDNOD    MASS\n")
        for a in range(0, len(x), bloque):
            f.write("".join(f"SPHEL /{id_elem_inicial + a + k:>8d}{id_part:>8d}"
                            f"{id_nodo_inicial + a + k:>8d}{m8[a + k]}\n"
                            for k in range(min(bloque, len(x) - a))))
        f.write("$\n$ Completa la tarjeta PART tipo SPHEL con tu material (ver exportar.py).\n"
                "$ Comprueba en el manual de tu version si admite h por particula; si no,\n"
                "$ PAM-CRASH la calcula con RATIO/Hmin/Hmax de la PART.\n")
    return ruta


def exportar_csv_variable(dist, ruta):
    x = dist.x if dist.dim == 3 else np.column_stack([dist.x, np.zeros(dist.N)])
    datos = np.column_stack([x, dist.masa, dist.V, dist.dx, dist.h])
    np.savetxt(ruta, datos, delimiter=";", fmt="%.9g",
               header="x;y;z;masa;volumen;dx;h", comments="")
    return ruta


def exportar_vtk_variable(dist, ruta, campos=None):
    """VTK legacy (ParaView) con dx, masa, h y los campos de las metricas."""
    x = dist.x if dist.dim == 3 else np.column_stack([dist.x, np.zeros(dist.N)])
    n = len(x)
    esc = {"dx": dist.dx, "masa": dist.masa, "h": dist.h}
    for k in ("err_rho", "Lambda", "gradGamma_h", "ratio", "salto", "dist_zona"):
        if campos and k in campos:
            esc[k] = np.where(np.isfinite(campos[k]), campos[k], -1.0)
    with open(ruta, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\nPython SPH inflation\nASCII\nDATASET POLYDATA\nPOINTS {n} float\n")
        np.savetxt(f, x, fmt="%.7g")
        f.write(f"VERTICES {n} {2 * n}\n")
        np.savetxt(f, np.column_stack([np.ones(n, int), np.arange(n)]), fmt="%d")
        f.write(f"POINT_DATA {n}\n")
        for nombre, v in esc.items():
            f.write(f"SCALARS {nombre} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, np.asarray(v, float), fmt="%.6g")
    return ruta


# ==========================================================================
# 7. Flujo completo (lo usan los scripts 30 y 31)
# ==========================================================================
def ejecutar_inflation(geom, campo, rho0=9.5e-7, h_factor_sim=1.2, iter_wvt=100,
                       iter_densidad=500, calibrar=True, carpeta="resultados", prefijo=None,
                       exportar=True, figura=True, mostrar=True, seccion=("y", 0.0)):
    import os
    os.makedirs(carpeta, exist_ok=True)
    prefijo = prefijo or f"{geom.nombre}_inflation_{geom.dim}D"
    base = os.path.join(carpeta, prefijo)
    n_est = estimar_n(geom, campo)
    print(f"N estimado = {n_est}")
    dist, hist = generar_inflation(geom, campo, rho0, h_factor_sim, iter_wvt=iter_wvt,
                                   iter_densidad=iter_densidad, calibrar=calibrar)
    res, campos = metricas_inflation(dist)
    print(f"  npz -> {dist.guardar(base + '.npz')}")
    if exportar:
        print(f"  PAM-CRASH -> {exportar_pamcrash_variable(dist, base + '_pamcrash.inc')}")
        print(f"  CSV       -> {exportar_csv_variable(dist, base + '.csv')}")
        print(f"  VTK       -> {exportar_vtk_variable(dist, base + '.vtk', campos)}  (ParaView)")
    if figura:
        from .visor_inflation import figura_inflation
        import matplotlib.pyplot as plt
        figura_inflation(dist, res, campos, seccion, guardar=base + "_resumen.png", mostrar=False)
        print(f"  figura    -> {base}_resumen.png")
        if mostrar:
            plt.show()
        else:
            plt.close("all")
    return dist, res, campos
