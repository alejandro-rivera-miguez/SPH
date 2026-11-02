"""
Geometria 3D a partir de un fichero .stl (ASCII o binario).

    geom = crear_geometria_3d("stl", archivo="stl/pajaro.stl",
                              escala=1.0, centrar=True)

Requisitos del .stl: superficie CERRADA (estanca), sin agujeros. Es lo
normal si lo exportas de CATIA, HyperMesh, ANSA, SolidWorks, Blender...
Las unidades del fichero se multiplican por 'escala' (p.ej. escala=1000 si
el .stl esta en metros y trabajas en mm).

Lo que se calcula (sin librerias externas, solo numpy/scipy):
  * distancia EXACTA de cualquier punto a la malla (punto mas cercano sobre
    el triangulo mas cercano)
  * signo dentro/fuera: con la normal del triangulo cuando el punto mas
    cercano cae dentro de una cara, y con trazado de rayos (paridad de
    cortes) cuando cae en una arista o vertice -> robusto tambien en zonas
    concavas y aristas vivas
  * volumen y area de la malla, caja envolvente
Con esto funcionan todos los scripts 3D (11-18) igual que con las
geometrias basicas.
"""
import os
import struct
import numpy as np
from scipy.spatial import cKDTree

from .geometria3d import Geometria3D


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------
def leer_stl(ruta):
    """Devuelve (vertices (n,3), caras (m,3)) con los vertices fusionados."""
    with open(ruta, "rb") as f:
        datos = f.read()
    tri = None
    if len(datos) >= 84:
        n = struct.unpack("<I", datos[80:84])[0]
        if 84 + 50 * n == len(datos):                  # binario
            reg = np.frombuffer(datos[84:84 + 50 * n],
                                dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                ("a", "<u2")]))
            tri = reg["v"].astype(np.float64)
    if tri is None:                                    # ASCII
        txt = datos.decode("latin-1").split()
        pts = []
        k = 0
        while k < len(txt):
            if txt[k] == "vertex":
                pts.append((float(txt[k + 1]), float(txt[k + 2]), float(txt[k + 3])))
                k += 4
            else:
                k += 1
        if len(pts) == 0 or len(pts) % 3:
            raise ValueError(f"No se pudo leer el STL: {ruta}")
        tri = np.array(pts, dtype=np.float64).reshape(-1, 3, 3)
    # fusionar vertices repetidos
    P = tri.reshape(-1, 3)
    tol = 1e-9 * max(1.0, float(np.ptp(P, axis=0).max()))
    clave = np.round(P / tol).astype(np.int64)
    _, idx, inv = np.unique(clave, axis=0, return_index=True, return_inverse=True)
    V = P[idx]
    F = inv.reshape(-1, 3)
    # quitar triangulos degenerados
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return V, F[ok]


def escribir_stl(ruta, V, F, nombre="python_sph"):
    """Escribe un STL binario (util para exportar geometrias de prueba)."""
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    reg = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    reg["n"], reg["v"] = n, T
    with open(ruta, "wb") as f:
        f.write(nombre.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(reg.tobytes())
    return ruta


# --------------------------------------------------------------------------
# Punto mas cercano sobre triangulos (Ericson, Real-Time Collision Detection)
# --------------------------------------------------------------------------
def _punto_cercano_triangulos(p, a, b, c):
    """Vectorizado: p, a, b, c (n,3). Devuelve q (n,3) y la region del punto
    mas cercano: 0 cara, 1/2/3 vertice A/B/C, 4 arista AB, 5 arista BC, 6 arista CA."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    q = np.empty_like(p)
    hecho = np.zeros(len(p), dtype=bool)
    region = np.zeros(len(p), dtype=np.int8)

    def asignar(m, val, reg):
        m = m & ~hecho
        q[m] = val[m]
        hecho[m] = True
        region[m] = reg

    asignar((d1 <= 0) & (d2 <= 0), a, 1)                                    # vertice A
    asignar((d3 >= 0) & (d4 <= d3), b, 2)                                   # vertice B
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        asignar((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab, 4)  # arista AB
        asignar((d6 >= 0) & (d5 <= d6), c, 3)                               # vertice C
        w = d2 / (d2 - d6)
        asignar((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac, 6)  # arista CA
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        asignar((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                b + w[:, None] * (c - b), 5)                                # arista BC
        den = 1.0 / (va + vb + vc)
        v, w = vb * den, vc * den
    m = ~hecho                                                              # interior
    q[m] = a[m] + v[m, None] * ab[m] + w[m, None] * ac[m]
    return q, region


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
class GeometriaSTL(Geometria3D):
    sdf_exacta = True

    def __init__(self, archivo, escala=1.0, centrar=True, nombre=None, k_candidatos=16):
        if not os.path.exists(archivo):
            raise FileNotFoundError(
                f"No encuentro el STL '{archivo}'. Ponlo en la carpeta 'stl' del proyecto "
                f"y usa archivo='stl/<nombre>.stl'")
        V, F = leer_stl(archivo)
        V = V * escala
        self.archivo = archivo
        self.nombre = nombre or os.path.splitext(os.path.basename(archivo))[0]
        T = V[F]
        vol = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0
        if vol < 0:                                  # normales hacia dentro -> girar
            F = F[:, ::-1].copy()
            vol = -vol
        if centrar:
            # centro de masas del solido
            T = V[F]
            vt = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])) / 6.0
            cm = (vt[:, None] * T.sum(axis=1) / 4.0).sum(axis=0) / vt.sum()
            V = V - cm
            self.desplazamiento = -cm
        else:
            self.desplazamiento = np.zeros(3)
        self.V, self.F = V, F
        self.T = V[F]
        self.volumen = float(vol)
        nrm = np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0])
        self.area = float(0.5 * np.linalg.norm(nrm, axis=1).sum())
        self.normales = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
        self.bbox = np.array([V.min(axis=0), V.max(axis=0)])
        self._comprobar_cerrada()
        self._pseudonormales()
        # muestras sobre los triangulos para encontrar candidatos rapido
        self.k = k_candidatos
        self._construir_arbol()

    # ------------------------------------------------------------------
    def _comprobar_cerrada(self):
        e = np.vstack([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e.sort(axis=1)
        _, cuenta = np.unique(e, axis=0, return_counts=True)
        self.aristas_abiertas = int(np.sum(cuenta == 1))
        self.aristas_no_manifold = int(np.sum(cuenta > 2))
        if self.aristas_abiertas or self.aristas_no_manifold:
            print(f"  AVISO STL '{self.nombre}': {self.aristas_abiertas} aristas abiertas y "
                  f"{self.aristas_no_manifold} aristas compartidas por >2 caras. "
                  f"La malla deberia ser cerrada; el dentro/fuera puede fallar cerca de esas zonas.")

    def _pseudonormales(self):
        """Normales de vertice (ponderadas por angulo) y de arista (suma de las dos
        caras): dan el signo dentro/fuera exacto en una malla cerrada
        (Baerentzen & Aanaes 2005)."""
        T, F, N = self.T, self.F, self.normales
        nv = np.zeros((len(self.V), 3))
        for k in range(3):
            e1 = T[:, (k + 1) % 3] - T[:, k]
            e2 = T[:, (k + 2) % 3] - T[:, k]
            cosang = np.einsum("ij,ij->i", e1, e2) / np.maximum(
                np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-30)
            ang = np.arccos(np.clip(cosang, -1, 1))
            np.add.at(nv, F[:, k], ang[:, None] * N)
        self._nv = nv / np.maximum(np.linalg.norm(nv, axis=1), 1e-30)[:, None]
        # aristas: AB (0-1), BC (1-2), CA (2-0) de cada triangulo
        e = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1).reshape(-1, 2)
        e = np.sort(e, axis=1)
        _, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.ravel()
        ne = np.zeros((inv.max() + 1, 3))
        np.add.at(ne, inv, np.repeat(N, 3, axis=0))
        ne /= np.maximum(np.linalg.norm(ne, axis=1), 1e-30)[:, None]
        self._ne = ne[inv].reshape(-1, 3, 3)            # (ntri, 3 aristas, 3)
        # tabla vertice -> triangulos (para refinar la busqueda)
        orden = np.argsort(F.ravel(), kind="stable")
        vert = F.ravel()[orden]
        tris = (orden // 3)
        cuenta = np.bincount(vert, minlength=len(self.V))
        vmax = int(cuenta.max())
        tabla = -np.ones((len(self.V), vmax), dtype=np.int64)
        inicio = np.concatenate([[0], np.cumsum(cuenta)[:-1]])
        pos = np.arange(len(vert)) - inicio[vert]
        tabla[vert, pos] = tris
        self._v2t = tabla

    def _construir_arbol(self):
        """Arbol de busqueda con el centro de cada triangulo (los triangulos
        grandes se subdividen para que ningun trozo supere ~1.5 x la arista
        mediana): asi los k centros mas cercanos contienen el triangulo mas
        cercano."""
        T = self.T
        lado = np.max(np.linalg.norm(T[:, [1, 2, 0]] - T, axis=2), axis=1)
        s = max(1.5 * float(np.median(lado)), self.escala / 2000.0)
        n_sub = np.clip(np.ceil(lado / s).astype(int), 1, 30)
        pts, tri = [], []
        for n in np.unique(n_sub):
            idx = np.where(n_sub == n)[0]
            i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            m = (i + j) <= n - 1
            u, v = (i[m] + 1 / 3) / n, (j[m] + 1 / 3) / n
            w = 1 - u - v
            P = (w[None, :, None] * T[idx, 0][:, None, :] + u[None, :, None] * T[idx, 1][:, None, :]
                 + v[None, :, None] * T[idx, 2][:, None, :])
            pts.append(P.reshape(-1, 3))
            tri.append(np.repeat(idx, len(u)))
        self._muestras = np.vstack(pts)
        self._tri_muestra = np.concatenate(tri)
        self._arbol = cKDTree(self._muestras)

    # ------------------------------------------------------------------
    def _consulta(self, p):
        """distancia firmada, punto mas cercano y normal exterior."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        n = len(p)
        k = min(self.k, len(self._muestras))
        _, idx = self._arbol.query(p, k=k)
        idx = idx.reshape(n, k)
        cand = self._tri_muestra[idx]                      # (n, k) triangulos candidatos
        mejor_d = np.full(n, np.inf)
        mejor_q = np.zeros((n, 3))
        mejor_t = np.zeros(n, dtype=int)
        mejor_reg = np.zeros(n, dtype=np.int8)
        for c in range(k):
            t = cand[:, c]
            q, reg = _punto_cercano_triangulos(p, self.T[t, 0], self.T[t, 1], self.T[t, 2])
            d = np.linalg.norm(p - q, axis=1)
            m = d < mejor_d - 1e-12
            mejor_d[m], mejor_q[m], mejor_t[m], mejor_reg[m] = d[m], q[m], t[m], reg[m]
        # refinado: si el punto mas cercano cae en una arista o vertice, probar
        # tambien todos los triangulos que comparten esos vertices
        ref = np.where(mejor_reg != 0)[0]
        if len(ref):
            Fb = self.F[mejor_t[ref]]
            reg = mejor_reg[ref]
            va_ = np.where(reg == 1, Fb[:, 0], np.where(reg == 2, Fb[:, 1], np.where(
                reg == 3, Fb[:, 2], np.where(reg == 4, Fb[:, 0], np.where(reg == 5, Fb[:, 1], Fb[:, 2])))))
            vb_ = np.where(reg == 4, Fb[:, 1], np.where(reg == 5, Fb[:, 2], np.where(
                reg == 6, Fb[:, 0], va_)))
            for vv in (va_, vb_):
                vecinos = self._v2t[vv]                       # (nref, vmax)
                for c in range(vecinos.shape[1]):
                    t = vecinos[:, c]
                    ok = t >= 0
                    if not ok.any():
                        break
                    rr, t = ref[ok], t[ok]
                    q, rg = _punto_cercano_triangulos(p[rr], self.T[t, 0], self.T[t, 1], self.T[t, 2])
                    d = np.linalg.norm(p[rr] - q, axis=1)
                    m = d < mejor_d[rr] - 1e-12
                    sel = rr[m]
                    mejor_d[sel], mejor_q[sel], mejor_t[sel], mejor_reg[sel] = d[m], q[m], t[m], rg[m]
        # signo con la pseudonormal de la cara / arista / vertice mas cercano
        pn = self.normales[mejor_t].copy()
        for r_ in (1, 2, 3):
            m = mejor_reg == r_
            pn[m] = self._nv[self.F[mejor_t[m], r_ - 1]]
        for r_, ia in ((4, 0), (5, 1), (6, 2)):
            m = mejor_reg == r_
            pn[m] = self._ne[mejor_t[m], ia]
        proy = np.einsum("ij,ij->i", p - mejor_q, pn)
        signo = np.where(proy >= 0, 1.0, -1.0)
        if self.aristas_abiertas or self.aristas_no_manifold:
            # malla no cerrada: confirmar con trazado de rayos fuera de las caras
            amb = (mejor_reg != 0) & (mejor_d > 1e-12 * self.escala)
            if np.any(amb):
                signo[amb] = np.where(self._dentro_por_rayos(p[amb]), -1.0, 1.0)
        dist = signo * mejor_d
        with np.errstate(invalid="ignore", divide="ignore"):
            nrm = (p - mejor_q) / mejor_d[:, None] * signo[:, None]
        sobre = mejor_d < 1e-10 * self.escala
        nrm[sobre] = self.normales[mejor_t[sobre]]
        return dist, mejor_q, nrm

    def _dentro_por_rayos(self, p, bloque=None):
        """Paridad del numero de cortes de un rayo con la malla (Moller-Trumbore)."""
        d = np.array([0.5773, 0.5774, 0.5770]) + np.array([1e-3, -2e-3, 3e-3])
        d /= np.linalg.norm(d)
        v0, e1, e2 = self.T[:, 0], self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0]
        h = np.cross(d, e2)
        a = np.einsum("ij,ij->i", e1, h)
        valido = np.abs(a) > 1e-14
        f = np.where(valido, 1.0 / np.where(valido, a, 1.0), 0.0)
        dentro = np.zeros(len(p), dtype=bool)
        bloque = bloque or max(1, int(1.5e6 // len(self.T)))
        for s in range(0, len(p), bloque):
            P = p[s:s + bloque]
            S = P[:, None, :] - v0[None, :, :]             # (b, m, 3)
            u = f[None, :] * np.einsum("bmj,mj->bm", S, h)
            Q = np.cross(S, e1[None, :, :])
            v = f[None, :] * np.einsum("bmj,j->bm", Q, d)
            t = f[None, :] * np.einsum("bmj,mj->bm", Q, e2)
            corte = valido[None, :] & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-12)
            dentro[s:s + bloque] = (corte.sum(axis=1) % 2) == 1
        return dentro

    # --- interfaz de Geometria3D ---------------------------------------
    def sdf(self, p):
        return self._consulta(p)[0]

    def gradiente(self, p, eps=None):
        return self._consulta(p)[2]

    def punto_cercano(self, p, c=0.0, iteraciones=None):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        d, q, n = self._consulta(p)
        if c == 0.0:
            return q, np.abs(d), n
        # superficie paralela: desplazar desde el punto mas cercano
        qc = q + n * c
        for _ in range(2):                                # refinar (zonas concavas)
            d2, q2, n2 = self._consulta(qc)
            qc = qc - n2 * (d2 - c)[:, None]
        return qc, np.linalg.norm(p - qc, axis=1), self._consulta(qc)[2]

    def malla(self, n=None, max_caras=6000):
        """Malla para dibujar (si es muy grande se dibuja una muestra de caras)."""
        F = self.F
        if len(F) > max_caras:
            rng = np.random.default_rng(0)
            F = F[rng.choice(len(F), max_caras, replace=False)]
        return self.V, F

    def __repr__(self):
        return (f"GeometriaSTL({self.nombre}: {len(self.F)} triangulos, V={self.volumen:.4g}, "
                f"A={self.area:.4g}, bbox={np.round(self.bbox, 2).tolist()})")
