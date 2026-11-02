"""
Geometrias 3D definidas por su funcion de distancia firmada (SDF).

    sdf(p) < 0 dentro,  > 0 fuera,  |sdf| = distancia exacta a la superficie

Con la SDF se resuelve todo lo que necesitan los algoritmos:
  contiene(p), distancia_firmada(p), punto_cercano(p) (proyeccion a la
  superficie), superficie_offset(c) (superficie paralela a distancia c),
  muestrear_offset(c, ds) (puntos repartidos uniformemente sobre ella).

Geometrias disponibles en crear_geometria_3d():
  esfera(R)            cubo(L)              caja(Lx, Ly, Lz)
  cilindro(R, L)       capsula(R)  -> PAJARO: cilindro con extremos
                                      semiesfericos, longitud total 4R
                                      (Siemann & Ritt 2019). Eje X = direccion
                                      de vuelo / impacto.
  elipsoide(a, b, c)   (SDF aproximada, se corrige iterando)
El eje de los cuerpos de revolucion es el eje X.
La geometria desde .stl se anadira en la siguiente fase con la misma interfaz.
"""
import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Clase base
# --------------------------------------------------------------------------
class Geometria3D:
    dim = 3
    nombre = "geometria3d"
    sdf_exacta = True

    # --- a definir por cada geometria -----------------------------------
    def sdf(self, p):
        raise NotImplementedError

    def malla(self, n=48):
        """Triangulacion de la superficie (vertices, caras) para dibujar."""
        raise NotImplementedError

    # --- interfaz comun ----------------------------------------------------
    def distancia_firmada(self, p):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        out = np.empty(len(p))
        for a in range(0, len(p), 400000):
            out[a:a + 400000] = self.sdf(p[a:a + 400000])
        return out

    def contiene(self, p):
        return self.distancia_firmada(p) < 0

    def gradiente(self, p, eps=None):
        """Normal exterior unitaria (gradiente de la SDF por diferencias centrales)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        eps = eps or 1e-4 * self.escala
        g = np.zeros_like(p)
        for k in range(3):
            e = np.zeros(3)
            e[k] = eps
            g[:, k] = (self.distancia_firmada(p + e) - self.distancia_firmada(p - e)) / (2 * eps)
        n = np.linalg.norm(g, axis=1)
        return g / np.maximum(n, 1e-30)[:, None]

    def punto_cercano(self, p, c=0.0, iteraciones=None):
        """Proyeccion sobre la superficie de nivel sdf = c.
        Devuelve (q, distancia, normal exterior)."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        it = iteraciones or (1 if self.sdf_exacta else 4)
        q = p.copy()
        for _ in range(it):
            d = self.distancia_firmada(q) - c
            n = self.gradiente(q)
            q = q - n * d[:, None]
        n = self.gradiente(q)
        return q, np.linalg.norm(p - q, axis=1), n

    @property
    def escala(self):
        return float(np.max(self.bbox[1] - self.bbox[0]))

    def superficie_offset(self, c, ds=None):
        return SuperficieNivel(self, c)

    def esquinas_offset(self, d):
        return np.zeros((0, 3))

    # --- muestreo de superficies ------------------------------------------
    def puntos_banda(self, c, paso, ancho=None):
        """Puntos de una malla regular (paso) cercanos a la superficie sdf=c,
        proyectados sobre ella. Tambien devuelve una estimacion del area."""
        ancho = ancho or paso
        marg = abs(c) + 2 * paso
        x = np.arange(self.bbox[0, 0] - marg, self.bbox[1, 0] + marg + paso, paso)
        y = np.arange(self.bbox[0, 1] - marg, self.bbox[1, 1] + marg + paso, paso)
        z = np.arange(self.bbox[0, 2] - marg, self.bbox[1, 2] + marg + paso, paso)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        yz = np.stack([Y.ravel(), Z.ravel()], 1)
        sel, cuenta = [], 0
        for xi in x:                                    # por rodajas: poca memoria
            p = np.column_stack([np.full(len(yz), xi), yz])
            d = self.distancia_firmada(p) - c
            cuenta += np.count_nonzero(np.abs(d) < ancho)
            sel.append(p[np.abs(d) < 0.5 * paso])
        area = cuenta * paso ** 3 / (2 * ancho)
        pts = np.vstack(sel)
        q, _, _ = self.punto_cercano(pts, c)
        return q, area

    def muestrear_offset(self, c, ds, relajar=60, semilla=0):
        """~Area/ds^2 puntos repartidos uniformemente sobre la superficie sdf=c
        (para las capas de particulas fijas o de contorno)."""
        rng = np.random.default_rng(semilla)
        cand, area = self.puntos_banda(c, ds / 3.0)
        n = max(4, int(round(area / ds ** 2)))
        x = cand[rng.choice(len(cand), min(n, len(cand)), replace=False)]
        return relajar_en_superficie(self, x, c, ds, relajar), area

    def puntos_superficie(self, ds):
        """Muestreo denso de la superficie real (para metricas)."""
        q, _ = self.puntos_banda(0.0, ds)
        return q

    def __repr__(self):
        return f"{self.__class__.__name__}({self.nombre}, V={self.volumen:.4g})"


class SuperficieNivel:
    """Superficie sdf = c, con la misma interfaz que en 2D."""

    def __init__(self, geom, c):
        self.geom, self.c = geom, c

    def punto_cercano(self, p):
        return self.geom.punto_cercano(p, self.c)


def relajar_en_superficie(geom, x, c, ds, iteraciones=60, xi=0.08, chi=0.01):
    """Reparte uniformemente puntos sobre la superficie sdf=c con una WVT
    restringida a la superficie (desplazamientos tangenciales + reproyeccion)."""
    x = x.copy()
    h = ds
    for _ in range(iteraciones):
        arbol = cKDTree(x)
        p = arbol.query_pairs(2 * h, output_type="ndarray")
        if len(p) == 0:
            break
        i = np.concatenate([p[:, 0], p[:, 1]])
        j = np.concatenate([p[:, 1], p[:, 0]])
        xij = x[i] - x[j]
        r = np.linalg.norm(xij, axis=1)
        f = 1.0 / (r / h + chi) ** 2 - 1.0 / (2 + chi) ** 2
        e = xij / np.maximum(r, 1e-12)[:, None]
        disp = np.stack([np.bincount(i, weights=f * e[:, k], minlength=len(x))
                         for k in range(3)], 1) * xi * h
        n = geom.gradiente(x)
        disp -= np.einsum("ij,ij->i", disp, n)[:, None] * n
        lim = 0.3 * ds
        nd = np.linalg.norm(disp, axis=1)
        disp *= np.minimum(1.0, lim / np.maximum(nd, 1e-30))[:, None]
        x, _, _ = geom.punto_cercano(x + disp, c)
    return x


# --------------------------------------------------------------------------
# Mallas para dibujar
# --------------------------------------------------------------------------
def _revolucion(perfil, n=48):
    """Superficie de revolucion alrededor del eje X. perfil = [(x, r), ...]
    desde un polo (r=0) al otro."""
    perfil = np.asarray(perfil, dtype=float)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = []
    for xk, rk in perfil:
        V.append(np.column_stack([np.full(n, xk), rk * np.cos(t), rk * np.sin(t)]))
    V = np.vstack(V)
    F = []
    m = len(perfil)
    for a in range(m - 1):
        for b in range(n):
            i0, i1 = a * n + b, a * n + (b + 1) % n
            j0, j1 = (a + 1) * n + b, (a + 1) * n + (b + 1) % n
            F += [(i0, j0, j1), (i0, j1, i1)]
    return V, np.array(F)


# --------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------
class Esfera(Geometria3D):
    def __init__(self, R=50.0):
        self.nombre, self.R = "esfera", R
        self.volumen = 4 / 3 * np.pi * R ** 3
        self.area = 4 * np.pi * R ** 2
        self.bbox = np.array([[-R] * 3, [R] * 3], dtype=float)

    def sdf(self, p):
        return np.linalg.norm(p, axis=1) - self.R

    def gradiente(self, p, eps=None):
        n = np.linalg.norm(p, axis=1)
        return p / np.maximum(n, 1e-30)[:, None]

    def malla(self, n=48):
        t = np.linspace(np.pi, 0, n // 2 + 1)
        return _revolucion(np.column_stack([self.R * np.cos(t), self.R * np.sin(t)]), n)


class Capsula(Geometria3D):
    """Pajaro: cilindro de radio R y longitud 2R con dos semiesferas (total 4R)."""

    def __init__(self, R=53.5, L_cilindro=None):
        self.nombre, self.R = "capsula", R
        self.a = (2 * R if L_cilindro is None else L_cilindro) / 2   # semi-longitud del cilindro
        self.volumen = np.pi * R ** 2 * 2 * self.a + 4 / 3 * np.pi * R ** 3
        self.area = 2 * np.pi * R * 2 * self.a + 4 * np.pi * R ** 2
        self.bbox = np.array([[-self.a - R, -R, -R], [self.a + R, R, R]], dtype=float)

    def _eje(self, p):
        c = np.zeros_like(p)
        c[:, 0] = np.clip(p[:, 0], -self.a, self.a)
        return c

    def sdf(self, p):
        return np.linalg.norm(p - self._eje(p), axis=1) - self.R

    def gradiente(self, p, eps=None):
        v = p - self._eje(p)
        n = np.linalg.norm(v, axis=1)
        return v / np.maximum(n, 1e-30)[:, None]

    def malla(self, n=48):
        t1 = np.linspace(np.pi, np.pi / 2, n // 4 + 1)
        t2 = np.linspace(np.pi / 2, 0, n // 4 + 1)
        izq = np.column_stack([-self.a + self.R * np.cos(t1), self.R * np.sin(t1)])
        der = np.column_stack([self.a + self.R * np.cos(t2), self.R * np.sin(t2)])
        return _revolucion(np.vstack([izq, der]), n)


class Cilindro(Geometria3D):
    def __init__(self, R=50.0, L=150.0):
        self.nombre, self.R, self.h = "cilindro", R, L / 2
        self.volumen = np.pi * R ** 2 * L
        self.area = 2 * np.pi * R * L + 2 * np.pi * R ** 2
        self.bbox = np.array([[-self.h, -R, -R], [self.h, R, R]], dtype=float)

    def sdf(self, p):
        r = np.linalg.norm(p[:, 1:], axis=1)
        d = np.stack([r - self.R, np.abs(p[:, 0]) - self.h], 1)
        return (np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0))

    def malla(self, n=48):
        R, h = self.R, self.h
        perfil = [(-h, 0), (-h, R / 2), (-h, R), (-h / 2, R), (0, R), (h / 2, R), (h, R),
                  (h, R / 2), (h, 0)]
        return _revolucion(perfil, n)


class Caja(Geometria3D):
    def __init__(self, Lx=100.0, Ly=100.0, Lz=100.0, nombre="caja"):
        self.nombre = nombre
        self.b = np.array([Lx, Ly, Lz], dtype=float) / 2
        self.volumen = float(np.prod(2 * self.b))
        self.area = 8 * (self.b[0] * self.b[1] + self.b[1] * self.b[2] + self.b[0] * self.b[2])
        self.bbox = np.array([-self.b, self.b])

    def sdf(self, p):
        q = np.abs(p) - self.b
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)

    def malla(self, n=8):
        b = self.b
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * b
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return V, F


class Elipsoide(Geometria3D):
    sdf_exacta = False

    def __init__(self, a=80.0, b=50.0, c=50.0):
        self.nombre = "elipsoide"
        self.r = np.array([a, b, c], dtype=float)
        self.volumen = 4 / 3 * np.pi * a * b * c
        p = 1.6075
        self.area = 4 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
        self.bbox = np.array([-self.r, self.r])

    def sdf(self, p):
        # aproximacion de primer orden: f/|grad f| (buena cerca de la superficie)
        k0 = np.linalg.norm(p / self.r, axis=1)
        k1 = np.linalg.norm(p / self.r ** 2, axis=1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-30)

    def malla(self, n=48):
        t = np.linspace(np.pi, 0, n // 2 + 1)
        V, F = _revolucion(np.column_stack([np.cos(t), np.sin(t)]), n)
        return V * self.r, F


def crear_geometria_3d(nombre, **p):
    """esfera(R) | cubo(L) | caja(Lx, Ly, Lz) | cilindro(R, L) | capsula(R)
    | elipsoide(a, b, c) | stl(archivo, escala=1, centrar=True).
    Unidades: mm. Centradas en el origen, eje X."""
    nombre = nombre.lower()
    if nombre == "esfera":
        return Esfera(p.get("R", 50.0))
    if nombre == "cubo":
        L = p.get("L", 100.0)
        return Caja(L, L, L, nombre="cubo")
    if nombre == "caja":
        return Caja(p.get("Lx", 150.0), p.get("Ly", 100.0), p.get("Lz", 60.0))
    if nombre == "cilindro":
        return Cilindro(p.get("R", 50.0), p.get("L", 150.0))
    if nombre in ("capsula", "pajaro"):
        return Capsula(p.get("R", 53.5), p.get("L_cilindro"))
    if nombre == "elipsoide":
        return Elipsoide(p.get("a", 80.0), p.get("b", 50.0), p.get("c", 50.0))
    if nombre == "stl":
        from .geometria_stl import GeometriaSTL
        if "archivo" not in p:
            raise ValueError('Para GEOMETRIA = "stl" indica PARAMS_GEOM = {"archivo": "stl/tu_pieza.stl"}')
        return GeometriaSTL(p["archivo"], escala=p.get("escala", 1.0),
                            centrar=p.get("centrar", True), nombre=p.get("nombre"))
    raise ValueError(f"Geometria 3D desconocida: {nombre}")


GEOMETRIAS_3D = ["esfera", "cubo", "caja", "cilindro", "capsula", "elipsoide", "stl"]
