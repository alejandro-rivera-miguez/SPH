"""
Geometrias 2D cerradas para rellenar con particulas.

Cada geometria se representa por su contorno (poligono denso, sentido
antihorario) y ofrece:
  - contiene(p)            -> bool, punto dentro
  - distancia_firmada(p)   -> <0 dentro, >0 fuera (distancia exacta al contorno)
  - punto_cercano(p)       -> punto del contorno mas cercano y normal exterior
  - remuestrear(ds)        -> puntos equiespaciados del contorno + normales
  - curva_offset(d, ds)    -> curva paralela a distancia d (d<0 hacia dentro)
  - esquinas_offset(d)     -> esquinas convexas desplazadas (para "hard points")

Geometrias disponibles en crear_geometria():
  circulo, cuadrado, rectangulo, triangulo, elipse, hexagono, estrella,
  capsula (pajaro 2D: rectangulo con extremos semicirculares, longitud 4R),
  poligono (lista de vertices que tu definas)
"""
import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Utilidades de poligonos
# --------------------------------------------------------------------------
def _area_poligono(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _densificar(v, ds):
    """Inserta puntos en cada lado para que ningun segmento supere ds."""
    out = []
    n = len(v)
    for k in range(n):
        a, b = v[k], v[(k + 1) % n]
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        t = np.arange(m)[:, None] / m
        out.append(a + t * (b - a))
    return np.vstack(out)


class Polilinea:
    """Curva cerrada (poligono) con consultas de distancia rapidas."""

    def __init__(self, vertices, densificar_ds=None):
        v = np.asarray(vertices, dtype=float)
        if np.allclose(v[0], v[-1]):
            v = v[:-1]
        if _area_poligono(v) < 0:          # forzar sentido antihorario
            v = v[::-1].copy()
        self.vertices_originales = v
        per = np.sum(np.linalg.norm(np.roll(v, -1, 0) - v, axis=1))
        if densificar_ds is None:
            densificar_ds = per / 4000.0
        self.v = _densificar(v, densificar_ds)
        self.perimetro = per
        self.area = _area_poligono(v)
        self._arbol = cKDTree(self.v, compact_nodes=False, balanced_tree=False)
        self._path = Path(np.vstack([self.v, self.v[:1]]), closed=True)

    # ---------------------------------------------------------------
    def contiene(self, p):
        p = np.atleast_2d(p)
        return self._path.contains_points(p)

    def punto_cercano(self, p):
        """Punto mas cercano del contorno, distancia (sin signo) y normal exterior."""
        p = np.atleast_2d(np.asarray(p, dtype=float))
        n = len(self.v)
        _, k = self._arbol.query(p, k=1)
        mejor_d = np.full(len(p), np.inf)
        mejor_q = np.zeros_like(p)
        mejor_t = np.zeros_like(p)
        for desp in (-1, 0):           # segmentos (k-1,k) y (k,k+1)
            a = self.v[(k + desp) % n]
            b = self.v[(k + desp + 1) % n]
            ab = b - a
            L2 = np.einsum("ij,ij->i", ab, ab)
            t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L2, 0.0, 1.0)
            q = a + t[:, None] * ab
            d = np.linalg.norm(p - q, axis=1)
            mejor = d < mejor_d
            mejor_d[mejor] = d[mejor]
            mejor_q[mejor] = q[mejor]
            mejor_t[mejor] = ab[mejor] / np.sqrt(L2[mejor])[:, None]
        # normal exterior de un poligono antihorario: tangente girada -90 grados
        normal = np.stack([mejor_t[:, 1], -mejor_t[:, 0]], axis=1)
        return mejor_q, mejor_d, normal

    def distancia_firmada(self, p):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        q, d, n = self.punto_cercano(p)
        proy = np.einsum("ij,ij->i", p - q, n)
        signo = np.sign(proy)
        # si el punto mas cercano es un vertice (proyeccion no alineada con la
        # normal) el signo es ambiguo -> test punto-en-poligono solo ahi
        amb = np.abs(proy) < 0.999 * d
        if np.any(amb):
            signo[amb] = np.where(self.contiene(p[amb]), -1.0, 1.0)
        return signo * d

    def remuestrear(self, ds):
        """Puntos equiespaciados (por longitud de arco) con normales exteriores."""
        v = self.v
        seg = np.linalg.norm(np.roll(v, -1, 0) - v, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1]
        n = max(3, int(round(L / ds)))
        sq = np.arange(n) * L / n
        vc = np.vstack([v, v[:1]])
        x = np.interp(sq, s, vc[:, 0])
        y = np.interp(sq, s, vc[:, 1])
        pts = np.stack([x, y], axis=1)
        # normales: media de las normales de los segmentos adyacentes (ec. 10 paper 6)
        t1 = np.roll(pts, -1, 0) - pts
        t0 = pts - np.roll(pts, 1, 0)
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t0 /= np.linalg.norm(t0, axis=1)[:, None]
        t = t0 + t1
        nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-30
        return pts, nrm


class Geometria2D:
    def __init__(self, nombre, vertices, area_exacta=None, esquinas=True,
                 angulo_esquina=20.0):
        self.nombre = nombre
        self.dim = 2
        self.contorno = Polilinea(vertices)
        self.area = area_exacta if area_exacta is not None else self.contorno.area
        self.volumen = self.area
        v = self.contorno.vertices_originales
        self.bbox = np.array([v.min(axis=0), v.max(axis=0)])
        self._esquinas = []
        if esquinas:
            self._detectar_esquinas(angulo_esquina)

    # -- consultas delegadas en el contorno --------------------------------
    def contiene(self, p):
        return self.contorno.contiene(p)

    def distancia_firmada(self, p):
        return self.contorno.distancia_firmada(p)

    def punto_cercano(self, p):
        return self.contorno.punto_cercano(p)

    def remuestrear(self, ds):
        return self.contorno.remuestrear(ds)

    @property
    def perimetro(self):
        return self.contorno.perimetro

    # -- esquinas ----------------------------------------------------------
    def _detectar_esquinas(self, ang_min):
        v = self.contorno.vertices_originales
        n = len(v)
        for k in range(n):
            a, b, c = v[k - 1], v[k], v[(k + 1) % n]
            u1 = (b - a) / np.linalg.norm(b - a)
            u2 = (c - b) / np.linalg.norm(c - b)
            giro = np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1)))
            convexa = (u1[0] * u2[1] - u1[1] * u2[0]) > 0
            if giro > ang_min and convexa:
                theta = np.pi - np.radians(giro)          # angulo interior
                bis = (-u1 + u2)
                bis /= np.linalg.norm(bis)                # bisectriz hacia dentro
                self._esquinas.append((b, bis, theta))

    def esquinas_offset(self, d):
        """Posicion de las esquinas convexas desplazadas una distancia d hacia
        dentro (d > 0). Para un angulo interior theta la esquina se mueve
        d / sin(theta/2) a lo largo de la bisectriz."""
        return np.array([b + bis * d / np.sin(th / 2) for b, bis, th in self._esquinas]
                        ).reshape(-1, 2)

    # -- curvas paralelas (offset) -----------------------------------------
    def curva_offset(self, d, ds):
        """Curva(s) de nivel de la distancia firmada = d (d<0 dentro).
        Se calcula sobre una malla fina con contourpy (incluido en matplotlib),
        asi funciona para cualquier forma (concava, con esquinas...)."""
        import contourpy
        paso = ds / 6.0
        marg = abs(d) + 2 * ds
        x = np.arange(self.bbox[0, 0] - marg, self.bbox[1, 0] + marg + paso, paso)
        y = np.arange(self.bbox[0, 1] - marg, self.bbox[1, 1] + marg + paso, paso)
        X, Y = np.meshgrid(x, y)
        Z = self.distancia_firmada(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
        lineas = contourpy.contour_generator(X, Y, Z).lines(d)
        curvas = [Polilinea(l) for l in lineas if len(l) > 5]
        return curvas

    def __repr__(self):
        return f"Geometria2D({self.nombre}, area={self.area:.4g})"


# --------------------------------------------------------------------------
# Fabrica de geometrias
# --------------------------------------------------------------------------
def _circulo(cx, cy, r, n=720):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def crear_geometria(nombre, **p):
    """Crea una geometria 2D centrada en el origen. Parametros (mm por defecto):

    circulo(R=50) | cuadrado(L=100) | rectangulo(L=150, H=80)
    triangulo(L=120)  (equilatero) | elipse(a=80, b=50) | hexagono(R=60)
    estrella(R=70, r=30, puntas=5) | capsula(R=53.5)  (longitud total 4R)
    poligono(vertices=[(x,y), ...])
    """
    nombre = nombre.lower()
    if nombre == "circulo":
        R = p.get("R", 50.0)
        return Geometria2D("circulo", _circulo(0, 0, R, 1440), np.pi * R ** 2, esquinas=False)
    if nombre == "cuadrado":
        L = p.get("L", 100.0) / 2
        return Geometria2D("cuadrado", [(-L, -L), (L, -L), (L, L), (-L, L)])
    if nombre == "rectangulo":
        L, H = p.get("L", 150.0) / 2, p.get("H", 80.0) / 2
        return Geometria2D("rectangulo", [(-L, -H), (L, -H), (L, H), (-L, H)])
    if nombre == "triangulo":
        L = p.get("L", 120.0)
        h = L * np.sqrt(3) / 2
        return Geometria2D("triangulo", [(-L / 2, -h / 3), (L / 2, -h / 3), (0, 2 * h / 3)])
    if nombre == "elipse":
        a, b = p.get("a", 80.0), p.get("b", 50.0)
        t = np.linspace(0, 2 * np.pi, 1440, endpoint=False)
        return Geometria2D("elipse", np.stack([a * np.cos(t), b * np.sin(t)], 1),
                           np.pi * a * b, esquinas=False)
    if nombre == "hexagono":
        R = p.get("R", 60.0)
        t = np.arange(6) * np.pi / 3
        return Geometria2D("hexagono", np.stack([R * np.cos(t), R * np.sin(t)], 1))
    if nombre == "estrella":
        R, r, k = p.get("R", 70.0), p.get("r", 30.0), p.get("puntas", 5)
        t = np.arange(2 * k) * np.pi / k + np.pi / 2
        rad = np.where(np.arange(2 * k) % 2 == 0, R, r)
        return Geometria2D("estrella", np.stack([rad * np.cos(t), rad * np.sin(t)], 1))
    if nombre in ("capsula", "pajaro"):
        # Cilindro con extremos semiesfericos, longitud total = 4R (Siemann 2019)
        R = p.get("R", 53.5)
        n = 720
        t1 = np.linspace(-np.pi / 2, np.pi / 2, n)
        t2 = np.linspace(np.pi / 2, 3 * np.pi / 2, n)
        der = np.stack([R + R * np.cos(t1), R * np.sin(t1)], 1)
        izq = np.stack([-R + R * np.cos(t2), R * np.sin(t2)], 1)
        v = np.vstack([der, izq])
        return Geometria2D("capsula", v, 4 * R * R + np.pi * R ** 2, esquinas=False)
    if nombre == "poligono":
        return Geometria2D("poligono", p["vertices"])
    raise ValueError(f"Geometria desconocida: {nombre}")


GEOMETRIAS_2D = ["circulo", "cuadrado", "rectangulo", "triangulo", "elipse",
                 "hexagono", "estrella", "capsula", "poligono"]
