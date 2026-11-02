"""
Redes regulares 2D recortadas a la geometria (pasos 1, 2 y 3).

  red_cubica      : red cuadrada (simple cubic), distancia dx entre vecinos.
  red_fcc         : "face centered" 2D tipo HyperMesh: celda cuadrada de lado
                    'a' con una particula extra en el centro de cada celda.
  red_hexagonal   : red triangular (empaquetado hexagonal compacto en 2D),
                    cada particula tiene 6 vecinos a la misma distancia.
                    variante='panal' da la red "honeycomb" (3 vecinos), que
                    Colagrossi (paper 5, tabla 1) muestra que es siempre inestable.

Parametro mismo_volumen:
  True  -> se ajusta el paso de la red para que cada particula ocupe un area
           dx^2 (mismo numero de particulas y misma masa que la red cubica).
           Es lo que hay que hacer para comparar redes de forma justa.
  False -> el paso de la red es exactamente dx (como lo hace HyperMesh con el
           "pitch"), y el numero de particulas cambia.
"""
import numpy as np


def _recortar(geom, pts, margen=0.0):
    """Se quedan los puntos dentro de la geometria (y a 'margen' del borde)."""
    dentro = geom.contiene(pts)
    if margen > 0:
        dentro &= geom.distancia_firmada(pts) < -margen
    return pts[dentro]


def _malla(geom, ax, ay, ox, oy, extra=2):
    (x0, y0), (x1, y1) = geom.bbox
    i0, i1 = int(np.floor((x0 - ox) / ax)) - extra, int(np.ceil((x1 - ox) / ax)) + extra
    j0, j1 = int(np.floor((y0 - oy) / ay)) - extra, int(np.ceil((y1 - oy) / ay)) + extra
    I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    return I.ravel(), J.ravel()


def red_cubica(geom, dx, desfase=(0.5, 0.5), margen=0.0):
    """Red cuadrada de paso dx. 'desfase' en fracciones de dx respecto al origen."""
    ox, oy = desfase[0] * dx, desfase[1] * dx
    I, J = _malla(geom, dx, dx, ox, oy)
    pts = np.stack([ox + I * dx, oy + J * dx], 1)
    return _recortar(geom, pts, margen)


def red_fcc(geom, dx, mismo_volumen=True, desfase=(0.5, 0.5), margen=0.0):
    """Red cuadrada centrada (equivalente 2D del FCC de HyperMesh).
    Con mismo_volumen=True el lado de celda es a = sqrt(2)*dx (2 particulas por
    celda -> area dx^2 por particula, vecino mas proximo = dx)."""
    a = np.sqrt(2.0) * dx if mismo_volumen else dx
    ox, oy = desfase[0] * a, desfase[1] * a
    I, J = _malla(geom, a, a, ox, oy)
    esq = np.stack([ox + I * a, oy + J * a], 1)
    cen = esq + 0.5 * a
    return _recortar(geom, np.vstack([esq, cen]), margen)


def red_hexagonal(geom, dx, mismo_volumen=True, variante="triangular",
                  desfase=(0.5, 0.5), margen=0.0):
    """Red hexagonal compacta (triangular). Area por particula = (sqrt3/2) s^2.
    Con mismo_volumen=True: s = dx*sqrt(2/sqrt3) ~ 1.0746 dx -> area dx^2."""
    if variante == "panal":
        return red_panal(geom, dx, mismo_volumen, margen=margen)
    s = dx * np.sqrt(2.0 / np.sqrt(3.0)) if mismo_volumen else dx
    ay = s * np.sqrt(3.0) / 2.0
    ox, oy = desfase[0] * s, desfase[1] * ay
    I, J = _malla(geom, s, ay, ox, oy)
    x = ox + I * s + 0.5 * s * (J % 2)
    y = oy + J * ay
    return _recortar(geom, np.stack([x, y], 1), margen)


def red_panal(geom, dx, mismo_volumen=True, margen=0.0):
    """Red de panal de abeja (honeycomb). Area por particula = (3 sqrt3/4) s^2."""
    s = dx * np.sqrt(4.0 / (3.0 * np.sqrt(3.0))) if mismo_volumen else dx
    # red triangular de paso s*sqrt(3) con base de 2 particulas
    a = s * np.sqrt(3.0)
    ay = a * np.sqrt(3.0) / 2.0
    I, J = _malla(geom, a, ay, 0.0, 0.0, extra=3)
    bx = I * a + 0.5 * a * (J % 2)
    by = J * ay
    p1 = np.stack([bx, by], 1)
    p2 = np.stack([bx, by + s], 1)
    return _recortar(geom, np.vstack([p1, p2]), margen)


def red_regular(geom, dx, tipo="cubica", **kw):
    tipo = tipo.lower()
    if tipo in ("cubica", "cubic", "sc"):
        return red_cubica(geom, dx, **{k: v for k, v in kw.items() if k != "mismo_volumen"})
    if tipo in ("fcc",):
        return red_fcc(geom, dx, **kw)
    if tipo in ("hexagonal", "hex", "triangular"):
        return red_hexagonal(geom, dx, **kw)
    if tipo in ("panal", "honeycomb"):
        return red_panal(geom, dx, **kw)
    raise ValueError(tipo)
