"""
Redes regulares 3D recortadas a la geometria.

  sc   : cubica simple                     6 vecinos,  1 particula/celda
  bcc  : cubica centrada en el cuerpo       8 vecinos,  2 particulas/celda
  fcc  : cubica centrada en las caras      12 vecinos,  4 particulas/celda
         (la opcion "face centered cubic" de HyperMesh)
  hcp  : hexagonal compacta (capas ABAB)   12 vecinos (empaquetado mas denso,
         igual que FCC; cambia el orden de apilamiento)

mismo_volumen=True: el paso se ajusta para que cada particula ocupe dx^3
(mismo N y misma masa en todas las redes -> comparacion justa).
mismo_volumen=False: el lado de la celda es dx (el "pitch" de HyperMesh).
"""
import numpy as np


def _malla(geom, a, base, desfase, extra=2):
    (x0, y0, z0), (x1, y1, z1) = geom.bbox
    o = np.asarray(desfase) * a
    rng = [np.arange(int(np.floor((lo - oo) / a)) - extra, int(np.ceil((hi - oo) / a)) + extra + 1)
           for lo, hi, oo in ((x0, x1, o[0]), (y0, y1, o[1]), (z0, z1, o[2]))]
    I, J, K = np.meshgrid(*rng, indexing="ij")
    celdas = np.stack([I.ravel(), J.ravel(), K.ravel()], 1) * a + o
    pts = (celdas[:, None, :] + np.asarray(base)[None, :, :] * a).reshape(-1, 3)
    return pts


def _recortar(geom, pts, margen=0.0):
    sd = geom.distancia_firmada(pts)
    return pts[sd < -margen]


def red_sc(geom, dx, desfase=(0.5, 0.5, 0.5), margen=0.0, **_):
    return _recortar(geom, _malla(geom, dx, [[0, 0, 0]], desfase), margen)


def red_bcc(geom, dx, mismo_volumen=True, desfase=(0.5, 0.5, 0.5), margen=0.0):
    a = dx * 2 ** (1 / 3) if mismo_volumen else dx
    base = [[0, 0, 0], [0.5, 0.5, 0.5]]
    return _recortar(geom, _malla(geom, a, base, desfase), margen)


def red_fcc3d(geom, dx, mismo_volumen=True, desfase=(0.25, 0.25, 0.25), margen=0.0):
    a = dx * 4 ** (1 / 3) if mismo_volumen else dx
    base = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    return _recortar(geom, _malla(geom, a, base, desfase), margen)


def red_hcp(geom, dx, mismo_volumen=True, margen=0.0):
    """Capas triangulares en el plano YZ apiladas a lo largo de X (eje del pajaro)."""
    s = dx * 2 ** (1 / 6) if mismo_volumen else dx      # distancia entre vecinos
    dy, dz, dxl = s, s * np.sqrt(3) / 2, s * np.sqrt(2.0 / 3.0)
    (x0, y0, z0), (x1, y1, z1) = geom.bbox
    I = np.arange(int(np.floor(x0 / dxl)) - 2, int(np.ceil(x1 / dxl)) + 3)
    J = np.arange(int(np.floor(y0 / dy)) - 2, int(np.ceil(y1 / dy)) + 3)
    K = np.arange(int(np.floor(z0 / dz)) - 2, int(np.ceil(z1 / dz)) + 3)
    I, J, K = np.meshgrid(I, J, K, indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    capaB = (I % 2).astype(float)
    x = (I + 0.5) * dxl
    y = (J + 0.5 * (K % 2) + 0.5 * capaB) * dy
    z = (K + capaB / 3.0) * dz
    return _recortar(geom, np.stack([x, y, z], 1), margen)


def red_regular_3d(geom, dx, tipo="sc", **kw):
    tipo = tipo.lower()
    if tipo in ("sc", "cubica", "cubic"):
        return red_sc(geom, dx, **{k: v for k, v in kw.items() if k != "mismo_volumen"})
    if tipo == "bcc":
        return red_bcc(geom, dx, **kw)
    if tipo == "fcc":
        return red_fcc3d(geom, dx, **kw)
    if tipo in ("hcp", "hexagonal"):
        return red_hcp(geom, dx, **{k: v for k, v in kw.items() if k != "desfase"})
    raise ValueError(tipo)
