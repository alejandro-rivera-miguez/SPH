"""
Exportacion de distribuciones de particulas.

  exportar_pamcrash(dist, ruta)  -> include de PAM-CRASH / VPS (formato fijo)
  exportar_csv(dist, ruta)       -> x, y, z, masa, h

Formato PAM-CRASH (columnas fijas de 8 caracteres):
  NODE  / IDNOD(9-16)   X(17-32)   Y(33-48)   Z(49-64)
  SPHEL / IDEL(9-16)  IDPRT(17-24)  IDNOD(25-32)  MASA(33-40)

  * La masa de cada particula va en el campo 33-40 de SPHEL (todas iguales).
  * La tarjeta PART tipo SPHEL (material, RATIO/Hmin/Hmax del smoothing length,
    contacto...) se deja como plantilla COMENTADA al final del fichero, para
    que la completes en tu modelo con tu material (Tait/Murnaghan, etc.).
  * IMPORTANTE: comprueba los campos con el manual de tu version de
    PAM-CRASH / VPS antes de lanzar el calculo.
"""
import numpy as np


def _f8(v):
    """Numero en 8 caracteres max (formato fijo PAM-CRASH)."""
    if v == 0:
        return "      0."
    for dec in range(6, -1, -1):
        s = f"{v:.{dec}E}"                      # p.ej. 4.4433E-05
        mant, exp = s.split("E")
        e = int(exp)
        s = f"{mant}E{e}"                         # 4.4433E-5
        if len(s) <= 8:
            return s.rjust(8)
    raise ValueError(f"No cabe en 8 caracteres: {v}")


def _f16(v):
    s = f"{v:.8g}"
    if "." not in s and "e" not in s.lower():
        s += "."
    if len(s) > 16:
        s = f"{v:.9E}"
    return s.rjust(16)


def exportar_pamcrash(dist, ruta, id_part=1, id_nodo_inicial=1, id_elem_inicial=1,
                      eje_extrusion_z=0.0, unidades="mm, kg, ms"):
    x = dist.x
    if dist.dim == 2:
        x = np.column_stack([x, np.full(len(x), eje_extrusion_z)])
    m = dist.masa
    lineas = []
    lineas.append("$" + "=" * 79)
    lineas.append(f"$ Particulas SPH generadas con Python SPH - metodo: {dist.metodo}")
    lineas.append(f"$ Geometria: {dist.geom.nombre}   N = {dist.N}   unidades: {unidades}")
    lineas.append(f"$ dx_eq = {dist.dx_eq:.6g}   h sugerido = {dist.h:.6g} "
                  f"(h/dx = {dist.h_factor})")
    lineas.append(f"$ rho0 = {dist.rho0:.6g}   masa por particula = {m:.6g}   "
                  f"masa total = {dist.masa_total:.6g}")
    if dist.dim == 2:
        lineas.append(f"$ Modelo 2D: masa calculada con espesor = {dist.espesor:.6g}")
    lineas.append("$" + "=" * 79)
    lineas.append("$#        IDNOD               X               Y               Z")
    for k, p in enumerate(x):
        lineas.append(f"NODE  /{id_nodo_inicial + k:>8d}{_f16(p[0])}{_f16(p[1])}{_f16(p[2])}")
    lineas.append("$#         IDEL   IDPRT   IDNOD    MASS")
    f8m = _f8(m)
    for k in range(len(x)):
        lineas.append(f"SPHEL /{id_elem_inicial + k:>8d}{id_part:>8d}"
                      f"{id_nodo_inicial + k:>8d}{f8m}")
    lineas += [
        "$",
        "$ ---- PLANTILLA PART SPHEL (descomentar y completar con tu material) ----",
        "$#         IDPRT   ATYPE   IDMAT IDVAMAT IDTHMAT  IDPMAT",
        f"$PART  /{id_part:>8d}   SPHEL       1       0       0       0",
        "$NAME PAJARO_SPH",
        "$#  DTELIM    TSCALF",
        "$",
        "$#   TCONT    EPSINI  COULFRIC",
        "$",
        "$#   RATIO      Hmin      Hmax       ETAINORMNPAIR   ALPHAmg    BETAmg NMON",
        "$",
        "$END_PART",
    ]
    with open(ruta, "w") as f:
        f.write("\n".join(lineas) + "\n")
    return ruta


def exportar_csv(dist, ruta):
    x = dist.x
    if dist.dim == 2:
        x = np.column_stack([x, np.zeros(len(x))])
    datos = np.column_stack([x, np.full(len(x), dist.masa), np.full(len(x), dist.h)])
    np.savetxt(ruta, datos, delimiter=",", header="x,y,z,masa,h", comments="",
               fmt="%.8g")
    return ruta


def exportar_vtk(dist, ruta, campos=None):
    """Fichero .vtk (legacy ASCII, POLYDATA) para abrir en ParaView.
    Incluye masa, h y, si se pasan, los campos de las metricas."""
    x = dist.x if dist.dim == 3 else np.column_stack([dist.x, np.zeros(dist.N)])
    n = len(x)
    L = ["# vtk DataFile Version 3.0", f"Python SPH {dist.metodo}", "ASCII",
         "DATASET POLYDATA", f"POINTS {n} float"]
    L += [f"{p[0]:.7g} {p[1]:.7g} {p[2]:.7g}" for p in x]
    L.append(f"VERTICES {n} {2 * n}")
    L += [f"1 {k}" for k in range(n)]
    L.append(f"POINT_DATA {n}")
    escalares = {"masa": np.full(n, dist.masa), "h": np.full(n, dist.h)}
    if campos:
        for k in ("Lambda", "rho_rel", "gradGamma_h", "dnn", "vecinos", "dist_borde"):
            if k in campos:
                escalares[k] = np.asarray(campos[k], dtype=float)
        escalares["interior"] = np.asarray(campos["interior"], dtype=float)
    for nombre, v in escalares.items():
        v = np.where(np.isfinite(v), v, -1.0)
        L += [f"SCALARS {nombre} float 1", "LOOKUP_TABLE default"]
        L += [f"{a:.6g}" for a in v]
    with open(ruta, "w") as f:
        f.write("\n".join(L) + "\n")
    return ruta
