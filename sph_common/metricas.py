"""
Metricas de calidad de una distribucion de particulas (postprocesado).

Lo que se calcula, y de donde sale:

  Lambda (Siemann & Ritt 2019, paper 4; propuesto por Groenenboom):
      Tensor de inercia SPH de cada particula
          I_i^ab = sum_j (m_j/rho_j) (x_i^a - x_j^a)(x_i^b - x_j^b) W_ij
      Lambda_i = lambda_min / lambda_max  (autovalores de I_i).
      Lambda = 1 -> vecindario isotropo.  Siemann obtiene 0.83 <= Lambda <= 1
      con WVT frente a >90 % de particulas por debajo de 0.9 en el modelo de
      referencia mallado. Se evalua solo en particulas INTERIORES (con el
      soporte del kernel completo), igual que en el paper.

  Gamma y grad(Gamma) (Colagrossi 2012, paper 5):
      Gamma_i = sum_j V_j W_ij      (=1 si la distribucion es perfecta)
      |grad Gamma_i| * h            (=0 en equilibrio; es la "fuerza" que
                                     mueve las particulas al empezar el calculo)

  Error de densidad (Negi & Ramachandran, paper 6, ec. 20):
      rho_i = sum_j m_j W_ij ;  L_inf(rho - rho0)/rho0 en el interior.

  Espaciado (Siemann, fig. 6): distancia al vecino mas cercano normalizada
      con dx_eq, y media de los N vecinos mas cercanos.

  Volumen de Voronoi (Siemann, fig. 5): area de la celda de Voronoi de cada
      particula interior / V0; ratio Vmax/Vmin.

  Superficie: hueco entre el contorno real y las particulas. Ideal = dx/2.
      Un contorno "en escalera" (redes regulares) da huecos irregulares y
      es lo que golpea primero en un impacto.
"""
import numpy as np
from scipy.spatial import cKDTree, Voronoi, ConvexHull

from .kernels import crear_kernel
from .vecinos import pares, sumar_por_particula


def _area_poligono(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def calcular_metricas(dist, kernel="w4", n_vecinos_espaciado=6, voronoi=True):
    x = dist.x
    N, dim = x.shape
    geom = dist.geom
    h = dist.h
    dx = dist.dx_eq
    V0 = dist.V0
    ker = crear_kernel(kernel, dim)
    rc = ker.radio(h)

    # --- vecinos y sumas SPH (densidad "por unidad de espesor" con V = V0) --
    i, j, xij, rij = pares(x, rc)
    W0 = ker.W(0.0, h)
    Wij = ker.W(rij, h)
    gamma = V0 * (W0 + sumar_por_particula(i, Wij, N))          # Gamma_i
    rho_rel = gamma.copy()                                       # rho_i/rho0 (masas iguales)
    dW = ker.dWdr(rij, h)
    with np.errstate(invalid="ignore", divide="ignore"):
        eij = np.where(rij[:, None] > 0, xij / rij[:, None], 0.0)
    grad_gamma = V0 * sumar_por_particula(i, dW[:, None] * eij, N)
    grad_gamma_h = np.linalg.norm(grad_gamma, axis=1) * h
    n_vec = np.bincount(i, minlength=N)

    # --- Lambda: tensor de inercia SPH ------------------------------------
    Vj = V0 / np.maximum(rho_rel[j], 1e-12) * 1.0               # m_j/rho_j
    pw = Vj * Wij
    I = np.zeros((N, dim, dim))
    for a in range(dim):
        for b in range(a, dim):
            I[:, a, b] = np.bincount(i, weights=xij[:, a] * xij[:, b] * pw, minlength=N)
            I[:, b, a] = I[:, a, b]
    ev = np.linalg.eigvalsh(I)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = np.where(ev[:, -1] > 0, ev[:, 0] / ev[:, -1], 0.0)

    # --- particulas interiores: soporte completo dentro del cuerpo ---------
    dfirm = geom.distancia_firmada(x)
    interior = dfirm < -rc
    if interior.sum() < 5:                     # cuerpo muy pequeno vs h
        interior = dfirm < -0.5 * rc

    # --- espaciado -------------------------------------------------------
    arbol = cKDTree(x)
    k = min(n_vecinos_espaciado + 1, N)
    dk, _ = arbol.query(x, k=k)
    dnn = dk[:, 1] / dx
    dmedia = dk[:, 1:].mean(axis=1) / dx

    # --- Voronoi (solo celdas interiores) ---------------------------------
    vor_rel = np.full(N, np.nan)
    if voronoi and N > 4:
        vor = Voronoi(x)
        for p in np.where(interior)[0]:
            reg = vor.regions[vor.point_region[p]]
            if len(reg) == 0 or -1 in reg:
                continue
            if dim == 2:
                vor_rel[p] = _area_poligono(vor.vertices[reg]) / V0
            else:
                vor_rel[p] = ConvexHull(vor.vertices[reg]).volume / V0

    # --- superficie: hueco contorno-particulas ---------------------------
    if dim == 2:
        bpts, _ = geom.remuestrear(dx / 4.0)
    else:
        bpts = geom.puntos_superficie(dx / 3.0)
    dgap, _ = arbol.query(bpts)
    gap = dgap / dx                          # ideal = 0.5
    capa_ext = dfirm > -1.0 * dx               # particulas de la primera capa
    prof_capa = -dfirm[capa_ext] / dx          # ideal = 0.5

    # --- resumen ----------------------------------------------------------
    def est(a):
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return dict(min=np.nan, media=np.nan, max=np.nan, std=np.nan)
        return dict(min=a.min(), media=a.mean(), max=a.max(), std=a.std())

    li = lam[interior]
    vr = vor_rel[interior]
    vr = vr[np.isfinite(vr)]
    r = {
        "metodo": dist.metodo,
        "N": N,
        "N_interior": int(interior.sum()),
        "dx_eq": dx,
        "h": h,
        "masa": dist.masa,
        "error_masa_%": 100 * (dist.masa_total / (dist.rho0 * geom.volumen *
                                                  (dist.espesor if dim == 2 else 1)) - 1),
        "Lambda_min": li.min() if len(li) else np.nan,
        "Lambda_media": li.mean() if len(li) else np.nan,
        "Lambda_P5": np.percentile(li, 5) if len(li) else np.nan,
        "%_Lambda>0.9": 100 * np.mean(li > 0.9) if len(li) else np.nan,
        "gradGamma_h_max": grad_gamma_h[interior].max() if interior.any() else np.nan,
        "gradGamma_h_media": grad_gamma_h[interior].mean() if interior.any() else np.nan,
        "gradGamma_h_sup_media": grad_gamma_h[capa_ext].mean() if capa_ext.any() else np.nan,
        "gradGamma_h_sup_CV_%": (100 * grad_gamma_h[capa_ext].std() / grad_gamma_h[capa_ext].mean()
                                 if capa_ext.any() else np.nan),
        "err_rho_Linf_%": 100 * np.abs(rho_rel[interior] - 1).max() if interior.any() else np.nan,
        "err_rho_std_%": 100 * rho_rel[interior].std() if interior.any() else np.nan,
        "err_rho_media_%": 100 * (rho_rel[interior].mean() - 1) if interior.any() else np.nan,
        "dnn_min": dnn.min(),
        "dnn_media": dnn.mean(),
        "dnn_CV_%": 100 * dnn.std() / dnn.mean(),
        "vecinos_media": n_vec.mean(),
        "vecinos_min": n_vec[interior].min() if interior.any() else n_vec.min(),
        "Vvor_max/min": vr.max() / vr.min() if len(vr) else np.nan,
        "Vvor_CV_%": 100 * vr.std() / vr.mean() if len(vr) else np.nan,
        "hueco_sup_media": np.nanmean(gap),
        "hueco_sup_max": np.nanmax(gap),
        "capa_ext_CV_%": 100 * prof_capa.std() / max(prof_capa.mean(), 1e-12),
    }
    campos = {
        "Lambda": lam, "rho_rel": rho_rel, "gradGamma_h": grad_gamma_h,
        "dnn": dnn, "dmedia": dmedia, "vecinos": n_vec, "Vvor": vor_rel,
        "interior": interior, "dist_borde": dfirm, "hueco": gap, "capa_ext": capa_ext,
    }
    return r, campos


# Descripcion corta de cada metrica (para tablas e informes)
DESCRIPCION = {
    "N": "numero de particulas",
    "Lambda_min": "Lambda minimo interior (1 = isotropo)  [paper 4]",
    "Lambda_media": "Lambda medio interior  [paper 4]",
    "Lambda_P5": "percentil 5 de Lambda (cola mala)",
    "%_Lambda>0.9": "% de particulas interiores con Lambda > 0.9",
    "gradGamma_h_max": "max |grad Gamma|*h interior (0 = equilibrio)  [paper 5]",
    "gradGamma_h_media": "media |grad Gamma|*h interior  [paper 5]",
    "gradGamma_h_sup_media": "media de |grad Gamma|*h en la capa exterior",
    "gradGamma_h_sup_CV_%": "irregularidad de |grad Gamma|*h en la capa exterior (0 = uniforme)",
    "err_rho_Linf_%": "error max de densidad SPH interior  [paper 6]",
    "err_rho_std_%": "desviacion tipica de la densidad SPH interior",
    "err_rho_media_%": "sesgo medio de densidad interior (pre-compresion inicial)",
    "dnn_min": "distancia minima entre particulas / dx",
    "dnn_CV_%": "dispersion de la distancia al vecino mas cercano",
    "vecinos_media": "vecinos medios dentro del soporte 2h",
    "Vvor_max/min": "volumen Voronoi max/min (1 = iguales)  [paper 4]",
    "Vvor_CV_%": "dispersion del volumen de Voronoi",
    "hueco_sup_media": "hueco medio contorno-particula / dx (ideal 0.5)",
    "hueco_sup_max": "hueco max contorno-particula / dx (ideal ~0.5-0.6)",
    "capa_ext_CV_%": "irregularidad de la capa exterior (0 = sigue el contorno)",
}

METRICAS_TABLA = ["N", "Lambda_min", "Lambda_media", "Lambda_P5", "%_Lambda>0.9",
                  "gradGamma_h_max", "gradGamma_h_media",
                  "gradGamma_h_sup_media", "gradGamma_h_sup_CV_%", "err_rho_Linf_%",
                  "err_rho_std_%", "err_rho_media_%", "dnn_min", "dnn_CV_%", "vecinos_media",
                  "Vvor_max/min", "Vvor_CV_%", "hueco_sup_media", "hueco_sup_max",
                  "capa_ext_CV_%"]


def imprimir_metricas(r):
    print(f"\n=== Metricas: {r['metodo']} ===")
    for k in METRICAS_TABLA:
        v = r[k]
        txt = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"
        print(f"  {k:<20s} {txt:>10s}   {DESCRIPCION.get(k, '')}")
