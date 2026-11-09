"""
Catalogo de metodos de empaquetado: nombre visible -> identificador -> funcion.
Lo usan la interfaz grafica y los comparadores.
"""
from .redes2d import red_cubica, red_fcc, red_hexagonal
from .redes3d import red_sc, red_fcc3d, red_hcp
from .packing import empaquetar_colagrossi, empaquetar_negi, empaquetar_wvt

METODOS_2D = {
    "Cubica simple": "01_cubica",
    "FCC (como HyperMesh)": "02_fcc",
    "Hexagonal compacta": "03_hexagonal",
    "Colagrossi V1 (Wendland)": "04_colagrossi_V1",
    "Colagrossi V2 (Wendland + W4)": "04_colagrossi_V2",
    "Negi (hibrido)": "05_negi",
    "WVT (Siemann, DLR)": "06_wvt",
}
METODOS_3D = {
    "Cubica simple": "11_cubica",
    "FCC (como HyperMesh)": "12_fcc",
    "Hexagonal compacta (HCP)": "13_hcp",
    "Colagrossi V1 (Wendland)": "14_colagrossi_V1",
    "Colagrossi V2 (Wendland + W4)": "14_colagrossi_V2",
    "Negi (hibrido)": "15_negi",
    "WVT (Siemann, DLR)": "16_wvt",
}


def generar(metodo, geom, dx, h_factor_sim=1.2, verbose=True):
    """Posiciones de las particulas para el identificador 'metodo'."""
    d3 = geom.dim == 3
    tipo = metodo.split("_", 1)[1]
    if tipo == "cubica":
        return red_sc(geom, dx) if d3 else red_cubica(geom, dx)
    if tipo == "fcc":
        return red_fcc3d(geom, dx, mismo_volumen=True) if d3 else red_fcc(geom, dx, mismo_volumen=True)
    if tipo in ("hexagonal", "hcp"):
        return red_hcp(geom, dx, mismo_volumen=True) if d3 else red_hexagonal(geom, dx, mismo_volumen=True)
    it = 3000 if d3 else 4000
    if tipo in ("colagrossi_V1", "colagrossi_V2"):
        x = empaquetar_colagrossi(geom, dx, kernel="wendland", max_iter=it, verbose=verbose)[0]
        if tipo == "colagrossi_V2":
            x = empaquetar_colagrossi(geom, dx, h_factor=h_factor_sim, alpha=5e-2, kernel="w4",
                                      x_inicial=x, repulsion=True, fijar_capa=1.0,
                                      max_iter=1000, min_iter=200, verbose=verbose)[0]
        return x
    if tipo == "negi":
        return empaquetar_negi(geom, dx, max_iter=4000 if d3 else 8000, verbose=verbose)[0]
    if tipo == "wvt":
        return empaquetar_wvt(geom, dx, max_iter=6000 if d3 else 10000, verbose=verbose)[0]
    raise ValueError(metodo)
