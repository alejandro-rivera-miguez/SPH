"""
Busqueda de vecinos con scipy.spatial.cKDTree (sustituye al NNPS de PySPH).
"""
import numpy as np
from scipy.spatial import cKDTree


def pares(x, radio):
    """Devuelve todos los pares (i, j) con |xi - xj| < radio, en AMBOS sentidos.

    Retorna
    -------
    i, j : arrays de indices
    xij  : (npares, dim) vector xi - xj
    rij  : (npares,) distancia
    """
    arbol = cKDTree(x)
    p = arbol.query_pairs(radio, output_type="ndarray")
    if len(p) == 0:
        vacio = np.zeros(0, dtype=int)
        return vacio, vacio, np.zeros((0, x.shape[1])), np.zeros(0)
    i = np.concatenate([p[:, 0], p[:, 1]])
    j = np.concatenate([p[:, 1], p[:, 0]])
    xij = x[i] - x[j]
    rij = np.sqrt(np.einsum("ij,ij->i", xij, xij))
    return i, j, xij, rij


class ListaVerlet:
    """Lista de vecinos con 'piel' (skin): solo se reconstruye el arbol cuando
    alguna particula se ha movido mas de skin/2. Acelera mucho los bucles
    iterativos de empaquetado."""

    def __init__(self, radio, skin, n_dest=None):
        self.n_dest = n_dest      # si se da, solo pares con i < n_dest
        self.radio = radio
        self.skin = skin
        self.x_ref = None
        self.reconstrucciones = 0

    def pares(self, x):
        if (self.x_ref is None or len(x) != len(self.x_ref) or
                np.max(np.sum((x - self.x_ref) ** 2, axis=1)) > (0.5 * self.skin) ** 2):
            arbol = cKDTree(x)
            p = arbol.query_pairs(self.radio + self.skin, output_type="ndarray")
            self.i = np.concatenate([p[:, 0], p[:, 1]]) if len(p) else np.zeros(0, int)
            self.j = np.concatenate([p[:, 1], p[:, 0]]) if len(p) else np.zeros(0, int)
            if self.n_dest is not None:
                m = self.i < self.n_dest
                self.i, self.j = self.i[m], self.j[m]
            self.x_ref = x.copy()
            self.reconstrucciones += 1
        xij = x[self.i] - x[self.j]
        rij = np.sqrt(np.einsum("ij,ij->i", xij, xij))
        m = rij < self.radio
        return self.i[m], self.j[m], xij[m], rij[m]


def sumar_por_particula(i, valores, n):
    """Suma valores (escalar o vector por par) agrupando por la particula i."""
    valores = np.asarray(valores)
    if valores.ndim == 1:
        return np.bincount(i, weights=valores, minlength=n)
    return np.stack([np.bincount(i, weights=valores[:, k], minlength=n)
                     for k in range(valores.shape[1])], axis=1)
