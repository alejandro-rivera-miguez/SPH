"""
Kernels SPH (funciones de suavizado) para 2D y 3D.

- Wendland C2 (soporte 2h): el "quintic Wendland" que usa Siemann (paper 4)
  y el que usan muchos solvers comerciales. Colagrossi (paper 5) tambien
  usa un Wendland.
- Quintic spline (soporte 3h): el que usan Negi & Ramachandran (paper 6).

Todas las funciones aceptan arrays de numpy (r >= 0).
"""
import numpy as np


class Kernel:
    """Clase base. Subclases definen sigma, W(q) y dW/dq (q = r/h)."""
    soporte = 2.0  # radio de soporte en unidades de h

    def __init__(self, dim=2):
        if dim not in (1, 2, 3):
            raise ValueError("dim debe ser 1, 2 o 3")
        self.dim = dim

    def sigma(self, h):
        raise NotImplementedError

    def _w(self, q):
        raise NotImplementedError

    def _dwdq(self, q):
        raise NotImplementedError

    def W(self, r, h):
        q = np.asarray(r) / h
        return self.sigma(h) * self._w(q)

    def dWdr(self, r, h):
        """Derivada radial dW/dr (<= 0). El gradiente es dWdr * (xi-xj)/r."""
        q = np.asarray(r) / h
        return self.sigma(h) * self._dwdq(q) / h

    def radio(self, h):
        return self.soporte * h


class WendlandC2(Kernel):
    """W = sigma (1 - q/2)^4 (1 + 2q),  0 <= q <= 2."""
    soporte = 2.0
    nombre = "Wendland C2"

    def sigma(self, h):
        if self.dim == 1:
            return 3.0 / (4.0 * h)
        if self.dim == 2:
            return 7.0 / (4.0 * np.pi * h ** 2)
        return 21.0 / (16.0 * np.pi * h ** 3)

    def _w(self, q):
        t = np.clip(1.0 - 0.5 * q, 0.0, None)
        return t ** 4 * (1.0 + 2.0 * q)

    def _dwdq(self, q):
        t = np.clip(1.0 - 0.5 * q, 0.0, None)
        return -5.0 * q * t ** 3


class QuinticSpline(Kernel):
    """Quintic spline de Morris (soporte 3h), ecuacion (14) del paper 6."""
    soporte = 3.0
    nombre = "Quintic spline"

    def sigma(self, h):
        if self.dim == 1:
            return 1.0 / (120.0 * h)
        if self.dim == 2:
            return 7.0 / (478.0 * np.pi * h ** 2)
        return 1.0 / (120.0 * np.pi * h ** 3)

    def _w(self, q):
        q = np.asarray(q, dtype=float)
        t3 = np.clip(3.0 - q, 0.0, None)
        t2 = np.clip(2.0 - q, 0.0, None)
        t1 = np.clip(1.0 - q, 0.0, None)
        return t3 ** 5 - 6.0 * t2 ** 5 + 15.0 * t1 ** 5

    def _dwdq(self, q):
        q = np.asarray(q, dtype=float)
        t3 = np.clip(3.0 - q, 0.0, None)
        t2 = np.clip(2.0 - q, 0.0, None)
        t1 = np.clip(1.0 - q, 0.0, None)
        return -5.0 * t3 ** 4 + 30.0 * t2 ** 4 - 75.0 * t1 ** 4


def crear_kernel(nombre="wendland", dim=2):
    nombre = nombre.lower()
    if nombre.startswith("wend"):
        return WendlandC2(dim)
    if nombre.startswith("quin"):
        return QuinticSpline(dim)
    raise ValueError(f"Kernel desconocido: {nombre}")
