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


class GaussianaSplineCubico(Kernel):
    """Gaussiana aproximada por splines cubicos (Radioss / OpenRadioss, manual
    de teoria 'SPH Approximation of a Function'). SOLO 3D. Soporte 2h.

        r <= h       : W = 3/(2 pi h^3) * [ 2/3 - (r/h)^2 + 1/2 (r/h)^3 ]
        h <= r <= 2h : W = 1/(4 pi h^3) * (2 - r/h)^3
        r >= 2h      : W = 0

    Derivada (q = r/h):
        q <= 1       : dW/dr = 3/(2 pi h^4) * [ -2 q + 3/2 q^2 ]
        1 <= q <= 2  : dW/dr = -3/(4 pi h^4) * (2 - q)^2
    """
    soporte = 2.0
    nombre = "Gaussiana spline cubico (Radioss)"

    def __init__(self, dim=3):
        if dim != 3:
            raise ValueError("El kernel gaussiano por splines cubicos solo esta implementado en 3D")
        super().__init__(dim)

    def sigma(self, h):
        return 1.0 / (np.pi * h ** 3)          # se reparte en los dos tramos de _w

    def _w(self, q):
        q = np.asarray(q, dtype=float)
        t1 = 1.5 * (2.0 / 3.0 - q ** 2 + 0.5 * q ** 3)       # 3/(2pi h^3) / (1/(pi h^3)) = 3/2
        t2 = 0.25 * np.clip(2.0 - q, 0.0, None) ** 3          # 1/(4pi h^3) / (1/(pi h^3)) = 1/4
        return np.where(q <= 1.0, t1, t2)

    def _dwdq(self, q):
        q = np.asarray(q, dtype=float)
        d1 = 1.5 * (-2.0 * q + 1.5 * q ** 2)
        d2 = -0.75 * np.clip(2.0 - q, 0.0, None) ** 2
        return np.where(q <= 1.0, d1, d2)


class W4BSpline(Kernel):
    """Kernel W4 B-spline (Monaghan & Lattanzio 1985). Soporte 2h, v = r/h.

        0 <= v <= 1 : W = sigma * ( 1 - 3/2 v^2 + 3/4 v^3 )
        1 <= v <= 2 : W = sigma * 1/4 (2 - v)^3
        v >= 2      : W = 0
        sigma = 2/(3h) (1D), 10/(7 pi h^2) (2D), 1/(pi h^3) (3D)

    Derivada:
        0 <= v <= 1 : dW/dr = sigma/h * ( -3 v + 9/4 v^2 )
        1 <= v <= 2 : dW/dr = sigma/h * ( -3/4 (2 - v)^2 )
    """
    soporte = 2.0
    nombre = "W4 B-spline"

    def sigma(self, h):
        if self.dim == 1:
            return 2.0 / (3.0 * h)
        if self.dim == 2:
            return 10.0 / (7.0 * np.pi * h ** 2)
        return 1.0 / (np.pi * h ** 3)

    def _w(self, q):
        q = np.asarray(q, dtype=float)
        t1 = 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3
        t2 = 0.25 * np.clip(2.0 - q, 0.0, None) ** 3
        return np.where(q <= 1.0, t1, t2)

    def _dwdq(self, q):
        q = np.asarray(q, dtype=float)
        d1 = -3.0 * q + 2.25 * q ** 2
        d2 = -0.75 * np.clip(2.0 - q, 0.0, None) ** 2
        return np.where(q <= 1.0, d1, d2)


def crear_kernel(nombre="wendland", dim=2):
    """nombre: "wendland" | "quintic" | "gauss_spline" (Radioss, solo 3D) | "w4" """
    nombre = nombre.lower()
    if nombre.startswith("wend"):
        return WendlandC2(dim)
    if nombre.startswith("quin"):
        return QuinticSpline(dim)
    if nombre.startswith("gauss") or nombre.startswith("radioss"):
        return GaussianaSplineCubico(dim)
    if nombre.startswith("w4") or nombre.startswith("bspline") or nombre.startswith("b-spline"):
        return W4BSpline(dim)
    raise ValueError(f"Kernel desconocido: {nombre}")
