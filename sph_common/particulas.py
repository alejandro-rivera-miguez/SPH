"""
Contenedor comun para una distribucion de particulas SPH.

Criterio de masa: TODAS las particulas tienen la MISMA masa
    m = rho0 * V_total / N
(en 2D, V_total = area * espesor). Asi el modelo conserva la masa exacta
del cuerpo y las particulas interaccionan entre si de forma simetrica.

Smoothing length: h = h_factor * dx_eq,  dx_eq = (V_total/N)^(1/dim)
(Siemann 2019 usa h = 1.9 * radio de particula ~ 1.18 dx en 3D.)
"""
import os
import numpy as np


class DistribucionSPH:
    def __init__(self, x, geometria, metodo, dx_nominal, rho0=9.5e-7,
                 espesor=None, h_factor=1.2, info=None):
        self.x = np.asarray(x, dtype=float)
        self.geom = geometria
        self.metodo = metodo
        self.dx_nominal = dx_nominal
        self.rho0 = rho0
        self.h_factor = h_factor
        self.info = dict(info or {})
        self.dim = self.x.shape[1]
        # en 2D cada particula representa un prisma de espesor 'espesor'
        self.espesor = espesor if espesor is not None else self.dx_eq if self.dim == 2 else 1.0

    # -------------------------------------------------------------------
    @property
    def N(self):
        return len(self.x)

    @property
    def volumen_total(self):
        return self.geom.volumen

    @property
    def dx_eq(self):
        """Espaciado equivalente: (V/N)^(1/dim)."""
        return (self.volumen_total / self.N) ** (1.0 / self.dim)

    @property
    def V0(self):
        """Volumen (area en 2D) de cada particula."""
        return self.volumen_total / self.N

    @property
    def masa(self):
        """Masa igual para todas las particulas (en 2D incluye el espesor)."""
        vol = self.volumen_total * (self.espesor if self.dim == 2 else 1.0)
        return self.rho0 * vol / self.N

    @property
    def masa_total(self):
        return self.masa * self.N

    @property
    def h(self):
        return self.h_factor * self.dx_eq

    # -------------------------------------------------------------------
    def resumen(self):
        return (f"[{self.metodo}] {self.geom.nombre}: N = {self.N}, "
                f"dx_nominal = {self.dx_nominal:.4g}, dx_eq = {self.dx_eq:.4g}, "
                f"h = {self.h:.4g}, masa/particula = {self.masa:.4e}, "
                f"masa total = {self.masa_total:.4e}")

    def guardar(self, carpeta="resultados"):
        os.makedirs(carpeta, exist_ok=True)
        ruta = os.path.join(carpeta, f"{self.geom.nombre}_{self.metodo}.npz")
        np.savez(ruta, x=self.x, metodo=self.metodo, geom=self.geom.nombre,
                 dx_nominal=self.dx_nominal, rho0=self.rho0, espesor=self.espesor,
                 h_factor=self.h_factor)
        return ruta

    @staticmethod
    def cargar(ruta, geometria):
        d = np.load(ruta, allow_pickle=False)
        return DistribucionSPH(d["x"], geometria, str(d["metodo"]),
                               float(d["dx_nominal"]), float(d["rho0"]),
                               float(d["espesor"]), float(d["h_factor"]))
