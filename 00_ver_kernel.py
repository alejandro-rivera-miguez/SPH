"""
PASO 0 - Ver los kernels SPH (funcion y derivada)
=================================================
Dibuja W(r) y dW/dr de los kernels de sph_common/kernels.py para comprobar
que estan bien implementados (comparalos con tus graficas de referencia).

Ademas comprueba numericamente:
  * normalizacion: la integral de W en todo su soporte debe dar 1
  * derivada: dW/dr analitica frente a diferencias finitas
  * continuidad en los cambios de tramo

Kernels: "w4" (W4 B-spline, PAM-CRASH), "gauss_spline" (Radioss, solo 3D),
         "wendland" (Wendland C2), "quintic" (quintic spline)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from sph_common.kernels import crear_kernel

# ------------------------------- CONFIGURACION -----------------------------
KERNELS = ["w4", "wendland"]   # uno o varios: "w4", "gauss_spline", "wendland", "quintic"
DIM = 3                        # 1, 2 o 3 (gauss_spline solo admite 3)
H = 1.0                        # con h = 1 el eje x es directamente v = r/h
NORMALIZADO = False            # True: dibuja W/sigma y (dW/dr)*h/sigma (forma sin constantes)
V_MAX = 3.2                    # hasta que r/h se dibuja
# ---------------------------------------------------------------------------


def comprobar(k, h):
    """Integral de normalizacion, error de la derivada y salto en los tramos."""
    soporte = k.radio(h)
    jac = {1: lambda r: 2.0, 2: lambda r: 2 * np.pi * r, 3: lambda r: 4 * np.pi * r * r}[k.dim]
    integral = quad(lambda r: jac(r) * float(k.W(r, h)), 0, soporte,
                    points=[h, 2 * h], limit=200)[0]
    r = np.linspace(0.01 * h, soporte - 0.01 * h, 2000)
    e = 1e-6 * h
    fd = (k.W(r + e, h) - k.W(r - e, h)) / (2 * e)
    err_der = np.max(np.abs(k.dWdr(r, h) - fd)) / np.max(np.abs(k.dWdr(r, h)))
    saltos = [abs(float(k.W(c * h - 1e-9, h) - k.W(c * h + 1e-9, h)))
              for c in (1.0, 2.0) if c * h < soporte]
    return integral, err_der, max(saltos) if saltos else 0.0


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    v = np.linspace(0, V_MAX, 800)
    r = v * H
    print(f"\n=== Kernels en {DIM}D, h = {H} ===")
    for nombre in KERNELS:
        try:
            k = crear_kernel(nombre, DIM)
        except ValueError as e:
            print(f"  {nombre}: {e}")
            continue
        W, dW = k.W(r, H), k.dWdr(r, H)
        if NORMALIZADO:
            W, dW = W / k.sigma(H), dW * H / k.sigma(H)
        ax[0].plot(v, W, lw=2, label=k.nombre)
        ax[1].plot(v, dW, lw=2, label=k.nombre)
        integral, err_der, salto = comprobar(k, H)
        imin = np.argmin(k.dWdr(r, H))
        print(f"  {k.nombre:36s} soporte = {k.soporte:.0f}h  sigma = {k.sigma(H):.6f}  "
              f"W(0) = {float(k.W(0.0, H)):.6f}")
        print(f"  {'':36s} integral = {integral:.8f} (debe ser 1)   "
              f"error derivada = {err_der:.1e}   salto entre tramos = {salto:.1e}")
        print(f"  {'':36s} minimo de dW/dr en r/h = {v[imin]:.3f}  (dW/dr = {k.dWdr(r, H)[imin]:.5f})")
    ax[0].set_title("W" + (" / sigma" if NORMALIZADO else ""))
    ax[1].set_title("dW/dr" + (" * h / sigma" if NORMALIZADO else ""))
    for a in ax:
        a.set_xlabel("v = r/h")
        a.axhline(0, color="k", lw=0.5)
        for c in (1, 2, 3):
            a.axvline(c, color="grey", ls="--", lw=0.6)
        a.grid(alpha=0.3)
        a.legend(fontsize=9)
    fig.suptitle(f"Kernels SPH en {DIM}D (h = {H})")
    fig.tight_layout()
    plt.show()
