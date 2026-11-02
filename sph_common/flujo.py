"""
Flujo comun a todos los scripts: metricas + visor + guardado + exportacion.
Sirve para 2D y 3D (se detecta con dist.dim).
"""
import os
import json
import webbrowser
import numpy as np

from .metricas import calcular_metricas, imprimir_metricas
from .visor import panel_resumen, dibujar_historial
from .exportar import exportar_pamcrash, exportar_csv, exportar_vtk


def procesar(dist, hist=None, carpeta="resultados", mostrar=True,
             exportar_pam=True, exportar_texto=True, kernel_metricas="wendland",
             visor_html=True, abrir_html=False, visor_interactivo=False):
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, f"{dist.geom.nombre}_{dist.metodo}")
    print(dist.resumen())
    r, campos = calcular_metricas(dist, kernel=kernel_metricas)
    imprimir_metricas(r)

    dist.guardar(carpeta)
    with open(base + "_metricas.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (float, np.floating, int, np.integer))
                       else v) for k, v in r.items()}, f, indent=2)
    if exportar_pam:
        print("  PAM-CRASH ->", exportar_pamcrash(dist, base + "_pamcrash.inc"))
    if exportar_texto:
        print("  CSV       ->", exportar_csv(dist, base + ".csv"))
    if hist:
        dibujar_historial(hist, titulo=f"Convergencia {dist.metodo}",
                          guardar=base + "_convergencia.png", mostrar=False)

    if dist.dim == 2:
        panel_resumen(dist, r, campos, guardar=base + "_resumen.png", mostrar=mostrar)
        return r, campos

    # ---------------- 3D -----------------
    from .visor3d import panel_resumen_3d, visor_interactivo_3d
    print("  VTK       ->", exportar_vtk(dist, base + ".vtk", campos))
    if visor_html:
        from .visor_html import exportar_html
        ruta = exportar_html(dist, campos, base + "_visor3D.html", r)
        print("  Visor 3D  ->", ruta, "(abrir con doble clic en el navegador)")
        if abrir_html:
            webbrowser.open("file://" + os.path.abspath(ruta))
    panel_resumen_3d(dist, r, campos, guardar=base + "_resumen.png",
                     mostrar=mostrar and not visor_interactivo)
    if visor_interactivo and mostrar:
        import matplotlib.pyplot as plt
        visor_interactivo_3d(dist, campos)
        plt.show()
    return r, campos
