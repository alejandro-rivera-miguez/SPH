"""
Flujo comun a todos los scripts: metricas + visor + guardado + exportacion.
Sirve para 2D y 3D (se detecta con dist.dim).
"""
import os
import json
import webbrowser
import numpy as np

from .metricas import calcular_metricas, imprimir_metricas
from .visor import panel_resumen, dibujar_historial, panel_superficie
from .exportar import exportar_pamcrash, exportar_csv, exportar_vtk


FIGURAS_POR_DEFECTO = ("resumen", "superficie", "convergencia")


def procesar(dist, hist=None, carpeta="resultados", mostrar=True,
             exportar_pam=True, exportar_texto=True, kernel_metricas="w4",
             visor_html=True, abrir_html=False, visor_interactivo=False,
             paneles=None, seccion=("y", 0.0), figuras=FIGURAS_POR_DEFECTO):
    """Metricas + figuras + guardado + exportacion de una distribucion.

    paneles : lista de paneles de la figura de resumen (ver sph_common/figuras.py),
              None = los de siempre.
    seccion : (eje, posicion) del corte por defecto de los mapas en 3D.
    figuras : que figuras se generan: "resumen", "superficie", "convergencia".
    """
    from .figuras import figura_paneles
    import matplotlib.pyplot as plt
    figuras = set(figuras or ())
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
    if dist.dim == 3:
        print("  VTK       ->", exportar_vtk(dist, base + ".vtk", campos))
        if visor_html:
            from .visor_html import exportar_html
            ruta = exportar_html(dist, campos, base + "_visor3D.html", r)
            print("  Visor 3D  ->", ruta, "(abrir con doble clic en el navegador)")
            if abrir_html:
                webbrowser.open("file://" + os.path.abspath(ruta))

    if hist and "convergencia" in figuras:
        dibujar_historial(hist, titulo=f"Convergencia {dist.metodo}",
                          guardar=base + "_convergencia.png", mostrar=False)
    if "superficie" in figuras:
        panel_superficie(dist, r, campos, guardar=base + "_superficie.png", mostrar=False)
    if "resumen" in figuras:
        figura_paneles(dist, r, campos, paneles=paneles, seccion=seccion,
                       guardar=base + "_resumen.png", mostrar=False)
    if dist.dim == 3 and visor_interactivo and mostrar:
        from .visor3d import visor_interactivo_3d
        visor_interactivo_3d(dist, campos)
    if mostrar:
        plt.show()
    else:
        plt.close("all")
    return r, campos
