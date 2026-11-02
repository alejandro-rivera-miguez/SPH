# Python SPH – Distribución y empaquetado de partículas (2D y 3D)

Este código no depende de PySPH: solo usa `numpy`, `scipy` y `matplotlib`.

## Instalación (una vez)

```
pip install -r requirements.txt
```

Abre la carpeta `Python SPH` en VS Code, abre cualquier script y pulsa **F5**
(o `python 11_cubica_simple_3D.py` en el terminal, **desde esta carpeta**).

## Cómo cambiar la geometría (y cualquier otro parámetro)

Cada script empieza con un bloque **CONFIGURACIÓN**. Solo hay que tocar ahí:

```python
GEOMETRIA = "capsula"     # nombre de la geometría
PARAMS_GEOM = {}          # sus dimensiones (vacío = valores por defecto)
DX = 6.65                 # distancia entre partículas [mm]
```

Ejemplos (3D):

```python
GEOMETRIA = "esfera";    PARAMS_GEOM = {"R": 50}
GEOMETRIA = "cubo";      PARAMS_GEOM = {"L": 100}
GEOMETRIA = "caja";      PARAMS_GEOM = {"Lx": 150, "Ly": 100, "Lz": 60}
GEOMETRIA = "cilindro";  PARAMS_GEOM = {"R": 50, "L": 150}
GEOMETRIA = "capsula";   PARAMS_GEOM = {"R": 53.5}                    # pájaro (longitud 4R)
GEOMETRIA = "capsula";   PARAMS_GEOM = {"R": 50, "L_cilindro": 150}   # pájaro más largo
GEOMETRIA = "elipsoide"; PARAMS_GEOM = {"a": 80, "b": 50, "c": 50}
```

Ejemplos (2D):

```python
GEOMETRIA = "circulo";   PARAMS_GEOM = {"R": 50}
GEOMETRIA = "estrella";  PARAMS_GEOM = {"R": 70, "r": 30, "puntas": 5}
GEOMETRIA = "poligono";  PARAMS_GEOM = {"vertices": [(0, 0), (100, 0), (60, 80)]}
```

- Todas las geometrías están centradas en el origen y en mm. En 3D, el eje de
  los cuerpos de revolución (cilindro y cápsula/pájaro) es el **eje X**, que es
  la dirección de impacto.
- **Usa la misma `GEOMETRIA`, `PARAMS_GEOM` y `DX` en todos los scripts que
  vayas a comparar.** El comparador (07 / 17) carga los resultados guardados
  por nombre de geometría y método.
- Para ver qué geometrías hay: `00_ver_geometrias_2D.py` y `10_ver_geometrias_3D.py`.

## Usar tu propio .stl

1. Copia el fichero en la carpeta **`stl`** del proyecto (por ejemplo `stl\pajaro_real.stl`).
2. Ábrelo con **`19_comprobar_stl.py`**:
   - pon `ARCHIVO = "stl/pajaro_real.stl"` y pulsa F5;
   - te dice si la malla está cerrada, su volumen, el tamaño, la masa y cuántas
     partículas saldrán con cada DX;
   - dibuja la malla y secciones con el dentro/fuera, para comprobar que se lee bien.
3. En cualquier script 3D (11 a 18) cambia:
   ```python
   GEOMETRIA = "stl"
   PARAMS_GEOM = {"archivo": "stl/pajaro_real.stl", "escala": 1.0, "centrar": True}
   ```
   - `escala`: factor por el que se multiplican las coordenadas. Usa 1000 si el
     STL está en metros y trabajas en mm.
   - `centrar`: lleva el centro de masas al origen.

   El resto (DX, métodos, visores, exportación a PAM-CRASH) funciona igual.
   Los resultados se guardan con el nombre del fichero
   (`resultados\pajaro_real_16_wvt_pamcrash.inc`, etc.).

Requisitos y detalles:
- Sirven el STL ASCII y el binario.
- La malla tiene que ser **cerrada** (estanca, sin agujeros). Si no lo es,
  el script 19 avisa.
- La dirección de impacto es la que traiga el STL: oriéntalo en tu CAD o
  preprocesador antes de exportarlo. Los visores no dependen de la orientación.
- Ejemplos para probar: `stl\ejemplo_pajaro_capsula.stl` (el mismo pájaro de
  Siemann) y `stl\ejemplo_toro.stl` (una forma cóncava).
- Si convertís las coordenadas de las partículas exportadas al sistema original
  del STL, recuerda que `centrar=True` las desplaza (el desplazamiento lo
  muestra el script 19). Con `centrar=False` se quedan en las coordenadas
  originales.

## Scripts 3D

| Script | Qué hace | Tiempo aprox.* |
|---|---|---|
| `10_ver_geometrias_3D.py` | Dibuja las geometrías 3D | 10 s |
| `11_cubica_simple_3D.py` | Paso 1: red cúbica simple | 5 s |
| `12_fcc_3D.py` | Paso 2: FCC (como el panel SPH de HyperMesh) | 5 s |
| `13_hexagonal_hcp_3D.py` | Paso 3: hexagonal compacto HCP (o BCC) | 5 s |
| `14_packing_colagrossi_3D.py` | Paso 4: Colagrossi 2012 (paper 5) | 4–8 min |
| `15_packing_negi_3D.py` | Paso 5: Negi 2021 (paper 6) | 3–6 min |
| `16_wvt_siemann_3D.py` | Paso 6: EWVT de Siemann/DLR (paper 4) | 1–3 min |
| `17_comparar_metodos_3D.py` | Compara todos los métodos | lo que tarden los que no estén guardados |
| `18_visor_3D.py` | Abre un resultado guardado en los visores, sin recalcular | 5 s |
| `19_comprobar_stl.py` | Comprueba un `.stl` (cerrado, volumen, nº de partículas) y lo dibuja | 10–30 s |

\* Pájaro de Siemann (R = 53,5 mm, 1,814 kg) con dx = 6,65 mm (unas 5440
partículas, el modelo "grueso" del paper), en un portátil normal. El tiempo
crece aproximadamente con N: el modelo fino del paper (40 824 partículas)
corresponde a dx ≈ 3,4 mm y tarda unas 8 veces más.

### Visualizadores 3D

Al ejecutar un script 3D se obtienen tres vistas:

1. **Figura de resumen** (matplotlib). Contiene:
   - el medio pájaro en 3D;
   - tres **secciones** (longitudinal y = 0, horizontal z = 0 y transversal
     x = 0) con las partículas a tamaño real, coloreadas por la distancia entre
     vecinos, Λ y el error de densidad;
   - histogramas;
   - la curva del espaciado a lo largo del eje (como la fig. 6 de Siemann).
2. **Ventana interactiva** (`VISOR_INTERACTIVO = True`). Un deslizador mueve un
   plano de corte por X, Y o Z. A la izquierda se ve el cuerpo cortado en 3D
   (se puede girar con el ratón) y a la derecha la sección a tamaño real. Se
   puede elegir el campo de color y el espesor de la loncha.
3. **Visor HTML** `resultados/<geom>_<metodo>_visor3D.html`. Se abre con doble
   clic en Chrome, Edge o Firefox; no necesita instalar nada ni internet.
   Dibuja las partículas como esferas y es el más fluido para revisar el
   volumen:
   - **girar:** botón izquierdo del ratón;
   - **mover:** botón derecho (o Mayús + izquierdo);
   - **zoom:** rueda;
   - **cortes:** "semiespacio" deja ver el interior y "loncha" muestra una
     sección de espesor elegido.

Además se guarda un `.vtk` para abrir en **ParaView**, si lo tienes.

## Scripts 2D

| Script | Qué hace | Tiempo aprox. |
|---|---|---|
| `00_ver_geometrias_2D.py` | Geometrías, esquinas y contorno desplazado dx/2 | 1 s |
| `01_cubica_simple_2D.py` | Paso 1: red cúbica simple | 1 s |
| `02_fcc_2D.py` | Paso 2: FCC 2D (en 2D es la red cuadrada girada 45°) | 1 s |
| `03_hexagonal_2D.py` | Paso 3: hexagonal compacto (o "panal") | 1 s |
| `04_packing_colagrossi_2D.py` | Paso 4: Colagrossi 2012 | 20–60 s |
| `05_packing_negi_2D.py` | Paso 5: Negi 2021 | 1–2 min |
| `06_wvt_siemann_2D.py` | Paso 6: EWVT de Siemann | 10–20 s |
| `07_comparar_metodos_2D.py` | Comparación de todos | — |

## Qué se guarda en `resultados/`

- `<geom>_<metodo>_resumen.png`: figura de resumen.
- `<geom>_<metodo>_convergencia.png`: historial de convergencia (métodos iterativos).
- `<geom>_<metodo>_metricas.json`: todas las métricas.
- `<geom>_<metodo>.npz`: posiciones. El comparador y el visor 18 las reutilizan sin recalcular.
- `<geom>_<metodo>_pamcrash.inc`: tarjetas `NODE` + `SPHEL` (masa en el campo 33-40), más una plantilla `PART SPHEL` comentada.
- `<geom>_<metodo>.csv`: x, y, z, masa, h.
- Solo en 3D: `<geom>_<metodo>_visor3D.html` y `<geom>_<metodo>.vtk`.
- `comparacion_<geom>.*` (2D) y `comparacion3D_<geom>.*` (3D): salida del comparador.

## Criterios adoptados

- **Misma masa en todas las partículas**: m = ρ0·V_total/N. En 3D, con ρ0 = 1,1317e-6 kg/mm³, el pájaro pesa 1,814 kg, como en Siemann.
- **Comparación justa entre redes**: con `MISMO_VOLUMEN = True`, cada partícula ocupa dx³ (dx² en 2D), así que todas las redes tienen la misma N y la misma masa.
- **h para la simulación**: h = 1.2·dx_eq por defecto. Siemann usa h = 1.9 × radio de partícula ≈ 1.18·dx.

### Cambios respecto a los papers en 3D (por coste de cálculo)

- **Colagrossi**: en 3D se empaqueta con h/dx = 1.4, frente a 2.0 en 2D.
- **Negi**: kernel Wendland con h = 1.3·dx en lugar de quintic con h = 1.2·dx.
  Además, el contenedor sigue la forma del cuerpo en vez de ser una caja.
  `KERNEL = "quintic"`, `HDX = 1.2` y `CONTENEDOR = "caja"` dan exactamente la
  versión del paper, unas 5 veces más lenta.

## Métricas (`sph_common/metricas.py`)

| Métrica | Origen | Ideal |
|---|---|---|
| Λ = λmin/λmax del tensor de inercia SPH | Siemann 2019 (paper 4) | 1 |
| \|∇Γ\|·h | Colagrossi 2012 (paper 5) | 0 (no hay reacomodo inicial) |
| error de densidad SPH (L∞, desviación típica y sesgo medio) | Negi 2021 (paper 6) | 0 |
| distancia al vecino más cercano / dx, dispersión | Siemann, fig. 6 | 1 y 0 % |
| volumen de Voronoi máx/mín y dispersión | Siemann, fig. 5 | 1 y 0 % |
| hueco superficie–partícula / dx | superficie de impacto | 0.5 |
| irregularidad de la capa exterior | contorno "en escalera" | 0 |

Λ y el volumen de Voronoi **no distinguen entre redes perfectas**: una red
perfecta ya da Λ = 1. Lo que penaliza a las redes es la superficie en
escalera (columnas "hueco" y "capa exterior") y sus direcciones preferentes.

## Estructura

```
sph_common/
  geometria2d.py   geometrías 2D (polígonos), distancia firmada, offsets, esquinas
  geometria3d.py   geometrías 3D por función de distancia firmada (SDF)
  geometria_stl.py lector de .stl y distancia firmada exacta a la malla
  redes2d.py       cúbica, FCC, hexagonal, panal (2D)
  redes3d.py       SC, BCC, FCC, HCP (3D)
  kernels.py       Wendland C2 y quintic spline (2D/3D)
  vecinos.py       búsqueda de vecinos (cKDTree) y listas de Verlet
  packing.py       Colagrossi, Negi, WVT (2D y 3D)
  metricas.py      postprocesado (Λ, Γ, densidad, Voronoi, superficie)
  visor.py         figuras 2D
  visor3d.py       figuras 3D, secciones y ventana interactiva
  visor_html.py    visor WebGL en el navegador
  exportar.py      PAM-CRASH (.inc), CSV y VTK
  particulas.py    clase DistribucionSPH (masa, h, guardar/cargar)
  flujo.py         métricas + figuras + exportación de cada script
```

## Siguientes fases

1. "Inflation" (resolución variable). La WVT ya acepta `h_func(x)`.
