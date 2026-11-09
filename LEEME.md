# Python SPH – Distribución y empaquetado de partículas (2D y 3D)

Este código no depende de PySPH: solo usa `numpy`, `scipy` y `matplotlib`.

## Instalación (una vez)

```
pip install -r requirements.txt
```

Abre la carpeta `Python SPH` en VS Code, abre cualquier script y pulsa **F5**
(o `python 11_cubica_simple_3D.py` en el terminal, **desde esta carpeta**).

## Novedades (8 nov 2026)

### Interfaz gráfica: `20_interfaz_grafica.py`
```
python 20_interfaz_grafica.py
```
(tkinter viene incluido con Python en Windows). Desde la ventana:
1. **Modelo**: 2D/3D, geometría (o *Elegir STL...*), parámetros (`R=53.5, L=150`), escala STL, dx, rho0 y h/dx del solver.
2. **Empaquetado**: método, *Generar* (o *Cargar guardado*), exportar y abrir el visor HTML.
3. **Visualización**: colorear por partículas, Λ, densidad, |∇Γ|·h, |∇Γ|·h superficie, dnn...; en 3D, plano de corte X/Y/Z con deslizadores de posición y espesor, y vista sección / 3D cortado / ambas.
4. **Comparar**: selecciona 2 o 3 métodos (Ctrl+clic) y una métrica; abre una ventana con los empaquetados arriba, la métrica abajo y el mejor en verde.

### `00_ver_kernel.py`
Dibuja W(r) y dW/dr de los kernels de `KERNELS` (W4, Wendland, quintic, gauss_spline) y comprueba que integran 1. W4: |dW/dr| máximo en r = 2h/3, y tiende a 0 en r → 0 (origen de la *pairing instability*).

### Elegir las figuras de cada script (01–06, 11–16)
En el bloque de configuración:
```
FIGURAS = ["resumen", "superficie", "convergencia"]   # quita las que no quieras
PANELES = ["particulas", "lambda", "densidad", "dnn", "hist_dnn", "hist_lambda", "gradgamma", "tabla"]
SECCION = ("y", 0.0)                                   # 3D: plano de corte por defecto
```
Paneles disponibles: `particulas, lambda, densidad, gradgamma, gradgamma_sup, dnn, hist_dnn, hist_lambda, hist_densidad, hist_gradgamma, perfil_superficie, perfil_dnn, vista3d, tabla`.
En 3D se puede fijar el corte por panel: `"lambda@z=10"`, `"densidad@x=-30"`.

### Comparadores 07 (2D) y 17 (3D)
```
FIG_EMPAQUETADOS = [... 6 métodos ...]            # figura _empaquetados.png
COMPARAR_METODOS = ["11_cubica", "14_colagrossi_V2", "16_wvt"]   # 2 o 3
COMPARAR_METRICAS = ["densidad", "gradgamma", "gradgamma_sup", "lambda"]   # una figura por métrica
```
Salidas en `resultados/comparacion2D_<geom>_*` y `comparacion3D_<geom>_*`:
`_empaquetados.png`, `_comparar_<metrica>.png` (arriba empaquetados, abajo métrica, mejor en verde y "+X % peor"),
`_tabla_diferencias.png`, y el CSV con el % de cada método frente al mejor y la columna "mejor".
En consola, `*` marca el mejor de cada fila. Cuando el mejor vale ≈0 (redes en densidad o ∇Γ interior) se da la diferencia absoluta en lugar del %.

## Inflation: resolución variable (scripts 30, 31 y 32)

Refina una o varias zonas (cabeza del pájaro, punto de impacto, superficie libre
de la piscina) con un crecimiento suave hasta el tamaño grueso. Scripts
independientes del resto:

| Script | Qué hace |
| --- | --- |
| `30_inflation_2D.py` | pruebas en 2D |
| `31_inflation_3D.py` | 3D, también con tu `.stl` |
| `32_interfaz_inflation.py` | interfaz: STL, zonas con clic, dx fino/grueso/tasa, N estimado, generar, ver, exportar |

**Campo de tamaño**

    dx(x) = min( dx_grueso , dx_zona + tasa * distancia_a_la_zona )

`tasa` = crecimiento por capa: 0.05 significa que cada capa es un 5 % mayor que
la anterior. Longitud de la transición = (dx_grueso - dx_fino) / tasa.
Recomendado 0.03-0.10. Fuera de la transición todo tiene dx_grueso.

**Zonas** (coordenadas de la geometría; con STL usa `centrar=False` para
trabajar en las coordenadas de tu modelo):

    dict(tipo="esfera", centro=(x, y, z), radio=R)
    dict(tipo="caja",   pmin=(...), pmax=(...))
    dict(tipo="losa",   punto=(...), normal=(0, 0, 1), espesor=e)   # piscina
    dict(tipo="punto",  centro=(...))
    (cada zona admite dx=... propio)

**Algoritmo** (`sph_common/inflation.py`):
1. Celdas adaptativas (octree) con el número de partículas que pide dx(x).
2. WVT con h variable (Siemann, paper 4), 100 iteraciones: ordena e isotropiza.
3. Corrección de densidad (Colagrossi con volúmenes variables y pared
   semianalítica, sin partículas fantasma), 500 iteraciones.
4. Calibración de volúmenes con el kernel del solver (W4, h = 1.2 dx):
   sum_j V_j W_ij = 1. **Masa total exacta**: m_i = rho0 * V_i.

**Exportación**: `.inc` de PAM-CRASH con la **masa de cada partícula** en SPHEL,
CSV (x, y, z, masa, volumen, dx, h), VTK para ParaView y `.npz`.
En 2D la masa es por mm de espesor (solo para pruebas).

**Rendimiento**: usa todos los núcleos (variable de entorno `SPH_HILOS` para
limitarlos). Portátil de 2 núcleos: 55 000 partículas en ~3 min y ~0,5 GB de
RAM; 250 000 partículas en ~20 min y 1,75 GB. Escala de forma lineal: memoria ≈ 0,7 GB por cada 100 000 partículas (3 millones ≈ 21 GB), y el tiempo baja con el número de núcleos.

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
| `14_packing_colagrossi_3D_V1.py` | Paso 4: Colagrossi 2012, solo Wendland | 4–8 min |
| `14_packing_colagrossi_3D_V2.py` | Paso 4: Colagrossi, Wendland + afinado con W4 | 5–10 min |
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
| `04_packing_colagrossi_2D_V1.py` | Paso 4: Colagrossi 2012, solo Wendland | 20–60 s |
| `04_packing_colagrossi_2D_V2.py` | Paso 4: Colagrossi, Wendland + afinado con W4 | 30–70 s |
| `05_packing_negi_2D.py` | Paso 5: Negi 2021 | 1–2 min |
| `06_wvt_siemann_2D.py` | Paso 6: EWVT de Siemann | 10–20 s |
| `07_comparar_metodos_2D.py` | Comparación de todos | — |

## Qué se guarda en `resultados/`

- `<geom>_<metodo>_resumen.png`: figura de resumen.
- `<geom>_<metodo>_superficie.png`: \|∇Γ\|·h en la capa exterior (mapa, perfil a lo largo de la superficie e histograma).
- `<geom>_<metodo>_convergencia.png`: historial de convergencia (métodos iterativos).
- `<geom>_<metodo>_metricas.json`: todas las métricas.
- `<geom>_<metodo>.npz`: posiciones. El comparador y el visor 18 las reutilizan sin recalcular.
- `<geom>_<metodo>_pamcrash.inc`: tarjetas `NODE` + `SPHEL` (masa en el campo 33-40), más una plantilla `PART SPHEL` comentada.
- `<geom>_<metodo>.csv`: x, y, z, masa, h.
- Solo en 3D: `<geom>_<metodo>_visor3D.html` y `<geom>_<metodo>.vtk`.
- `comparacion_<geom>.*` (2D) y `comparacion3D_<geom>.*` (3D): salida del comparador. `..._superficie.png` compara el CV de superficie de todos los métodos.

## Criterios adoptados

- **Misma masa en todas las partículas**: m = ρ0·V_total/N. En 3D, con ρ0 = 1,1317e-6 kg/mm³, el pájaro pesa 1,814 kg, como en Siemann.
- **Comparación justa entre redes**: con `MISMO_VOLUMEN = True`, cada partícula ocupa dx³ (dx² en 2D), así que todas las redes tienen la misma N y la misma masa.
- **h para la simulación**: h = H_FACTOR_SIM·dx_eq (1,2 por defecto). Siemann usa h = 1.9 × radio de partícula ≈ 1.18·dx. **Pon en H_FACTOR_SIM el h/dx que uses en PAM-CRASH**: las métricas y el afinado W4 se calculan con él.
- **Kernel de las métricas: W4 B-spline**, el de PAM-CRASH (`calcular_metricas(..., kernel="w4")`). Los kernels disponibles en `kernels.py` son `wendland`, `quintic`, `w4` y `gauss_spline` (Radioss, solo 3D; en 3D es la misma función que el W4).

### Colagrossi V1 y V2

- **V1**: empaquetado solo con Wendland. El Wendland no se empareja con ningún h.
- **V2**: V1 + una segunda fase con W4 a h = H_FACTOR_SIM·dx, para llevar la distribución al equilibrio del kernel de PAM-CRASH. Lleva:
  - **repulsión de Negi**: sin ella, el W4 sufre *pairing instability* (las partículas se colocan una encima de otra), porque su derivada se anula en r → 0;
  - **ALPHA_W4 = 0,05**: amortiguamiento más alto, para que converja sin oscilar;
  - **FIJAR_CAPA = 1**: congela la capa exterior para conservar la superficie regular de la fase 1.
- Resultado típico (2D, cápsula): \|∇Γ\|·h máx. interior 0,023 (V1) frente a 0,007 (V2), con el mismo o mejor CV de superficie.

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
| \|∇Γ\|·h máx. interior | Colagrossi 2012 (paper 5) | 0 (no hay reacomodo inicial) |
| **CV de \|∇Γ\|·h en la capa exterior** | uniformidad de la fuerza espuria en la superficie | 0 % |
| error de densidad SPH (L∞, desviación típica y sesgo medio) | Negi 2021 (paper 6) | 0 |
| distancia al vecino más cercano / dx, dispersión | Siemann, fig. 6 | 1 y 0 % |
| volumen de Voronoi máx/mín y dispersión | Siemann, fig. 5 | 1 y 0 % |
| hueco superficie–partícula / dx | superficie de impacto | 0.5 |
| irregularidad de la capa exterior | contorno "en escalera" | 0 |

Todas se calculan con el kernel W4 y H_FACTOR_SIM.

**Métricas de referencia para elegir método:**
- **\|∇Γ\|·h máx. interior**: lo bien equilibrado que está el interior. Sirve para comparar los métodos de empaquetado entre sí; las redes dan 0 aquí porque su interior está en equilibrio perfecto.
- **CV de \|∇Γ\|·h en la capa exterior**: en una superficie libre \|∇Γ\|·h nunca es 0, pero debe ser uniforme. La superficie en escalera de las redes da un 17–29 %; los empaquetados, un 2–6 %. Es la cifra que muestra que el empaquetado es mejor que las redes cristalinas. **Depende de h: compara siempre con el mismo H_FACTOR_SIM.**

Λ y el volumen de Voronoi **no distinguen entre redes perfectas**: una red
perfecta ya da Λ = 1.

## Estructura

```
sph_common/
  geometria2d.py   geometrías 2D (polígonos), distancia firmada, offsets, esquinas
  geometria3d.py   geometrías 3D por función de distancia firmada (SDF)
  geometria_stl.py lector de .stl y distancia firmada exacta a la malla
  redes2d.py       cúbica, FCC, hexagonal, panal (2D)
  redes3d.py       SC, BCC, FCC, HCP (3D)
  kernels.py       Wendland C2, quintic spline, W4 B-spline y gaussiana por splines (Radioss, 3D)
  vecinos.py       búsqueda de vecinos (cKDTree) y listas de Verlet
  packing.py       Colagrossi, Negi, WVT (2D y 3D)
  metricas.py      postprocesado (Λ, Γ, densidad, Voronoi, superficie)
  visor.py         figuras 2D
  visor3d.py       figuras 3D, secciones y ventana interactiva
  visor_html.py    visor WebGL en el navegador
  exportar.py      PAM-CRASH (.inc), CSV y VTK
  particulas.py    clase DistribucionSPH (masa, h, guardar/cargar)
  flujo.py         métricas + figuras + exportación de cada script
  figuras.py       paneles configurables, figuras de comparación y % frente al mejor
  catalogo.py      lista de métodos y generar() (usado por la interfaz)
  inflation.py     resolución variable: zonas, campo dx(x), WVT variable, densidad, masas, exportación
  visor_inflation.py figuras del inflation
```

## Siguientes fases

1. Inflation: hecho (scripts 30-32). Pendiente: validar en PAM-CRASH con masa variable.
