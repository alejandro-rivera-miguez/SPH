"""
INTERFAZ GRAFICA - Python SPH
=============================
Ejecutalo con F5 (o: python 20_interfaz_grafica.py desde esta carpeta).

  1. Modelo:        2D / 3D, geometria por defecto o un .stl (boton "Elegir STL"),
                    dx, densidad y h/dx de la simulacion (el de PAM-CRASH).
  2. Empaquetado:   metodo (cubica, FCC, hexagonal, Colagrossi V1/V2, Negi, WVT)
                    -> "Generar" (o "Cargar guardado" si ya existe en resultados/).
                    Con "Guardar y exportar" se escriben el .npz, el include de
                    PAM-CRASH, el CSV y (3D) el VTK y el visor HTML.
  3. Visualizacion: campo de color (particulas, Lambda, densidad, |grad Gamma|*h,
                    capa exterior, distancia al vecino), y en 3D el plano de corte
                    X/Y/Z, su posicion, el espesor de la loncha y la vista
                    (seccion, cuerpo cortado en 3D o ambas). La vista 3D se gira
                    con el raton; la barra de herramientas permite zoom y guardar.
  4. Comparar:      elige 2 o 3 metodos y una metrica -> figura con los
                    empaquetados arriba, la metrica abajo y el mejor en verde.

Solo usa tkinter (incluido en Python para Windows), numpy, scipy y matplotlib.
"""
import io
import os
import sys
import queue
import threading
import traceback
import webbrowser

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sph_common import (crear_geometria, crear_geometria_3d, DistribucionSPH,
                        calcular_metricas, GEOMETRIAS_2D, GEOMETRIAS_3D)
from sph_common.catalogo import METODOS_2D, METODOS_3D, generar
from sph_common.figuras import (_mapa, _campo, figura_comparar, METRICAS_COMPARACION,
                                diferencia_frente_al_mejor)
from sph_common.exportar import exportar_pamcrash, exportar_csv, exportar_vtk

CARPETA = "resultados"
CAMPOS = {
    "Particulas": "particulas",
    "Lambda (interior)": "lambda",
    "Error de densidad (%)": "densidad",
    "|grad Gamma|*h (interior)": "gradgamma",
    "|grad Gamma|*h (capa exterior)": "gradgamma_sup",
    "Distancia al vecino / dx": "dnn",
}
METRICAS_COMP = {
    "|grad Gamma|*h interior": "gradgamma",
    "|grad Gamma|*h superficie (CV)": "gradgamma_sup",
    "Error de densidad": "densidad",
    "Lambda": "lambda",
    "Distancia al vecino": "dnn",
}
RESUMEN = [("N", "N"), ("|gradGamma|h max int.", "gradGamma_h_max"),
           ("CV |gradGamma|h superf. %", "gradGamma_h_sup_CV_%"),
           ("desv. densidad int. %", "err_rho_std_%"), ("sesgo densidad %", "err_rho_media_%"),
           ("Lambda medio", "Lambda_media"), ("Lambda min", "Lambda_min"),
           ("dnn min / dx", "dnn_min"), ("hueco superf. max / dx", "hueco_sup_max"),
           ("capa exterior CV %", "capa_ext_CV_%")]


class _Escritor(io.TextIOBase):
    """Redirige print() a la cola de la interfaz (seguro entre hilos)."""

    def __init__(self, cola):
        self.cola = cola

    def write(self, s):
        if s:
            self.cola.put(("log", s))
        return len(s)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python SPH - empaquetado de particulas")
        self.geometry("1500x900")
        self.cola = queue.Queue()
        self.cache = {}          # (clave de modelo, metodo) -> (dist, r, campos)
        self.actual = None       # (dist, r, campos)
        self.ocupado = False
        self._redibujo_pendiente = None
        self._construir()
        self._cambiar_dim()
        self._id_sondeo = self.after(100, self._sondear)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    def _cerrar(self):
        try:
            self.after_cancel(self._id_sondeo)
        except Exception:
            pass
        self.quit()
        self.destroy()

    # ------------------------------------------------------------------
    # Construccion de la ventana
    # ------------------------------------------------------------------
    def _construir(self):
        izq = ttk.Frame(self, padding=6)
        izq.pack(side="left", fill="y")
        der = ttk.Frame(self, padding=4)
        der.pack(side="left", fill="both", expand=True)

        # --- 1. Modelo -------------------------------------------------
        f = ttk.LabelFrame(izq, text="1. Modelo", padding=6)
        f.pack(fill="x", pady=3)
        self.v_dim = tk.StringVar(value="3D")
        self._fila(f, "Dimension", ttk.Combobox(f, textvariable=self.v_dim, values=["2D", "3D"],
                                                state="readonly", width=24), 0)
        self.v_dim.trace_add("write", lambda *a: self._cambiar_dim())
        self.v_geom = tk.StringVar()
        self.cb_geom = ttk.Combobox(f, textvariable=self.v_geom, state="readonly", width=24)
        self._fila(f, "Geometria", self.cb_geom, 1)
        self.v_geom.trace_add("write", lambda *a: self._cambiar_geom())
        self.v_params = tk.StringVar(value="")
        self._fila(f, "Parametros", ttk.Entry(f, textvariable=self.v_params, width=26), 2)
        ttk.Label(f, text="p.ej.  R=53.5   o   R=50, L=150", foreground="#777").grid(
            row=3, column=1, sticky="w")
        self.v_stl = tk.StringVar(value="")
        self.b_stl = ttk.Button(f, text="Elegir STL...", command=self._elegir_stl)
        self.b_stl.grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(f, textvariable=self.v_stl, width=28, foreground="#225").grid(
            row=4, column=1, sticky="w")
        self.v_escala = tk.StringVar(value="1.0")
        self.e_escala = ttk.Entry(f, textvariable=self.v_escala, width=26)
        self._fila(f, "Escala STL", self.e_escala, 5)
        self.v_dx = tk.StringVar(value="6.65")
        self._fila(f, "dx [mm]", ttk.Entry(f, textvariable=self.v_dx, width=26), 6)
        self.v_rho = tk.StringVar(value="1.1317e-6")
        self._fila(f, "rho0 [kg/mm3]", ttk.Entry(f, textvariable=self.v_rho, width=26), 7)
        self.v_hsim = tk.StringVar(value="1.2")
        self._fila(f, "h/dx simulacion", ttk.Entry(f, textvariable=self.v_hsim, width=26), 8)

        # --- 2. Empaquetado --------------------------------------------
        f = ttk.LabelFrame(izq, text="2. Empaquetado", padding=6)
        f.pack(fill="x", pady=3)
        self.v_metodo = tk.StringVar()
        self.cb_metodo = ttk.Combobox(f, textvariable=self.v_metodo, state="readonly", width=30)
        self._fila(f, "Metodo", self.cb_metodo, 0)
        self.v_exportar = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Guardar y exportar (npz, PAM-CRASH, CSV, VTK, HTML)",
                        variable=self.v_exportar).grid(row=1, column=0, columnspan=2, sticky="w")
        bf = ttk.Frame(f)
        bf.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        self.b_generar = ttk.Button(bf, text="Generar", command=self._generar)
        self.b_generar.pack(side="left", padx=2)
        self.b_cargar = ttk.Button(bf, text="Cargar guardado", command=self._cargar)
        self.b_cargar.pack(side="left", padx=2)
        self.b_html = ttk.Button(bf, text="Abrir visor HTML", command=self._abrir_html)
        self.b_html.pack(side="left", padx=2)
        self.barra = ttk.Progressbar(f, mode="indeterminate", length=280)
        self.barra.grid(row=3, column=0, columnspan=2, sticky="we", pady=2)

        # --- 3. Visualizacion -------------------------------------------
        f = ttk.LabelFrame(izq, text="3. Visualizacion", padding=6)
        f.pack(fill="x", pady=3)
        self.v_campo = tk.StringVar(value="Particulas")
        cb = ttk.Combobox(f, textvariable=self.v_campo, values=list(CAMPOS), state="readonly",
                          width=30)
        self._fila(f, "Colorear por", cb, 0)
        self.v_vista = tk.StringVar(value="Ambas")
        self.cb_vista = ttk.Combobox(f, textvariable=self.v_vista, state="readonly", width=30,
                                     values=["Ambas", "Seccion", "3D cortado"])
        self._fila(f, "Vista (3D)", self.cb_vista, 1)
        self.v_plano = tk.StringVar(value="Y")
        self.cb_plano = ttk.Combobox(f, textvariable=self.v_plano, values=["X", "Y", "Z"],
                                     state="readonly", width=30)
        self._fila(f, "Plano de corte", self.cb_plano, 2)
        self.v_pos = tk.DoubleVar(value=0.5)
        self.s_pos = ttk.Scale(f, from_=0.0, to=1.0, variable=self.v_pos, length=200,
                               command=lambda e: self._redibujar_luego())
        self._fila(f, "Posicion", self.s_pos, 3)
        self.l_pos = ttk.Label(f, text="", foreground="#555")
        self.l_pos.grid(row=4, column=1, sticky="w")
        self.v_esp = tk.DoubleVar(value=1.0)
        self.s_esp = ttk.Scale(f, from_=0.5, to=6.0, variable=self.v_esp, length=200,
                               command=lambda e: self._redibujar_luego())
        self._fila(f, "Espesor (x dx)", self.s_esp, 5)
        for v in (self.v_campo, self.v_vista, self.v_plano):
            v.trace_add("write", lambda *a: self._redibujar_luego())

        # --- 4. Comparar -------------------------------------------------
        f = ttk.LabelFrame(izq, text="4. Comparar (2-3 metodos)", padding=6)
        f.pack(fill="x", pady=3)
        self.lb = tk.Listbox(f, selectmode="multiple", height=7, exportselection=False, width=34)
        self.lb.grid(row=0, column=0, columnspan=2, sticky="we")
        self.v_metcomp = tk.StringVar(value=list(METRICAS_COMP)[0])
        self._fila(f, "Metrica", ttk.Combobox(f, textvariable=self.v_metcomp,
                                              values=list(METRICAS_COMP), state="readonly",
                                              width=30), 1)
        self.b_comp = ttk.Button(f, text="Comparar seleccionados", command=self._comparar)
        self.b_comp.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)

        # --- derecha: figura arriba; metricas + log abajo ---------------------
        abajo = ttk.Frame(der)
        abajo.pack(side="bottom", fill="x")
        f = ttk.LabelFrame(abajo, text="Metricas (kernel W4)", padding=2)
        f.pack(side="left", fill="y")
        self.t_met = tk.Text(f, height=12, width=44, font=("Consolas", 9))
        self.t_met.pack(fill="both", expand=True)
        f = ttk.LabelFrame(abajo, text="Registro", padding=2)
        f.pack(side="left", fill="both", expand=True)
        self.t_log = tk.Text(f, height=12, font=("Consolas", 9), bg="#111", fg="#ddd")
        self.t_log.pack(fill="both", expand=True)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        tb = NavigationToolbar2Tk(self.canvas, der, pack_toolbar=False)
        tb.update()
        tb.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._mensaje_inicial()

    def _fila(self, padre, texto, widget, fila):
        ttk.Label(padre, text=texto).grid(row=fila, column=0, sticky="w", padx=(0, 6), pady=1)
        widget.grid(row=fila, column=1, sticky="w", pady=1)

    def _mensaje_inicial(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "Elige modelo y metodo y pulsa  Generar  o  Cargar guardado",
                ha="center", va="center", fontsize=13, color="#555")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Entradas
    # ------------------------------------------------------------------
    def _cambiar_dim(self):
        d3 = self.v_dim.get() == "3D"
        geos = [g for g in (GEOMETRIAS_3D if d3 else GEOMETRIAS_2D) if g not in ("stl", "poligono")]
        if d3:
            geos.append("stl")
        self.cb_geom["values"] = geos
        self.v_geom.set("capsula")
        metodos = METODOS_3D if d3 else METODOS_2D
        self.cb_metodo["values"] = list(metodos)
        self.v_metodo.set(list(metodos)[-1])
        self.v_dx.set("6.65" if d3 else "4.0")
        self.v_rho.set("1.1317e-6" if d3 else "9.5e-7")
        estado = "readonly" if d3 else "disabled"
        for w in (self.cb_vista, self.cb_plano):
            w.configure(state=estado)
        for w in (self.s_pos, self.s_esp, self.b_html):
            w.state(["!disabled"] if d3 else ["disabled"])
        self._cambiar_geom()
        self._actualizar_lista()

    def _cambiar_geom(self):
        stl = self.v_geom.get() == "stl"
        self.b_stl.state(["!disabled"] if stl else ["disabled"])
        self.e_escala.state(["!disabled"] if stl else ["disabled"])

    def _elegir_stl(self):
        ruta = filedialog.askopenfilename(title="Elige un fichero STL",
                                          initialdir=os.path.abspath("stl"),
                                          filetypes=[("STL", "*.stl *.STL"), ("Todos", "*.*")])
        if ruta:
            self.v_stl.set(ruta)
            self.v_geom.set("stl")

    def _parametros(self):
        p = {}
        txt = self.v_params.get().strip()
        if txt:
            for trozo in txt.replace(";", ",").split(","):
                if "=" in trozo:
                    k, v = trozo.split("=", 1)
                    p[k.strip()] = float(v)
        if self.v_geom.get() == "stl":
            if not self.v_stl.get():
                raise ValueError("Elige un fichero STL con el boton 'Elegir STL...'")
            p = {"archivo": self.v_stl.get(), "escala": float(self.v_escala.get())}
        return p

    def _modelo(self):
        d3 = self.v_dim.get() == "3D"
        nombre = self.v_geom.get()
        params = self._parametros()
        geom = (crear_geometria_3d if d3 else crear_geometria)(nombre, **params)
        dx = float(self.v_dx.get())
        clave = (d3, nombre, tuple(sorted((k, str(v)) for k, v in params.items())), dx)
        return geom, dx, float(self.v_rho.get()), float(self.v_hsim.get()), clave

    def _metodos(self):
        return METODOS_3D if self.v_dim.get() == "3D" else METODOS_2D

    # ------------------------------------------------------------------
    # Trabajo en segundo plano
    # ------------------------------------------------------------------
    def _lanzar(self, tarea, *args):
        if self.ocupado:
            messagebox.showinfo("Python SPH", "Espera a que termine el calculo en curso.")
            return
        self.ocupado = True
        self.barra.start(12)
        for b in (self.b_generar, self.b_cargar, self.b_comp):
            b.state(["disabled"])

        def trabajo():
            viejo = sys.stdout
            sys.stdout = _Escritor(self.cola)
            try:
                res = tarea(*args)
                self.cola.put(("ok", res))
            except Exception as e:
                self.cola.put(("log", traceback.format_exc()))
                self.cola.put(("error", str(e)))
            finally:
                sys.stdout = viejo
        threading.Thread(target=trabajo, daemon=True).start()

    def _sondear(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                if tipo == "log":
                    self.t_log.insert("end", dato)
                    self.t_log.see("end")
                elif tipo in ("ok", "error"):
                    self.ocupado = False
                    self.barra.stop()
                    for b in (self.b_generar, self.b_cargar, self.b_comp):
                        b.state(["!disabled"])
                    if tipo == "error":
                        messagebox.showerror("Python SPH", dato)
                    elif dato is not None:
                        accion, valor = dato
                        if accion == "mostrar":
                            self.actual = valor
                            self._actualizar_lista()
                            self._mostrar_metricas()
                            self._ajustar_slider()
                            self._redibujar()
                        elif accion == "comparar":
                            self._actualizar_lista()
                            self._ventana_comparar(*valor)
        except queue.Empty:
            pass
        self._id_sondeo = self.after(100, self._sondear)

    def _calcular(self, geom, dx, rho, hsim, clave, metodo, x=None, exportar=False):
        if x is None:
            print(f"\n>>> Generando {metodo} ({geom.nombre}, dx = {dx}) ...")
            x = generar(metodo, geom, dx, hsim)
        dist = DistribucionSPH(x, geom, metodo, dx, rho0=rho, h_factor=hsim)
        print(dist.resumen())
        r, campos = calcular_metricas(dist)
        self.cache[(clave, metodo)] = (dist, r, campos)
        if exportar:
            os.makedirs(CARPETA, exist_ok=True)
            base = os.path.join(CARPETA, f"{geom.nombre}_{metodo}")
            dist.guardar(CARPETA)
            print("  PAM-CRASH ->", exportar_pamcrash(dist, base + "_pamcrash.inc"))
            print("  CSV       ->", exportar_csv(dist, base + ".csv"))
            if geom.dim == 3:
                from sph_common.visor_html import exportar_html
                print("  VTK       ->", exportar_vtk(dist, base + ".vtk", campos))
                print("  Visor 3D  ->", exportar_html(dist, campos, base + "_visor3D.html", r))
        return dist, r, campos

    def _generar(self):
        try:
            geom_args = self._modelo()
        except Exception as e:
            messagebox.showerror("Python SPH", str(e))
            return
        metodo = self._metodos()[self.v_metodo.get()]
        exportar = self.v_exportar.get()
        self._lanzar(lambda: ("mostrar", self._calcular(*geom_args, metodo, exportar=exportar)))

    def _cargar(self):
        try:
            geom, dx, rho, hsim, clave = self._modelo()
        except Exception as e:
            messagebox.showerror("Python SPH", str(e))
            return
        metodo = self._metodos()[self.v_metodo.get()]
        ruta = os.path.join(CARPETA, f"{geom.nombre}_{metodo}.npz")
        if not os.path.exists(ruta):
            messagebox.showinfo("Python SPH", f"No existe {ruta}. Pulsa Generar.")
            return

        def tarea():
            d = DistribucionSPH.cargar(ruta, geom)
            print(f"\n>>> Cargado {ruta}")
            if abs(d.dx_nominal - dx) > 1e-9:
                print(f"  OJO: el guardado tiene dx = {d.dx_nominal} (no {dx})")
            return "mostrar", self._calcular(geom, d.dx_nominal, rho, hsim, clave, metodo, x=d.x)
        self._lanzar(tarea)

    def _abrir_html(self):
        if not self.actual or self.actual[0].dim != 3:
            return
        dist, r, campos = self.actual
        from sph_common.visor_html import exportar_html
        os.makedirs(CARPETA, exist_ok=True)
        ruta = exportar_html(dist, campos, os.path.join(
            CARPETA, f"{dist.geom.nombre}_{dist.metodo}_visor3D.html"), r)
        webbrowser.open("file://" + os.path.abspath(ruta))

    # ------------------------------------------------------------------
    # Metricas y lista de comparacion
    # ------------------------------------------------------------------
    def _mostrar_metricas(self):
        dist, r, campos = self.actual
        self.t_met.delete("1.0", "end")
        self.t_met.insert("end", f"{dist.metodo}  ({dist.geom.nombre})\n")
        self.t_met.insert("end", f"dx_eq = {dist.dx_eq:.4g}   h = {dist.h:.4g}\n")
        self.t_met.insert("end", f"masa/part. = {dist.masa:.4e}\nmasa total = {dist.masa_total:.4g}\n\n")
        for et, k in RESUMEN:
            v = r[k]
            t = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"
            self.t_met.insert("end", f"{et:<26s}{t:>12s}\n")

    def _actualizar_lista(self):
        """Metodos disponibles para comparar: calculados en esta sesion o guardados."""
        self.lb.delete(0, "end")
        try:
            geom, dx, rho, hsim, clave = self._modelo()
            nombre = geom.nombre
        except Exception:
            clave, nombre = None, None
        for vis, met in self._metodos().items():
            marca = ""
            if clave and (clave, met) in self.cache:
                marca = "  [calculado]"
            elif nombre and os.path.exists(os.path.join(CARPETA, f"{nombre}_{met}.npz")):
                marca = "  [guardado]"
            self.lb.insert("end", vis + marca)

    def _comparar(self):
        sel = [self.lb.get(i).split("  [")[0] for i in self.lb.curselection()]
        if not 2 <= len(sel) <= 3:
            messagebox.showinfo("Python SPH", "Selecciona 2 o 3 metodos en la lista.")
            return
        try:
            geom, dx, rho, hsim, clave = self._modelo()
        except Exception as e:
            messagebox.showerror("Python SPH", str(e))
            return
        metodos = [self._metodos()[s] for s in sel]
        metrica = METRICAS_COMP[self.v_metcomp.get()]

        def tarea():
            lista = []
            for m in metodos:
                if (clave, m) in self.cache:
                    lista.append(self.cache[(clave, m)])
                    continue
                ruta = os.path.join(CARPETA, f"{geom.nombre}_{m}.npz")
                if os.path.exists(ruta):
                    d = DistribucionSPH.cargar(ruta, geom)
                    lista.append(self._calcular(geom, d.dx_nominal, rho, hsim, clave, m, x=d.x))
                else:
                    lista.append(self._calcular(geom, dx, rho, hsim, clave, m))
            return "comparar", (lista, metodos, metrica)
        self._lanzar(tarea)

    def _ventana_comparar(self, lista, metodos, metrica):
        seccion = (self.v_plano.get().lower(), self._posicion_corte(lista[0][0]))
        fig = figura_comparar(lista, metodos, metrica, seccion=seccion, mostrar=False)
        plt.close(fig)
        top = tk.Toplevel(self)
        top.title(f"Comparacion: {metrica}")
        # ventana ajustada a la pantalla; la figura se reescala al tamano de la ventana
        W = min(self.winfo_screenwidth() - 60, int(fig.get_figwidth() * 100))
        H = min(self.winfo_screenheight() - 100, int(fig.get_figheight() * 100) + 40)
        top.geometry(f"{W}x{H}+20+20")
        cv = FigureCanvasTkAgg(fig, master=top)
        tb = NavigationToolbar2Tk(cv, top, pack_toolbar=False)
        tb.update()
        tb.pack(side="top", fill="x")
        cv.get_tk_widget().pack(fill="both", expand=True)
        cv.draw()

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    def _posicion_corte(self, dist):
        k = "xyz".index(self.v_plano.get().lower())
        lo, hi = dist.geom.bbox[0][k], dist.geom.bbox[1][k]
        return lo + self.v_pos.get() * (hi - lo)

    def _ajustar_slider(self):
        if self.actual and self.actual[0].dim == 3:
            self.v_pos.set(0.5)

    def _redibujar_luego(self):
        if self._redibujo_pendiente:
            self.after_cancel(self._redibujo_pendiente)
        self._redibujo_pendiente = self.after(150, self._redibujar)

    def _redibujar(self):
        self._redibujo_pendiente = None
        if not self.actual:
            return
        dist, r, campos = self.actual
        nombre = CAMPOS[self.v_campo.get()]
        self.fig.clf()
        titulo = f"{dist.metodo}  (N = {dist.N})"
        if dist.dim == 2:
            ax = self.fig.add_subplot(111)
            _mapa(ax, dist, campos, nombre, ("y", 0.0), titulo=titulo)
            self.l_pos.configure(text="")
        else:
            eje = self.v_plano.get().lower()
            pos = self._posicion_corte(dist)
            esp = self.v_esp.get() * dist.dx_eq
            self.l_pos.configure(text=f"{eje} = {pos:.2f} mm   espesor = {esp:.2f} mm")
            vista = self.v_vista.get()
            if vista in ("Ambas", "3D cortado"):
                ax3 = self.fig.add_subplot(1, 2 if vista == "Ambas" else 1, 1, projection="3d")
                self._dibujar_3d(ax3, dist, campos, nombre, eje, pos)
            if vista in ("Ambas", "Seccion"):
                ax = self.fig.add_subplot(1, 2 if vista == "Ambas" else 1, 2 if vista == "Ambas" else 1)
                _mapa(ax, dist, campos, nombre, (eje, pos), titulo=titulo, espesor=esp)
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _dibujar_3d(self, ax, dist, campos, nombre, eje, pos):
        from sph_common.visor3d import _ejes_iguales
        k = "xyz".index(eje)
        sel = dist.x[:, k] <= pos
        x = dist.x[sel]
        if nombre == "particulas":
            ax.scatter(*x.T, s=5, c=-campos["dist_borde"][sel] / dist.dx_eq, cmap="viridis",
                       depthshade=False)
            et = "color = profundidad"
        else:
            vals, cmap, a, b, et, masc = _campo(nombre, dist, campos)
            if masc is not None:
                vals = np.where(masc, vals, np.nan)
            v = vals[sel]
            fin = np.isfinite(v)
            ax.scatter(*x[~fin].T, s=4, c="#cccccc", depthshade=False)
            sc = ax.scatter(*x[fin].T, s=6, c=v[fin], cmap=cmap, vmin=a, vmax=b, depthshade=False)
            self.fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=et)
        _ejes_iguales(ax, dist.geom)
        ax.set_title(f"cuerpo cortado: {eje} <= {pos:.1f}", fontsize=10)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    App().mainloop()
