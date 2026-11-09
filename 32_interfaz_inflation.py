"""
INTERFAZ GRAFICA - INFLATION (resolucion variable)
==================================================
Ejecutalo con F5 (o: python 32_interfaz_inflation.py desde esta carpeta).

  1. Modelo:      2D / 3D, geometria por defecto o tu .stl ("Elegir STL").
                  "Centrar STL" desactivado = se usan las coordenadas del STL
                  (las de tu modelo), para definir las zonas en esas coordenadas.
  2. Resolucion:  dx fino, dx grueso y tasa de crecimiento por capa (%).
                  Se muestran la longitud de la transicion y el N estimado.
  3. Zonas:       esfera, caja, losa (banda alrededor de un plano, p.ej. la
                  superficie libre de la piscina) o punto. Rellena los campos y
                  pulsa "Anadir". Con "Colocar con clic" el siguiente clic en la
                  figura pone ahi el centro de la zona (en el plano de corte).
  4. Generar:     celdas adaptativas -> WVT con h variable -> correccion de
                  densidad -> calibracion de volumenes. Exporta PAM-CRASH (masa
                  por particula), CSV, VTK y npz.
  5. Ver:         vista previa del dx pedido (antes de generar), particulas,
                  error de densidad, Lambda, espaciado real/pedido, perfil o la
                  figura resumen; en 3D, plano X/Y/Z y su posicion.
"""
import io
import os
import sys
import queue
import threading
import traceback

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from sph_common import crear_geometria, crear_geometria_3d, GEOMETRIAS_2D, GEOMETRIAS_3D
from sph_common import inflation as inf
from sph_common.visor_inflation import (dibujar_campo, dibujar_particulas, figura_inflation,
                                        ejes_seccion, NOMBRE)

CARPETA = "resultados"
TIPOS_ZONA = {"Esfera": "esfera", "Caja": "caja", "Losa (plano)": "losa", "Punto": "punto"}
VISTAS = ["dx pedido (vista previa)", "Particulas (dx)", "Error de densidad (%)", "Lambda",
          "Espaciado real / pedido", "Salto entre vecinas", "Perfil dx - distancia",
          "Resumen (6 paneles, ventana aparte)"]
RESUMEN = [("N", "N"), ("N zona fina", "N_zona_fina"), ("masa total", "masa_total"),
           ("error masa %", "error_masa_%"), ("masa max/min", "masa_max/min"),
           ("desv. densidad %", "err_rho_std_%"), ("sesgo densidad %", "err_rho_media_%"),
           ("max error densidad %", "err_rho_Linf_%"), ("dx real/pedido", "dx_real/dx_media"),
           ("salto vecinas max", "salto_vecinas_max"), ("Lambda medio", "Lambda_media"),
           ("Lambda min", "Lambda_min"), ("CV |gradG|h superf. %", "gradGamma_h_sup_CV_%"),
           ("dnn min / dx", "dnn_min")]


class _Escritor(io.TextIOBase):
    def __init__(self, cola):
        self.cola = cola

    def write(self, s):
        if s:
            self.cola.put(("log", s))
        return len(s)


def _vector(texto, dim):
    v = [float(a) for a in texto.replace(";", ",").split(",") if a.strip()]
    if len(v) < dim:
        v += [0.0] * (dim - len(v))
    return tuple(v[:dim])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python SPH - inflation (resolucion variable)")
        self.geometry("1500x930")
        self.cola = queue.Queue()
        self.ocupado = False
        self.zonas = []                 # lista de dicts
        self.geom = None
        self.dist = None                # DistribucionVariable
        self.res = self.campos = None
        self.ruta_stl = None
        self._clic_pendiente = False
        self._id_redib = None
        self._construir()
        self.protocol("WM_DELETE_WINDOW", self._cerrar)
        self._cambio_dim()
        self._id_sondeo = self.after(100, self._sondear)

    def _cerrar(self):
        try:
            self.after_cancel(self._id_sondeo)
        except Exception:
            pass
        self.quit()
        self.destroy()

    # ------------------------------------------------------------------
    # Ventana
    # ------------------------------------------------------------------
    def _construir(self):
        izq = ttk.Frame(self, padding=5)
        izq.pack(side="left", fill="y")
        der = ttk.Frame(self, padding=4)
        der.pack(side="left", fill="both", expand=True)

        def fila(f, r, texto, var, ancho=16):
            ttk.Label(f, text=texto).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(f, textvariable=var, width=ancho)
            e.grid(row=r, column=1, sticky="we", padx=2, pady=1)
            return e

        # 1. Modelo ------------------------------------------------------
        f = ttk.LabelFrame(izq, text="1. Modelo", padding=4)
        f.pack(fill="x", pady=2)
        self.v_dim = tk.StringVar(value="3D")
        ttk.Label(f, text="Dimension").grid(row=0, column=0, sticky="w")
        c = ttk.Combobox(f, textvariable=self.v_dim, values=["2D", "3D"], state="readonly", width=14)
        c.grid(row=0, column=1, sticky="we")
        c.bind("<<ComboboxSelected>>", lambda e: self._cambio_dim())
        self.v_geom = tk.StringVar()
        ttk.Label(f, text="Geometria").grid(row=1, column=0, sticky="w")
        self.c_geom = ttk.Combobox(f, textvariable=self.v_geom, state="readonly", width=14)
        self.c_geom.grid(row=1, column=1, sticky="we")
        self.c_geom.bind("<<ComboboxSelected>>", lambda e: self._cambio_modelo())
        self.v_par = tk.StringVar()
        e = fila(f, 2, "Parametros", self.v_par)
        e.bind("<Return>", lambda ev: self._cambio_modelo())
        self.b_stl = ttk.Button(f, text="Elegir STL...", command=self._elegir_stl)
        self.b_stl.grid(row=3, column=0, sticky="we")
        self.l_stl = ttk.Label(f, text="(ninguno)", width=22)
        self.l_stl.grid(row=3, column=1, sticky="w")
        self.v_esc = tk.StringVar(value="1.0")
        e = fila(f, 4, "Escala STL", self.v_esc)
        e.bind("<Return>", lambda ev: self._cambio_modelo())
        self.v_centrar = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Centrar STL (si no: coordenadas del STL)", variable=self.v_centrar,
                        command=self._cambio_modelo).grid(row=5, column=0, columnspan=2, sticky="w")
        self.v_rho = tk.StringVar(value="9.5e-7")
        fila(f, 6, "rho0 [kg/mm3]", self.v_rho)
        self.v_hsim = tk.StringVar(value="1.2")
        fila(f, 7, "h/dx simulacion", self.v_hsim)
        self.l_bbox = ttk.Label(f, text="", foreground="#555", wraplength=300)
        self.l_bbox.grid(row=8, column=0, columnspan=2, sticky="w")

        # 2. Resolucion --------------------------------------------------
        f = ttk.LabelFrame(izq, text="2. Resolucion", padding=4)
        f.pack(fill="x", pady=2)
        self.v_fino, self.v_grueso, self.v_tasa = tk.StringVar(value="3.0"), tk.StringVar(value="6.65"), tk.StringVar(value="8")
        for r, (t, v) in enumerate([("dx fino [mm]", self.v_fino), ("dx grueso [mm]", self.v_grueso),
                                    ("tasa [% por capa]", self.v_tasa)]):
            e = fila(f, r, t, v)
            e.bind("<Return>", lambda ev: self._programar_redibujo(estimar=True))
            e.bind("<FocusOut>", lambda ev: self._programar_redibujo(estimar=True))
        self.l_info = ttk.Label(f, text="", foreground="#1a5", wraplength=300)
        self.l_info.grid(row=3, column=0, columnspan=2, sticky="w")

        # 3. Zonas -------------------------------------------------------
        f = ttk.LabelFrame(izq, text="3. Zonas de refinado", padding=4)
        f.pack(fill="x", pady=2)
        self.v_tipo = tk.StringVar(value="Esfera")
        ttk.Label(f, text="Tipo").grid(row=0, column=0, sticky="w")
        c = ttk.Combobox(f, textvariable=self.v_tipo, values=list(TIPOS_ZONA), state="readonly", width=14)
        c.grid(row=0, column=1, sticky="we")
        c.bind("<<ComboboxSelected>>", lambda e: self._cambio_tipo())
        self.v_c, self.v_R = tk.StringVar(value="-107, 0, 0"), tk.StringVar(value="25")
        self.v_lo, self.v_hi = tk.StringVar(value="-110, -20, -20"), tk.StringVar(value="-80, 20, 20")
        self.v_n, self.v_e = tk.StringVar(value="0, 0, 1"), tk.StringVar(value="10")
        self.v_dxz = tk.StringVar(value="")
        self.e_zona = {}
        for r, (k, t, v) in enumerate([("c", "Centro / punto", self.v_c), ("R", "Radio", self.v_R),
                                       ("lo", "Esquina min", self.v_lo), ("hi", "Esquina max", self.v_hi),
                                       ("n", "Normal", self.v_n), ("e", "Espesor", self.v_e),
                                       ("dx", "dx propio (opc.)", self.v_dxz)], start=1):
            self.e_zona[k] = fila(f, r, t, v)
        b = ttk.Frame(f)
        b.grid(row=8, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(b, text="Anadir", command=self._anadir_zona).pack(side="left", expand=True, fill="x")
        ttk.Button(b, text="Modificar", command=self._modificar_zona).pack(side="left", expand=True, fill="x")
        ttk.Button(b, text="Quitar", command=self._quitar_zona).pack(side="left", expand=True, fill="x")
        self.v_clic = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Colocar el centro con un clic en la figura",
                        variable=self.v_clic).grid(row=9, column=0, columnspan=2, sticky="w")
        self.lb = tk.Listbox(f, height=4, width=46, exportselection=False)
        self.lb.grid(row=10, column=0, columnspan=2, sticky="we")
        self.lb.bind("<<ListboxSelect>>", lambda e: self._zona_a_campos())

        # 4. Generar -----------------------------------------------------
        f = ttk.LabelFrame(izq, text="4. Generar", padding=4)
        f.pack(fill="x", pady=2)
        self.v_iw, self.v_id = tk.StringVar(value="100"), tk.StringVar(value="500")
        fila(f, 0, "Iteraciones WVT", self.v_iw, 8)
        fila(f, 1, "Iter. densidad", self.v_id, 8)
        self.v_cal = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Calibrar volumenes con el kernel W4", variable=self.v_cal).grid(
            row=2, column=0, columnspan=2, sticky="w")
        b = ttk.Frame(f)
        b.grid(row=3, column=0, columnspan=2, sticky="we")
        self.b_gen = ttk.Button(b, text="Generar", command=self._generar)
        self.b_gen.pack(side="left", expand=True, fill="x")
        self.b_exp = ttk.Button(b, text="Exportar", command=self._exportar)
        self.b_exp.pack(side="left", expand=True, fill="x")
        self.b_car = ttk.Button(b, text="Cargar .npz", command=self._cargar)
        self.b_car.pack(side="left", expand=True, fill="x")
        self.barra = ttk.Progressbar(f, mode="determinate", maximum=1000)
        self.barra.grid(row=4, column=0, columnspan=2, sticky="we", pady=2)

        # derecha: figura + metricas + registro ---------------------------
        abajo = ttk.Frame(der)
        abajo.pack(side="bottom", fill="x")
        f = ttk.LabelFrame(abajo, text="Metricas (W4, h variable)", padding=2)
        f.pack(side="left", fill="y")
        self.t_met = tk.Text(f, height=12, width=40, font=("Consolas", 9))
        self.t_met.pack(fill="both", expand=True)
        f = ttk.LabelFrame(abajo, text="Registro", padding=2)
        f.pack(side="left", fill="both", expand=True)
        self.t_log = tk.Text(f, height=12, font=("Consolas", 9), bg="#111", fg="#ddd")
        self.t_log.pack(fill="both", expand=True)
        # 5. Ver (encima de la figura) ---------------------------------
        f = ttk.LabelFrame(der, text="5. Ver", padding=3)
        f.pack(side="top", fill="x")
        self.v_vista = tk.StringVar(value=VISTAS[0])
        ttk.Label(f, text="Mostrar").pack(side="left")
        c = ttk.Combobox(f, textvariable=self.v_vista, values=VISTAS, state="readonly", width=32)
        c.pack(side="left", padx=4)
        c.bind("<<ComboboxSelected>>", lambda e: self._programar_redibujo())
        self.v_plano = tk.StringVar(value="Y")
        ttk.Label(f, text="   Plano de corte (3D)").pack(side="left")
        c = ttk.Combobox(f, textvariable=self.v_plano, values=["X", "Y", "Z"], state="readonly", width=4)
        c.pack(side="left", padx=4)
        c.bind("<<ComboboxSelected>>", lambda e: self._programar_redibujo())
        self.v_pos = tk.DoubleVar(value=0.5)
        ttk.Label(f, text="   Posicion").pack(side="left")
        ttk.Scale(f, from_=0.0, to=1.0, variable=self.v_pos, length=260,
                  command=lambda e: self._programar_redibujo()).pack(side="left", padx=4)
        self.l_pos = ttk.Label(f, text="", foreground="#555", width=16)
        self.l_pos.pack(side="left")
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=der)
        tb = NavigationToolbar2Tk(self.canvas, der, pack_toolbar=False)
        tb.update()
        tb.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._clic)

    # ------------------------------------------------------------------
    # Modelo y campo
    # ------------------------------------------------------------------
    def _cambio_dim(self):
        d3 = self.v_dim.get() == "3D"
        self.c_geom["values"] = GEOMETRIAS_3D if d3 else [g for g in GEOMETRIAS_2D if g != "poligono"]
        self.v_geom.set("capsula")
        for v, t3, t2 in [(self.v_c, "-107, 0, 0", "-107, 0"), (self.v_lo, "-110, -20, -20", "-110, -20"),
                          (self.v_hi, "-80, 20, 20", "-80, 20"), (self.v_n, "0, 0, 1", "0, 1")]:
            v.set(t3 if d3 else t2)
        if not d3:
            self.v_fino.set("1.5"); self.v_grueso.set("4.0"); self.v_R.set("15")
        else:
            self.v_fino.set("3.0"); self.v_grueso.set("6.65"); self.v_R.set("25")
        self.zonas = []
        self._anadir_zona(silencio=True)
        self._cambio_tipo()
        self._cambio_modelo()

    def _parametros(self):
        p = {}
        for trozo in self.v_par.get().replace(";", ",").split(","):
            if "=" in trozo:
                k, v = trozo.split("=", 1)
                p[k.strip()] = float(v)
        return p

    def _crear_geom(self):
        d3 = self.v_dim.get() == "3D"
        nombre = self.v_geom.get()
        p = self._parametros()
        if nombre == "stl":
            if not self.ruta_stl:
                raise ValueError("Elige un fichero STL con 'Elegir STL...'")
            p = dict(archivo=self.ruta_stl, escala=float(self.v_esc.get()), centrar=self.v_centrar.get())
        return (crear_geometria_3d if d3 else crear_geometria)(nombre, **p)

    def _cambio_modelo(self):
        es_stl = self.v_geom.get() == "stl"
        self.b_stl.state(["!disabled"] if es_stl else ["disabled"])
        if es_stl and not self.ruta_stl:
            self.geom = None
            self.l_bbox.config(text="Elige un STL")
            self.fig.clf(); self.canvas.draw_idle()
            return
        try:
            self.geom = self._crear_geom()
        except Exception as e:
            messagebox.showerror("Python SPH", str(e))
            return
        lo, hi = self.geom.bbox
        f = lambda v: "(" + ", ".join(f"{a:.4g}" for a in v) + ")"
        self.l_bbox.config(text=f"{self.geom.nombre}: caja {f(lo)} - {f(hi)}\n"
                                f"volumen = {self.geom.volumen:.6g}")
        self.dist = self.res = self.campos = None
        self.v_vista.set(VISTAS[0])
        self._programar_redibujo(estimar=True)

    def _elegir_stl(self):
        r = filedialog.askopenfilename(title="Elegir STL", filetypes=[("STL", "*.stl *.STL"), ("Todos", "*.*")],
                                       initialdir=os.path.abspath("stl") if os.path.isdir("stl") else None)
        if r:
            self.ruta_stl = r
            self.l_stl.config(text=os.path.basename(r))
            self._cambio_modelo()

    def _campo(self):
        tasa = float(self.v_tasa.get()) / 100.0
        return inf.CampoResolucion(float(self.v_fino.get()), float(self.v_grueso.get()), tasa, self.zonas)

    # ------------------------------------------------------------------
    # Zonas
    # ------------------------------------------------------------------
    def _cambio_tipo(self):
        t = TIPOS_ZONA[self.v_tipo.get()]
        activos = {"esfera": ("c", "R"), "punto": ("c",), "caja": ("lo", "hi"), "losa": ("c", "n", "e")}[t]
        for k, e in self.e_zona.items():
            e.state(["!disabled"] if (k in activos or k == "dx") else ["disabled"])

    def _zona_de_campos(self):
        dim = 3 if self.v_dim.get() == "3D" else 2
        t = TIPOS_ZONA[self.v_tipo.get()]
        dxz = self.v_dxz.get().strip()
        z = dict(tipo=t, dx=float(dxz) if dxz else None)
        if t in ("esfera", "punto"):
            z["centro"] = _vector(self.v_c.get(), dim)
            if t == "esfera":
                z["radio"] = float(self.v_R.get())
        elif t == "caja":
            z["pmin"], z["pmax"] = _vector(self.v_lo.get(), dim), _vector(self.v_hi.get(), dim)
        else:
            z["punto"], z["normal"] = _vector(self.v_c.get(), dim), _vector(self.v_n.get(), dim)
            z["espesor"] = float(self.v_e.get())
        inf.Zona.de_dict(z)                     # valida
        return z

    def _refrescar_lista(self):
        self.lb.delete(0, "end")
        for k, z in enumerate(self.zonas):
            self.lb.insert("end", f"{k + 1}. {inf.Zona.de_dict(z).describir()}")
        self.dist = self.res = self.campos = None
        if not self.v_vista.get().startswith("dx pedido"):
            self.v_vista.set(VISTAS[0])
        self._programar_redibujo(estimar=True)

    def _anadir_zona(self, silencio=False):
        try:
            self.zonas.append(self._zona_de_campos())
        except Exception as e:
            if not silencio:
                messagebox.showerror("Python SPH", f"Zona no valida: {e}")
            return
        self._refrescar_lista()
        self.lb.selection_clear(0, "end"); self.lb.selection_set(len(self.zonas) - 1)

    def _modificar_zona(self):
        sel = self.lb.curselection()
        if not sel:
            return self._anadir_zona()
        try:
            self.zonas[sel[0]] = self._zona_de_campos()
        except Exception as e:
            messagebox.showerror("Python SPH", f"Zona no valida: {e}")
            return
        self._refrescar_lista()
        self.lb.selection_set(sel[0])

    def _quitar_zona(self):
        sel = self.lb.curselection()
        if sel:
            del self.zonas[sel[0]]
            self._refrescar_lista()

    def _zona_a_campos(self):
        sel = self.lb.curselection()
        if not sel:
            return
        z = self.zonas[sel[0]]
        f = lambda v: ", ".join(f"{a:g}" for a in v)
        inv = {v: k for k, v in TIPOS_ZONA.items()}
        self.v_tipo.set(inv[z["tipo"]])
        if "centro" in z: self.v_c.set(f(z["centro"]))
        if "punto" in z: self.v_c.set(f(z["punto"]))
        if "radio" in z: self.v_R.set(f"{z['radio']:g}")
        if "pmin" in z: self.v_lo.set(f(z["pmin"])); self.v_hi.set(f(z["pmax"]))
        if "normal" in z: self.v_n.set(f(z["normal"])); self.v_e.set(f"{z['espesor']:g}")
        self.v_dxz.set("" if not z.get("dx") else f"{z['dx']:g}")
        self._cambio_tipo()

    def _clic(self, ev):
        if not self.v_clic.get() or ev.inaxes is None or self.geom is None:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode:
            return                               # zoom/pan activo
        dim = self.geom.dim
        a, b, k = ejes_seccion(dim, self._seccion())
        p = np.zeros(dim)
        p[a], p[b] = ev.xdata, ev.ydata
        if k is not None:
            p[k] = self._seccion()[1]
        t = TIPOS_ZONA[self.v_tipo.get()]
        f = lambda v: ", ".join(f"{x:.4g}" for x in v)
        if t == "caja":
            lo, hi = np.array(_vector(self.v_lo.get(), dim)), np.array(_vector(self.v_hi.get(), dim))
            m = 0.5 * (hi - lo)
            self.v_lo.set(f(p - m)); self.v_hi.set(f(p + m))
        else:
            self.v_c.set(f(p))
        sel = self.lb.curselection()
        (self._modificar_zona if sel else self._anadir_zona)()

    # ------------------------------------------------------------------
    # Dibujo
    # ------------------------------------------------------------------
    def _seccion(self):
        if self.geom is None or self.geom.dim == 2:
            return ("z", 0.0)
        k = "xyz".index(self.v_plano.get().lower())
        lo, hi = self.geom.bbox[0][k], self.geom.bbox[1][k]
        return (self.v_plano.get().lower(), float(lo + self.v_pos.get() * (hi - lo)))

    def _programar_redibujo(self, estimar=False):
        if self._id_redib:
            self.after_cancel(self._id_redib)
        self._id_redib = self.after(300, self._redibujar)
        if estimar:
            self._estimar()

    def _estimar(self):
        if self.geom is None:
            return
        try:
            campo = self._campo()
        except Exception as e:
            self.l_info.config(text=f"Revisa la resolucion: {e}", foreground="#c00")
            return
        L = campo.longitud_transicion
        capas = np.log(campo.dx_grueso / campo.dx_min) / np.log(1 + campo.tasa)
        self.l_info.config(text=f"transicion = {L:.4g} mm (~{capas:.0f} capas)\nN estimado: calculando...",
                           foreground="#1a5")
        geom = self.geom

        def tarea():
            try:
                n = inf.estimar_n(geom, campo)
                self.cola.put(("estimacion", (n, L, capas)))
            except Exception as e:
                self.cola.put(("log", f"estimacion: {e}\n"))
        threading.Thread(target=tarea, daemon=True).start()

    def _redibujar(self):
        self._id_redib = None
        if self.geom is None:
            return
        vista = self.v_vista.get()
        sec = self._seccion()
        self.l_pos.config(text="" if self.geom.dim == 2 else f"{sec[0]} = {sec[1]:.4g} mm")
        if vista.startswith("Resumen"):
            if self.dist is None:
                messagebox.showinfo("Python SPH", "Genera (o carga) primero una distribucion.")
                self.v_vista.set(VISTAS[0])
            else:
                fig = figura_inflation(self.dist, self.res, self.campos, sec, mostrar=False)
                plt.close(fig)
                top = tk.Toplevel(self)
                top.title("Resumen del inflation")
                top.geometry(f"{min(self.winfo_screenwidth() - 60, 1700)}x{min(self.winfo_screenheight() - 100, 950)}+20+20")
                cv = FigureCanvasTkAgg(fig, master=top)
                NavigationToolbar2Tk(cv, top).update()
                cv.get_tk_widget().pack(fill="both", expand=True)
                cv.draw()
                self.v_vista.set(VISTAS[1])
            vista = self.v_vista.get()
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        try:
            if vista.startswith("dx pedido") or self.dist is None:
                dibujar_campo(ax, self.geom, self._campo(), sec)
            elif vista.startswith("Perfil"):
                self._perfil(ax)
            else:
                c = self.campos
                I = c["interior"]
                op = {"Particulas (dx)": (None, "viridis_r", None, None, "dx"),
                      "Error de densidad (%)": (np.where(I, c["err_rho"], np.nan), "RdBu_r", -1.5, 1.5,
                                                "error de densidad SPH (%)"),
                      "Lambda": (np.where(I, c["Lambda"], np.nan), "viridis", 0.8, 1.0, "Lambda"),
                      "Espaciado real / pedido": (np.where(I, c["ratio"], np.nan), "RdBu_r", 0.95, 1.05,
                                                  "dx real / dx pedido"),
                      "Salto entre vecinas": (c["salto"], "magma_r", 1.0, None, "max dx_j / dx_i")}[vista]
                dibujar_particulas(ax, self.dist, sec, op[0], cmap=op[1], vmin=op[2], vmax=op[3],
                                   etiqueta=op[4])
                ax.set_title(f"{vista}  -  N = {self.dist.N}", fontsize=10)
        except Exception as e:
            self.t_log.insert("end", traceback.format_exc())
            ax.text(0.5, 0.5, f"Error al dibujar:\n{e}", ha="center", transform=ax.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _perfil(self, ax):
        c, d = self.campos, self.dist
        dz, I = c["dist_zona"], c["interior"]
        if np.all(np.isinf(dz)):
            ax.text(0.5, 0.5, "sin zonas", ha="center", transform=ax.transAxes)
            return
        sel = np.where(I)[0]
        if len(sel) > 30000:
            sel = np.random.default_rng(0).choice(sel, 30000, replace=False)
        ax.plot(dz[sel], c["dx_real"][sel], ".", ms=2, alpha=0.4, label="dx real (interior)")
        dd = np.linspace(0, dz[np.isfinite(dz)].max(), 300)
        ax.plot(dd, np.min([np.minimum(d.campo.dx_grueso, (z.dx or d.campo.dx_fino) + d.campo.tasa * dd)
                            for z in d.campo.zonas], axis=0), "k-", lw=2, label="dx pedido")
        ax.set_xlabel("distancia a la zona"); ax.set_ylabel("dx"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("espaciado real frente al pedido", fontsize=10)

    # ------------------------------------------------------------------
    # Calculo en segundo plano
    # ------------------------------------------------------------------
    def _botones(self, activos):
        for b in (self.b_gen, self.b_exp, self.b_car):
            b.state(["!disabled"] if activos else ["disabled"])

    def _lanzar(self, tarea):
        if self.ocupado:
            messagebox.showinfo("Python SPH", "Espera a que termine el calculo en curso.")
            return
        self.ocupado = True
        self._botones(False)
        self.barra["value"] = 0

        def trabajo():
            viejo = sys.stdout
            sys.stdout = _Escritor(self.cola)
            try:
                self.cola.put(("ok", tarea()))
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
                    self.t_log.insert("end", dato); self.t_log.see("end")
                elif tipo == "progreso":
                    it, n = dato
                    self.barra["value"] = 1000 * it / max(n, 1)
                elif tipo == "estimacion":
                    n, L, capas = dato
                    t = n * 3.2e-3 / max(1.0, (inf.N_HILOS / 2) ** 0.7)
                    self.l_info.config(text=f"transicion = {L:.4g} mm (~{capas:.0f} capas)\n"
                                            f"N estimado = {n:,}  (~{t / 60:.0f} min aqui, {inf.N_HILOS} hilos)"
                                       .replace(",", "."))
                elif tipo in ("ok", "error"):
                    self.ocupado = False
                    self._botones(True)
                    self.barra["value"] = 1000 if tipo == "ok" else 0
                    if tipo == "error":
                        messagebox.showerror("Python SPH", dato)
                    elif dato:
                        accion, valor = dato
                        if accion == "mostrar":
                            self.dist, self.res, self.campos = valor
                            self._mostrar_metricas()
                            self.v_vista.set(VISTAS[1])
                            self._programar_redibujo()
                        elif accion == "info":
                            messagebox.showinfo("Python SPH", valor)
        except queue.Empty:
            pass
        self._id_sondeo = self.after(100, self._sondear)

    def _generar(self):
        if self.geom is None:
            messagebox.showinfo("Python SPH", "Define primero la geometria.")
            return
        try:
            campo = self._campo()
            iw, idn = int(self.v_iw.get()), int(self.v_id.get())
            rho0, hs = float(self.v_rho.get()), float(self.v_hsim.get())
        except Exception as e:
            messagebox.showerror("Python SPH", str(e))
            return
        n = inf.estimar_n(self.geom, campo)
        if n > 400_000 and not messagebox.askyesno(
                "Python SPH", f"Se generaran unas {n:,} particulas. Puede tardar mucho en este "
                              f"ordenador. Continuar?".replace(",", ".")):
            return
        geom, cal = self.geom, self.v_cal.get()

        def progreso(it, total, texto):
            self.cola.put(("progreso", (it, total)))

        def tarea():
            d, h = inf.generar_inflation(geom, campo, rho0, hs, iter_wvt=iw, iter_densidad=idn,
                                         calibrar=cal, progreso=progreso)
            r, c = inf.metricas_inflation(d)
            return "mostrar", (d, r, c)
        self._lanzar(tarea)

    def _mostrar_metricas(self):
        self.t_met.delete("1.0", "end")
        self.t_met.insert("end", f"{self.dist.geom.nombre}  ({self.dist.dim}D)\n")
        self.t_met.insert("end", f"dx {self.dist.dx.min():.4g} - {self.dist.dx.max():.4g}\n\n")
        for t, k in RESUMEN:
            v = self.res.get(k)
            s = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.4g}"
            self.t_met.insert("end", f"{t:<24s}{s:>12s}\n")

    def _exportar(self):
        if self.dist is None:
            messagebox.showinfo("Python SPH", "Genera primero una distribucion.")
            return
        d, c = self.dist, self.campos
        os.makedirs(CARPETA, exist_ok=True)
        base = os.path.join(CARPETA, f"{d.geom.nombre}_inflation_{d.dim}D")

        def tarea():
            print(f"npz       -> {d.guardar(base + '.npz')}")
            print(f"PAM-CRASH -> {inf.exportar_pamcrash_variable(d, base + '_pamcrash.inc')}")
            print(f"CSV       -> {inf.exportar_csv_variable(d, base + '.csv')}")
            print(f"VTK       -> {inf.exportar_vtk_variable(d, base + '.vtk', c)}")
            return "info", f"Exportado en {os.path.abspath(CARPETA)}:\n{os.path.basename(base)}_*"
        self._lanzar(tarea)

    def _cargar(self):
        if self.geom is None:
            messagebox.showinfo("Python SPH", "Define primero la geometria (la misma que al generar).")
            return
        r = filedialog.askopenfilename(title="Cargar distribucion", filetypes=[("npz", "*.npz")],
                                       initialdir=os.path.abspath(CARPETA) if os.path.isdir(CARPETA) else None)
        if not r:
            return
        geom = self.geom

        def tarea():
            d = inf.cargar_distribucion(r, geom)
            print(d.resumen())
            res, c = inf.metricas_inflation(d)
            return "mostrar", (d, res, c)
        self._lanzar(tarea)


if __name__ == "__main__":
    App().mainloop()
