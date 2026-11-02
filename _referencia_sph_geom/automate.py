#!/usr/bin/env python
import os

from itertools import cycle, product
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_cases
from pysph.solver.utils import get_files, load, iter_output
from math import sqrt
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
backend = ' --openmp '

SCHEMES = ['new', 'jiang', 'colagrossi']

params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)

def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    else:
        return params

def make_table(column_names, row_data, output_fname, sort_cols=None,
               multicol=None, **extra_kw):
    import pandas as pd
    col_data = [[] for i in range(len(column_names))]
    for row in row_data:
        for i, item in enumerate(row):
            col_data[i].append(item)
    d = {n: col_data[i] for i, n in enumerate(column_names)}
    df = pd.DataFrame(d, columns=column_names)
    if sort_cols:
        sort_by = [column_names[i] for i in sort_cols]
    else:
        sort_by = column_names[-1]
    df.sort_values(by=sort_by, inplace=True)
    parent = os.path.dirname(output_fname)
    if not os.path.exists(parent):
        os.mkdir(parent)
    with open(output_fname, 'w') as fp:
        fp.write(df.to_latex(index=False, escape=False, **extra_kw))
        fp.close()
    if multicol is not None:
        import fileinput
        for line in fileinput.FileInput(output_fname, inplace=1):
            if "\\toprule" in line:
                line = line.replace(line, multicol)
            print(line, end='')


class Block2D(Problem):
    '''Test for stability for periodic lattice
    '''
    def get_name(self):
        return 'stability_2d'

    def _get_file(self):
        return 'code/case_2d_periodic.py --tf 200. --max-step 15000 --pfreq 500 --dx 0.05'

    def _get_file_name(self):
        return 'case_2d_periodic'

    def _get_dim(self):
        return 2

    def setup(self):
        self.rad = [1.0]
        self.shape = ['hexa', 'rect']
        self.method = ['ND', 'ND_RF']
        cmd = 'python ' + self._get_file() + backend

        get_path = self.input_path
        self.case_info = {
            f'{method}_shape_{shape}': dict(
                rad=rad, shape=shape, method=method, no_agitate=None
                )
            for method, rad, shape in product(self.method,
                                              self.rad, self.shape)
        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_perturbed()

    def _get_title(self, method,  shape):
        return method + '_' + shape

    def _plot_perturbed(self):
        case_info = [x for x in self.case_info
                     if 'no_agitate' in self.case_info[x]]
        self._plot_for_cases(case_info, self.get_name() + '_perturbed.png')

    def _get_maxdiff(self, rho, shape):
        return max(abs(rho - 1.00))

    def _plot_for_cases(self, case_info, filename):
        fig = plt.figure(figsize=(5,5))
        for i, case in enumerate(case_info):
            case_obj = self.case_info[case]['case']
            shape = case_obj.params['shape']
            method = case_obj.params['method']
            label = self._get_title(method, shape)
            fname = self._get_file_name()
            files = get_files(self.input_path(case), fname)
            rhodiff = []
            t = []
            itr = 0
            for sd, arrays in iter_output(files[1:-1]):
                t.append(itr)
                itr += 500
                pa = arrays['free']
                rhomax = self._get_maxdiff(pa.rho, shape)
                rhodiff.append(rhomax)
            if shape=='hexa':
                if method == 'ND':
                    plt.semilogy(t, rhodiff, '.-k', label=label)
                elif method == 'ND_RF':
                    plt.semilogy(t, rhodiff, '--k', label=label)
            else:
                if method == 'ND':
                    plt.semilogy(t, rhodiff, ':k', label=label)
                elif method == 'ND_RF':
                    plt.semilogy(t, rhodiff, '-k', label=label)

        plt.grid()
        plt.legend()
        plt.ylabel(r'$L_{\infty}(\rho - \rho_{o})$')
        plt.xlabel('iterations')
        plt.tight_layout()
        plt.savefig(self.output_path(filename), dpi=300)
        plt.close()


class ProjectFrequecnyTest(Problem):
    '''arbitrary shaped objects packing'''

    def get_name(self):
        return "pfreq_test"

    def setup(self):
        self.dfreq = [1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80]
        cmd = 'python code/case_2d_bounded_cylinder.py --tf 200.0 --max-step 20000 --pfreq 5000 --adaptive-timestep --dx 0.1 --nu 2.0 --k 0.0004 --pb 1' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'pf_{dfreq}': dict(
                dfreq=dfreq
                )
            for dfreq in self.dfreq

        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._get_table()

    def _get_table(self):
        column_names = ['PF', r'$N_s$', 'f', 'df']
        colformat = 'clll'

        rows = []
        df = []
        ns = []

        for case in self.cases:
            filename = case.input_path('l2.npz')
            data = np.load(filename)
            ef = data['ef']
            f = data['f']
            edf = data['edf']
            dfd = data['df']
            l2f = np.sqrt(sum((ef-f)**2)/len(f))
            l2df = np.sqrt(sum((edf-dfd)**2)/len(dfd))

            row = []
            data = case.data
            dfreq = case.params['dfreq']
            rs = data['rs']
            Ns = len(rs)
            
            df.append(dfreq)
            ns.append(l2df)

            row.append(dfreq)
            row.append(Ns)
            row.append(l2f)
            row.append(l2df)

            rows.append(row)

        make_table(column_names,
                   rows,
                   self.output_path(self.get_name() + '_table.tex'),
                   sort_cols=[0],
                   column_format=colformat)

        plt.figure()
        plt.plot(df, ns , 'o-')
        plt.grid()
        plt.xlabel('Projection frequency')
        plt.ylabel(r'$L_2(\nabla \Gamma_i)$')
        plt.tight_layout()
        plt.savefig(self.output_path(self.get_name() + '_fig.png'), dpi=300)
        plt.close()


class Star(Problem):
    '''arbitrary shaped objects packing'''

    def get_name(self):
        return "arbitrary_object_2d"

    def setup(self):
        self.dx = [0.05, 0.075, 0.1]
        cmd = 'python code/case_2d_bounded_object.py --tf 200.0 --max-step 20000 --pfreq 5000 --adaptive-timestep' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'nx_{dx}': dict(
                dx=dx
                )
            for dx in self.dx

        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_solid_particles()
        self._convergence()

    def _get_title_method(self, scheme):
        if  scheme == 'new':
            return 'Hybrid'
        elif scheme == 'colagrossi':
            return 'Standard'
        elif scheme == 'jiang':
            return 'Coupled'

    def _get_file_name(self):
        return 'case_2d_bounded_object'

    def _get_boundary(self):
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'code', 'data', 'starfish_sorted.csv')
        xa, ya = np.loadtxt(filename, unpack=True)
        shiftx = 0.5 * (max(xa) + min(xa))
        shifty = 0.5 * (max(ya) + min(ya))
        xa -= shiftx
        ya -= shifty
        return xa, ya

    def _plot_solid_particles(self):
        import matplotlib.pyplot as plt
        xa, ya = self._get_boundary()
        for case in self.cases:
            data = case.data
            dx = case.params['dx']
            xs = data['xs']
            ys = data['ys']
            rs = data['rs']
            xf = data['xf']
            yf = data['yf']
            rf = data['rf']
            fig = plt.figure()
            x = np.concatenate((xs,xf))
            y = np.concatenate((ys,yf))
            r = np.concatenate((rs,rf))
            plt.scatter(x, y, c=r, s=400*dx, marker='.',
                        rasterized=True)
            plt.plot(xa,ya, '-k', linewidth=5*dx)
            plt.colorbar(label=r'$\rho$')
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.title(r'$\Delta x = %.3f$'%dx)
            filename = os.path.join(self.output_path(), self.get_name() + case.name+'.png')
            plt.savefig(filename, dpi=300)
            plt.close()

    def _initial_particle_comparison(self, dx0=0.1, s=6, lim=[-1.5, 1.5]):
        from pysph.solver.utils import get_files, load
        xb, yb = self._get_boundary()
        fig = plt.figure(figsize=(7, 3.2))
        grid = plt.GridSpec(1, 3, hspace=0.2, wspace=0.1)
        ax = []
        ax.append(fig.add_subplot(grid[0, 0]))
        ax.append(fig.add_subplot(grid[0, 1]))
        ax[-1].yaxis.set_tick_params(labelbottom=False)
        ax.append(fig.add_subplot(grid[0, 2]))
        ax[-1].yaxis.set_tick_params(labelbottom=False)
        axi = 0

        for i, case in enumerate(self.cases):
            if case.params['dx'] == dx0:
                scheme = case.params['scheme']
                data = case.data
                xs = data['xs']
                ys = data['ys']
                rs = data['rs']
                xf = data['xf']
                yf = data['yf']
                rf = data['rf']
                x = np.concatenate((xs, xf))
                y = np.concatenate((ys, yf))
                r = np.concatenate((rs, rf))
                img = ax[axi].scatter(
                    x, y, c=r, s=s, rasterized=True, vmin=.95, vmax=1.05)
                ax[axi].plot(xb, yb, '--k', linewidth=0.5)
                ax[axi].set_title(self._get_title_method(scheme))
                ax[axi].set_ylim(lim)
                ax[axi].set_xlim(lim)
                axi += 1

        fig.colorbar(
            img, pad=0.1, ax=ax, label=r'$\rho$', orientation="horizontal")

        fig.savefig(self.output_path(self.get_name() + '_p.png'), pad_inches=0.5, dpi=300)
        plt.close()

    def _initial_particle_comparison_fs(self, dx0=0.1, s=4, lim=[-1.5, 1.5]):
        from pysph.solver.utils import get_files, load
        xb, yb = self._get_boundary()
        fig = plt.figure(figsize=(6, 2))
        grid = plt.GridSpec(1, 3, hspace=0.2, wspace=0.1)
        ax = []
        ax.append(fig.add_subplot(grid[0, 0]))
        ax.append(fig.add_subplot(grid[0, 1]))
        ax[-1].yaxis.set_tick_params(labelbottom=False)
        ax.append(fig.add_subplot(grid[0, 2]))
        ax[-1].yaxis.set_tick_params(labelbottom=False)
        axi = 0

        for i, case in enumerate(self.cases):
            if case.params['dx'] == dx0:
                scheme = case.params['scheme']
                data = case.data
                xs = data['xs']
                ys = data['ys']
                rs = data['rs']
                xf = data['xf']
                yf = data['yf']
                rf = data['rf']
                img = ax[axi].scatter(
                    xs, ys, s=s, c='r', rasterized=True)
                img = ax[axi].scatter(
                    xf, yf, s=s, c='b', rasterized=True)
                ax[axi].plot(xb, yb, '--k', linewidth=0.5)
                ax[axi].set_title(self._get_title_method(scheme))
                ax[axi].set_ylim(lim)
                ax[axi].set_xlim(lim)
                axi += 1

        fig.savefig(self.output_path(self.get_name() + 'p_sf.png'), pad_inches=0, dpi=300)
        plt.close()

    def _convergence(self):
        norms = {}
        for i, case in enumerate(self.cases):
            filename = case.input_path('l2.npz')
            data = np.load(filename)
            ef = data['ef']
            f = data['f']
            edf = data['edf']
            df = data['df']
            l2f = np.sqrt(sum((ef-f)**2)/len(f))
            l2df = np.sqrt(sum((edf-df)**2)/len(df))

            if 'scheme' in case.params:
                scheme = case.params['scheme']
            else:
                scheme = 'new'
            dx = case.params['dx']
            if scheme in norms:
                norms[scheme][dx] = [l2f, l2df]
            else:
                norms[scheme] = {}
                norms[scheme][dx] = [l2f, l2df]

        print(norms)
        fig = plt.figure(figsize=(5, 5))
        linestyle = cycle(['-r', '--b', ':g'])
        for scheme in norms:
            dxs = []
            l2f = []
            for dx in self.dx:
                dxs.append(1/dx)
                l2f.append(norms[scheme][dx][0])
            label = self._get_title_method(scheme)
            ln = next(linestyle)
            plt.semilogy(dxs, l2f, ln, marker='o', label=label)

        plt.grid(True, 'both')
        plt.legend()
        plt.ylabel(r'$L_{2}(f - f_{o})$')
        plt.xlabel(r'Particles in unit length')
        plt.gca().set_yticklabels([], minor=True)
        plt.tight_layout()
        plt.savefig(self.output_path(self.get_name() + '_conv.png'), dpi=300)
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        linestyle = cycle(['-r', '--b', ':g'])
        for scheme in norms:
            dxs = []
            l2df = []
            for dx in self.dx:
                dxs.append(1/dx)
                l2df.append(norms[scheme][dx][1])
            label = self._get_title_method(scheme)
            ln = next(linestyle)
            plt.semilogy(dxs, l2df, ln, marker='o', label=label)

        plt.grid(True, 'both')
        plt.legend()
        plt.ylabel(r'$L_{2}(f\ ^\prime - f\ ^{\prime}_{o})$')
        plt.xlabel(r'Particles in unit length')
        plt.gca().set_yticklabels([], minor=True)
        plt.tight_layout()
        plt.savefig(self.output_path(self.get_name() +'_conv_df.png'), dpi=300)
        plt.close()


class ConvergenceLimit(Star):
    '''convergence limit test'''

    def get_name(self):
        return "convergence_test"


    def _get_boundary(self):
        import os
        parent = os.path.dirname(__file__)
        airfoil_nodes = os.path.join(parent, 'code', 'data', 'NACA0015.dat')
        xa, ya = np.loadtxt(airfoil_nodes, unpack=True)
        shift = 0.5 * (max(xa) + min(xa))
        xa -= shift
        return xa, ya

    def setup(self):
        self.dx = [0.02]
        self.tol = [1e-2, 1e-2/2, 1e-2/4]
        cmd = 'python code/case_2d_bounded_airfoil.py --tf 200.0 --max-step 40000 --pfreq 5000 --nu 50' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'tol_{tol}': dict(
                dx=dx, tol=tol
                )
            for dx, tol in product(self.dx, self.tol)

        }

        print(self.case_info)

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_solid_particles()
        self._convergence()

    def _convergence(self):
        cols = ['Tolerance', 'Iterations', r'$L_{\infty}(\rho_i)$']
        rows = []
        proj_itr = 1300
        for i, case in enumerate(self.cases):
            row = []
            tol = case.params['tol']
            files = get_files(case.input_path())
            data = load(files[-1])
            itr = data['solver_data']['count']
            filename = case.input_path('l2.npz')
            data = case.data
            dx = case.params['dx']
            rs = data['rs']
            rf = data['rf']
            r = np.concatenate((rs, rf))
            err = max(abs(r - 1.0))
            # err = np.sqrt(sum((r - 1.0)**2)/len(r))
            row.append('%.1e'%(tol/100))
            row.append(itr - proj_itr)
            row.append('%.3f'%err)

            rows.append(row)

        fname = self.output_path('tol.tex')
        make_table(cols, rows, fname, column_format='cll')

    def _plot_solid_particles(self):
        import matplotlib.pyplot as plt
        xa, ya = self._get_boundary()
        for case in self.cases:
            data = case.data
            dx = case.params['dx']
            xs = data['xs']
            ys = data['ys']
            rs = data['rs']
            xf = data['xf']
            yf = data['yf']
            rf = data['rf']
            fig = plt.figure()
            x = np.concatenate((xs,xf))
            y = np.concatenate((ys,yf))
            r = np.concatenate((rs,rf))
            plt.scatter(x, y, c=r, s=30, marker='.',
                        rasterized=True)
            plt.plot(xa,ya, ':k', linewidth=0.5)
            cbar = plt.colorbar(label=r'$\rho$', orientation='horizontal')
            cbar.ax.locator_params(nbins=5)
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.title(r'$\Delta x = %.3f$'%dx)
            filename = os.path.join(self.output_path(), self.get_name() + case.name+'.png')
            plt.savefig(filename, dpi=300)
            plt.close()


class Cylinder(Star):
    def get_name(self):
        return "cylinder_2d"

    def _get_boundary(self):
        self.dc = dc = 2.0
        dx = 0.1
        theta = np.mgrid[0:2*np.pi:dx]
        xb = 0.5*dc*np.cos(theta)
        yb = 0.5*dc*np.sin(theta)
        xb, yb = [np.concatenate((t, [t[0]])) for t in (xb, yb)]
        return xb, yb

    def setup(self):
        self.dx = [0.02, 0.05, 0.1]
        cmd = 'python code/case_2d_bounded_cylinder.py --tf 200.0 --max-step 20000 --pfreq 5000' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'{scheme}_nx_{dx}': dict(
                dx=dx, scheme=scheme
                )
            for dx, scheme in product(self.dx, SCHEMES)

        }


        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def _get_file_name(self):
        return 'case_2d_bounded_cylinder'

    def run(self):
        self.make_output_dir()
        self._plot_solid_particles()
        self._initial_particle_comparison()
        self._initial_particle_comparison_fs()
        self._convergence()


class Airfoil(Star):
    def get_name(self):
        return "airfoil_2d"

    def _get_boundary(self):
        import os
        parent = os.path.dirname(__file__)
        airfoil_nodes = os.path.join(parent, 'code', 'data', 'NACA0015.dat')
        xa, ya = np.loadtxt(airfoil_nodes, unpack=True)
        shift = 0.5 * (max(xa) + min(xa))
        xa -= shift
        return xa, ya

    def setup(self):
        self.dx = [0.02, 0.01]
        cmd = 'python code/case_2d_bounded_airfoil.py --tf 200.0 --max-step 20000 --pfreq 5000  --adaptive-timestep' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'nx_{dx}': dict(
                dx=dx
                )
            for dx in self.dx

        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def _get_file_name(self):
        return 'case_2d_bounded_airfoil'


class Block3D(Block2D):
    '''Test for stability for periodic lattice
    '''
    def get_name(self):
        return 'stability_3d'

    def _get_dim(self):
        return 3

    def _get_file(self):
        return 'code/case_3d_bounded_lattice.py --tf 200.0 --max-step 15000 --pfreq 500 --dx 0.05'

    def _get_file_name(self):
        return 'case_3d_bounded_lattice'

    def run(self):
        self.make_output_dir()
        self._plot_perturbed()

    def _get_maxdiff(self, rho, shape):
        return max(abs(rho - 1.00))


class Sphere(Problem):
    '''Flow past sphere
    '''
    def get_name(self):
        return 'sphere'

    def _get_file(self):
        return 'code/case_3d_bounded_sphere.py --pfreq 5000 --tf 200 --max-step 20000'

    def _get_file_name(self):
        return 'case_3d_bounded_sphere'

    def setup(self):
        self.dx = [0.1, 0.075, 0.05]
        self.scheme = ['new', 'jiang']
        cmd = 'python ' + self._get_file() + backend

        get_path = self.input_path
        self.case_info = {
            f'{s}_dx_{dx}': dict(
                dx=dx, scheme=s, tf=200.0
                ) for dx, s in product(self.dx, self.scheme)
        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=16),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_particles()
        self._convergence()

    def _get_title_method(self, scheme):
        if  scheme == 'new':
            return 'Hybrid'
        elif scheme == 'colagrossi':
            return 'Standard'
        elif scheme == 'jiang':
            return 'Coupled'

    def _plot_particles(self):
        import subprocess
        cmd = 'pysph dump_vtk '
        currdir = os.path.dirname(os.path.abspath(__file__))
        spherfile = os.path.join(currdir, 'code', 'data', 'ellipsoid.stl')
        for case in self.case_info:
            fname = self._get_file_name()
            files = get_files(self.input_path(case), fname)
            subprocess.run(['pysph', 'dump_vtk' , files[-1]])
            remove = files[-1].split(fname)
            num = remove[-1].split('.')
            newfile = files[-1].split(remove[-1])[0] + '_sphere' + num[0] + '.vtu'
            outfile = self.output_path('%s.png'%case)
            subprocess.run([
                "pvpython", "vis_3d.py" , '--in-files', spherfile, newfile,
                '-o', outfile, '-m', 'stl_with_body'])

    def _convergence(self):
        norms = {}
        for i, case in enumerate(self.cases):
            filename = case.input_path('l2.npz')
            data = np.load(filename)
            print(filename)
            ef = data['ef']
            f = data['f']
            edf = data['edf']
            df = data['df']
            l2f = np.sqrt(sum((ef-f)**2)/len(f))
            l2df = np.sqrt(sum((edf-df)**2)/len(df))

            if 'scheme' in case.params:
                scheme = case.params['scheme']
            else:
                scheme = 'new'
            dx = case.params['dx']
            if scheme in norms:
                norms[scheme][dx] = [l2f, l2df]
            else:
                norms[scheme] = {}
                norms[scheme][dx] = [l2f, l2df]

        print(norms)
        linestyle = cycle(['-r', '--b'])
        fig = plt.figure(figsize=(5, 5))
        for scheme in norms:
            dxs = []
            l2f = []
            for dx in self.dx:
                dxs.append(1/dx)
                l2f.append(norms[scheme][dx][0])
            label = self._get_title_method(scheme)
            ln = next(linestyle)
            plt.semilogy(dxs, l2f, ln, marker='o', label=label)

        plt.grid(True, 'both')
        plt.legend()
        plt.ylabel(r'$L_{2}(f - f_{o})$')
        plt.xlabel(r'Particles in unit length')
        plt.gca().set_yticklabels([], minor=True)
        plt.tight_layout()
        plt.savefig(self.output_path(self.get_name() + '_conv.png'), dpi=300)
        plt.close()

        linestyle = cycle(['-r', '--b'])
        fig = plt.figure(figsize=(5, 5))
        for scheme in norms:
            dxs = []
            l2df = []
            for dx in self.dx:
                dxs.append(1/dx)
                l2df.append(norms[scheme][dx][1])
            label = self._get_title_method(scheme)
            ln = next(linestyle)
            plt.semilogy(dxs, l2df, ln, marker='o', label=label)

        plt.grid(True, 'both')
        plt.legend()
        plt.ylabel(r'$L_{2}(f\ ^\prime - f\ ^{\prime}_{o})$')
        plt.xlabel(r'Particles in unit length')
        plt.gca().set_yticklabels([], minor=True)
        plt.tight_layout()
        plt.savefig(self.output_path(self.get_name() +'_conv_df.png'), dpi=300)
        plt.close()


class Bunny(Problem):
    '''3d packing'''

    def get_name(self):
        return "bunny"

    def _get_file(self):
        return 'code/case_3d_bunny.py --pfreq 5000 --tf 200 --max-step 20000'

    def _get_file_name(self):
        return 'case_3d_bunny'

    def _get_process_file(self):
        return 'bunny.npz'

    def setup(self):
        cmd = 'python ' + self._get_file() + backend

        get_path = self.input_path
        self.case_info = {
            f'bunny': dict(
                tf = 200.0
                )
        }


        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=4, n_thread=16),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_particles()

    def _plot_particles(self):
        import subprocess
        cmd = 'pysph dump_vtk '
        currdir = os.path.dirname(os.path.abspath(__file__))
        spherfile = os.path.join(currdir, 'code', 'data', 'bun_zipper_res2.stl')
        for case in self.case_info:
            fname = self._get_file_name()
            files = get_files(self.input_path(case), fname)
            subprocess.run(['pysph', 'dump_vtk' , files[-1]])
            remove = files[-1].split(fname)
            num = remove[-1].split('.')
            newfile = files[-1].split(remove[-1])[0] + '_bunny' + num[0] + '.vtu'
            outfile = self.output_path('%s.png'%case)
            subprocess.run([
                "pvpython", "vis_3d.py" , '--in-files', spherfile, newfile,
                '-o', outfile, '-m', 'stl_with_bunny'])
            outfile = self.output_path('%s_1.png'%case)
            subprocess.run([
                "pvpython", "vis_3d.py" , '--in-files', spherfile, newfile,
                '-o', outfile, '-m', 'stl_with_bunny1'])
            outfile = self.output_path('%s_2.png'%case)
            subprocess.run([
                "pvpython", "vis_3d.py" , '--in-files', spherfile, newfile,
                '-o', outfile, '-m', 'stl_with_bunny2'])

    def _plot_all_particles(self):
        pass


class Force(Problem):
    def get_name(self):
        return 'force'

    def setup(self):
        cmd = 'python code/repulsion_force_plot.py'
        self.force = ['lj', 'new']

        get_path = self.input_path
        self.case_info = {
            f'{f}': dict(
                eq=f
                ) for f in self.force
        }


        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=1, n_thread=1),
                **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_force()

    def _get_label(self, name):
        if name == 'lj':
            return 'LJP'
        elif name == 'new':
            return 'RF'


    def _plot_force(self):
        plt.figure()
        linestyle = cycle(['-r', '--b'])
        for case in self.cases:
            data = case.data
            name = case.params['eq']
            x = data['x']
            au = data['au']
            ls = next(linestyle)
            label = self._get_label(name)
            plt.semilogy(x, abs(au), ls, label=label.upper())

        plt.grid()
        plt.legend()
        plt.xlabel(r'$r/\Delta s$')
        plt.ylabel(r'$F/k_r$')
        plt.ylim([0.01,1e5])
        # plt.xlim([0.0, 2])

        plt.tight_layout()
        outfile = self.output_path('force.png')
        plt.savefig(outfile, dpi=300)
        plt.close()


class ZWall(Cylinder):
    def get_name(self):
        return "zig_zag_2d"

    def _get_boundary(self):
        import os
        parent = os.path.dirname(__file__)
        airfoil_nodes = os.path.join(parent, 'code', 'data', 'zig-zag.csv')
        xa, ya = np.loadtxt(airfoil_nodes, unpack=True)
        shift = 0.5 * (max(xa) + min(xa))
        xa -= shift
        return xa, ya

    def setup(self):
        self.dx = [0.025, 0.05, 0.075]
        cmd = 'python code/case_2d_zig_zag_wall.py --tf 200.0 --max-step 20000 --pfreq 5000' +\
            backend

        get_path = self.input_path
        self.case_info = {
            f'{s}_dx_{dx}': dict(
                dx=dx, scheme=s
                )
            for dx, s in product(self.dx, SCHEMES)

        }

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=2, n_thread=4),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_solid_particles()
        self._initial_particle_comparison(dx0=0.05, lim=[-0.75, 0.75])
        self._initial_particle_comparison_fs(dx0=0.05, lim=[-0.75, 0.75])
        self._convergence()

if __name__ == '__main__':
    PROBLEMS = [
        Block2D, Star, Block3D, Sphere, Bunny, Cylinder, Airfoil, ZWall,
        Force, ProjectFrequecnyTest, ConvergenceLimit]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
