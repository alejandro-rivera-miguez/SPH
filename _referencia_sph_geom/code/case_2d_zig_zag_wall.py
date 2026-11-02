"""
2d test zig-zag wall 
"""
from __future__ import print_function
import numpy as np
import os
from math import sqrt, sin, pi
import sys


# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator, IntegratorStep
from pysph.sph.integrator_step import EulerStep
from pysph.sph.equation import Group, Equation
from pysph.solver.application import Application
from pysph.tools.geometry import remove_overlap_particles
from pysph.sph.scheme import SchemeChooser

# local imports
from particle_packing import ParticlePacking
from jiang import JiangParticlePacking
from colagrossi import ColagrossiParticlePacking
from particle_packing import (
    calculate_normal_2d_surface,
    shift_surface_inside)


class ZigZag2d(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.02
        self.L = 1.5
        self.B = 1.5
        self.dim = 2

    def wall_jiang(self, hdx, L, B, name):
        dx = self.scheme.scheme.dx
        free = self.scheme.scheme.create_free_particles(hdx, L , B, name=name)
        
        x = free.x
        y = free.y
        cond0 = (y - (-0.5-dx/2) < 1e-14) & (x + y - dx/2 < 1e-14)
        x0 = x[cond0] 
        y0 = y[cond0] 

        cond1 = (y-(0.5-dx/2)<1e-14) & (x + y - dx/2 > 1e-14)
        x1 = x[cond1] 
        y1 = y[cond1] 

        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))

        m = free.m[0]
        h = free.h[0]
        rho = free.rho[0]
        wall = get_particle_array(name=name, x=x, y=y, m=m, h=h, rho=rho)

        return wall

    def free_colagrossi(self, hdx, L, B, name):
        dx = self.scheme.scheme.dx
        free = self.scheme.scheme.create_free_particles(hdx, L , B, name=name)
        
        x = free.x
        y = free.y
        cond0 = (y - (0.5) > 1e-14) & (x + y > 1e-14)
        x0 = x[cond0] 
        y0 = y[cond0] 

        cond1 = (y-(-0.5)>1e-14) & (x + y < 1e-14)
        x1 = x[cond1] 
        y1 = y[cond1] 

        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))

        m = free.m[0]
        h = free.h[0]
        rho = free.rho[0]
        wall = get_particle_array(name=name, x=x, y=y, m=m, h=h, rho=rho)

        return wall

    def wall_colagrossi(self, dx, hdx, rho, nodes, name='free'):
        h = hdx * dx
        x_ = nodes.x
        y_ = nodes.y
        xn = nodes.xn
        yn = nodes.yn

        x1 = x_[40]
        y1 = y_[40]
        xn1 = xn[40]
        yn1 = yn[40]
        x2 = x_[60]
        y2 = y_[60]
        xn2 = xn[60]
        yn2 = yn[60]

        x = []
        y = []

        for l in range(6):
            newx1 = x1 - xn1 * (l* dx + dx/2) / sin(pi/8)
            newy1 = y1 - yn1 * (l* dx + dx/2) / sin(pi/8)
            newx2 = x2 - xn2 * (l* dx + dx/2) / sin(pi/8)
            newy2 = y2 - yn2 * (l* dx + dx/2) / sin(pi/8)
            d = sqrt((newx1 - newx2)**2 + (newy1 - newy2)**2)
            n_pnts = int(d/dx + 0.5)
            _x0 = np.linspace(newx1, newx2, n_pnts)
            _y0 = np.linspace(newy1, newy2, n_pnts)
            _x1 = -1*np.arange(-newx1, 1.5+dx, dx )
            _y1 = np.ones_like(_x1) * (y_[0] - yn[0] *(l*dx+dx/2))
            _x2 = np.arange(newx2, 1.5+dx, dx )
            _y2 = np.ones_like(_x2) * (y_[-1] - yn[-1] *(l*dx+dx/2))
            _x = list(np.concatenate((_x0[1:-1], _x1, _x2)))
            _y = list(np.concatenate((_y0[1:-1], _y1, _y2)))
            # _x = list(_x0)
            # _y = list(_y0)
            x.extend(_x)
            y.extend(_y)

        wall = get_particle_array(name=name, x=x, y=y, m=rho*dx**2, h=h, rho=rho)

        return wall

    def create_particles_colagrossi(self):
        self.dx = self.scheme.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.free_colagrossi(self.hdx, L , B, name='free')
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'zig-zag.csv')
        wall_bound = self.scheme.scheme.create_boundary_node(
            filename, hdx, shift=False, invert=True, name='wall_nodes',
            isclosed=False)
        wall = self.wall_colagrossi(self.dx, hdx, 1.0, wall_bound, name='wall') 
        frozen = self.scheme.scheme.create_frozen_container(hdx, L, B, name='frozen', l=7)

        particles = [free, wall, frozen, wall_bound]

        self.scheme.setup_properties(particles)

        return particles


    def create_particles_jiang(self):
        self.dx = self.scheme.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.scheme.scheme.create_free_particles(hdx, L , B, name='free')
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'zig-zag.csv')
        wall_bound = self.scheme.scheme.create_boundary_node(
            filename, hdx, shift=True, invert=True, name='wall_nodes',
            isclosed=False)
        wall_outer = self.scheme.scheme.create_boundary_node(
            filename, hdx, shift=True, invert=False, name='wall_outer',
            isclosed=False)
        wall = self.wall_jiang(hdx, L, B, name='wall') 
        frozen = self.scheme.scheme.create_frozen_container(
            hdx, L, B, name='frozen', l=7)
        remove_overlap_particles(free, wall, self.dx, dim=2)

        particles = [free, wall, frozen, wall_bound, wall_outer]

        self.scheme.setup_properties(particles)

        return particles

    def create_particles_new(self):
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.scheme.scheme.create_free_particles(L , B, name='free' )
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'zig-zag.csv')
        wall_nodes = self.scheme.scheme.create_boundary_node(
            filename, shift=True, invert=True, name='wall_nodes',
            isclosed=False)
        wall = get_particle_array(name='wall')
        frozen = self.scheme.scheme.create_frozen_container(L, B, name='frozen')

        particles = [free, wall, frozen, wall_nodes]

        self.scheme.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def create_particles(self):
        if self.options.scheme == 'new':
            return self.create_particles_new()
        elif self.options.scheme == 'jiang':
            return self.create_particles_jiang()
        elif self.options.scheme == 'colagrossi':
            return self.create_particles_colagrossi()

    def create_scheme(self):
        hardpoints = {
            40: [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0],
            60: [1.0 / sqrt(2), -1.0 / sqrt(2), 0.0]
        }
        new = ParticlePacking(
            fluids=['free'], solids={'wall':'wall_nodes'}, frozen=['frozen'],
            dim=self.dim, hdx=self.hdx, dx=self.dx, hardpoints=hardpoints,
            filter_layers=False, use_prediction=False, reduce_dfreq=False)

        colagrossi = ColagrossiParticlePacking(
            fluids=['free'], solids={'wall':'wall_nodes'},
            frozen=['frozen'], dim=self.dim)

        jiang = JiangParticlePacking(
            fluids=['free'], solids={'wall':['wall_nodes', 'wall_outer']},
            frozen=['frozen'], dim=self.dim, hardpoints=hardpoints)

        s = SchemeChooser(
            default='new', new=new, colagrossi=colagrossi, jiang=jiang
        )
        return s

    def configure_scheme(self):
        scheme = self.scheme
        if not (self.options.scheme == 'new'):
            self.hdx = 1.0
        scheme.configure_solver()

    def post_step(self, solver):
        self.scheme.scheme.post_step(self.particles, solver)

    def post_process(self, info_fname):
        import os
        from pysph.solver.utils import load
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        res = os.path.join(self.output_dir, 'results.npz')
        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        wall = data['arrays']['wall']
        wall_nodes = data['arrays']['wall_nodes']
        frozen = data['arrays']['frozen']
        if self.options.scheme == 'new':
            self.scheme.scheme.post_process(
                free, wall, wall_nodes, frozen, self.scheme.scheme.dx, res)
        else:
            self.scheme.scheme.post_process(free, wall, res)
        self.function_approx()

    def function_approx(self):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.solver.utils import load

        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['wall']
        frozen = data['arrays']['frozen']
        pa_arr = [free, solid, frozen]

        for pa in pa_arr:
            pa.add_property('f')
            x = pa.x
            y = pa.y
            f = np.sin(x**2 + y**2)
            pa.f[:] = f

        m = free.m[0]
        dx = self.scheme.scheme.dx
        hdx = 1.0 - 25. * (dx-0.1)/4
        h = hdx * dx 
        free.h[:] = h
        frozen.h[:] = h
        solid.h[:] = h
        rho = free.rho[0]
        L = B = self.L
        nl = 10*dx
        x0, y0 = np.mgrid[-L-nl+dx: L+nl: 2 *dx, -B-nl+dx/2: B+nl: dx]
        x1, y1 = np.mgrid[-L-nl: L+nl: 2 *dx, -B-nl: B+nl: dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        source0 = get_particle_array(
            name='source0', x=x, y=y, m=m, h=h, rho=rho, f=0, V=0)

        source0.f = np.sin(x**2 + y**2)


        x, y = np.mgrid[-L:L:200j, -L:L:200j]
        cond0 = (abs(y) < 0.6) & (x + y - 4*dx < 1e-14)
        cond1 = (abs(y) < 0.6) & (x + y - 4*dx> 1e-14) & (y > 0.4)
        cond = cond0 | cond1
        x = x[cond]
        y = y[cond]
        dest = get_particle_array(
            name='dest', x=x, y=y, m=m, h=h, rho=rho, f=0, df=0)
        dest1 = get_particle_array(
            name='dest1', x=x, y=y, m=m, h=h, rho=rho, f=0, df=0)

        pa_arr.extend([dest, dest1, source0])

        from particle_packing import SPHApprox, SPHDerivativeApprox, SummationDensity
        eqs = [
            Group(equations=[
                SummationDensity(dest='source0', sources=['source0'])
            ]),
            Group(equations=[
                SPHApprox(dest='dest', sources=['free', 'wall', 'frozen']),
                SPHDerivativeApprox(dest='dest',
                                    sources=['free', 'wall', 'frozen']),
                SPHApprox(dest='dest1', sources=['source0']),
                SPHDerivativeApprox(dest='dest1', sources=['source0'])
            ])
        ]

        eval = SPHEvaluator(pa_arr, equations=eqs, dim=2, backend='cython', kernel=QuinticSpline(dim=2))
        eval.evaluate()

        exactf = dest1.f 
        exactdf = dest1.df 
        f = dest.f
        df = dest.df

        res = os.path.join(self.output_dir, 'l2.npz')
        np.savez(res, ef=exactf, f=f, edf=exactdf, df=df)


if __name__ == '__main__':
    app = ZigZag2d()
    app.run()
    app.post_process(app.info_filename)
