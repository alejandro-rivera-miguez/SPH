"""
2d test airfoil with periodic boundary
"""
from __future__ import print_function
import numpy as np
import os


# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.tools.geometry import remove_overlap_particles
from pysph.sph.equation import Group
from math import sqrt

# local imports
from particle_packing import ParticlePacking
from particle_packing import (calculate_normal_2d_surface,
                             shift_surface_inside)


class Shape2d(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.05
        self.dim = 2
        self.L = 1.5
        self.B = 1.5

    def create_particles(self):
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.scheme.create_free_particles(L , B, name='free' )
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'starfish_sorted.csv')
        shape_nodes = self.scheme.create_boundary_node(filename, invert=True, shift=True,
                                                       name='shape_nodes')
        shape = get_particle_array(name='shape')
        frozen = self.scheme.create_frozen_container(L, B, name='frozen')

        particles = [free, shape, frozen, shape_nodes]

        self.scheme.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def create_scheme(self):
        hardpoints = None
        s = ParticlePacking(
            fluids=['free'], solids={'shape': 'shape_nodes'},
            frozen=['frozen'],
            dim=self.dim, dx=self.dx, hardpoints=hardpoints,
            use_prediction=False, filter_layers=False, reduce_dfreq=False,
            hdx=self.hdx)

        s.configure_solver(dt=1e-4)
        return s

    def post_step(self, solver):
        self.scheme.post_step(self.particles, solver)

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
        solid = data['arrays']['shape']
        solid_nodes = data['arrays']['shape_nodes']
        frozen = data['arrays']['frozen']
        self.scheme.post_process(
            free, solid, solid_nodes, frozen, self.scheme.dx, res)
        self.function_approx()

    def function_approx(self):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.solver.utils import load


        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['shape']
        frozen = data['arrays']['frozen']
        pa_arr = [free, solid, frozen]

        for pa in pa_arr:
            pa.add_property('f')
            x = pa.x
            y = pa.y
            f = np.sin(x**2 + y**2)
            pa.f[:] = f

        m = free.m[0]
        h = free.h[0]
        rho = free.rho[0]
        L = B = self.L
        nl = 5
        dx = self.dx
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
                SPHApprox(dest='dest', sources=['free', 'shape', 'frozen']),
                SPHDerivativeApprox(dest='dest',
                                    sources=['free', 'shape', 'frozen']),
                SPHApprox(dest='dest1', sources=['source0']),
                SPHDerivativeApprox(dest='dest1', sources=['source0'])
            ])
        ]

        eval = SPHEvaluator(pa_arr, equations=eqs, dim=2, backend='cython')
        eval.evaluate()

        exactf = dest1.f 
        exactdf = dest1.df 
        f = dest.f
        df = dest.df

        res = os.path.join(self.output_dir, 'l2.npz')
        np.savez(res, ef=exactf, f=f, edf=exactdf, df=df)


if __name__ == '__main__':
    app = Shape2d()
    app.run()
    app.post_process(app.info_filename)
