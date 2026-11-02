"""
Case uav
"""
from __future__ import print_function
import numpy as np
import os


# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.equation import Group

# local imports
from particle_packing import ParticlePacking, shift_surface_inside
from case_3d_bounded_sphere import Sphere


class Bunny(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.02
        self.L = 0.6
        self.B = 0.6
        self.H = 0.75
        self.dim = 3

    def create_particles(self):
        self.dx = self.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        H = self.H
        free = self.scheme.create_free_particles(L , B, H=H, name='free' )
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'bun_zipper_res2.stl')
        bunny_nodes = self.scheme.create_boundary_node(filename, scale=10.0, shift=False,
                                                       name='bunny_nodes')
        bunny = get_particle_array(name='bunny')
        frozen = self.scheme.create_frozen_container(L, B, H=H, name='frozen')

        particles = [free, bunny, frozen, bunny_nodes]

        self.scheme.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def create_scheme(self):
        s = ParticlePacking(
            fluids=['free'], solids={'bunny': 'bunny_nodes'},
            frozen=['frozen'],
            dim=self.dim, dx=self.dx, hdx=self.hdx)

        s.configure_solver(dt=1e-5)
        return s

    def post_step(self, solver):
        self.scheme.post_step(self.particles, solver)

    def post_process(self, info_fname):
        import os
        from pysph.solver.utils import load
        import numpy as np
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        res = os.path.join(self.output_dir, 'results.npz')
        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['bunny']
        solid_nodes = data['arrays']['bunny_nodes']
        frozen = data['arrays']['frozen']
        self.scheme.post_process(
            free, solid, solid_nodes, frozen, self.scheme.dx, res)
        self.function_approx()

    def function_approx(self):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.solver.utils import load


        filename = self.output_files[-1]
        print(filename)
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['bunny']
        frozen = data['arrays']['frozen']
        pa_arr = [free, solid, frozen]

        for pa in pa_arr:
            pa.add_property('f')
            x = pa.x
            y = pa.y
            z = pa.z
            f = np.sin(x**2 + y**2 + z**2)
            pa.f[:] = f

        m = free.m[0]
        h = free.h[0]
        rho = free.rho[0]
        L = self.L
        B = self.B
        H = self.H
        nl = 5
        dx = self.dx
        x0, y0, z0 = np.mgrid[-L - nl + dx:L + nl:2*dx,
                              -B - nl + dx/2:B + nl:dx,
                              -H - nl + dx/2:H + nl:dx]
        x1, y1, z1 = np.mgrid[-L - nl:L + nl:2*dx,
                              -B - nl:B + nl:dx,
                              -H - nl:H + nl:dx]
        x0, y0, z0 = [t.ravel() for t in (x0, y0, z0)]
        x1, y1, z1 = [t.ravel() for t in (x1, y1, z1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))
        source0 = get_particle_array(
            name='source0', x=x, y=y, z=z, m=m, h=h, rho=rho, f=0, V=0)

        source0.f = np.sin(x**2 + y**2 + z**2)

        x, y, z = np.mgrid[-L:L:20j, -B:B:20j, -B:B:20j]
        x, y, z = [t.ravel() for t in (x, y, z)]
        dest = get_particle_array(
            name='dest', x=x, y=y, z=z, m=m, h=h, rho=rho, f=0, df=0)
        dest1 = get_particle_array(
            name='dest1', x=x, y=y, z=z, m=m, h=h, rho=rho, f=0, df=0)

        pa_arr.extend([dest, dest1, source0])

        from particle_packing import SPHApprox, SPHDerivativeApprox, SummationDensity
        eqs = [
            Group(equations=[
                SummationDensity(dest='source0', sources=['source0'])
            ]),
            Group(equations=[
                SPHApprox(dest='dest', sources=['free', 'bunny', 'frozen']),
                SPHDerivativeApprox(dest='dest',
                                    sources=['free', 'bunny', 'frozen']),
                SPHApprox(dest='dest1', sources=['source0']),
                SPHDerivativeApprox(dest='dest1', sources=['source0'])
            ])
        ]

        eval = SPHEvaluator(pa_arr, equations=eqs, dim=3, backend='cython')
        eval.evaluate()

        exactf = dest1.f 
        exactdf = dest1.df 
        f = dest.f
        df = dest.df

        res = os.path.join(self.output_dir, 'l2.npz')
        np.savez(res, ef=exactf, f=f, edf=exactdf, df=df)


if __name__ == '__main__':
    app = Bunny()
    app.run()
    # app.post_process(app.info_filename)
