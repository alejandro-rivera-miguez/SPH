"""
Case Sphere
"""
from __future__ import print_function
import numpy as np
import os


# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.tools.geometry import remove_overlap_particles
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group

# local imports
from particle_packing import ParticlePacking, shift_surface_inside
from jiang import JiangParticlePacking


class Sphere(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.1
        self.L = 1.5
        self.B = 1.5
        self.H = 1.5
        self.dim = 3

    def create_sphere_new(self):
        dx = self.dx
        hdx = self.hdx
        m = dx**3
        rho = 1.0
        h = hdx * dx
        L = self.L
        B = self.B
        H = self.H
        nl = 5
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
        cond = x**2 + (y/0.5)**2 + (z/0.75)**2 - 1.0 < 1e-14

        sphere = get_particle_array(
            name='sphere', x=x[cond], y=y[cond], z=z[cond], m=m, rho=rho, h=h) 
        return sphere

    def create_particles_new(self):
        hdx = self.hdx
        L = self.L
        B = self.B
        H = self.H
        free = self.scheme.scheme.create_free_particles(L, B, H=H, name='free' )
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'ellipsoid.stl')
        sphere_nodes = self.scheme.scheme.create_boundary_node(filename, shift=True,
                                                       name='sphere_nodes')
        sphere = get_particle_array(name='sphere')
        frozen = self.scheme.scheme.create_frozen_container(L, B, H=H, name='frozen')

        particles = [free, sphere, frozen, sphere_nodes]

        self.scheme.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def create_particles_jiang(self):
        hdx = self.hdx
        L = self.L
        B = self.B
        H = self.H
        free = self.scheme.scheme.create_free_particles(hdx, L, B, H=H, name='free' )
        import os
        parent = os.path.dirname(__file__)
        filename = os.path.join(parent, 'data', 'ellipsoid.stl')
        sphere_nodes = self.scheme.scheme.create_boundary_node(
            filename, hdx, shift=True, name='sphere_nodes')
        sphere_outer = self.scheme.scheme.create_boundary_node(
            filename, hdx, shift=True, name='sphere_outer', invert=True)

        sphere = self.create_sphere_new() 
        frozen = self.scheme.scheme.create_frozen_container(hdx, L, B, H=H, name='frozen')

        remove_overlap_particles(free, sphere, self.dx)

        particles = [free, sphere, frozen, sphere_nodes, sphere_outer]

        self.scheme.setup_properties(particles)
        return particles

    def create_particles(self):
        self.dx = self.scheme.scheme.dx
        if self.options.scheme == 'new':
            return self.create_particles_new()
        elif self.options.scheme == 'jiang':
            return self.create_particles_jiang()

    def create_scheme(self):
        new = ParticlePacking(
            fluids=['free'], solids={'sphere': 'sphere_nodes'},
            frozen=['frozen'], dim=self.dim,
            dx=self.dx, hdx=self.hdx)

        jiang = JiangParticlePacking(
            fluids=['free'], solids={'sphere':['sphere_nodes',
            'sphere_outer']}, frozen=['frozen'], dim=self.dim)

        s = SchemeChooser(
            default='new', new=new, jiang=jiang
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
        solid = data['arrays']['sphere']
        solid_nodes = data['arrays']['sphere_nodes']
        frozen = data['arrays']['frozen']
        if self.options.scheme == 'new':
            self.scheme.scheme.post_process(
                free, solid, solid_nodes, frozen, self.scheme.scheme.dx, res)
        else:
            self.scheme.scheme.post_process(free, solid, res)
        self.function_approx()

    def function_approx(self):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.solver.utils import load


        filename = self.output_files[-1]
        data = load(filename)
        print(filename)
        free = data['arrays']['free']
        solid = data['arrays']['sphere']
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
        dx = self.scheme.scheme.dx
        hdx = 1.0 - 25. * (dx-0.1)/4
        h = hdx * dx 
        free.h[:] = h
        frozen.h[:] = h
        solid.h[:] = h
        rho = free.rho[0]
        L = self.L
        B = self.B
        H = self.H
        nl = 10 * dx
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
                SPHApprox(dest='dest', sources=['free', 'sphere', 'frozen']),
                SPHDerivativeApprox(dest='dest',
                                    sources=['free', 'sphere', 'frozen']),
                SPHApprox(dest='dest1', sources=['source0']),
                SPHDerivativeApprox(dest='dest1', sources=['source0'])
            ])
        ]

        from pysph.base.kernels import QuinticSpline
        eval = SPHEvaluator(pa_arr, equations=eqs, dim=3, backend='cython', kernel=QuinticSpline(dim=3))
        eval.evaluate()

        exactf = dest1.f 
        exactdf = dest1.df 
        f = dest.f
        df = dest.df

        res = os.path.join(self.output_dir, 'l2.npz')
        np.savez(res, ef=exactf, f=f, edf=exactdf, df=df)


if __name__ == '__main__':
    app = Sphere()
    app.run()
    app.post_process(app.info_filename)
