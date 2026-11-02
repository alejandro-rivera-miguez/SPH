"""
2d test periodic with solid cylinder with external boundary
"""
from __future__ import print_function
import numpy as np
import os
from math import sqrt
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
    shift_surface_inside
    )


class Cylinder2d(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.1
        self.d = 2.0
        self.L = 1.5
        self.B = 1.5
        self.dim = 2

    def _create_cylinder_new(self):
        dx = self.dx
        hdx = self.hdx
        r = self.d / 2
        rho = 1.0
        m = dx*dx*rho
        h = hdx*dx
        theta = np.mgrid[0:2*np.pi:self.dx]
        boundary = [0.5*self.d*np.cos(theta), 0.5*self.d*np.sin(theta)]
        xn, yn, x, y, area = calculate_normal_2d_surface(boundary, shift=dx/2)
        n_pnts = sum(area)/dx
        cylinder = get_particle_array(x=x, y=y, m=m, rho=rho, h=h,
                                      name='cylinder_nodes', xn=xn, yn=yn, zn=0,
                                      area=area)
        cylinder.add_constant('n_pnts', n_pnts)
        return cylinder

    def _create_cylinder_jiang(self):
        dx = self.dx
        hdx = self.hdx
        rho = 1.0
        m = dx*dx*rho
        h = hdx*dx
        theta = np.mgrid[0:2*np.pi:self.dx]
        boundary = [0.5*self.d*np.cos(theta), 0.5*self.d*np.sin(theta)]
        xn, yn, x, y, area = calculate_normal_2d_surface(boundary, dx/2)
        cylinder = get_particle_array(x=x, y=y, m=m, rho=rho, h=h,
                                      name='cylinder_nodes', xn=xn, yn=yn, zn=0,
                                      area=area)
        return cylinder

    def _create_cylinder_outer_jiang(self):
        dx = self.dx
        hdx = self.hdx
        rho = 1.0
        m = dx*dx*rho
        h = hdx*dx
        theta = np.mgrid[0:2*np.pi:self.dx]
        boundary = [0.5*self.d*np.cos(theta), 0.5*self.d*np.sin(theta)]
        xn, yn, x, y, area = calculate_normal_2d_surface(boundary, -dx/2)
        cylinder = get_particle_array(x=x, y=y, m=m, rho=rho, h=h,
                                      name='cylinder_outer', xn=-xn, yn=-yn, zn=0,
                                      area=area)
        return cylinder

    def cylinder_jiang(self, dx, hdx, rho, L, B, H=0, dim=2, name='free'):
        import numpy as np
        h = hdx*dx
        m = rho*dx**dim
        x0, y0 = np.mgrid[-L+dx: L: 2 *dx, -B: B+dx/2: dx]
        x1, y1 = np.mgrid[-L: L+dx: 2 *dx, -B+dx/2: B: dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
        cond = x**2 + y**2 + z**2 - (1.0)**2 < 1e-14
        x = x[cond]
        y = y[cond]
        z = z[cond]

        free = get_particle_array(
            x=x, y=y, z=z, m=m, rho=rho, h=h, name=name)
        return free

    def cylinder_colagrossi(self, dx, hdx, rho, L, B, H=0, dim=2, name='free'):
        import numpy as np
        h0 = hdx * dx
        x = [0.0]
        y = [0.0]
        r = dx/2
        nt = 0
        while r - self.d / 2 < 0.00001:
            nnew = int(np.pi*r**2/dx**2 + 0.5)
            tomake = nnew-nt
            theta = np.linspace(0., 2.*np.pi, tomake + 1)
            for t in theta[:-1]:
                x.append(r*np.cos(t))
                y.append(r*np.sin(t))
            nt = nnew
            r = r + dx
        x = np.array(x)
        y = np.array(y)
        volume = dx*dx
        solid = get_particle_array(
            name=name, x=x, y=y, m=volume*rho, rho=rho, h=h0)
        return solid

    def create_free_particles_colagrossi(self, dx, hdx, rho, L, B, H=0, dim=2, name='free'):
        import numpy as np
        h = hdx*dx
        m = rho*dx**dim
        x0, y0 = np.mgrid[-L+dx: L: 2 *dx, -B: B+dx/2: dx]
        x1, y1 = np.mgrid[-L: L+dx: 2 *dx, -B+dx/2: B: dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
        cond = x**2 + y**2 + z**2 - (1.0)**2 > 1e-14
        x = x[cond]
        y = y[cond]
        z = z[cond]
        # temporary fix for cylinder
        free = get_particle_array(x=x, y=y, z=z, m=m, rho=rho, h=h,
                                        name=name)
        return free


    def create_particles_colagrossi(self):
        self.dx = self.scheme.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.create_free_particles_colagrossi(self.dx, hdx, 1.0, L , B, name='free')
        cylinder_bound = self._create_cylinder_jiang()
        cylinder = self.cylinder_colagrossi(self.dx, hdx, 1.0, L, B, name='cylinder')
        frozen = self.scheme.scheme.create_frozen_container(hdx, L, B, name='frozen', l=7)

        particles = [free, cylinder, frozen, cylinder_bound]

        self.scheme.setup_properties(particles)

        return particles

    def create_particles_jiang(self):
        self.dx = self.scheme.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.scheme.scheme.create_free_particles(hdx, L , B, name='free')
        cylinder_bound = self._create_cylinder_jiang()
        cylinder_outer = self._create_cylinder_outer_jiang()
        cylinder = self.cylinder_jiang(self.dx, hdx, 1.0, L, B, name='cylinder')
        frozen = self.scheme.scheme.create_frozen_container(hdx, L, B, name='frozen', l=7)
        remove_overlap_particles(free, cylinder, self.dx, dim=2)

        particles = [free, cylinder, frozen, cylinder_bound, cylinder_outer]

        self.scheme.setup_properties(particles)

        return particles

    def create_particles_new(self):
        self.dx = self.scheme.scheme.dx
        hdx = self.hdx
        L = self.L
        B = self.B
        free = self.scheme.scheme.create_free_particles(L , B, name='free' )
        cylinder_bound = self._create_cylinder_new()
        cylinder = get_particle_array(name='cylinder')
        frozen = self.scheme.scheme.create_frozen_container(L, B, name='frozen')

        particles = [free, cylinder, frozen, cylinder_bound]

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
        new = ParticlePacking(
            fluids=['free'], solids={'cylinder':'cylinder_nodes'},
            frozen=['frozen'], dim=self.dim, hdx=self.hdx,
            use_prediction=False, filter_layers=False, reduce_dfreq=False)

        colagrossi = ColagrossiParticlePacking(
            fluids=['free'], solids={'cylinder':'cylinder_nodes'},
            frozen=['frozen'], dim=self.dim)

        jiang = JiangParticlePacking(
            fluids=['free'], solids={'cylinder':['cylinder_nodes',
            'cylinder_outer']}, frozen=['frozen'], dim=self.dim)

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
        name = 'cylinder_' + self.options.scheme + '.npz'
        dir = os.path.join(os.path.split(self.output_dir)[0], 'preprocess')
        print(dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        newfile = os.path.join(dir , name)
        res = os.path.join(self.output_dir, 'results.npz')
        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['cylinder']
        solid_nodes = data['arrays']['cylinder_nodes']
        frozen = data['arrays']['frozen']
        if self.options.scheme == 'new':
            self.scheme.scheme.post_process(
                free, solid, solid_nodes, frozen, self.scheme.scheme.dx, res)
        else:
            self.scheme.scheme.post_process(free, solid, res)

        from shutil import copyfile
        copyfile(res, newfile)
        self.function_approx()

    def function_approx(self):
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.solver.utils import load

        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        solid = data['arrays']['cylinder']
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
        solid.h[:] = h
        frozen.h[:] = h
        rho = free.rho[0]
        L = B = self.L
        r = self.d / 2
        nl = 10 * dx
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
        cond = x**2 + y**2 - (r-dx)**2 > 1e-14
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
                SPHApprox(dest='dest', sources=['free', 'cylinder', 'frozen']),
                SPHDerivativeApprox(dest='dest',
                                    sources=['free', 'cylinder', 'frozen']),
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
    app = Cylinder2d()
    app.run()
    app.post_process(app.info_filename)
