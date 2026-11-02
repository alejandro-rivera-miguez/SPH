"""
2d test periodic only fluid
"""
from __future__ import print_function
import numpy as np
import os
from math import sqrt


# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.base.nnps import DomainManager
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator, IntegratorStep
from pysph.sph.integrator_step import EulerStep
from pysph.sph.equation import Group, Equation
from pysph.solver.application import Application
from pysph.sph.scheme import add_bool_argument
from pysph.tools.geometry import remove_overlap_particles


class Case2dPeriodic(Application):
    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.05
        self.rho = 1.0
        self.dim = 2
        self.pb = 2000
        self.k = 1.0
        self.nu = -0.0
        self.L = 0.5
        self.method = 'ND'
        self.shape = 'rect'
        self.rad = 0.95/1.2
        self.id = 0 

    def add_user_options(self, group):
        group.add_argument(
            "--method", action="store", type=str, dest="method", default='ND',
            help="Method for particle acceleration, available options are 'ND', ND_RF'"
        )
        group.add_argument(
            "--nu", action="store", type=float, dest="nu", default='0.05',
            help="dynamics viscosity"
        )
        group.add_argument(
            "--rad", action="store", type=float, dest="rad", default='1.2',
            help="influence of repulsion times dx"
        )
        group.add_argument(
            "--pb", action="store", type=float, dest="pb", default='1000',
            help="background pressure"
        )
        group.add_argument(
            "--k", action="store", type=float, dest="k", default='1',
            help="repulsion constant"
        )
        group.add_argument(
            "--shape", action="store", type=str, dest="shape", default='rect',
            help="shape of lattice 'hexa', 'rect'"
        )
        group.add_argument(
            "--dx", action="store", type=float, dest="dx", default=0.1,
            help="dx"
        )
        group.add_argument(
            "--fac", action="store", type=float, dest="fac", default=0.5,
            help="factor of distance"
        )

        add_bool_argument(
            group, "agitate", dest="agitate",
            default='None', help="If True then agitation will be done"
        )

    def consume_user_options(self):
        self.method = self.options.method
        self.shape = self.options.shape
        self.agitate = self.options.agitate
        dx = self.dx = self.options.dx
        self.pb = 1.0
        self.k = 0.004 * dx
        self.nu = .2/dx 
        self.dt = 0.1 * self.hdx * dx 
        self.fac = self.options.fac
        print(self.pb, self.k, self.nu, self.dt)

    def _create_free(self):
        dx = self.dx
        hdx = self.hdx
        rho = self.rho
        L = self.L
        m = dx*dx*rho
        h = hdx*dx
        x, y = np.mgrid[-L+dx/2: L: dx, -L+dx/2: L: dx]
        x, y = [t.ravel() for t in (x, y)]
        dist = np.sqrt(x**2 + y**2)
        self.id = id = np.where(dist == min(dist))[0][-1]
        if not self.agitate:
            x[id] += self.fac * dx
            y[id] += self.fac * dx
        free = get_particle_array(x=x, y=y, m=m, rho=rho, h=h,
                                      name='free')
        return free

    def _create_frozen(self):
        dx = self.dx
        hdx = self.hdx
        rho = self.rho
        L = self.L
        m = dx*dx*rho
        h = hdx*dx
        nl = 7.0 * dx
        x, y = np.mgrid[-L-nl+dx/2: L+nl: dx, -L-nl+dx/2: L+nl: dx]
        x, y = [t.ravel() for t in (x, y)]
        frozen = get_particle_array(x=x, y=y, m=m, rho=rho, h=h,
                                      name='frozen')
        return frozen

    def _create_free_hexa(self):
        from particle_packing import create_free_particles
        dx = self.dx
        hdx = self.hdx
        rho = self.rho
        L = self.L
        free = create_free_particles(dx, hdx, rho, L, L)
        x = free.x
        y = free.y
        dist = np.sqrt(x**2 + y**2)
        self.id = id = np.where(dist == min(dist))[0][-1]
        if not self.agitate:
            free.x[id] += self.fac * dx
            free.y[id] += self.fac * dx
        return free

    def _create_frozen_hexa(self):
        from particle_packing import create_frozen_container
        dx = self.dx
        hdx = self.hdx
        rho = self.rho
        L = self.L
        frozen = create_frozen_container(dx, hdx, rho, L, L, l=7)
        return frozen

    def _get_agitator(self, free):
        x = free.x
        y = free.y
        z = free.z
        dist = np.sqrt(x**2 + y**2 + z**2)
        id = np.where(dist == min(dist))[0][-1]
        pa_add = free.extract_particles([id])
        free.remove_particles([id])
        x0 = pa_add.x.copy()
        y0 = pa_add.y.copy()
        agitator = get_particle_array(name = 'agitator',
                                   **pa_add.get_property_arrays(),
                                   x0=x0, y0=y0)

        return agitator

    def create_particles(self):
        if self.shape == 'rect':
            free = self._create_free()
            frozen = self._create_frozen()
        elif self.shape == 'hexa':
            free = self._create_free_hexa()
            frozen = self._create_frozen_hexa()

        remove_overlap_particles(frozen, free, self.dx, dim=2)
        # dx = self.dx
        # np.random.seed(10)
        # free.x += (np.random.random(len(free.x)) - 0.5) * 2* self.fac * dx
        # free.y += (np.random.random(len(free.x)) - 0.5) * 2* self.fac * dx

        particles = [free, frozen]
        if self.agitate == True:
            agitator = self._get_agitator(free)
            particles.append(agitator)


        props = ['x', 'y', 'z', 'u', 'v','w', 'p', 'V', 'h', 'm',
                 'rho', 'interior', 'au', 'av', 'aw', 'gid',
                 'pid', 'tag', 'V', 'dt_adapt', 'rho_orig', 'inid', 'E',
                 'xn', 'yn', 'zn', 'dmin', 'nearest', 'sol_neartag',
                 'farthest', 'vmag', 'x0', 'y0', 'z0']
        for prop in props:
            for pa in particles:
                pa.add_property(prop)

        for pa in particles:
            pa.add_constant('visc', 1.0)
            pa.add_constant('addvel', 1.0)
            pa.add_constant('split', 1.0)
            pa.add_constant('stopwatch', 1.0)
            pa.add_constant('nu', self.nu)
            pa.add_constant('pb', self.pb)
            pa.add_constant('k', self.k)
            pa.add_constant('rad', self.dx*0.75)
        output_array = ['x', 'y', 'z', 'u', 'v','w', 'p', 'V', 'h', 'm',
                        'E', 'xn' , 'yn', 'zn', 'vmag', 'au', 'av', 'aw']
        for pa in particles:
            pa.add_output_arrays(output_array)

        return particles

    def create_solver(self):
        from particle_packing import InteriorStep
        kernel = QuinticSpline(dim=self.dim)

        integrator = EulerIntegrator(free=InteriorStep())
        dt = self.dt
        tf = 0.2
        pfreq = 10
        solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                        dt=dt, tf=tf, pfreq=pfreq)

        return solver

    def _number_density_eqns(self):
        from particle_packing import (
            NumberDensityGradient, SummationDensity,
            ViscousDamping)

        all = ['free', 'frozen']
        if self.agitate:
            all.append('agitator')

        eqns = []

        g1 =[SummationDensity(dest='free', sources=all),
             SummationDensity(dest='frozen', sources=all)]
        if self.agitate:
            g1.append(SummationDensity(dest='agitator', sources=all))
        eqns.append(Group(g1))

        g2 = Group(
            equations=[
                NumberDensityGradient(dest='free', sources=all),
                ViscousDamping(dest='free', sources=[]),
            ], real=False)
        eqns.append(g2)

        return eqns

    def _number_density_LJ_eqns(self):

        from particle_packing import (
            NumberDensityGradient, SummationDensity,
            ViscousDamping, RepulsionForce)

        all = ['free', 'frozen']
        if self.agitate:
            all.append('agitator')

        eqns = []

        g1 =[SummationDensity(dest='free', sources=all),
             SummationDensity(dest='frozen', sources=all)]
        if self.agitate:
            g1.append(SummationDensity(dest='agitator', sources=all))
        eqns.append(Group(g1))

        g2 = Group(
            equations=[
                NumberDensityGradient(dest='free', sources=all),
                RepulsionForce(dest='free', sources=all, hdx=self.rad),
                ViscousDamping(dest='free', sources=all),
            ], real=False)
        eqns.append(g2)

        return eqns

    def create_equations(self):
        if self.method == 'ND':
            return self._number_density_eqns()
        elif self.method == 'ND_RF':
            return self._number_density_LJ_eqns()

    def post_step(self, solver):
        if self.agitate == False:
            return
        pa_agitator = None
        pa_free = None
        for pa in self.particles:
            if pa.name == 'agitator':
                pa_agitator = pa
            elif pa.name == 'free':
                pa_free = pa
        pa_agitator.stopwatch[0] += solver.dt
        if solver.t > 1.0:
            if pa_agitator.get_number_of_particles() == 0:
                return
            x = pa_agitator.x
            ids = np.where(x < 1e14)[0]
            pa_agitator.extract_particles(ids, pa_free)
            pa_agitator.remove_particles(ids)
            self.id = len(pa_free.x) - 1

    def post_process(self, info_fname):
        import os
        from pysph.solver.utils import load
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        res = os.path.join(self.output_dir, 'results.png')
        filename = self.output_files
        dx = self.dx
        L = self.L
        N = self.id
        xs = []
        ts = []
        for files in filename:
            data = load(files)
            free = data['arrays']['free']
            t = data['solver_data']['t']
            print(free.au[N])
            xs.append(free.x[N])
            ts.append(t) 


if __name__ == '__main__':
    app = Case2dPeriodic()
    app.run()
    app.post_process(app.info_filename)
