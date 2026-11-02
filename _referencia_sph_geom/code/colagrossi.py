'''API for motion of points on the boundary of
the geometry'''

from pysph.sph.equation import Equation, Group
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import remove_overlap_particles
from pysph.base.kernels import CubicSpline
from itertools import combinations
from math import pi, sqrt, exp
from compyle.api import declare
import numpy


# intergrator
class InteriorStep(IntegratorStep):
    """Euler integrator for free particles
    """
    def stage1(self, d_idx, d_x, d_y, d_z, d_h, d_u, d_v,
               d_w, d_au, d_av, d_aw, dt=0.0):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_aw[d_idx]

# Equations
class SummationDensity(Equation):
    """Standard summation density

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes) 
    """
    def initialize(self, d_idx, d_V, d_rho):
        d_V[d_idx] = 0.0
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_V, d_rho, d_m, s_m, WIJ):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += s_m[s_idx]*WIJ


class NumberDensityGradient(Equation):
    """Evaluate the number density gradient

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes)
    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_m, d_rho, s_rho,
             d_au, d_av, d_aw, d_V, s_V, DWIJ, XIJ, s_m, s_p,
             d_pb):

        pi = 1.0/d_rho[d_idx]**2
        pj = 1.0/s_rho[s_idx]**2

        tmp = -d_pb[0] * d_m[d_idx] * s_m[s_idx] * (pi + pj)

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ViscousDamping(Equation):
    """Evaluate acceleration due to damping 

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes)
    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_rho, d_m, d_V,
             d_au, d_av, d_aw,
             d_u, d_v, d_w, s_m, t, d_nu):
        etai = d_nu[0] 
        d_au[d_idx] += -etai * d_u[d_idx]
        d_av[d_idx] += -etai * d_v[d_idx]
        d_aw[d_idx] += -etai * d_w[d_idx]


class ColagrossiParticlePacking(Scheme):
    """An API for a standard particle packing scheme.

    Methods
    -------
    create_frozen_container(L, B, H=0, l=5, name='frozen')
        Creates the outer container of ghost particles 

    create_free_particles(L, B, H=0, name='free')
        Creates array of free particles

    create_boundary_node(filename, scale=1.0, shift=True,
                         invert=False, name='solid_nodes', isclosed=True)
        Creates boundary node arrays 

    _is_volume_converged(pa)
        Checks for convergence 

    post_process(free, solid, solid_nodes, frozen, dx, filename)
        Method to split free particles into solid and fluid 

    post_step(particles, solver)
        methods runs post every iteration 
    """

    def __init__(
        self, fluids, solids, frozen, dim, dx=0.1, nu=None,
        pb=None, gamma=None, dfreq=100, rho0=None, hardpoints=None):
        """Parameters
        ----------

        fluids: list
            list of free particles 

        solids: dict
            dict with boundary and boundary nodes pair 

        frozen: list
            list of frozen particles arrays

        dim: int
            dimension of the problem

        dx: double
            expexted resolution 

        nu: double    
            damping constant

        pb: double
            background pressure

        gamma: double
            constant for background pressure 

        dfreq: int
            frequency to check for convergence

        rho0: double
            expected density

        hardpoints: dict
            dictionary of hard point index and correspinding normal
        """
        import numpy as np
        self.fluids = fluids
        self.solids = solids
        self.frozen = frozen
        self.solver = None
        self.dim = dim
        self.dx = dx
        self.solid_pairs = {}
        self.nu = nu
        self.pb = pb
        self.gamma = gamma 
        self.t_avg_proj = np.ones(6)
        self.old_rhodiff = 1.0
        self.hardpoints = {} if hardpoints is None else hardpoints
        self.rho0 = 1.0
        self.ang_vel = 2 * numpy.pi / 0.2
        self.radius = self.dx / 2
        self.dfreq = dfreq 
        self.surface_points = 0
        self.remove_agitator = False
        self.fz_eval = None
        self.sol_eval = None
        self.cf_eval = None
        self.check = -1.0
        self.converge = []
        if self.dim == 2:
            self.rho0 = 0.9999567333467625
        elif self.dim == 3:
            self.rho0 = 1.002791787248099

    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument
        group.add_argument(
             "--dfreq", action="store", type=float, dest="dfreq",
             default=None,
             help="particle deletion frequency."
         )
        group.add_argument(
             "--pb", action="store", type=float, dest="pb",
             default=None,
             help="Background pressure"
         )
        group.add_argument(
             "--rho0", action="store", type=float, dest="rho0",
             default=None,
             help="Density evaluate using summation density"
         )
        group.add_argument(
             "--nu", action="store", type=float, dest="nu",
             default=None,
             help="Dynamic viscosity"
         )
        group.add_argument(
             "--dx", action="store", type=float, dest="dx",
             default=None,
             help="Set particle spacing value"
         )

    def consume_user_options(self, options):
        vars = ['dfreq', 'pb', 'nu', 'dx', 'rho0']

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

        if self.pb is None:
            self.pb = 10.0 
            self.nu = sqrt(self.pb * 2.0 * self.dx**2)
        else:
            if self.nu is None:
                raise AttributeError("Pass values for nu and k")


    def create_frozen_container(self, hdx, L, B, H=0, l=5, name='frozen'):
        from particle_packing import create_frozen_container
        return create_frozen_container(
            self.dx, hdx, 1.0, L, B, H=H, l=l, dim=self.dim, name=name,
            )

    def create_free_particles(self, hdx, L, B, H=0, name='free'):
        from particle_packing import create_free_particles
        return create_free_particles(
            self.dx, hdx, 1.0, L, B, H=H, dim=self.dim, name=name,
            )

    def create_boundary_node(self, filename, hdx, scale=1.0, shift=True,
                           invert=False, name='solid_nodes', isclosed=True):
        if self.dim == 2:
            from particle_packing import create_surface_from_file
            return create_surface_from_file(
                filename, self.dx, hdx, 1.0, invert=invert, shift=shift,
                name=name, isclosed=isclosed)
        elif self.dim == 3:
            from particle_packing import create_surface_from_stl
            return create_surface_from_stl(
                filename, self.dx, hdx, 1.0, scale=scale, shift=shift,
                name=name, )


    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EulerIntegrator

        cls = EulerIntegrator
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = InteriorStep()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver( dim=self.dim, integrator=integrator,
                              kernel=kernel, n_damp=10,
                              pfreq=3000, tf=30, **kw )

    def get_equations(self):

        equations = self.get_equations_step1()
        print(equations)
        return equations

    def get_equations_step1(self):
        all = list(self.solids.keys()) + self.fluids + self.frozen
        equations = []

        g1=[]
        dest = list(self.solids.keys()) + self.frozen
        for name in self.fluids:
            g1.append(SummationDensity(dest=name, sources=all))
        for name in dest:
            g1.append(SummationDensity(dest=name, sources=self.fluids+[name]))

        dest = list(self.solids.keys())
        equations.append(Group(equations=g1, real=False))

        g3 = []
        dest = self.fluids
        for name in dest:
            if self.nu > 1e-14:
                g3.append(ViscousDamping(dest=name, sources=[]))
            g3.append(
                NumberDensityGradient(dest=name, sources=all))
        equations.append(Group(equations=g3, real=False))

        return equations

    def setup_properties(self, particles, clean=True):
        """Add following properties to the particle array

        x, y, z: coordinate of the position
        u, v, w: components of velocity
        p: pressure
        V: Volume
        h: support radius
        m: mass
        rho: density
        au, av, aw: components of acceleration
        xn, yn, zn: normal of boundary nodes
        """
        props = ['x', 'y', 'z', 'u', 'v','w', 'p', 'V', 'h', 'm', 'rho',
                 'au', 'av', 'aw', 'xn', 'yn', 'zn']
        output_props = props 
        newarr = []
        for pa in particles:
            prop_to_ensure = props.copy()
            self._ensure_properties(pa, prop_to_ensure, clean=False)
            pa.add_constant('nu', self.nu)
            pa.add_constant('pb', self.pb)
            pa.set_output_arrays(output_props)
        particles.extend(newarr)

    def _is_volume_converged(self, pa):
        import numpy as np
        u = pa.u
        v = pa.v
        w = pa.w
        h = pa.h[0]
        vel = np.sqrt(u**2 + v**2 + w**2)
        maxvel = max(vel)
        rel_dist = maxvel * self.solver.dt / h * 100
        self.converge.append([rel_dist, self.solver.t])
        print('Convergence = ', rel_dist)
        if (rel_dist - 0.01 < 1e-14) and (len(self.converge)>10):
            self.solver.tf = self.solver.t

    def post_process(self, free, solid, filename):
        import numpy as np

        xfluid = free.x
        yfluid = free.y
        zfluid = free.z
        vfluid = free.V
        rfluid = free.rho

        xsolid = solid.x
        ysolid = solid.y
        zsolid = solid.z
        vsolid = solid.V
        rsolid = solid.rho

        np.savez(
            filename, xs=xsolid, ys=ysolid, zs=zsolid, vs=vsolid, rs=rsolid,
            xf=xfluid, yf=yfluid, zf=zfluid, vf=vfluid, rf=rfluid)

    def post_step(self, particles, solver):
        import numpy as np
        pa_fluid = None
        dfreq = self.dfreq
        if self.solver.count % dfreq == 0:
            for pa in particles:
                if self.fluids[0] == pa.name:
                    pa_fluid = pa

            self._is_volume_converged(pa_fluid)
