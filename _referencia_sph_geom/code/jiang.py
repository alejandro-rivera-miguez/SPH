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

FLIP_TIME = 6.0 # time at which the internal and external are coupled

#kernel
def akinci_kernel(dim=2, rij=1.0, h=1.0):
    """Kernel proposed by Akinci for cohesion force

    Parameters:

    dim: dimension of the problem
    rij: distance from the ith particle
    h: support radius

    Return:

    val*fac: the kernel value
    """
    fac = 32. / pi 
    h1 = 1. / h
    q = rij * h1

    # get the kernel normalizing factor
    if dim == 1:
        fac = fac * h1
    elif dim == 2:
        fac = fac * h1 * h1
    elif dim == 3:
        fac = fac * h1 * h1 * h1

    tmp2 = (1. - q)**3 * q**3
    if (q > 1.0):
        val = 0.0

    elif (q > 0.5):
        val = 2 * tmp2 - 1.0**6/64 
    else:
        val = tmp2 

    return val * fac

# intergrator
class InteriorStep(IntegratorStep):
    """Euler integrator for free particles
    """
    def stage1(self, d_idx, d_x, d_y, d_z, d_h, d_u, d_v,
               d_w, d_au, d_av, d_aw, d_xn, d_yn, d_zn,
               dt=0.0):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_aw[d_idx]


class SolidStep(InteriorStep):
    """Euler integrator for boundary particles
    """
    def stage1(self, d_idx, d_x, d_y, d_z, d_h, d_u, d_v,
               d_w, d_au, d_av, d_aw, d_proj, d_xn, d_yn, d_zn, d_hard,
               dt=0.0):
        if d_hard[d_idx] < 0.5:
            V = (d_u[d_idx]*d_xn[d_idx] + d_v[d_idx]*d_yn[d_idx] +
                d_w[d_idx]*d_zn[d_idx])
            if (abs(d_proj[d_idx]) - d_h[d_idx] * 0.05 < 1e-14) and (V>1e-14): 
                d_x[d_idx] = d_x[d_idx] + dt * (d_u[d_idx] - d_xn[d_idx] * V)
                d_y[d_idx] = d_y[d_idx] + dt * (d_v[d_idx] - d_yn[d_idx] * V)
                d_z[d_idx] = d_z[d_idx] + dt * (d_w[d_idx] - d_zn[d_idx] * V)
            else:
                d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
                d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
                d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

            d_u[d_idx] = d_u[d_idx] + dt * d_au[d_idx]
            d_v[d_idx] = d_v[d_idx] + dt * d_av[d_idx]
            d_w[d_idx] = d_w[d_idx] + dt * d_aw[d_idx]


def _condition_post(t, dt):
    """condition for equations after coupling
    """
    if t > FLIP_TIME:
        return True
    else:
        return False


def _condition_pre(t, dt):
    """condition for equations before coupling
    """
    if t > FLIP_TIME:
        return False
    else:
        return True


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


class CohesionForce(Equation):
    """Evaluate the acceleration due to cohesion force 

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        Self
    """
    def _get_helpers_(self):
        return [akinci_kernel]

    def __init__(self, dest, sources, gamma, dim):
        self.gamma = gamma
        self.dim = dim
        super(CohesionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho,
             d_au, d_av, d_aw, d_V, s_V, XIJ, RIJ, HIJ,
             s_m):
        wij = akinci_kernel(self.dim, RIJ, HIJ*2.0)
        k = 2.0 / (d_rho[d_idx] + s_rho[s_idx])
        const = -self.gamma * d_m[d_idx] * s_m[s_idx] * k * wij / (RIJ + 1e-6)

        d_au[d_idx] += const * XIJ[0]
        d_av[d_idx] += const * XIJ[1]
        d_aw[d_idx] += const * XIJ[2]


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


class FindNearestNode(Equation):
    """Find nearest boundary node

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        Boundary nodes array
    """
    def initialize(self, d_idx, d_nearest, d_neartag, d_proj, d_xn,
                   d_yn, d_zn):
        d_nearest[d_idx] = 10000.0
        d_neartag[d_idx] = -1
        d_proj[d_idx] = -1000.0
        d_xn[d_idx] = 0.0
        d_yn[d_idx] = 0.0
        d_zn[d_idx] = 0.0

    def loop(self, d_idx, s_idx, RIJ, d_nearest, d_neartag, t, s_hard,
             d_xn, d_yn, d_zn, s_xn, s_yn, s_zn, XIJ, d_proj):
        proj = XIJ[0]*s_xn[s_idx] + XIJ[1]*s_yn[s_idx] + XIJ[2]*s_zn[s_idx]
        if (RIJ < d_nearest[d_idx]) and (s_hard[s_idx] < 0.5):
            d_nearest[d_idx] = RIJ
            d_proj[d_idx] = proj
            d_neartag[d_idx] = s_idx
            d_xn[d_idx] = s_xn[s_idx]
            d_yn[d_idx] = s_yn[s_idx]
            d_zn[d_idx] = s_zn[s_idx]


class ProjectNode(Equation):
    """project boundary/free particles back to boundary 

    Parameters:

    dest: pysph.base.utils.ParticleArray 
        Free or boundary particle 

    sources: list of pysph.base.utils.ParticleArray
        Boundary nodes array
    """
    def loop(self, d_idx, s_idx, d_neartag, d_x, d_y, d_z, s_xn, s_yn,
             s_zn, XIJ, d_proj, d_h, t, dt, d_hard):
        pfreq = declare('int')
        if d_hard[d_idx] < 0.5:
            pfreq = int(t / dt)
            if (t < 1e-14):
                if s_idx == d_neartag[d_idx]:
                    proj = XIJ[0]*s_xn[s_idx] + XIJ[1]*s_yn[s_idx] + XIJ[2]*s_zn[s_idx]
                    if proj > 0.0:
                        d_x[d_idx] -= 2.5 * s_xn[s_idx] * proj
                        d_y[d_idx] -= 2.5 * s_yn[s_idx] * proj
                        d_z[d_idx] -= 2.5 * s_zn[s_idx] * proj
            else:
                if abs(d_proj[d_idx]) - d_h[d_idx] * 0.05 < 1e-14:
                    if s_idx == d_neartag[d_idx]:
                        proj = XIJ[0]*s_xn[s_idx] + XIJ[1]*s_yn[s_idx] + XIJ[2]*s_zn[s_idx]
                        d_x[d_idx] -= s_xn[s_idx] * proj
                        d_y[d_idx] -= s_yn[s_idx] * proj
                        d_z[d_idx] -= s_zn[s_idx] * proj
                elif d_proj[d_idx] > 1e-14:
                    if s_idx == d_neartag[d_idx]:
                        proj = XIJ[0]*s_xn[s_idx] + XIJ[1]*s_yn[s_idx] + XIJ[2]*s_zn[s_idx]
                        d_x[d_idx] -= s_xn[s_idx] * proj
                        d_y[d_idx] -= s_yn[s_idx] * proj
                        d_z[d_idx] -= s_zn[s_idx] * proj



class JiangParticlePacking(Scheme):
    """An API for a hybrid particle packing scheme.

    Methods
    -------
    create_frozen_container(L, B, H=0, l=5, name='frozen')
        Creates the outer container of ghost particles 

    create_free_particles(L, B, H=0, name='free')
        Creates array of free particles

    create_agitators(free, solid_nodes, L, B, H=0, name='agitator')
        Creates agitator particles [Not used] 

    create_boundary_node(filename, scale=1.0, shift=True,
                         invert=False, name='solid_nodes', isclosed=True)
        Creates boundary node arrays 

    _is_volume_converged(pa)
        Checks for convergence 

    post_process(free, solid, solid_nodes, frozen, dx, filename)
        Method to split free particles into solid and fluid 

    setup_hardpoints(pa_solid_nodes, pa_fluid, pa_solid)
        Sets a free particle near to a hard point on the boundary node 

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
            constant for cohesive force 

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
        self.nu = nu
        self.pb = pb
        self.gamma = gamma 
        self.hardpoints = {} if hardpoints is None else hardpoints
        self.rho0 = 1.0
        self.dfreq = dfreq 
        self.converge = []
        if self.dim == 2:
            self.rho0 = 0.9999567333467625
        elif self.dim == 3:
            self.rho0 = 1.002791787248099
        print(self.dim, self.dx, self.nu)

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
             "--gamma", action="store", type=float, dest="gamma",
             default=None,
             help="Spring Constant"
         )
        group.add_argument(
             "--dx", action="store", type=float, dest="dx",
             default=None,
             help="Set particle spacing value"
         )

    def consume_user_options(self, options):
        vars = ['dfreq', 'pb', 'nu', 'gamma', 'dx', 'rho0']

        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

        if self.pb is None:
            self.pb = 10.0 
        if self.gamma is None:
            self.gamma =  20.
        if self.nu is None:
            self.nu = 1.0  

        print(self.pb, self.nu, self.gamma)

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
                name=name, isclosed=isclosed, hard=self.hardpoints)
        elif self.dim == 3:
            from particle_packing import create_surface_from_stl
            return create_surface_from_stl(
                filename, self.dx, hdx, 1.0, scale=scale, shift=shift,
                name=name, invert=invert)


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
                steppers[name] = SolidStep()

        for name in self.solids:
            if name not in steppers:
                steppers[name] = SolidStep()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        dt = 1e-3 
        print('time step = ', dt)
        self.solver = Solver( dim=self.dim, integrator=integrator,
                              kernel=kernel, n_damp=10,
                              pfreq=3000, tf=30, dt=dt, **kw )

    def get_equations(self):

        equations = self.get_equations_step1()
        equations.extend(self.get_equations_step2())
        print(equations)
        return equations

    def get_equations_step1(self):
        solids = list(self.solids.keys())
        fluids = self.fluids 
        frozen = self.frozen
        equations = []

        g1=[]
        sources = frozen + solids
        for name in solids:
            g1.append(SummationDensity(dest=name, sources=sources))
            g1.append(FindNearestNode(
                dest=name, sources=[self.solids[name][0]]))
            if self.hardpoints:
                from particle_packing import FindNearestNodeToHardPoint
                g1.append(FindNearestNodeToHardPoint(
                    dest=self.solids[name][0], sources=[name]))

        sources = fluids + frozen
        for name in fluids:
            g1.append(SummationDensity(dest=name, sources=sources))
            for solid in solids: 
                g1.append(FindNearestNode(dest=name, sources=[self.solids[solid][1]]))
                if self.hardpoints:
                    from particle_packing import FindNearestNodeToHardPoint
                    g1.append(FindNearestNodeToHardPoint(
                        dest=self.solids[solid][1], sources=[name]))
        for name in frozen:
            g1.append(SummationDensity(dest=name, sources=sources))
        equations.append(Group(equations=g1, real=False, condition=_condition_pre))

        g2=[]
        for name in solids:
            g2.append(ProjectNode(dest=name, sources=[self.solids[name][0]]))
        for name in fluids:
            for solid in solids: 
                g2.append(ProjectNode(dest=name, sources=[self.solids[solid][1]]))
        equations.append(Group(equations=g2, real=False, condition=_condition_pre))

        g3 = []
        sources = frozen + solids
        for name in solids:
            if self.nu > 1e-14:
                g3.append(ViscousDamping(dest=name, sources=[]))
            g3.append(
                NumberDensityGradient(dest=name, sources=sources))
            g3.append(
                CohesionForce(dest=name, sources=sources, gamma=self.gamma,
                              dim=self.dim))
        sources = fluids + frozen
        for name in fluids:
            if self.nu > 1e-14:
                g3.append(ViscousDamping(dest=name, sources=[]))
            g3.append(
                NumberDensityGradient(dest=name, sources=sources))
            g3.append(
                CohesionForce(dest=name, sources=sources, gamma=self.gamma,
                              dim=self.dim))
        equations.append(Group(equations=g3, real=False, condition=_condition_pre))
        return equations

    def get_equations_step2(self):
        all = list(self.solids.keys()) + self.fluids + self.frozen
        equations = []

        g1=[]
        dest = list(self.solids.keys()) + self.fluids
        for name in all:
            g1.append(SummationDensity(dest=name, sources=all))
        dest = list(self.solids.keys())
        for name in dest:
            g1.append(FindNearestNode(dest=name, sources=[self.solids[name][0]]))
        # for name in self.fluids:
        #     g1.append(FindNearestNodeNot(dest=name, sources=[self.solids[dest[0]][1]]))
        equations.append(Group(equations=g1, real=False, condition=_condition_post))

        g2=[]
        for name in dest:
            g2.append(ProjectNode(dest=name, sources=[self.solids[name][0]]))
        # for name in self.fluids:
        #     g2.append(ProjectNodeNot(dest=name, sources=[self.solids[dest[0]][1]]))
        equations.append(Group(equations=g2, real=False, condition=_condition_post))

        g3 = []
        dest = list(self.solids.keys()) + self.fluids
        for name in dest:
            if self.nu > 1e-14:
                g3.append(ViscousDamping(dest=name, sources=[]))
            g3.append(
                NumberDensityGradient(dest=name, sources=all))
        equations.append(Group(equations=g3, real=False, condition=_condition_post))

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
        interior: 1 of interior particle
        au, av, aw: components of acceleration
        area: area of the boundary nodes
        hard: 1 if the particle is hard pint
        xc, yc, zc: centroid of boundary nodes
        xn, yn, zn: normal of boundary nodes
        dmin: minimum distance value
        nearest: distance of nearest particle
        proj: projected distance
        neartag: id of nearest particle
        fartag: id of farthest particle
        nu: damping constant
        pb: background pressure
        """
        props = [
            'x', 'y', 'z', 'u', 'v','w', 'p', 'V', 'h', 'm', 'rho',
            'interior', 'au', 'av', 'aw', 'V', 'area', 'hard', 'xc', 'yc',
            'zc', 'xn', 'yn', 'zn', 'nearest', 'proj']
        output_props = [
            'x', 'y', 'z', 'u', 'v', 'w', 'interior', 'V', 'rho', 'xn', 'yn',
            'zn', 'm', 'au', 'av', 'aw', 'h', 'neartag', 'nearest']

        newarr = []
        for pa in particles:
            prop_to_ensure = props.copy()
            self._ensure_properties(pa, prop_to_ensure, clean=False)
            pa.add_property('neartag', type='int')
            pa.add_property('fartag', type='int')
            pa.add_constant('nu', self.nu)
            pa.add_constant('pb', self.pb)
            pa.set_output_arrays(output_props)
        particles.extend(newarr)
        print(self.pb, self.nu, self.gamma)

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

    def setup_hardpoints(self, pa_nodes, pa):
        if self.solver.t < 1e-14:
            ### replace nearest point from hard pint
            import numpy as np
            xh = pa_nodes.x
            yh = pa_nodes.y
            zh = pa_nodes.z
            neartag = pa_nodes.neartag

            for id in list(self.hardpoints.keys()):
                pa_id = neartag[id]
                print(id, pa_id)
                pa.x[pa_id] = xh[id]
                pa.y[pa_id] = yh[id]
                pa.z[pa_id] = zh[id]
                pa.hard[pa_id] = 1.0

    def post_step(self, particles, solver):
        import numpy as np
        pa_fluid = None
        pa_solid = None
        pa_frozen = None
        pa_solid_i = None
        pa_solid_o = None
        dfreq = self.dfreq
        t = solver.t
        dt = solver.dt
        if solver.count % dfreq == 0:
            for name in self.solids:
                for pa in particles:
                    if name == pa.name:
                        pa_solid = pa
                    elif self.solids[name][0] == pa.name:
                        pa_solid_i = pa
                    elif self.solids[name][1] == pa.name:
                        pa_solid_o = pa
                    elif self.fluids[0] == pa.name:
                        pa_fluid = pa
                    elif self.frozen[0] == pa.name:
                        pa_frozen = pa

                if self.hardpoints:
                    self.setup_hardpoints(pa_solid_i, pa_solid) 
                    self.setup_hardpoints(pa_solid_o, pa_fluid) 

                N = len(pa_solid.x)
                print('no of solids', N)
                print(_condition_post(t, dt))

                if _condition_post(t, dt):
                    pa_fluid.hard[:] = 0.0
                    print('cond post')
                    self._is_volume_converged(pa_fluid)
