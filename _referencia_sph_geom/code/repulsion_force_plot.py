"""
test for repulsion force
"""
from __future__ import print_function
import numpy as np
import os
import sys
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
from pysph.tools.sph_evaluator import SPHEvaluator

from particle_packing import RepulsionForce


class LenardJones(Equation):
    def __init__(self, dest, sources, k, dx, hdx):
        self.k = k
        self.dx = dx
        self.hdx = hdx
        super(LenardJones, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, RIJ, XIJ):
        if RIJ > 1e-14:
            c = self.hdx * self.dx * 2**(1/6)
            tmp = 12. * self.k * ( c**12 / RIJ**13 - c**6 / RIJ**7)
            d_au[d_idx] += tmp * XIJ[0]/RIJ
            d_av[d_idx] += tmp * XIJ[1]/RIJ
            d_aw[d_idx] += tmp * XIJ[2]/RIJ


class RepulsionForcePlots():
    def __init__(self, argv):
        self.argv = argv[1:]

    def initialize(self):
        self.hdx = 1.2
        self.dx = 0.05
        self.d = 2.0
        self.rho = 1.0
        self.dim = 1
        self.k = 1.0

    def parser(self):
        from argparse import ArgumentParser
        self.parser = ArgumentParser()

        self.parser.add_argument('--eq', dest='eq', type=str,
                                  default='new', help='the equation to be used')

        self.parser.add_argument('-d', dest='out', type=str,
                                  default='.', help='output directory')

        self.options = self.parser.parse_args(self.argv)
        print(self.options)


    def create_particles(self):
        x = np.linspace(0.1, 2, 1000)
        dest = get_particle_array(name='dest', x=x, h=1.0)

        x = [0]
        source = get_particle_array(name='source', x=x, h=1.0)

        particles = [dest, source]
        props = ['au', 'av', 'aw']
        for pa in particles:
            pa.add_constant('k', self.k)
            for p in props:
                pa.add_property(p)

        return [dest, source]

    def _new_force(self):
        equations = []
        eqns = Group(equations= [
            RepulsionForce(
                dest='dest', sources=['source'], hdx=1.0)], real=True)

        equations.append(eqns)
        return equations

    def _lenard_jones_force(self):
        equations = []
        eqns = Group(equations= [
            LenardJones(dest='dest', sources=['source'],
                                    dx=1.0, hdx=1.0, k=1.0)], real=True)

        equations.append(eqns)
        return equations

    def get_equations(self):
        if self.options.eq == 'new':
            return self._new_force()
        elif self.options.eq == 'lj':
            return self._lenard_jones_force()

    def run(self):
        self.initialize()
        self.particles = self.create_particles()
        eqns = self.get_equations()
        sphevaluator = SPHEvaluator(self.particles, eqns, dim=self.dim,
                                    kernel=QuinticSpline(dim=self.dim),
                                    backend='cython')
        sphevaluator.update()
        sphevaluator.evaluate()
        self.post_process()


    def post_process(self):
        outputdir = self.options.out
        filename = os.path.join(outputdir, 'results.npz')
        pa = self.particles[0]
        x = pa.x
        au = pa.au
        np.savez(filename, x=x, au=au)
        print('Done')


if __name__ == '__main__':
    app = RepulsionForcePlots(sys.argv)
    app.parser()
    app.run()
