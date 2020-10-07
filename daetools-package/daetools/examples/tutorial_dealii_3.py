#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_3.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
__doc__ = """
In this example the Cahn-Hilliard equation is solved using the finite element method.
This equation describes the process of phase separation, where two components of a
binary mixture separate and form domains pure in each component.

.. code-block:: none

   dc/dt - D*nabla^2(mu) = 0, in Omega
   mu = c^3 - c - gamma*nabla^2(c)

The mesh is a simple square (0-100)x(0-100):

.. image:: _static/square.png
   :width: 300 px

The concentration plot at t = 500s:

.. image:: _static/tutorial_dealii_3-results.png
   :height: 400 px

Animation:
    
.. image:: _static/tutorial_dealii_3-animation.gif
   :height: 400 px
"""

import os, sys, numpy, json, tempfile, random
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_2D(name='c',
                                          description='Concentration',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1),
                dealiiFiniteElementDOF_2D(name='mu',
                                          description='Chemical potential',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        # This mesh is a coarse one (30x30 cells); there are finer meshes available: 50x50, 100x100, 200x200
        mesh_file  = os.path.join(meshes_dir, 'square(0,100)x(0,100)-50x50.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('CahnHilliard', self, 'Cahn-Hilliard equation', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create some auxiliary objects for readability
        phi_c_i  =  phi_2D('c', fe_i, fe_q)
        phi_c_j  =  phi_2D('c', fe_j, fe_q)
        dphi_c_i = dphi_2D('c', fe_i, fe_q)
        dphi_c_j = dphi_2D('c', fe_j, fe_q)
        
        phi_mu_i  =  phi_2D('mu', fe_i, fe_q)
        phi_mu_j  =  phi_2D('mu', fe_j, fe_q)
        dphi_mu_i = dphi_2D('mu', fe_i, fe_q)
        dphi_mu_j = dphi_2D('mu', fe_j, fe_q)
        
        # FE approximation of a quantity at the specified quadrature point (adouble object)
        c      = dof_approximation_2D('c', fe_q)
        normal = normal_2D(fe_q)
        xyz    = xyz_2D(fe_q)
        JxW    = JxW_2D(fe_q)

        # Boundary IDs
        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        dirichletBC = {}
        surfaceIntegrals = {}

        self.useWikipedia_fc = True

        # 1) f(c) from the Wikipedia (https://en.wikipedia.org/wiki/Cahn-Hilliard_equation)
        if self.useWikipedia_fc:
            Diffusivity = 1.0
            gamma       = 1.0
            def f(c):
                return c**3 - c

        # 2) f(c) used by Raymond Smith (M.Z.Bazant's group, MIT) for phase-separating battery electrodes
        if not self.useWikipedia_fc:
            Diffusivity = 1
            gamma       = 1
            Omg_a       = 3.4
            log_fe = feExpression_2D.log
            def f(c):
                # The original expression is:
                #   log_fe(c/(1-c)) + Omg_a*(1-2*c)
                # However, the one below is much more computationally efficient and requires less memory,
                # since a Finite Element approximation of a DoF is an expensive operation:
                #   sum(phi_j * dof(j))
                # For vector-valued DoFs it is even more demanding for the approximation is:
                #   sum(phi_vector_j * dof(j)) 
                # where phi_vector_j is a rank=1 Tensor.
                return log_fe(1 + 1/(1-c)) + Omg_a - (2*Omg_a)*c

        # FE weak form terms
        c_accumulation    = (phi_c_i * phi_c_j) * JxW
        mu_diffusion_c_eq = (dphi_c_i * dphi_mu_j) * Diffusivity * JxW
        mu                = phi_mu_i *  phi_mu_j * JxW
        c_diffusion_mu_eq = (-dphi_mu_i * dphi_c_j) * gamma * JxW
        fun_c             = (phi_mu_i * JxW) * f(c)

        cell_Aij = mu_diffusion_c_eq + mu + c_diffusion_mu_eq + c_diffusion_mu_eq
        cell_Mij = c_accumulation
        cell_Fi  = fun_c
        
        weakForm = dealiiFiniteElementWeakForm_2D(Aij = cell_Aij,
                                                  Mij = cell_Mij,
                                                  Fi  = cell_Fi,
                                                  functionsDirichletBC = dirichletBC,
                                                  surfaceIntegrals = surfaceIntegrals)

        print('Cahn-Hilliard equation:')
        print('    Aij = %s' % str(cell_Aij))
        print('    Mij = %s' % str(cell_Mij))
        print('    Fi  = %s' % str(cell_Fi))
        print('    surfaceIntegrals  = %s' % str([item for item in surfaceIntegrals]))

        # Setting the weak form of the FE system will declare a set of equations:
        # [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integral equations
        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_3")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        numpy.random.seed(124)

        def c_with_noise_wiki(index, overallIndex):
            c0     = 0.0
            stddev = 0.1
            return numpy.random.normal(c0, stddev)

        def c_with_noise_ray(index, overallIndex):
            c0     = 0.5
            stddev = 0.1
            return numpy.random.normal(c0, stddev)

        if self.m.useWikipedia_fc:
            setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'c', c_with_noise_wiki)
        else:
            setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'c', c_with_noise_ray)
  
def run(**kwargs):
    guiRun = kwargs.get('guiRun', False)
    
    simulation = simTutorial()

    # Create SuperLU LA solver
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    # Create and setup two data reporters:
    datareporter = daeDelegateDataReporter()
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if guiRun:
        results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_3-')
        daeQtMessage("deal.II", "The simulation results will be located in: %s" % results_folder)
    else:
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_3-results')
        print("The simulation results will be located in: %s" % results_folder)
    
    # 1. deal.II (exports only FE DOFs in .vtk format to the specified directory)
    feDataReporter = simulation.m.fe_system.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    if not feDataReporter.Connect(results_folder, simName):
        sys.exit()
        
    # 2. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    return daeActivity.simulate(simulation, reportingInterval = 5, 
                                            timeHorizon       = 500,
                                            lasolver          = lasolver,
                                            datareporter      = datareporter,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
