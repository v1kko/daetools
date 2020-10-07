#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_9.py
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
In this example the 2D lid driven cavity problem is solved 
(`deal.II step-57 <https://www.dealii.org/8.5.0/doxygen/deal.II/step_57.html>`_
and `Lid-driven cavity problem <http://www.cfd-online.com/Wiki/Lid-driven_cavity_problem>`_).

The problem can be described using the incompressible Navier-Stokes equations:

.. code-block:: none

   du/dt + u div(u) + nabla(p) = 0, in Omega
   -div(u) = 0 in Omega

The mesh is a simple square (0,1)x(0,1):

.. image:: _static/square(0,1)x(0,1)-50x50.png
   :width: 300 px

The temperature and the velocity vectors plot:

.. image:: _static/tutorial_dealii_9-results.png
   :height: 400 px

Animation:
    
.. image:: _static/tutorial_dealii_9-animation.gif
   :height: 400 px
"""

import os, sys, numpy, json, tempfile, random
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU as superlu

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class WallVelocityFunction_2D(adoubleFunction_2D):
    def __init__(self, n_components = 1):
        adoubleFunction_2D.__init__(self, n_components)
        
        self.n_components = n_components

    def vector_value(self, point):
        values = [adouble(0.0)] * self.n_components
        values[0] = adouble(0.0) # ux component
        values[1] = adouble(0.0) # uy component
        return values

# Velocity x-component:
u_lid = 0.1
class LidVelocityFunction_2D(adoubleFunction_2D):
    def __init__(self, n_components = 1):
        adoubleFunction_2D.__init__(self, n_components)
        
        self.n_components = n_components

    def vector_value(self, point):
        values = [adouble(0.0)] * self.n_components
        values[0] = adouble(u_lid) # ux component
        values[1] = adouble(0.0)   # uy component
        return values

u_t = daeVariableType("u_t", unit(),  0.0, 1E20, 0, 1e-07)
p_t = daeVariableType("p_t", unit(),  0.0, 1E20, 0, 1e-07)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        FE_degree = 1
        dofs = [dealiiFiniteElementDOF_2D(name         = 'u',
                                          description  = 'Velocity',
                                          fe           = FE_Q_2D(FE_degree+1),
                                          multiplicity = 2,
                                          variableType = u_t),
                dealiiFiniteElementDOF_2D(name         = 'p',
                                          description  = 'Pressure',
                                          fe           = FE_Q_2D(FE_degree),
                                          multiplicity = 1,
                                          variableType = p_t)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'square(0,1)x(0,1)-64x64.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('NavierStokes', self, 'The NavierStokes equations', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Boundary IDs
        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        # Create some auxiliary objects for readability
        phi_p_i         =  phi_2D('p', fe_i, fe_q)
        phi_p_j         =  phi_2D('p', fe_j, fe_q)
        dphi_p_i        = dphi_2D('p', fe_i, fe_q)
        dphi_p_j        = dphi_2D('p', fe_j, fe_q)
        
        phi_vector_u_i  =         phi_vector_2D('u', fe_i, fe_q)
        phi_vector_u_j  =         phi_vector_2D('u', fe_j, fe_q)
        dphi_vector_u_i =        dphi_vector_2D('u', fe_i, fe_q)
        dphi_vector_u_j =        dphi_vector_2D('u', fe_j, fe_q)
        div_phi_u_i     =            div_phi_2D('u', fe_i, fe_q)
        div_phi_u_j     =            div_phi_2D('u', fe_j, fe_q)
        
        scalar_product  = feExpression_2D.scalar_product
        
        # FE approximation of the gradient of u at the specified quadrature point (Tensor<2,dim,adouble> object)
        du_dof = vector_dof_gradient_approximation_2D('u', fe_q)
        u_dof  = vector_dof_approximation_2D('u', fe_q)
        du_dof = vector_dof_gradient_approximation_2D('u', fe_q)
        
        normal  = normal_2D(fe_q)
        xyz     = xyz_2D(fe_q)
        JxW     = JxW_2D(fe_q)
        
        mu = 1.0/400

        dirichletBC = {}
        dirichletBC[left_edge]   = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', WallVelocityFunction_2D(self.n_components))]
        dirichletBC[top_edge]    = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', LidVelocityFunction_2D(self.n_components))]
        dirichletBC[right_edge]  = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', WallVelocityFunction_2D(self.n_components))]
        dirichletBC[bottom_edge] = [('p', adoubleConstantFunction_2D(adouble(0.0))),
                                    ('u', WallVelocityFunction_2D(self.n_components))]
        
        # Contributions from the Navie-Stokes equation:
        Aij_u_viscosity  = mu * scalar_product(dphi_vector_u_i, dphi_vector_u_j) * JxW
        #Aij_u_convection = (JxW, du_dof*phi_vector_u_i, phi_vector_u_j)
        Aij_u_convection1 = (u_dof, phi_vector_u_i, dphi_vector_u_j * JxW)
        #Aij_u_convection2 = (0.5 * JxW, du_dof * phi_vector_u_i, phi_vector_u_j)
        Aij_p_gradient   = -(div_phi_u_i * phi_p_j) * JxW
        
        # Contributions from the continuity equation:
        Aij_continuity   = -(phi_p_i * div_phi_u_j) * JxW

        # Total contributions:
        Mij = 0 * JxW
        Aij = [Aij_u_viscosity + Aij_p_gradient + Aij_continuity, Aij_u_convection1]
        Fi  = 0 * JxW
        
        weakForm = dealiiFiniteElementWeakForm_2D(Aij = Aij,
                                                  Mij = Mij,
                                                  Fi  = Fi,
                                                  functionsDirichletBC = dirichletBC)

        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_9")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        pass
        
def run(**kwargs):
    guiRun = kwargs.get('guiRun', False)
    
    simulation = simTutorial()

    # Create SuperLU LA solver
    lasolver = superlu.daeCreateSuperLUSolver()

    # Create and setup two data reporters:
    datareporter = daeDelegateDataReporter()
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if guiRun:
        results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_9-')
        daeQtMessage("deal.II", "The simulation results will be located in: %s" % results_folder)
    else:
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_9-results')
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

    return daeActivity.simulate(simulation, reportingInterval = 1, 
                                            timeHorizon       = 1,
                                            lasolver          = lasolver,
                                            datareporter      = datareporter,
                                            **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
