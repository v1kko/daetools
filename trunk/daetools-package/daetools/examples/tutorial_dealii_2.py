#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_2.py
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
In this example a simple transient heat convection-diffusion equation is solved.

.. code-block:: none

   dT/dt - kappa/(rho*cp)*nabla^2(T) + nabla.(uT) = g(T) in Omega

The fluid flows from the left side to the right with constant velocity of 0.01 m/s.
The inlet temperature for 0.2 <= y <= 0.3 is iven by the following expression:

.. code-block:: none

   T_left = T_base + T_offset*|sin(pi*t/25)| on dOmega

creating a bubble-like regions of higher temperature that flow towards the right end
and slowly diffuse into the bulk flow of the fluid due to the heat conduction.

The mesh is rectangular with the refined elements close to the left/right ends:

.. image:: _static/rect(1.5,0.5)-100x50.png
   :width: 500 px

The temperature plot at t = 500s:

.. image:: _static/tutorial_dealii_2-results.png
   :height: 400 px

Animation:
    
.. image:: _static/tutorial_dealii_2-animation.gif
   :height: 400 px
"""

import os, sys, numpy, json, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Nota bene:
#   This function is derived from Function_2D class and returns "double" value/gradient
class VelocityFunction_2D(Function_2D):
    def __init__(self, velocity, direction, n_components = 1):
        """
        Arguments:
          velocity  - float, velocity magnitude
          direction - Tensor<1,dim>, unit vector
        """
        Function_2D.__init__(self, n_components)
        self.m_velocity = Tensor_1_2D()
        self.m_velocity[0] = velocity * direction[0]
        self.m_velocity[1] = velocity * direction[1]

    def gradient(self, point, component = 0):
        return self.m_velocity

    def vector_gradient(self, point):
        return [self.value(point, c) for c in range(self.n_components)]

class TemperatureSource_2D(adoubleFunction_2D):
    def __init__(self, ymin, ymax, T_base, T_offset, n_components = 1):
        """
        The function creates bubble-like regions of fluid with a higher temperature.
        Arguments:
          ymin     - float
          ymax     - float
          T_base   - float
          T_offset - float
        Return value:
          T_base + T_offset * |sin(t/25)|
        """
        adoubleFunction_2D.__init__(self, n_components)

        self.ymin = ymin
        self.ymax = ymax
        self.T_base   = adouble(T_base)
        self.T_offset = adouble(T_offset)

    def value(self, point, component = 0):
        if point.y > self.ymin and point.y < self.ymax:
            return self.T_base + self.T_offset*numpy.fabs(numpy.sin(numpy.pi*Time()/25))
        else:
            return self.T_base

    def vector_value(self, point):
        return [self.value(point, c) for c in range(self.n_components)]


class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'rect(1.5,0.5)-100x50.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('HeatConvection', self, 'Transient heat convection', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        """
        # Overall indexes of the DOFs at the boundaries:
        # 0 (left edge), 1 (top edge), 2 (right edge) and 3 (bottom edge)
        bdofs = self.fe_system.GetBoundaryDOFs('T', [0])
        print(numpy.where(bdofs))
        bdofs = self.fe_system.GetBoundaryDOFs('T', [1])
        print(numpy.where(bdofs))
        bdofs = self.fe_system.GetBoundaryDOFs('T', [2])
        print(numpy.where(bdofs))
        bdofs = self.fe_system.GetBoundaryDOFs('T', [3])
        print(numpy.where(bdofs))
        """

        # Thermo-physical properties of the liquid (water).
        # The specific heat conductivity is normally 0.6 W/mK,
        # however, here we used much larger value to amplify the effect of conduction
        rho   = 1000.0  # kg/m**3
        cp    = 4181.0  # J/(kg*K)
        kappa =  100.0  # W/(m*K)
        # Thermal diffusivity (m**2/s)
        alpha = kappa/(rho * cp)

        # Velocity is in the positive x-axis direction
        velocity  = 0.01   # The velocity magnitude, m/s
        direction = (1, 0) # The velocity direction (unit vector)

        # The dimensions of the 2D domain is a rectangle: x=[0,2] and y=[0,0.5]
        ymin = 0.2
        ymax = 0.3
        T_base   = 300 # Base temperature, K
        T_offset = 50  # Offset temperature, K

        # Create some auxiliary objects for readability
        phi_i  =  phi_2D('T', fe_i, fe_q)
        phi_j  =  phi_2D('T', fe_j, fe_q)
        dphi_i = dphi_2D('T', fe_i, fe_q)
        dphi_j = dphi_2D('T', fe_j, fe_q)
        xyz    = xyz_2D(fe_q)
        JxW    = JxW_2D(fe_q)

        # Boundary IDs
        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        dirichletBC = {}
        dirichletBC[left_edge]  = [
                                    ('T',  TemperatureSource_2D(ymin, ymax, T_base, T_offset, self.n_components)),
                                  ]

        # Function<dim> wrapper
        self.fun_u_grad = VelocityFunction_2D(velocity, direction)
        # Function<dim>::gradient wrapper
        u_grad = function_gradient_2D("u", self.fun_u_grad, xyz)

        # FE weak form terms
        accumulation = (phi_i * phi_j) * JxW
        diffusion    = (dphi_i * dphi_j) * alpha * JxW
        convection   = phi_i * (u_grad * dphi_j) * JxW
        source       = phi_i * 0.0 * JxW

        cell_Aij = diffusion + convection
        cell_Mij = accumulation
        cell_Fi  = source
        
        weakForm = dealiiFiniteElementWeakForm_2D(Aij = cell_Aij,
                                                  Mij = cell_Mij,
                                                  Fi  = cell_Fi,
                                                  functionsDirichletBC = dirichletBC)

        print('Transient heat convection equations:')
        print('    Aij = %s' % str(cell_Aij))
        print('    Mij = %s' % str(cell_Mij))
        print('    Fi  = %s' % str(cell_Fi))

        # Setting the weak form of the FE system will declare a set of equations:
        # [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integral equations
        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_2")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # setFEInitialConditions(daeFiniteElementModel, dealiiFiniteElementSystem_xD, str, float|callable)
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'T', 300.0)

def run(**kwargs):
    guiRun = kwargs.get('guiRun', False)
    
    simulation = simTutorial()

    # Create SuperLU LA solver
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    # Create and setup two data reporters:
    datareporter = daeDelegateDataReporter()
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if guiRun:
        results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_2-')
        daeQtMessage("deal.II", "The simulation results will be located in: %s" % results_folder)
    else:
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_2-results')
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

    daeActivity.simulate(simulation, reportingInterval = 2, 
                                     timeHorizon       = 200,
                                     lasolver          = lasolver,
                                     datareporter      = datareporter,
                                     **kwargs)

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
