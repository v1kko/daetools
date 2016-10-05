#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_5.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
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
Flow through porous media. Darcy's law (deal.II step-20).

.. code-block:: none
    .. math::

        K^{-1} {\mathbf u} + \nabla p &=  0 \qquad {\textrm{in}\ } \Omega \\
         -{\textrm{div}}\ {\mathbf u} &= -f \qquad {\textrm{in}\ } \Omega \\
                                    p &=  g \qquad {\textrm{on}\ } \partial \Omega

Mesh:

.. image:: _static/square.png
   :width: 300 px

Results at t = 500s:

.. image:: _static/tutorial_dealii_4-results.png
   :width: 500 px
"""

import os, sys, numpy, json, tempfile, random
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class permeabilityFunction_2D(TensorFunction_2_2D):
    def __init__(self, N):
        TensorFunction_2_2D.__init__(self)

        numpy.random.seed(1000)
        self.centers = 2 * numpy.random.rand(N,2) - 1
        # Create a Tensor<rank=2,dim=2> object to serve as a return value (to make the function faster)
        self.inv_kappa = Tensor_2_2D()

    def value(self, point, component = 0):
        # 1) Sinusoidal (a function of the distance to the flowline)
        #distance_to_flowline = numpy.fabs(point[1] - 0.2*numpy.sin(10*point[0]))
        #permeability = numpy.exp(-(distance_to_flowline*distance_to_flowline)/0.01)
        #norm_permeability = max(permeability, 0.001)

        # 2) Random permeability field
        x2 = numpy.square(point[0]-self.centers[:,0])
        y2 = numpy.square(point[1]-self.centers[:,1])
        permeability = numpy.sum( numpy.exp(-(x2 + y2) / 0.01) )
        norm_permeability = max(permeability, 0.005)

        # Set-up the inverse permeability tensor (only the diagonal items)
        self.inv_kappa[0][0] = 1.0 / norm_permeability
        self.inv_kappa[1][1] = 1.0 / norm_permeability

        return self.inv_kappa

class pBoundaryFunction_2D(Function_2D):
    def __init__(self, n_components = 1):
        Function_2D.__init__(self, n_components)

    def value(self, p, component = 0):
        alpha = 0.3
        beta  = 1.0
        return -(alpha*p[0]*p[1]*p[1]/2.0 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6.0)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_2D(name='u',
                                          description='Velocity',
                                          fe = FE_RaviartThomas_2D(0),
                                          multiplicity=2),
                dealiiFiniteElementDOF_2D(name='p',
                                          description='Pressure',
                                          fe = FE_DGQ_2D(0),
                                          multiplicity=1)]
        self.n_components = int(numpy.sum([dof.Multiplicity for dof in dofs]))

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'square-step-20.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('PorousMedia', self, 'Flow through porous media', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        functions = {}
        functions['p_boundary'] = pBoundaryFunction_2D(self.n_components)

        # Boundary IDs
        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        dirichletBC = {}

        faceFi = {
                   left_edge:   -(phi_vector_2D('u', fe_i, fe_q) * normal_2D(fe_q)) * function_value_2D("p_boundary", xyz_2D(fe_q)) * JxW_2D(fe_q),
                   top_edge:    -(phi_vector_2D('u', fe_i, fe_q) * normal_2D(fe_q)) * function_value_2D("p_boundary", xyz_2D(fe_q)) * JxW_2D(fe_q),
                   right_edge:  -(phi_vector_2D('u', fe_i, fe_q) * normal_2D(fe_q)) * function_value_2D("p_boundary", xyz_2D(fe_q)) * JxW_2D(fe_q),
                   bottom_edge: -(phi_vector_2D('u', fe_i, fe_q) * normal_2D(fe_q)) * function_value_2D("p_boundary", xyz_2D(fe_q)) * JxW_2D(fe_q)
                 }

        #exp_fe = feExpression_2D.exp
        #p = dof_approximation_2D('p', fe_q)
        #alfa0 = 1.0
        #gamma = 0.5
        #alfa = alfa0 #* exp_fe(-gamma*p)

        self.fun_k_inverse = permeabilityFunction_2D(40)
        k_inverse = tensor2_function_value_2D('k_inverse', self.fun_k_inverse, xyz_2D(fe_q))

        accumulation = 0.0 * JxW_2D(fe_q)
        velocity     = (phi_vector_2D('u', fe_i, fe_q) * k_inverse * phi_vector_2D('u', fe_j, fe_q)) * JxW_2D(fe_q)
        p_gradient   = -(div_phi_2D('u', fe_i, fe_q) * phi_2D('p', fe_j, fe_q)) * JxW_2D(fe_q)
        continuity   = -(phi_2D('p', fe_i, fe_q) * div_phi_2D('u', fe_j, fe_q)) * JxW_2D(fe_q)
        source       = 0.0 * JxW_2D(fe_q)

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = velocity + p_gradient + continuity,
                                                  Mij = accumulation,
                                                  Fi  = source,
                                                  faceAij = {},
                                                  faceFi  = faceFi,
                                                  functions = functions,
                                                  functionsDirichletBC = dirichletBC)

        print('Darcy law equations:')
        print('    Aij = %s' % str(weakForm.Aij))
        print('    Mij = %s' % str(weakForm.Mij))
        print('    Fi  = %s' % str(weakForm.Fi))
        print('    faceAij = %s' % str([item for item in weakForm.faceAij]))
        print('    faceFi  = %s' % str([item for item in weakForm.faceFi]))

        # Setting the weak form of the FE system will declare a set of equations:
        # [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integral equations
        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II_5")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        pass

# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_5-')

    # Create two data reporters:
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

    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))

    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 5
    simulation.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation=simulation, datareporter = datareporter, lasolver=lasolver)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()

    lasolver = pySuperLU.daeCreateSuperLUSolver()
    daesolver.SetLASolver(lasolver)

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_5-results')

    # Create two data reporters:
    # 1. DealII
    feDataReporter = simulation.m.fe_system.CreateDataReporter()
    datareporter.AddDataReporter(feDataReporter)
    if not feDataReporter.Connect(results_folder, simName):
        sys.exit()

    # 2. TCP/IP
    tcpipDataReporter = daeTCPIPDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    if not tcpipDataReporter.Connect("", simName):
        sys.exit()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 5
    simulation.TimeHorizon = 500

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    lasolver.SaveAsXPM(simulation.m.Name + '.xpm')

    # Save the model report and the runtime model report
    simulation.m.fe_model.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.fe_model.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
