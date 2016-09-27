#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                        tutorial_dealii_1.py
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
In this tutorial the DAE Tools support for finite element method is presented.


n implementation

Mesh:

.. image:: _static/step-49.png
   :width: 400 px

Results at t = 500s:

.. image:: _static/tutorial_dealii_1-results.png
   :width: 600 px

"""

import os, sys, numpy, json, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.Q0_total = daeVariable("Q0_total", no_t, self, "Total heat passing through the boundary with id=0")
        self.Q1_total = daeVariable("Q1_total", no_t, self, "Total heat passing through the boundary with id=1")
        self.Q2_total = daeVariable("Q2_total", no_t, self, "Total heat passing through the boundary with id=2")

        # The starting point is the daeFiniteElementModel class that contains an implementation
        # of the daeFiniteElementObject class: dealiiFiniteElementSystem which is a wrapper
        # around deal.II FESystem<dim> class and handles all finite element related details.

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'step-49.msh')
        
        dofs = [dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]


        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom
          
        self.fe_model = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction FE problem', self.fe_system)
       
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        rho   = 8960.0  # kg/m**3
        cp    =  385.0  # J/(kg*K)
        kappa =  401.0  # W/(m*K)

        alpha = kappa/(rho * cp)

        # We use deal.II ConstantFunction class to specify a constant value.
        # Since we have only one DOF we do not need to specify n_components in the constructor
        # (the default value is 1) and do not need to handle values of multiple components.
        functions = {}
        functions['Diffusivity'] = ConstantFunction_2D(alpha)
        functions['Generation']  = ConstantFunction_2D(0.0)

        # Nota bene:
        #   For the Dirichlet BCs only the adouble versions of Function<dim> class can be used.
        #   The values allowed include constants and expressions on daeVariable/daeParameter objects.
        dirichletBC    = {}
        dirichletBC[0] = [('T', adoubleConstantFunction_2D( adouble(200)) )] # outer boundary
        dirichletBC[1] = [('T', adoubleConstantFunction_2D( adouble(350)) )] # inner ellipse
        dirichletBC[2] = [('T', adoubleConstantFunction_2D( adouble(250)) )] # inner rectangle

        boundaryIntegrals = {
                               0 : [(self.Q0_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))],
                               1 : [(self.Q1_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))],
                               2 : [(self.Q2_total(), (-alpha * (dphi_2D('T', fe_i, fe_q) * normal_2D(fe_q)) * JxW_2D(fe_q)) * dof_2D('T', fe_i))]
                            }

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = (dphi_2D('T', fe_i, fe_q) * dphi_2D('T', fe_j, fe_q)) * function_value_2D('Diffusivity', xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  Mij = (phi_2D('T', fe_i, fe_q) * phi_2D('T', fe_j, fe_q)) * JxW_2D(fe_q),
                                                  Fi  = phi_2D('T', fe_i, fe_q) * function_value_2D("Generation", xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  faceAij = {},
                                                  faceFi  = {},
                                                  functions = functions,
                                                  functionsDirichletBC = dirichletBC,
                                                  boundaryIntegrals = boundaryIntegrals)

        print('Heat conduction equation:')
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
        self.m = modTutorial("tutorial_deal_II_1")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        # setFEInitialConditions(daeFiniteElementModel, dealiiFiniteElementSystem_xD, str, float|callable)
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'T', 300)

# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_1-')

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

    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))
    
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
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
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_1-results')

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
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)
    
    # Save the model report and the runtime model report
    simulation.m.fe_model.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.fe_model.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")
    
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
