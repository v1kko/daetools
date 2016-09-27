#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_3.py
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
Mesh:

.. image:: _static/square.png
   :width: 400 px

Results at t = 500s:

.. image:: _static/tutorial_dealii_3-results.png
   :width: 600 px

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

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'square.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('HeatConduction', self, 'Transient heat conduction equation', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        D0    = 0.02
        kappa = 1e-2
        Omg_a = 3.4

        left_edge   = 0
        top_edge    = 1
        right_edge  = 2
        bottom_edge = 3

        functions = {}
        dirichletBC = {}
        boundaryIntegrals = {}

        # f(c) from the wikipedia
        c = dof_approximation_2D('c', fe_q) # FE approximation of a quantity at the specified quadrature point
        fc_Fi  = c**3 - c

        weakForm = dealiiFiniteElementWeakForm_2D(
            Aij = (   dphi_2D('c',  fe_i, fe_q) * dphi_2D('mu', fe_j, fe_q) * D0
                    +  phi_2D('mu', fe_i, fe_q) *  phi_2D('mu', fe_j, fe_q)
                    - dphi_2D('mu', fe_i, fe_q) * dphi_2D('c',  fe_j, fe_q) * kappa
                  ) * JxW_2D(fe_q),
            Mij = (phi_2D('c', fe_i, fe_q) * phi_2D('c', fe_j, fe_q)) * JxW_2D(fe_q),
            Fi  = (phi_2D('mu', fe_i, fe_q) * JxW_2D(fe_q)) * fc_Fi,
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
        print('    boundaryIntegrals  = %s' % str([item for item in weakForm.boundaryIntegrals]))

        # Setting the weak form of the FE system will declare a set of equations:
        # [Mij]{dx/dt} + [Aij]{x} = {Fi} and boundary integral equations
        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        numpy.random.seed(124)
        def ic_with_noise(index):
            ic = 0.5
            return ic + random.uniform(-0.1*ic, 0.1*ic)

        #setFEInitialConditions(daeFiniteElementModel, dealiiFiniteElementSystem_xD, str, float|callable)
        setFEInitialConditions(self.m.fe_model, self.m.fe_system, 'c', ic_with_noise)

# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation   = simTutorial()
    lasolver = pySuperLU.daeCreateSuperLUSolver()

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_3-')

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
    simulation.TimeHorizon       = 1.0
    simulation.ReportingInterval = simulation.TimeHorizon/100
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
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_3-results')

    # Create two data reporters:
    # 1. deal.II
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
    simulation.TimeHorizon       = 1.0
    simulation.ReportingInterval = simulation.TimeHorizon/100

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
