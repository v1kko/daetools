#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_dealii_6.py
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
A simple steady-state diffusion and first-order reaction in an irregular catalyst shape
(Proc. 6th Int. Conf. on Mathematical Modelling, Math. Comput. Modelling, Vol. 11, 375-319, 1988)
applying Dirichlet and Robin type of boundary conditions.

.. code-block:: none

   D_eA * nabla^2(C_A) - k_r * C_A = 0 in Omega
   D_eA * nabla(C_A) = k_m * (C_A - C_Ab) on dOmega1
   C_A = C_Ab on dOmega2

The catalyst pellet mesh:

.. image:: _static/ssdr.png
   :width: 400 px

The concentration plot:

.. image:: _static/tutorial_dealii_6-results1.png
   :width: 500 px

The concentration plot for Ca=Cab on all boundaries:

.. image:: _static/tutorial_dealii_6-results2.png
   :width: 500 px
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

        dofs = [dealiiFiniteElementDOF_2D(name='Ca',
                                          description='Concentration',
                                          fe = FE_Q_2D(1),
                                          multiplicity=1)]

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'ssdr.msh')

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_system = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs)          # degrees of freedom

        self.fe_model = daeFiniteElementModel('DiffusionReaction', self, 'Diffusion-reaction in a catalyst', self.fe_system)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        De  = 0.1 # Diffusivity, m**2/s
        km  = 0.1 # Mass transfer coefficient, mol
        kr  = 1.0 # First-order reaction rate constant
        Cab = 1.0 # Boundary concentration

        # Create some auxiliary objects for readability
        phi_i  =  phi_2D('Ca', fe_i, fe_q)
        phi_j  =  phi_2D('Ca', fe_j, fe_q)
        dphi_i = dphi_2D('Ca', fe_i, fe_q)
        dphi_j = dphi_2D('Ca', fe_j, fe_q)
        normal = normal_2D(fe_q)
        xyz    = xyz_2D(fe_q)
        JxW    = JxW_2D(fe_q)

        dirichletBC = {}
        dirichletBC[1] = [('Ca', adoubleConstantFunction_2D(adouble(Cab)))]

        # FE weak form terms
        diffusion    = -(dphi_i * dphi_j) * De * JxW
        reaction     = -kr * phi_i * phi_j * JxW
        accumulation = 0.0 * JxW
        rhs          = 0.0 * JxW
        # Robin type BC's:
        faceAij = {
                    2: km * phi_i * phi_j * JxW
                  }
        faceFi  = {
                    2: km * Cab * phi_i * JxW
                  }

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = diffusion + reaction,
                                                  Mij = accumulation,
                                                  Fi  = rhs,
                                                  boundaryFaceAij = faceAij,
                                                  boundaryFaceFi  = faceFi,
                                                  functionsDirichletBC = dirichletBC)

        self.fe_system.WeakForm = weakForm

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_dealii_6")
        self.m.Description = __doc__
        self.m.fe_model.Description = __doc__

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
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_6-')

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
    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 1
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
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_6-results')

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

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 1

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

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
