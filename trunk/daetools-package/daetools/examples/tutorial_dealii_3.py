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
"""

import os, sys, numpy, json, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.deal_II import *
from daetools.solvers.superlu import pySuperLU
#from daetools.solvers.trilinos import pyTrilinos
#from daetools.solvers.aztecoo_options import daeAztecOptions

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# Neumann BC use either value or gradient
# Dirichlet BC use vector_value with n_component = multiplicity of the equation
# Other functions use value
class fnConstantFunction(Function_2D):
    def __init__(self, val, n_components = 1, active_component = 0):
        Function_2D.__init__(self, n_components)
        self.m_value            = float(val)
        self.m_active_component = int(active_component)

    def value(self, point, component = 0):
        res = 0.0
        if component == self.m_active_component:
            res = self.m_value
        #print('value at %s = %s (component = %d)' % (point, res, component))
        return res

    def vector_value(self, point):
        res = [self.value(point, c) for c in range(self.n_components)]
        #print('vector_value at %s = %s' % (point, res))
        return res

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        dofs = [dealiiFiniteElementDOF_2D(name='T',
                                          description='Temperature',
                                          multiplicity=1),
                dealiiFiniteElementDOF_2D(name='T2',
                                          description='Temperature 2',
                                          multiplicity=1)]
        n_components = len(dofs)

        functions    = {}
        functions['Diffusivity'] = ConstantFunction_2D(401.0/(8960*385))
        functions['Generation']  = ConstantFunction_2D(0.0)

        dirichletBC    = {}
        dirichletBC[0] = ('T',  fnConstantFunction(200, n_components, 0)) # left edge
        dirichletBC[1] = ('T',  fnConstantFunction(350, n_components, 0)) # top edge
        dirichletBC[2] = ('T2', fnConstantFunction(250, n_components, 1)) # right edge

        meshes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
        mesh_file  = os.path.join(meshes_dir, 'square.msh')

        weakForm = dealiiFiniteElementWeakForm_2D(Aij = (dphi_2D('T', fe_i, fe_q) * dphi_2D('T', fe_j, fe_q)) * function_value_2D("Diffusivity", xyz_2D(fe_q)) * JxW_2D(fe_q) \
                                                      + (dphi_2D('T2', fe_i, fe_q) * dphi_2D('T2', fe_j, fe_q)) * function_value_2D("Diffusivity", xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  Mij = (phi_2D('T', fe_i, fe_q) * phi_2D('T', fe_j, fe_q)) * JxW_2D(fe_q) \
                                                      + (phi_2D('T2', fe_i, fe_q) * phi_2D('T2', fe_j, fe_q)) * JxW_2D(fe_q),
                                                  Fi  = phi_2D('T', fe_i, fe_q) * function_value_2D("Generation", xyz_2D(fe_q)) * JxW_2D(fe_q),
                                                  faceAij = {},
                                                  faceFi  = {},
                                                  functions = functions,
                                                  functionsDirichletBC = dirichletBC)

        print('Heat conduction equation:')
        print('    Aij = %s' % str(weakForm.Aij))
        print('    Mij = %s' % str(weakForm.Mij))
        print('    Fi  = %s' % str(weakForm.Fi))
        print('    faceAij = %s' % str([item for item in weakForm.faceAij]))
        print('    faceFi  = %s' % str([item for item in weakForm.faceFi]))

        # Store the object so it does not go out of scope while still in use by daetools
        self.fe_dealII = dealiiFiniteElementSystem_2D(meshFilename    = mesh_file,     # path to mesh
                                                      polynomialOrder = 1,             # polynomial order
                                                      quadrature      = QGauss_2D(3),  # quadrature formula
                                                      faceQuadrature  = QGauss_1D(3),  # face quadrature formula
                                                      dofs            = dofs,          # degrees of freedom
                                                      weakForm        = weakForm)      # FE system in weak form

        self.fe = daeFiniteElementModel('HeatConduction', self, 'Multi-scalar transient heat conduction FE problem', self.fe_dealII)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_deal_II_3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        m_dt = self.m.fe_dealII.Msystem()

        # Vector where every item marks the boundar
        #dof_to_boundary = self.m.fe_dealII.GetDOFtoBoundaryMap()
        #print list(dof_to_boundary)

        # dofIndexesMap relates global DOF indexes to points within daetools variables

        # Todo: use a function from daeSimulation
        dofIndexesMap = {}
        for variable in self.m.fe.Variables:
            if variable.Name == 'T':
                ic = 300
            elif variable.Name == 'T2':
                ic = 200
            else:
                raise RuntimeError('Unknown variable [%s] found' % variable.Name)

            for i in range(variable.NumberOfPoints):
                dofIndexesMap[variable.OverallIndex + i] = (variable, i, ic)

        for row in range(m_dt.n):
            # Iterate over columns and set initial conditions.
            # If an item in the dt matrix is zero skip it (it is at the boundary - not a diff. variable).
            for column in self.m.fe_dealII.RowIndices(row):
                if m_dt(row, column) != 0:
                    variable, index, ic = dofIndexesMap[column]
                    variable.SetInitialCondition(index, ic)
                    #print('%s(%d) initial condition = %f' % (variable.Name, column, ic))

# Use daeSimulator class
def guiRun(app):
    datareporter = daeDelegateDataReporter()
    simulation = simTutorial()
    tcpipDataReporter = daeTCPIPDataReporter()
    feDataReporter    = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(tcpipDataReporter.Connect("", simName) == False):
        sys.exit()
    results_folder = tempfile.mkdtemp(suffix = '-results', prefix = 'tutorial_deal_II_3-')
    feDataReporter.Connect(results_folder, simName)
    try:
        from PyQt4 import QtCore, QtGui
        QtGui.QMessageBox.warning(None, "deal.II", "The simulation results will be located in: %s" % results_folder)
    except Exception as e:
        print(str(e))

    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation=simulation, datareporter = datareporter)
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
    """
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "")
    lasolver.NumIters  = 500
    lasolver.Tolerance = 1e-6
    paramListAztec = lasolver.AztecOOOptions
    paramListAztec.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
    paramListAztec.set_int("AZ_kspace",    500)
    paramListAztec.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_reorder",   0)
    paramListAztec.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
    paramListAztec.set_int("AZ_keep_info", 1)
    paramListAztec.set_int("AZ_output",    daeAztecOptions.AZ_all) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}
    paramListAztec.set_int("AZ_precond",         daeAztecOptions.AZ_dom_decomp)
    paramListAztec.set_int("AZ_subdomain_solve", daeAztecOptions.AZ_ilut)
    paramListAztec.set_int("AZ_overlap",         daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_graph_fill",      1)
    paramListAztec.Print()
    daesolver.SetLASolver(lasolver)
    """

    # Create two data reporters: TCP/IP and DealII
    tcpipDataReporter = daeTCPIPDataReporter()
    feDataReporter    = simulation.m.fe_dealII.CreateDataReporter()
    datareporter.AddDataReporter(tcpipDataReporter)
    datareporter.AddDataReporter(feDataReporter)

    # Connect datareporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(tcpipDataReporter.Connect("", simName) == False):
        sys.exit()
    feDataReporter.Connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorial_deal_II_3-results'), simName)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 500

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    #simulation.m.fe.SaveModelReport(simulation.m.fe.Name + ".xml")
    #simulation.m.fe.SaveRuntimeModelReport(simulation.m.fe.Name + "-rt.xml")

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
