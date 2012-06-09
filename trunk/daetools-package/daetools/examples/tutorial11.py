#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             tutorial11.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

"""
 This tutorial shows the use of Trilinos group of solvers: Amesos and AztecOO iterative
 linear equation solvers with different preconditioners (built-in AztecOO, Ifpack or ML)
 and corresponding linear solver options.
 
 ACHTUNG, ACHTUNG!!
 Iterative solvers are not fully working yet and this example is given just as a showcase
 and for preconditioner options experimenting purposes.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime
try:
    from daetools.solvers.trilinos import pyTrilinos
    from daetools.solvers.aztecoo_options import daeAztecOptions
except ImportError as e:
    print('Unable to import Trilinos LA solver: {0}'.format(e))

# Standard variable types are defined in daeVariableTypes.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, m, "X axis domain")
        self.y = daeDomain("y", self, m, "Y axis domain")

        self.Qb = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Qt = daeParameter("Q_t",         W/(m**2), self, "Heat flux at the top edge of the plate")
        self.ro = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k  = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")

        self.T = daeVariable("T", temperature_t, self)
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)
        self.T.Description = "Temperature of the plate, K"

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = 1e-6 * (  self.ro() * self.cp() * self.T.dt(x, y) - self.k() * \
                               (self.T.d2(self.x, x, y) + self.T.d2(self.y, x, y))  )

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = 1e-6 * ( - self.k() * self.T.d(self.y, x, y) - self.Qb()  )

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = 1e-6 * ( - self.k() * self.T.d(self.y, x, y) - self.Qt()  )

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        eq = self.CreateEquation("BC_righ", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial11")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        n = 5
        self.m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.ro.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Qt.SetValue(0 * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

# Function to create the Trilinos linear equation solver.
# AztecOO solvers do not work well yet
def createLASolver():
    print "Supported Trilinos solvers:", str(pyTrilinos.daeTrilinosSupportedSolvers())
    
    # Amesos SuperLU solver
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu", "")
    
    # AztecOO built-in preconditioners are specified through AZ_precond option
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "")

    # Ifpack preconditioner can be one of: [ILU, ILUT, PointRelaxation, BlockRelaxation, IC, ICT]
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO_Ifpack", "PointRelaxation")
    
    # ML preconditioner can be one of: [SA, DD, DD-ML, DD-ML-LU, maxwell, NSSA]
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO_ML", "maxwell")
    
    return lasolver

# Function to set-up the Trilinos linear equation solver. Possible choices:
#  - Direct: {Amesos_KLU, Amesos_Superlu, Amesos_Umfpack, Amesos_Lapack}
#  - Iterative: {AztecOO, AztecOO_Ifpack, AztecOO_ML}
def setOptions(lasolver):
    #######################################################
    # Amesos_Superlu solver
    #######################################################
    if lasolver.Name == "Amesos_Superlu":
        paramListAmesos = lasolver.GetAmesosOptions()

        # Amesos status options:
        paramListAmesos.set_int("OutputLevel", 0)
        paramListAmesos.set_int("DebugLevel", 0)
        paramListAmesos.set_bool("PrintTiming", False)
        paramListAmesos.set_bool("PrintStatus", False)
        paramListAmesos.set_bool("ComputeVectorNorms", False)
        paramListAmesos.set_bool("ComputeTrueResidual", False)

        # Amesos control options:
        paramListAmesos.set_bool("AddZeroToDiag", False)
        paramListAmesos.set_float("AddToDiag", 0.0)
        paramListAmesos.set_bool("Refactorize", False)
        paramListAmesos.set_float("RcondThreshold", 0.0)
        paramListAmesos.set_int("MaxProcs", 0)
        paramListAmesos.set_string("MatrixProperty", "")
        paramListAmesos.set_int("ScaleMethod", 0);
        paramListAmesos.set_bool("Reindex", False)

    #######################################################
    # AztecOO solver options consist of:
    #  - solver options (given below)
    #  - preconditioner options given
    #######################################################
    if ("AztecOO" in lasolver.Name) or ("AztecOO_Ifpack" in lasolver.Name) or ("AztecOO_ML" in lasolver.Name):
        paramListAztec = lasolver.GetAztecOOOptions()

        lasolver.NumIters  = 500
        lasolver.Tolerance = 1e-6
        paramListAztec.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
        paramListAztec.set_int("AZ_kspace",    500)
        paramListAztec.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
        paramListAztec.set_int("AZ_reorder",   0)
        paramListAztec.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
        paramListAztec.set_int("AZ_keep_info", 1)
        paramListAztec.set_int("AZ_output",    daeAztecOptions.AZ_all) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}
        paramListAztec.Print()

    #######################################################
    # AztecOO preconditioner options
    #######################################################
    if "AztecOO_Ifpack" in lasolver.Name:
        # 2b) Ifpack preconditioner:
        paramListIfpack = lasolver.GetIfpackOptions()
        paramListIfpack.set_string("relaxation: type",               "Jacobi")
        paramListIfpack.set_float ("relaxation: min diagonal value", 1e-2)
        paramListIfpack.set_int   ("relaxation: sweeps",             5)
        #paramListIfpack.set_float("fact: ilut level-of-fill",        3.0)
        #paramListIfpack.set_float("fact: absolute threshold",        1e8)
        #paramListIfpack.set_float("fact: relative threshold",        0.0)
        paramListIfpack.Print()

    elif "AztecOO_ML" in lasolver.Name:
        # 2c) ML preconditioner:
        paramListML = lasolver.GetMLOptions()
        paramListML.set_bool("reuse: enable", True)
        paramListML.Print()

    elif "AztecOO" in lasolver.Name:
        # 2a) AztecOO built-in preconditioner:
        paramListAztec = lasolver.GetAztecOOOptions()
        paramListAztec.set_int("AZ_precond",         daeAztecOptions.AZ_dom_decomp)
        paramListAztec.set_int("AZ_subdomain_solve", daeAztecOptions.AZ_ilut)
        paramListAztec.set_int("AZ_overlap",         daeAztecOptions.AZ_none)
        paramListAztec.set_int("AZ_graph_fill",      1)
        #paramListAztec.set_int("AZ_type_overlap",    daeAztecOptions.AZ_standard)
        #paramListAztec.set_float("AZ_ilut_fill",     3.0)
        #paramListAztec.set_float("AZ_drop",          0.0)
        #paramListAztec.set_float("AZ_athresh",       1e8)
        #paramListAztec.set_float("AZ_rthresh",       0.0)
        paramListAztec.Print()
    
# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    la = createLASolver()
    simulator = daeSimulator(app, simulation = sim, lasolver = la, lasolver_setoptions_fn=setOptions)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    daesolver    = daeIDAS()
    lasolver     = createLASolver()
    daesolver.SetLASolver(lasolver)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 1000

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Set the solver options
    setOptions(lasolver)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

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
