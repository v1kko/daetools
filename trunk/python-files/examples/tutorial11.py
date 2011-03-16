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
 This tutorial shows the use of Trilinos AztecOO iterative Krylov linear equation solvers.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime
import daetools.pyTrilinos as pyTrilinos
from aztecoo_options import daeAztecOptions

typeNone         = daeVariableType("None",         "-",      0, 1E10,   0, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",    100, 1000, 500, 1e-5)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, "X axis domain")
        self.y = daeDomain("y", self, "Y axis domain")

        self.Qb = daeParameter("Q_b",      eReal, self, "Heat flux at the bottom edge of the plate, W/m2")
        self.Qt = daeParameter("Q_t",      eReal, self, "Heat flux at the top edge of the plate, W/m2")
        self.ro = daeParameter("&rho;",    eReal, self, "Density of the plate, kg/m3")
        self.cp = daeParameter("c_p",      eReal, self, "Specific heat capacity of the plate, J/kgK")
        self.k  = daeParameter("&lambda;", eReal, self, "Thermal conductivity of the plate, W/mK")

        self.T = daeVariable("T", typeTemperature, self)
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
        self.m.x.CreateDistributed(eCFDM, 2, 15, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, 15, 0, 0.1)

        self.m.k.SetValue(401)
        self.m.cp.SetValue(385)
        self.m.ro.SetValue(8960)
        self.m.Qb.SetValue(1e6)
        self.m.Qt.SetValue(0)

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300)

#lasolver.SetAztecOption(daeAztecOptions.AZ_output,          daeAztecOptions.AZ_none)
#lasolver.SetAztecOption(daeAztecOptions.AZ_diagnostics,     daeAztecOptions.AZ_none)
#lasolver.SetAztecOption(daeAztecOptions.AZ_solver,          daeAztecOptions.AZ_gmres)
#lasolver.SetAztecOption(daeAztecOptions.AZ_precond,         daeAztecOptions.AZ_dom_decomp)
#lasolver.SetAztecOption(daeAztecOptions.AZ_subdomain_solve, daeAztecOptions.AZ_ilut)
#lasolver.SetAztecParameter(daeAztecOptions.AZ_ilut_fill,    3.0)
#lasolver.SetAztecOption(daeAztecOptions.AZ_kspace,          500)
#lasolver.SetAztecOption(daeAztecOptions.AZ_overlap,         1)
#lasolver.SetAztecParameter(daeAztecOptions.AZ_athresh,      1e8)
#lasolver.SetAztecParameter(daeAztecOptions.AZ_rthresh,      0)

"""
Function to create and set-up the Trilinos linear equation solver. Possible choices:
 - Direct:
   {Amesos_KLU, Amesos_Superlu, Amesos_Umfpack, Amesos_Lapack}
 - Iterative:
   {AztecOO, AztecOO_Ifpack, AztecOO_ML}
"""
def CreateLASolver():
    #
    # 1) Amesos solver:
    #
    """
    lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu")

    paramListAmesos = pyTrilinos.TeuchosParameterList()

    # Amesos status options:
    paramListAmesos.set("OutputLevel", 0)
    paramListAmesos.set("DebugLevel", 0)
    paramListAmesos.set("PrintTiming", False)
    paramListAmesos.set("PrintStatus", False)
    paramListAmesos.set("ComputeVectorNorms", False)
    paramListAmesos.set("ComputeTrueResidual", False)

    # Amesos control options:
    paramListAmesos.set("AddZeroToDiag", False)
    paramListAmesos.set("AddToDiag", 0.0)
    paramListAmesos.set("Refactorize", False)
    paramListAmesos.set("RcondThreshold", 0.0)
    paramListAmesos.set("MaxProcs", 0)
    paramListAmesos.set("MatrixProperty", "")
    paramListAmesos.set("ScaleMethod", 0);
    paramListAmesos.set("Reindex", False)

    lasolver.SetAmesosOptions(paramListAmesos)
    """

    #
    # 2) AztecOO solver:
    #
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO_Ifpack", "PointRelaxation")
    paramListAztec = lasolver.GetAztecOOOptions()

    # AztecOO solver options:
    lasolver.NumIters  = 500
    lasolver.Tolerance = 1e-6
    paramListAztec.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
    paramListAztec.set_int("AZ_kspace",    500)
    paramListAztec.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_reorder",   0)
    paramListAztec.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
    paramListAztec.set_int("AZ_keep_info", 1)
    paramListAztec.set_int("AZ_output",    daeAztecOptions.AZ_all) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}

    #
    # 2a) AztecOO built-in preconditioner:
    #
    """
    paramListAztec.set_int("AZ_precond",         daeAztecOptions.AZ_dom_decomp)
    paramListAztec.set_int("AZ_subdomain_solve", daeAztecOptions.AZ_ilut)
    paramListAztec.set_int("AZ_overlap",         daeAztecOptions.AZ_none)
    paramListAztec.set_int("AZ_graph_fill",      1)
    paramListAztec.set_int("AZ_type_overlap",    daeAztecOptions.AZ_standard)
    paramListAztec.set_float("AZ_ilut_fill",     3.0)
    paramListAztec.set_float("AZ_drop",          0.0)
    paramListAztec.set_float("AZ_athresh",       1e8)
    paramListAztec.set_float("AZ_rthresh",       0.0)
    """
    paramListAztec.Print()

    #
    # 2b) Ifpack preconditioner:
    #

    paramListIfpack = lasolver.GetIfpackOptions()
    paramListIfpack.set_string("relaxation: type",               "Jacobi")
    paramListIfpack.set_float ("relaxation: min diagonal value", 1e-2)
    paramListIfpack.set_int   ("relaxation: sweeps",             5)
    #paramListIfpack.set_float("fact: ilut level-of-fill",   3.0)
    #paramListIfpack.set_float("fact: absolute threshold",   1e8)
    #paramListIfpack.set_float("fact: relative threshold",   0.0)
    paramListIfpack.Print()

    #
    # 2c) ML preconditioner:
    #
    """
    paramListML = lasolver.GetMLOptions()
    paramListML.set("reuse: enable", true)
    """

    return lasolver

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    la = CreateLASolver()
    simulator = daeSimulator(app, simulation=sim, lasolver=la)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    daesolver    = daeIDAS()
    lasolver     = CreateLASolver()
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

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
    else:
        consoleRun()
