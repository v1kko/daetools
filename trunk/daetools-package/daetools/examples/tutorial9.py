#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial9.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
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
In this example we use the same conduction problem as in the tutorial 1.

Here we introduce:

 - Third party linear equations solvers

Currently there are 3rd party linear equations solvers:

- SuperLU: sequential sparse direct solver defined in pySuperLU module (BSD licence)
- SuperLU_MT: multi-threaded sparse direct solver defined in pySuperLU_MT module (BSD licence)
- Trilinos: sequential sparse direct/iterative solver defined in pyTrilinos module (GNU Lesser GPL)
- IntelPardiso: multi-threaded sparse direct solver defined in pyIntelPardiso module (proprietary)
- Pardiso: multi-threaded sparse direct solver defined in pyPardiso module (proprietary)
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

try:
    # First import desired solver's module:
    from daetools.solvers.trilinos import pyTrilinos
    #from daetools.solvers.superlu import pySuperLU
    #from daetools.solvers.superlu_mt import pySuperLU_MT
    #from daetools.solvers.intel_pardiso import pyIntelPardiso
    #from daetools.solvers.pardiso import pyPardiso
except ImportError as e:
    print(('Unable to import LA solver: {0}'.format(e)))

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")
        self.y  = daeDomain("y", self, m, "Y axis domain")

        self.Qb = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Qt = daeParameter("Q_t",         W/(m**2), self, "Heat flux at the top edge of the plate")
        self.ro = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k  = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.ro() * self.cp() * self.T.dt(x, y) - self.k() * \
                     (self.T.d2(self.x, x, y) + self.T.d2(self.y, x, y))

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qb()

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qt()

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial9")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        n = 25

        self.m.x.CreateStructuredGrid(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateStructuredGrid(eCFDM, 2, n, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.ro.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Qt.SetValue(0 * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # The default linear solver is Sundials dense sequential solver (LU decomposition).
    # The following 3rd party direct linear solvers are supported:
    #   - Pardiso (multi-threaded - OMP)
    #   - IntelPardiso (multi-threaded - OMP)
    #   - SuperLU (sequential)
    #   - SuperLU_MT (multi-threaded - pthreads, OMP)
    #   - Trilinos Amesos (sequential): Klu, SuperLU, Lapack, Umfpack
    # If you are using Pardiso or IntelPardiso solvers you have to export their bin directories,
    # using, for instance, LD_LIBRARY_PATH shell variable (for more details see their documentation).
    # If you are using OpenMP capable solvers you should set the number of threads
    # (typically to the number of cores), for instance:
    #    export OMP_NUM_THREADS=4
    # or if using IntelPardiso solver:
    #    export MKL_NUM_THREADS=24
    # You can place the above command at the end of $HOME/.bashrc (or type it in shell, before simulation).

    # Import desired solver module (uncomment it from below) and set it using SetLASolver function:
    #print("Supported Trilinos 3rd party LA solvers:", str(pyTrilinos.daeTrilinosSupportedSolvers()))
    #lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
    lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu", "")
    #lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Lapack", "")
    #lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
    #lasolver     = pyIntelPardiso.daeCreateIntelPardisoSolver()
    #lasolver     = pyPardiso.daeCreatePardisoSolver()
    daesolver.SetLASolver(lasolver)

    """
    # Get Pardiso/IntelPardiso parameters (iparm[64] list of integers)
    iparm = lasolver.get_iparm()
    iparm_def = list(iparm) # Save it for comparison
    # Change some options
    # Achtung, Achtung!!
    # The meaning of items in iparm[64] is NOT identical in Pardiso and IntelPardiso solvers!!
    iparm[ 7] = 2 # Max. number of iterative refinement steps (common for both Pardiso and IntelPardiso)
    #iparm[27] = 1 # in Pardiso it means:      use METIS parallel reordering
                   # in IntelPardiso it means: use single precision (do not change it!)
    # Set them back
    lasolver.set_iparm(iparm)    
    iparm = lasolver.get_iparm()
    # Print the side by side comparison
    print('iparm     default new')
    for i in range(64):
        print 'iparm[%2d] %7d %3d' % (i, iparm_def[i], iparm[i])
    """
    
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
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
