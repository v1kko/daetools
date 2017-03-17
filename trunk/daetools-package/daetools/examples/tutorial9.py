#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial9.py
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
This tutorial introduces the following concepts:

- Third party **direct** linear equations solvers

Currently there are the following linear equations solvers available:

- SuperLU: sequential sparse direct solver defined in pySuperLU module (BSD licence)
- SuperLU_MT: multi-threaded sparse direct solver defined in pySuperLU_MT module (BSD licence)
- Trilinos Amesos: sequential sparse direct solver defined in pyTrilinos module (GNU Lesser GPL)
- IntelPardiso: multi-threaded sparse direct solver defined in pyIntelPardiso module (proprietary)
- Pardiso: multi-threaded sparse direct solver defined in pyPardiso module (proprietary)

In this example we use the same conduction problem as in the tutorial 1.

The temperature plot (at t=100s, x=0.5, y=*):

.. image:: _static/tutorial9-results.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *
# First import desired solver's module:
from daetools.solvers.trilinos import pyTrilinos
#from daetools.solvers.superlu import pySuperLU
#from daetools.solvers.superlu_mt import pySuperLU_MT
#from daetools.solvers.intel_pardiso import pyIntelPardiso
#from daetools.solvers.pardiso import pyPardiso

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")
        self.y  = daeDomain("y", self, m, "Y axis domain")

        self.Qb  = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Tt  = daeParameter("T_t",                K, self, "Temperature at the top edge of the plate")
        self.rho = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp  = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k   = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Heat balance equation valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.rho() * self.cp() * dt(self.T(x,y)) - self.k() * \
                     (d2(self.T(x,y), self.x) + d2(self.T(x,y), self.y))

        eq = self.CreateEquation("BC_bottom", "Neumann boundary conditions at the bottom edge (constant flux)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * d(self.T(x,y), self.y, eCFDM) - self.Qb()

        eq = self.CreateEquation("BC_top", "Dirichlet boundary conditions at the top edge (constant temperature)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = self.T(x,y) - self.Tt()

        eq = self.CreateEquation("BC_left", "Neumann boundary conditions at the left edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

        eq = self.CreateEquation("BC_right", "Neumann boundary conditions at the right edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial9")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(20, 0, 0.1)
        self.m.y.CreateStructuredGrid(20, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.rho.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Tt.SetValue(300 * K)

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
    lasolver     = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
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
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
