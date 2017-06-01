#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial11.py
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
This tutorial describes the use of iterative linear solvers (AztecOO from the Trilinos project)
with different preconditioners (built-in AztecOO, Ifpack or ML) and corresponding solver options.
Also, the range of Trilins Amesos solver options are shown.

The model is very similar to the model in tutorial 1, except for the different boundary conditions
and that the equations are written in a different way to maximise the number of items around the
diagonal (creating the problem with the diagonally dominant matrix).
These type of systems can be solved using very simple preconditioners such as Jacobi. To do so,
the interoperability with the NumPy package has been exploited and the package itertools used to
iterate through the distribution domains in x and y directions.

The equations are distributed in such a way that the following incidence matrix is obtained:

.. code-block:: none

    |XXX                                 |
    | X     X     X                      |
    |  X     X     X                     |
    |   X     X     X                    |
    |    X     X     X                   |
    |   XXX                              |
    |      XXX                           |
    | X    XXX    X                      |
    |  X    XXX    X                     |
    |   X    XXX    X                    |
    |    X    XXX    X                   |
    |         XXX                        |
    |            XXX                     |
    |       X    XXX    X                |
    |        X    XXX    X               |
    |         X    XXX    X              |
    |          X    XXX    X             |
    |               XXX                  |
    |                  XXX               |
    |             X    XXX    X          |
    |              X    XXX    X         |
    |               X    XXX    X        |
    |                X    XXX    X       |
    |                     XXX            |
    |                        XXX         |
    |                   X    XXX    X    |
    |                    X    XXX    X   |
    |                     X    XXX    X  |
    |                      X    XXX    X |
    |                           XXX      |
    |                              XXX   |
    |                   X     X     X    |
    |                    X     X     X   |
    |                     X     X     X  |
    |                      X     X     X |
    |                                 XXX|

The temperature plot (at t=100s, x=0.5, y=*):

.. image:: _static/tutorial11-results.png
   :width: 500px
"""

import sys, numpy, itertools
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.solvers.trilinos import pyTrilinos
from daetools.solvers.aztecoo_options import daeAztecOptions

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, m, "X axis domain")
        self.y = daeDomain("y", self, m, "Y axis domain")

        self.Qb  = daeParameter("Q_b",         W/(m**2), self, "Heat flux at the bottom edge of the plate")
        self.Qt  = daeParameter("Q_t",         W/(m**2), self, "Heat flux at the top edge of the plate")
        self.rho = daeParameter("&rho;",      kg/(m**3), self, "Density of the plate")
        self.cp  = daeParameter("c_p",         J/(kg*K), self, "Specific heat capacity of the plate")
        self.k   = daeParameter("&lambda;_p",   W/(m*K), self, "Thermal conductivity of the plate")

        self.T = daeVariable("T", temperature_t, self)
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)
        self.T.Description = "Temperature of the plate"

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # For readability, get the adouble objects from parameters/variables
        # and create numpy arrays for T and its derivatives in tim and space
        # This will also save a lot of memory (no duplicate adouble objects in equations)
        Nx  = self.x.NumberOfPoints
        Ny  = self.y.NumberOfPoints
        rho = self.rho()
        cp  = self.cp()
        k   = self.k()
        Qb  = self.Qb()
        Qt  = self.Qt()

        # Create numpy ndarrays to keep daetools adouble objects:
        #   T, dT/dt, dT/dx, d2T/dx2, dT/dy and d2T/dy2
        T      = numpy.empty((Nx,Ny), dtype=object)
        dTdt   = numpy.empty((Nx,Ny), dtype=object)
        dTdx   = numpy.empty((Nx,Ny), dtype=object)
        dTdy   = numpy.empty((Nx,Ny), dtype=object)
        d2Tdx2 = numpy.empty((Nx,Ny), dtype=object)
        d2Tdy2 = numpy.empty((Nx,Ny), dtype=object)

        # Fill the ndarrays with daetools adouble objects:
        for x in range(Nx):
            for y in range(Ny):
                T[x,y]      = self.T(x,y)
                dTdt[x,y]   = dt(self.T(x,y))
                dTdx[x,y]   = d (self.T(x,y), self.x, eCFDM)
                dTdy[x,y]   = d (self.T(x,y), self.y, eCFDM)
                d2Tdx2[x,y] = d2(self.T(x,y), self.x, eCFDM)
                d2Tdy2[x,y] = d2(self.T(x,y), self.y, eCFDM)

        # Get the flat list of indexes from the ranges of indexes in x and y domains
        indexes = [(x,y) for x,y in itertools.product(range(Nx), range(Ny))]

        """
        Populate the equation types based on the location in the 2D domain:

          Y axis
            ^
            |
        Ly -| L T T T T T T T T T R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
            | L i i i i i i i i i R
         0 -| L B B B B B B B B B R
            --|-------------------|-------> X axis
              0                   Lx
        """
        eq_types = numpy.empty((Nx,Ny), dtype=object)
        eq_types[ : , : ] = 'i'  # inner region
        eq_types[ : ,  0] = 'B'  # bottom boundary
        eq_types[ : , -1] = 'T'  # top boundary
        eq_types[  0, : ] = 'L'  # left boundary
        eq_types[ -1, : ] = 'R'  # right boundary
        print(eq_types.T) # print it transposed to visalise it more easily

        # Finally, create equations based on the equation type
        for x,y in indexes:
            eq_type = eq_types[x,y]
            eq = self.CreateEquation("HeatBalance", "")
            if eq_type == 'i':
                eq.Residual = rho*cp*dTdt[x,y] - k*(d2Tdx2[x,y] + d2Tdy2[x,y])
                eq.Name = 'HeatBalance(inner)'

            elif eq_type == 'L':
                eq.Residual = dTdx[x,y]
                eq.Name = 'BC(left)'

            elif eq_type == 'R':
                eq.Residual = dTdx[x,y]
                eq.Name = 'BC(right)'

            elif eq_type == 'T':
                eq.Residual = -k*dTdy[x,y] - Qt
                eq.Name = 'BC(top)'

            elif eq_type == 'B':
                eq.Residual = -k*dTdy[x,y] - Qb
                eq.Name = 'BC(bottom)'

            else:
                raise RuntimeError('Invalid equation type: %s' % eq_type)

            eq.Name = eq.Name + '(%02d,%02d)' % (x,y)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial11")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateStructuredGrid(20, 0, 0.1)
        self.m.y.CreateStructuredGrid(20, 0, 0.1)

        self.m.k.SetValue(401 * W/(m*K))
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.rho.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6 * W/(m**2))
        self.m.Qt.SetValue(0 * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300 * K)

# Function to create the Trilinos linear equation solver.
# AztecOO solvers do not work well yet
def createLASolver():
    print("Supported Trilinos solvers:", str(pyTrilinos.daeTrilinosSupportedSolvers()))
    
    # Amesos SuperLU solver
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu", "")

    # AztecOO built-in preconditioners are specified through AZ_precond option
    lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "")

    # Ifpack preconditioner can be one of: [ILU, ILUT, PointRelaxation, BlockRelaxation, IC, ICT]
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO_Ifpack", "PointRelaxation")
    
    # ML preconditioner can be one of: [SA, DD, DD-ML, DD-ML-LU, maxwell, NSSA]
    #lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO_ML", "maxwell")
    
    return lasolver

# Function to set-up the Trilinos linear equation solver. Possible choices:
#  - Direct: {Amesos_KLU, Amesos_Superlu, Amesos_Umfpack, Amesos_Lapack}
#  - Iterative: {AztecOO, AztecOO_Ifpack, AztecOO_ML}
def setOptions(lasolver):
    #######################################################
    # Amesos_Superlu solver
    #######################################################
    if lasolver.Name == "Amesos_Superlu":
        paramListAmesos = lasolver.AmesosOptions

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
        paramListAztec = lasolver.AztecOOOptions

        lasolver.NumIters  = 500
        lasolver.Tolerance = 1e-3
        paramListAztec.set_int("AZ_solver",    daeAztecOptions.AZ_gmres)
        paramListAztec.set_int("AZ_kspace",    500)
        paramListAztec.set_int("AZ_scaling",   daeAztecOptions.AZ_none)
        paramListAztec.set_int("AZ_reorder",   0)
        paramListAztec.set_int("AZ_conv",      daeAztecOptions.AZ_r0)
        paramListAztec.set_int("AZ_keep_info", 1)
        paramListAztec.set_int("AZ_output",    daeAztecOptions.AZ_warnings) # {AZ_all, AZ_none, AZ_last, AZ_summary, AZ_warnings}
        paramListAztec.Print()

    #######################################################
    # AztecOO preconditioner options
    #######################################################
    if "AztecOO_Ifpack" in lasolver.Name:
        # 2b) Ifpack preconditioner:
        paramListIfpack = lasolver.IfpackOptions
        paramListIfpack.set_string("relaxation: type",               "Jacobi")
        paramListIfpack.set_float ("relaxation: min diagonal value", 1e-2)
        paramListIfpack.set_int   ("relaxation: sweeps",             5)
        #paramListIfpack.set_float("fact: ilut level-of-fill",        3.0)
        #paramListIfpack.set_float("fact: absolute threshold",        1e8)
        #paramListIfpack.set_float("fact: relative threshold",        0.0)
        paramListIfpack.Print()

    elif "AztecOO_ML" in lasolver.Name:
        # 2c) ML preconditioner:
        paramListML = lasolver.MLOptions
        paramListML.set_bool("reuse: enable", True)
        paramListML.Print()

    elif "AztecOO" in lasolver.Name:
        # 2a) AztecOO built-in preconditioner:
        paramListAztec = lasolver.AztecOOOptions
        paramListAztec.set_int("AZ_precond",         daeAztecOptions.AZ_Jacobi)
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
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
