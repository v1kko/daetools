#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             tutorial3.py
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
In this example we use the same conduction problem as in the tutorial 1.
Here we introduce:
 - Arrays of variable values
 - Functions that operate on arrays of values
 - Functions that create constants and arrays of constant values (Constant and Array)
 - Non-uniform domain grids
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in daeVariableTypes.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

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

        # Here we define two new variables to hold the average temperature and the sum of heat fluxes
        self.Tave = daeVariable("T_ave", temperature_t, self, "The average temperature")
        self.Qsum = daeVariable("Q_sum", heat_flux_t,   self, "The sum of heat fluxes at the bottom edge of the plate")
        self.Qmul = daeVariable("Q_mul", heat_flux_t,   self, "Heat flux multiplied by a vector (units: K) and divided by a constant (units: K)")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate, K")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
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

        # There are several function that return arrays of values (or time- or partial-derivatives)
        # such as daeParameter and daeVariable functions array(), which return an array of parameter/variable values
        # To obtain the array of values it is necessary to define points from all domains that the parameter
        # or variable is distributed on. Functions that return array of values accept the following arguments:
        #  - daeIndexRange objects
        #  - plain integers (to select a single index from a domain)
        #  - python lists (to select a list of indexes from a domain)
        #  - python slices (to select a range of indexes from a domain: start_index, end_index, step)
        #  - character '*' (to select all points from a domain)
        #  - integer -1 (to select all points from a domain)
        #  - empty python list [] (to select all points from a domain)
        #
        # daeIndexRange constructor has three variants:
        #  1. The first one accepts a single argument: Domain
        #     in that case the array will contain all points from the domains
        #  2. The second one accepts 2 arguments: Domain and Indexes
        #     the argument indexes is a list of indexes within the domain and the array will contain the values 
        #     at those points
        #  3. The third one accepts 4 arguments: Domain, StartIndex, EndIndex, Step
        #     Basically this defines a slice on the array of points in the domain
        #     StartIndex is the starting index, EndIndex is the last index and Step is used to iterate over
        #     this sub-domain [StartIndex, EndIndex). For example if we want values at even indexes in the domain
        #     we can write: xr = daeDomainIndex(self.x, 0, -1, 2)
        #
        # In this example we calculate:
        #  a) the average temperature of the plate
        #  b) the sum of heat fluxes at the bottom edge of the plate (at y = 0)
        #
        # To calculate the average and the sum of heat fluxes we can use functions 'average' and 'sum' from daeModel class.
        # For the list of all available functions please have a look on pyDAE API Reference, module Core.

        eq = self.CreateEquation("T_ave", "The average temperature of the plate")
        eq.Residual = self.Tave() - self.average( self.T.array( '*', '*' ) )
        # An equivalent to the equation above is:
        #   a) xr = daeIndexRange(self.x)
        #      yr = daeIndexRange(self.y)
        #      eq.Residual = self.Tave() - self.average( self.T.array( xr, yr ) )
        #   b) eq.Residual = self.Tave() - self.average( self.T.array( '*', -1 ) )
        #   c) eq.Residual = self.Tave() - self.average( self.T.array( [], '*' ) )
        #   d) eq.Residual = self.Tave() - self.average( self.T.array( -1, slice(0, -1) ) )
        #
        # To select only certain points from a domain we can use a list or a slice:
        #   - self.T.array( '*', [1, 3, 7] )  returns all points from domain x and points 1, 3, 7 from domain y 
        #   - self.T.array( '*', slice(3, 9, 2) )  returns all points from domain x and points 3, 5, 7 from domain y 

        eq = self.CreateEquation("Q_sum", "The sum of heat fluxes at the bottom edge of the plate")
        eq.Residual = self.Qsum() + self.k() * self.sum( self.T.d_array(self.y, '*', 0) )
        
        # This equations is just a mental gymnastics to illustrate various functions (array, Constant, Vector)
        #  - The function Constant() creates a constant quantity that contains a value and units 
        #  - The function Array() creates an array of constant quantities that contain a value and units
        # Both functions accept plain floats (for instance, Constant(4.5) returns a dimensionless constant 4.5)
        #
        # The equation below expands into the following:
        #                                          ∂T(*, 0)
        #             [2K, 2K, 2K, ..., 2K] * k * ----------
        #                                             ∂y          2K         ∂T(0, 0)           2K         ∂T(xn, 0)
        # Qmul = -∑ ------------------------------------------ = ---- * k * ---------- + ... + ---- * k * -----------
        #                             2K                          2K           ∂y               2K            ∂y
        #
        # Achtung: the value of Qmul must be identical to Qsum!
        eq = self.CreateEquation("Q_mul", "Heat flux multiplied by a vector (units: K) and divided by a constant (units: K)")
        values = [2 * K for i in xrange(self.x.NumberOfPoints)] # creates list: [2K, 2K, 2K, ..., 2K] with length of x.NumberOfPoints
        eq.Residual = self.Qmul() + self.sum( Array(values) * self.k() * self.T.d_array(self.y, '*', 0) / Constant(2 * K) )

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial3")
        self.m.Description = "This tutorial explains how to define arrays of variable values and " \
                             "functions that operate on these arrays, constants and vectors, " \
                             "and how to define a non-uniform domain grid."

    def SetUpParametersAndDomains(self):
        n = 10

        self.m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1)

        # Points in distributed domains can be changed after the domain is defined by the CreateDistributed function.
        # In certain situations it is not desired to have a uniform distribution of the points within the given interval (LB, UB)
        # In these cases, a non-uniform grid can be specified by using the Points property od daeDomain.
        # A good candidates for the non-uniform grid are cases where we have a very stiff fronts at one side of the domain.
        # In these cases it is desirable to place more points at that part od the domain.
        # Here, we first print the points before changing them and then set the new values.
        self.Log.Message("  Before:" + str(self.m.y.Points), 0)
        self.m.y.Points = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.070, 0.100]
        self.Log.Message("  After:" + str(self.m.y.Points), 0)

        self.m.k.SetValue(401   * W/(m*K))
        self.m.cp.SetValue(385  * J/(kg*K))
        self.m.ro.SetValue(8960 * kg/(m**3))
        self.m.Qb.SetValue(1e6  * W/(m**2))
        self.m.Qt.SetValue(0    * W/(m**2))

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300*K)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 5
    sim.TimeHorizon       = 200
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 5
    simulation.TimeHorizon = 200

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
