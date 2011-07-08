#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             tutorial2.py
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
 - Arrays (discrete distribution domains)
 - Distributed parameters
 - Number of degrees of freedom and how to fix it
 - Initial guess of the variables
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in daeVariableTypes.py

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, "X axis domain")
        self.y  = daeDomain("y", self, "Y axis domain")

        # Nq is an array (a discrete distribution domain) with 2 elements
        self.Nq  = daeDomain("Nq", self, "Number of heat fluxes")

        # In this example the heat capacity is not a constant value but the temperature dependent (at every point in x and y domains).
        # To calculate cp a simple temperature dependency is proposed which depends on 2 parameters: a and b
        self.cp = daeVariable("c_p", specific_heat_capacity_t, self, "Specific heat capacity of the plate, J/kgK")
        self.cp.DistributeOnDomain(self.x)
        self.cp.DistributeOnDomain(self.y)

        self.a = daeParameter("a", eReal, self, "Coefficient for calculation of cp")
        self.b = daeParameter("b", eReal, self, "Coefficient for calculation of cp")

        # To introduce arrays (discrete distribution domains) parameters Qb and Qt are combined
        # into a single variable Q distributed on the domain Nq (that is as an array of fluxes)
        self.Q  = daeParameter("Q", eReal, self, "Heat flux array at the edges of the plate (bottom/top), W/m2")
        self.Q.DistributeOnDomain(self.Nq)

        # In this example the thermal conductivity is a distributed parameter (on domains x and y)
        self.k  = daeParameter("&lambda;",  eReal, self, "Thermal conductivity of the plate, W/mK")
        self.k.DistributeOnDomain(self.x)
        self.k.DistributeOnDomain(self.y)

        # In this example the density is now a variable
        self.ro = daeVariable("&rho;", density_t, self, "Density of the plate, kg/m3")

        # Domains that variables/parameters are distributed on can be specified in a constructor:
        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate, K", [self.x, self.y])
        # Another way would be by using DistributeOnDomain() function:
        #self.T.DistributeOnDomain(self.x)
        #self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.ro() * self.cp(x, y) * self.T.dt(x, y) - self.k(x, y) * \
                     (self.T.d2(self.x, x, y) + self.T.d2(self.y, x, y))

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        # Now we use Q(0) as the heat flux into the bottom edge
        eq.Residual = - self.k(x, y) * self.T.d(self.y, x, y) - self.Q(0)

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        # Now we use Q(1) as the heat flux at the top edge
        eq.Residual = - self.k(x, y) * self.T.d(self.y, x, y) - self.Q(1)

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        # Heat capacity as a function of the temperature
        eq = self.CreateEquation("C_p", "Equation to calculate the specific heat capacity of the plate as a function of the temperature.")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = self.cp(x,y) - self.a() - self.b() * self.T(x, y)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial2")
        self.m.Description = "This tutorial explains how to define Arrays (discrete distribution domains) and " \
                             "distributed parameters, how to calculate the number of degrees of freedom (NDOF) " \
                             "and how to fix it, and how to set initial guesses of the variables."

    def SetUpParametersAndDomains(self):
        n = 25

        self.m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1)

        # Nq is an array of size 2
        self.m.Nq.CreateArray(2)

        self.m.a.SetValue(367.0)
        self.m.b.SetValue(0.07)

        self.m.Q.SetValue(0, 1e6)
        self.m.Q.SetValue(1, 0.0)

        for x in range(0, self.m.x.NumberOfPoints):
            for y in range(0, self.m.y.NumberOfPoints):
                self.m.k.SetValue(x, y, 401)

    def SetUpVariables(self):
        # In the above model we defined 2*N*N+1 variables and 2*N*N equations,
        # meaning that the number of degrees of freedom (NDoF) is equal to: 2*N*N+1 - 2*N*N = 1
        # Therefore, we have to assign a value of one of the variables.
        # This variable cannot be chosen randomly, but must be chosen so that the combination
        # of defined equations and assigned variables produce a well posed system (that is a set of 2*N*N independent equations).
        # In our case the only candidate is ro. However, in more complex models there can be many independent combinations of variables.
        # The degrees of freedom can be fixed by assigning the variable value by using a function AssignValue:
        self.m.ro.AssignValue(8960)

        # To help the DAE solver it is possible to set initial guesses of of the variables.
        # Closer the initial guess is to the solution - faster the solver will converge to the solution
        # Just for fun, here we will try to obstruct the solver by setting the initial guess which is rather far from the solution.
        # Despite that, the solver will successfully initialize the system!
        self.m.T.SetInitialGuesses(1000)
        for x in range(self.m.x.NumberOfPoints):
            for y in range(self.m.y.NumberOfPoints):
                self.m.cp.SetInitialGuess(x, y, 1000)

        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300)

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

    # daeThirdPartyVariable class is used when interfacing DAE Tools with a 3rd party software.
    # The variable obviously must not be a state variable (that is it must have assigned value).
    # The class has attribute Value for a quick access to the variable value and it hides the
    # particulars about getting/setting a variable value.
    # It should be straightforward to create a wrapper around DAE Tools simulation,
    # which sets the inputs from the 3rd party software, runs a simulation with daeSimulation.ReRun()
    # and finally gets the results from DAE Tools. 
    # This procedure can be repeated as many times as necessary.
    ro = daeVariableWrapper(simulation.m.ro, 'ro')
    print ro.Name
    print ro.Value
    ro.Value = 9000
    print ro.Value

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
