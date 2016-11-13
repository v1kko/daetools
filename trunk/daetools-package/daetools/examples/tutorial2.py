#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial2.py
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
This tutorial introduces the following concepts:

- Arrays (discrete distribution domains)
- Distributed parameters
- Degrees of freedom
- Setting an initial guess for variables (used by a DAE solver during an initial phase)

The model in this example is very similar to the model used in the tutorial 1.
The differences are:

- The heat capacity is temperature dependent
- Different boundary conditions are applied

The temperature plot (at t=100s, x=0.5, y=*):

.. image:: _static/tutorial2-results.png
   :width: 500px
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kW

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, m, "X axis domain")
        self.y  = daeDomain("y", self, m, "Y axis domain")

        # Nq is an array (a discrete distribution domain) with 2 elements
        self.Nq  = daeDomain("Nq", self, unit(), "Number of heat fluxes")

        # In this example the heat capacity is not a constant value but the temperature dependent (at every point in x and y domains).
        # To calculate cp a simple temperature dependency is proposed which depends on 2 parameters: a and b
        self.cp = daeVariable("c_p", specific_heat_capacity_t, self, "Specific heat capacity of the plate")
        self.cp.DistributeOnDomain(self.x)
        self.cp.DistributeOnDomain(self.y)

        self.a = daeParameter("a", J/(kg*K),      self, "Coefficient for calculation of cp")
        self.b = daeParameter("b", J/(kg*(K**2)), self, "Coefficient for calculation of cp")

        # To introduce arrays (discrete distribution domains) parameters Qb and Qt are combined
        # into a single variable Q distributed on the domain Nq (that is as an array of fluxes)
        self.Q  = daeParameter("Q", W/(m**2), self, "Heat flux array at the edges of the plate (bottom/top)")
        self.Q.DistributeOnDomain(self.Nq)

        # In this example the thermal conductivity is a distributed parameter (on domains x and y)
        self.k  = daeParameter("&lambda;",  W/(m*K), self, "Thermal conductivity of the plate")
        self.k.DistributeOnDomain(self.x)
        self.k.DistributeOnDomain(self.y)

        # In this example the density is now a variable
        self.rho = daeVariable("&rho;", density_t, self, "Density of the plate")

        # Domains that variables/parameters are distributed on can be specified in a constructor:
        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate", [self.x, self.y])
        # Another way of distributing a variable would be by using DistributeOnDomain() function:
        #self.T.DistributeOnDomain(self.x)
        #self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Heat balance equation valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.rho() * self.cp(x, y) * self.T.dt(x, y) - self.k(x, y) * \
                     (d2(self.T(x,y), self.x, eCFDM) + d2(self.T(x,y), self.y, eCFDM))

        eq = self.CreateEquation("BC_bottom", "Neumann boundary conditions at the bottom edge (constant flux)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        # Now we use Q(0) as the heat flux into the bottom edge
        eq.Residual = - self.k(x, y) * d(self.T(x,y), self.y, eCFDM) - self.Q(0)

        eq = self.CreateEquation("BC_top", "Neumann boundary conditions at the top edge (constant flux)")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        # Now we use Q(1) as the heat flux at the top edge
        eq.Residual = - self.k(x, y) * d(self.T(x,y), self.y, eCFDM) - self.Q(1)

        eq = self.CreateEquation("BC_left", "Neumann boundary conditions at the left edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

        eq = self.CreateEquation("BC_right", " Neumann boundary conditions at the right edge (insulated)")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = d(self.T(x,y), self.x, eCFDM)

        # Heat capacity as a function of the temperature
        eq = self.CreateEquation("C_p", "Equation to calculate the specific heat capacity of the plate as a function of the temperature.")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eClosedClosed)
        eq.Residual = self.cp(x,y) - self.a() - self.b() * self.T(x, y)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial2")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        n = 10

        self.m.x.CreateStructuredGrid(n, 0, 0.1)
        self.m.y.CreateStructuredGrid(n, 0, 0.1)

        # Nq is an array of size 2
        self.m.Nq.CreateArray(2)

        self.m.a.SetValue(367.0 * J/(kg*K))
        self.m.b.SetValue(0.07  * J/(kg*(K**2)))

        self.m.Q.SetValue(0, 1e6 * W/(m**2))
        self.m.Q.SetValue(1, 0.0 * W/(m**2))

        Nx = self.m.x.NumberOfPoints
        Ny = self.m.y.NumberOfPoints
        
        # There are several ways to set a value of distributed parameters:
        #  a) Call SetValues(ndarray-of-floats-or-quantities) to set all values at once
        #  b) Call SetValues(float/quantity) to set all values to the same value
        #  c) In a loop call SetValue([index1, index2], float/quantity) to set a value for individual points
        #  d) In a loop call SetValue(index1, index2, float/quantity) to set a value for individual points
        
        # All the following four ways are equivalent:
        # a) Use an array of quantity objects
        #    Use numpy to create an empty two-dimensional numpy array with datatype=object and set all values to 0.401 kW/(m*K).
        #    The values will be converted to the units in the parameter 'k': W/(m*K) thus the value will be 401 W/(m*K).
        k = numpy.empty((Nx,Ny), dtype=object)
        k[:] = 0.401 * kW/(m*K)
        print('Parameter lambda values:')
        print(str(k))
        self.m.k.SetValues(k)
        
        # b) Use a single float value for all points (the units are implicitly W/(m*K)):
        #self.m.k.SetValues(401)
        
        # c) Iterate over domains and use a list of indexes to set values for individual points:
        #for x in range(Nx):
        #    for y in range(Ny):
        #        self.m.k.SetValue([x,y], 401 * W/(m*K))

        # d) Iterate over domains and set values for individual points:
        #for x in range(Nx):
        #    for y in range(Ny):
        #        self.m.k.SetValue(x, y, 401 * W/(m*K))
        
    def SetUpVariables(self):
        Nx = self.m.x.NumberOfPoints
        Ny = self.m.y.NumberOfPoints
        
        # In the above model we defined 2*N*N+1 variables and 2*N*N equations,
        # meaning that the number of degrees of freedom (DOF) is equal to: 2*N*N+1 - 2*N*N = 1
        # Therefore, we have to assign a value of one of the variables.
        # This variable cannot be chosen randomly, but must be chosen so that the combination
        # of defined equations and assigned variables produce a well posed system (that is a set of 2*N*N independent equations).
        # In our case the only candidate is rho. However, in more complex models there can be many independent combinations of variables.
        # The degrees of freedom can be fixed by assigning the variable value by using a function AssignValue:
        self.m.rho.AssignValue(8960 * kg/(m**3))

        # To help the DAE solver it is possible to set initial guesses of of the variables.
        # Closer the initial guess is to the solution - faster the solver will converge to the solution
        # Just for fun, here we will try to obstruct the solver by setting the initial guess which is rather far from the solution.
        # Despite that, the solver will successfully initialize the system! 
        # There are several ways to do it:
        #  a) SetInitialGuesses(float/quantity) - in a single call sets all to the same value:
        #          self.m.T.SetInitialGuesses(1000*K)
        #  b) SetInitialGuesses(ndarray-of-floats-or-quantities) - in a single call sets individual values:
        #          self.m.T.SetInitialGuesses([1000, 1001, ...])
        #  c) SetInitialGuess(index1, index2, ..., float/quantity) - sets an individual value:
        #          self.m.T.SetInitialGuess(1, 3, 1000*K)
        #  d) SetInitialGuess(list-of-indexes, float/quantity) - sets an individual value:
        #          self.m.T.SetInitialGuess([1, 3], 1000*K)
        # The following daeVariable functions can be called in a similar fashion: 
        #  - AssignValue(s) and ReAssignValue(s)
        #  - SetInitialCondition(s) and ReSetInitialCondition(s)
        #  - SetInitialGuess(es)
        # and the following daeParameter functions:
        #  - SetValue(s) and GetValue
        #
        # All the following calls are equivalent:
        # a) Use a single value
        self.m.T.SetInitialGuesses(1000 * K)
        
        # b) Use an array of quantity objects:
        #    Again, use numpy to create an empty two-dimensional numpy array with datatype=object and set all values to 1000 K.
        #init_guesses = numpy.empty((Nx,Ny), dtype=object)
        #init_guesses[:] = 1000 * K 
        #self.m.T.SetInitialGuesses(init_guesses)        
        
        # c) Loop over domains to set an initial guess for individual points
        for x in range(Nx):
            for y in range(Ny):
                self.m.cp.SetInitialGuess(x, y, 1000 * J/(kg*K))

        # Initial conditions can be set in all four above-mentioned ways.
        # Note: In this case init. conditions must be set for open-ended domains (excluding boundary points).
        # a) Use an array of quantity objects:
        #    Again, use numpy to create an empty two-dimensional array with datatype=object.
        #    However we do not set ALL values to 300 K but only those that correspond to the points in the domain interior,
        #    thus leaving points at the boundaries to None (which by design means they will be excluded).
        ic = numpy.empty((Nx,Ny), dtype=object)
        ic[1:Nx-1, 1:Ny-1] = 300 * K
        print('Initial conditions for T:')
        print(ic)
        self.m.T.SetInitialConditions(ic)        
        
        # b) Loop over domains to set initial conditions for individual points
        #for x in range(1, Nx-1):
        #    for y in range(1, Ny-1):
        #        self.m.T.SetInitialCondition([x,y], 300 * K)
                
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
