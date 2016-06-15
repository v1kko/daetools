#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial3.py
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
In this example we use the same conduction problem as in the tutorial 1.

Here we introduce:

- Arrays of variable values
- Functions that operate on arrays of values
- Functions that create constants and arrays of constant values (Constant and Array)
- Non-uniform domain grids
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
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
        self.Tave   = daeVariable("T_ave",  temperature_t, self, "The average temperature")
        self.Qsum   = daeVariable("Q_sum",  heat_flux_t,   self, "The sum of heat fluxes at the bottom edge of the plate")
        self.Qsum1  = daeVariable("Q_sum1", heat_flux_t,   self, "The sum of heat fluxes at the bottom edge of the plate (more complicated way, as an illustration)")
        self.Qsum2  = daeVariable("Q_sum2", heat_flux_t,   self, "The sum of heat fluxes at the bottom edge of the plate (numpy version)")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # All equations use the default discretisation method (central finite difference)
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.ro() * self.cp() * dt(self.T(x,y)) - \
                      self.k() * (d2(self.T(x,y), self.x) + d2(self.T(x,y), self.y))

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * d(self.T(x,y), self.y) - self.Qb()

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = - self.k() * d(self.T(x,y), self.y) - self.Qt()

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = d(self.T(x,y), self.x)

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = d(self.T(x,y), self.x)

        # There are several functions that return arrays of values (or time- or partial-derivatives)
        # such as daeParameter and daeVariable functions array(), which return an array of parameter/variable values.
        # To obtain the array of values it is necessary to define points from all domains that the parameter
        # or variable is distributed on. Functions that return array of values accept the following arguments:
        #  - daeIndexRange objects
        #  - plain integers (to select a single index from a domain)
        #  - python lists (to select a list of indexes from a domain),
        #  - python slices (to select a range of indexes from a domain: start_index, end_index, step)
        #  - character '*' (to select all points from a domain)
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
        # To calculate the average and the sum of heat fluxes we can use functions 'Average' and 'Sum'.
        # Available functions are: Sum, Product, Average, Integral, Min, Max.

        eq = self.CreateEquation("T_ave", "The average temperature of the plate")
        eq.Residual = self.Tave() - Average( self.T.array( '*', '*' ) )

        # An equivalent to the equation above is:
        #   a) xr = daeIndexRange(self.x)
        #      yr = daeIndexRange(self.y)
        #      eq.Residual = self.Tave() - Average( self.T.array( xr, yr ) )
        #   b) eq.Residual = self.Tave() - Average( self.T.array( '*', '*' ) )
        #   c) eq.Residual = self.Tave() - Average( self.T.array( [], '*' ) )
        #   d) eq.Residual = self.Tave() - Average( self.T.array( '*', slice(0,-1) ) )
        #   e) eq.Residual = self.Tave() - Average( self.T.array( [],  slice(0,-1) ) )
        #
        # To select only certain points from a domain we can use a list or a slice:
        #   - self.T.array( '*', [1, 3, 7] )  returns all points from domain x and points 1,3,7 from domain y
        #   - self.T.array( '*', slice(3, 9, 2) )  returns all points from domain x and points 3,9,2 from domain y

        eq = self.CreateEquation("Q_sum", "The sum of heat fluxes at the bottom edge of the plate")
        Tarray = self.T.array('*', 0) # array of T values along x axis and for y = 0 (adouble_array object)
        eq.Residual = self.Qsum() + self.k() * Sum( d_array(Tarray, self.y) )
        
        Nx = self.x.NumberOfPoints
        # These equations are just a mental gymnastics to illustrate various functions such as
        # daeVariable's array() and d_array() and global functions Constant() and Array().
        #  - daeVariable.array() creates an array of values stored in a adouble_array object
        #  - daeVariable.d_array() creates an array of partial derivatives stored in a adouble_array object
        #  - The function Constant() creates a constant quantity that contains a value and units
        #  - The function Array() creates an array of constant quantities that contain a value and units
        # Both functions also accept plain floats (for instance, Constant(4.5) returns a dimensionless constant 4.5)
        #
        # The equation below expands into the following:
        #                                          ∂T(*, 0)
        #             [2K, 2K, 2K, ..., 2K] * k * ----------
        #                                             ∂y          2K         ∂T(0, 0)           2K         ∂T(xn, 0)
        # Qsum1 = -∑ ------------------------------------------ = ---- * k * ---------- + ... + ---- * k * -----------
        #                             2K                          2K           ∂y               2K            ∂y
        #
        # Achtung: the value of Qsum1 must be identical to Qsum!
        eq = self.CreateEquation("Q_mul", "Heat flux multiplied by a vector (units: K) and divided by a constant (units: K)")
        values = [2 * K for i in range(Nx)]  # creates list: [2K, 2K, 2K, ..., 2K] with length of x.NumberOfPoints
        Tarray = self.T.array('*', 0)        # array of T values along x axis and for y = 0 (adouble_array object)
        dTdy_array = d_array(Tarray, self.y) # array of dT/dy partial derivatives (adouble_array object)
        eq.Residual = self.Qsum1() + Sum( Array(values) * self.k() * dTdy_array / Constant(2 * K) )

        # Often, it is desired to apply numpy/scipy numerical functions on arrays of adouble objects.
        # In those cases the functions such as array(), d_array(), dt_array(), Array() etc
        # are NOT applicable since they return adouble_array objects.
        # However, we can create a numpy array of adouble objects, apply numpy functions on them
        # and finally create adouble_array object from resulting numpy arrays of adouble objects, if necessary.
        #
        # In this example, we will demonstrate interoperability between daetools and numpy.
        # As an illustration, we can construct a sum of heat fluxes at the bottom edge of the plate
        # (which should be identical to the previous Q_sum equation).
        # 1. First, create an empty numpy array as a container for daetools adouble objects
        Qbottom = numpy.empty(Nx, dtype=object)
        # 2. Then, fill the created numpy array with adouble objects.
        #    The result is: [k*dT[0,0]/dt, k*dT[1,0]/dt, ..., k*dT[Nx-1,0]/dt]
        #    There are two ways:
        #      - all items at once
        #      - item by item
        #    In this case, we populate all items at once
        Qbottom[:] = [self.k() * d(self.T(x,0), self.y) for x in range(Nx)]
        # 3. Finally, create an equation
        eq = self.CreateEquation("Q_sum2", "The sum of heat fluxes at the bottom edge of the plate (numpy version)")
        eq.Residual = self.Qsum2() + numpy.sum(Qbottom)

        # If adouble_aray is needed after operations on a numpy array we can use two functions:
        #   a) static function adouble_array.FromList(python-list)
        #   b) static function adouble_array.FromNumpyArray(numpy-array)
        # Both return an adouble_array object.
        ad_arr_Qbottom = adouble_array.FromNumpyArray(Qbottom)
        print('ad_arr_Qbottom:')
        print(ad_arr_Qbottom)
        
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        n = 10

        self.m.x.CreateStructuredGrid(n, 0, 0.1)
        self.m.y.CreateStructuredGrid(n, 0, 0.1)

        # Points of structured grids can be changed after the domain is defined by the CreateStructuredGrid function.
        # In certain situations it is desired to create a non-uniform grid within the given interval (LB, UB).
        # In these cases, a non-uniform grid can be specified by changing the daeDomain.Points property.
        # Good candidates for non-uniform grids are cases where there is a stiff front at one side of a domain.
        # First, create an uniform grid and then create a new list of points and assign it to the Points property.
        # Nota bene: the number of points must remain the same.
        old_grid = self.m.y.Points # numpy array
        new_grid = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.070, 0.100]
        self.Log.Message("  Before: %s" % old_grid, 0)
        self.m.y.Points = new_grid
        self.Log.Message("  After: %s" % new_grid, 0)
        
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
