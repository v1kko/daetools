#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial14.py
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
***********************************************************************************
"""
__doc__ = """
In this tutorial we introduce the external functions concept that can handle and execute
functions in external libraries. The daeScalarExternalFunction-derived external function
object is used to calculate the heat transferred and to interpolate a set of values
using the scipy.interpolate.interp1d object.

In this example we use the same model as in the tutorial 5 with few additional equations.

The simulation output should show the following messages at the end of simulation:

.. code-block:: none

   ...
   scipy.interp1d statistics:
     interp1d called 1703 times (cache value used 770 times)

The plot of the 'Heat_ext' variable:

.. image:: _static/tutorial14-results.png
   :width: 500px

The plot of the 'Value_interp' variable:

.. image:: _static/tutorial14-results2.png
   :width: 500px
"""

import sys
import numpy, scipy.interpolate
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class extfnHeatTransferred(daeScalarExternalFunction):
    def __init__(self, Name, Model, units, m, cp, dT):
        # Instantiate the scalar external function by specifying
        # the arguments dictionary {'name' : adouble-object}
        arguments = {}
        arguments["m"]  = m
        arguments["cp"] = cp
        arguments["dT"] = dT

        daeScalarExternalFunction.__init__(self, Name, Model, units, arguments)
    
    def Calculate(self, values):
        # Calculate function is used to calculate a value and a derivative (if requested)
        # of the external function per given argument. Here the simple function is given by:
        #    f(m, cp, dT/dt) = m * cp * dT/dt

        # Procedure:
        # 1. Get the arguments from the dictionary values: {'arg-name' : adouble-object}.
        #    Every adouble object has two properties: Value and Derivative that can be
        #    used to evaluate function or its partial derivatives per its arguments
        #    (partial derivatives are used to fill in a Jacobian matrix necessary to solve
        #    a system of non-linear equations using the Newton method).
        m  = values["m"]
        cp = values["cp"]
        dT = values["dT"]
        
        # 2. Always calculate the value of a function (derivative part is zero by default)
        res = adouble(m.Value * cp.Value * dT.Value)
        
        # 3. If a function derivative per one of its arguments is requested,
        #    a derivative part of that argument will be non-zero.
        #    In that case, investigate which derivative is requested and calculate it
        #    using the chain rule: f'(x) = x' * df(x)/dx
        if m.Derivative != 0:
            # A derivative per 'm' was requested
            res.Derivative = m.Derivative * (cp.Value * dT.Value)
        elif cp.Derivative != 0:
            # A derivative per 'cp' was requested
            res.Derivative = cp.Derivative * (m.Value * dT.Value)
        elif dT.Derivative != 0:
            # A derivative per 'dT' was requested
            res.Derivative = dT.Derivative * (m.Value * cp.Value)
        
        #print('Heat(m=(%f,%f), cp=(%f,%f), dT=(%f,%f)) = (%f,%f)' % (m.Value,m.Derivative,
        #                                                             cp.Value,cp.Derivative,
        #                                                             dT.Value,dT.Derivative,
        #                                                             res.Value,res.Derivative))

        # 4. Return the result as a adouble object (contains both value and derivative)
        return res
        
class extfn_interp1d(daeScalarExternalFunction):
    def __init__(self, Name, Model, units, times, values, Time):
        arguments = {}
        arguments["t"]  = Time

        # Instantiate interp1d object and initialize interpolation using supplied (x,y) values
        self.interp = scipy.interpolate.interp1d(times, values)

        # During the solver iterations, the function is called very often with the same arguments
        # Therefore, cache the last interpolated value to speed up a simulation
        self.cache = None

        # Counters for performance (just an info; not really needed)
        self.counter       = 0
        self.cache_counter = 0

        daeScalarExternalFunction.__init__(self, Name, Model, units, arguments)

    def Calculate(self, values):
        # Increase the call counter every time the function is called
        self.counter += 1

        # Get the argument from the dictionary of arguments' values.
        time = values["t"].Value

        # Here we do not need to return a derivative for it is not a function of variables.
        # See the remarks above if thats not the case.

        # First check if an interpolated value was already calculated during the previous call
        # If it was return the cached value (derivative part is always equal to zero in this case)
        if self.cache:
            if self.cache[0] == time:
                self.cache_counter += 1
                return adouble(self.cache[1])
                
        # The time received is not in the cache and has to be interpolated.
        # Convert the result to float datatype since daetools can't accept
        # numpy.float64 types as arguments at the moment.
        interp_value = float(self.interp(time))
        res = adouble(interp_value, 0)

        # Save it in the cache for later use
        self.cache = (time, res.Value)

        return res
        
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin   = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T     = daeVariable("T",     temperature_t, self, "Temperature of the plate")
        
        self.Heat     = daeVariable("Heat",     power_t, self, "Heat transferred")
        self.Heat_ext = daeVariable("Heat_ext", power_t, self, "Heat transferred calculated using an external function")

        self.Value        = daeVariable("Value",        time_t, self, "Simple value")
        self.Value_interp = daeVariable("Value_interp", time_t, self, "Simple value calculated using an external function that wraps scipy.interp1d")
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        #
        # Scalar external function #1
        #
        # Create external function
        # It has to be created in DeclareEquations since it accesses the params/vars values
        self.exfnHeat = extfnHeatTransferred("Heat", self, W, self.m(), self.cp(), dt(self.T()))

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        eq = self.CreateEquation("Heat", "")
        eq.Residual = self.Heat() - self.m() * self.cp() * dt(self.T())

        eq = self.CreateEquation("Heat_ext", "")
        eq.Residual = self.Heat_ext() - self.exfnHeat()

        #
        # Scalar external function #2
        #
        # Create scipy interp1d interpolation external function
        times  = numpy.arange(0.0, 1000.0)
        values = 2*times
        self.interp1d = extfn_interp1d("interp1d", self, s, times, values, Time())

        eq = self.CreateEquation("Value", "")
        eq.Residual = self.Value() - 2*Time()

        eq = self.CreateEquation("Value_interp", "")
        eq.Residual = self.Value_interp() - self.interp1d()

        ####################################################
        self.stnRegulator = self.STN("Regulator")

        self.STATE("Heating")

        eq = self.CreateEquation("Q_in", "The heater is on")
        eq.Residual = self.Qin() - Constant(1500 * W)

        # Here the Time() function is used to get the current time (time elapsed) in the simulation
        self.SWITCH_TO("Cooling",   self.T() > Constant(340 * K))
        self.SWITCH_TO("HeaterOff", Time()   > Constant(350 * s))

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.SWITCH_TO("Heating",   self.T() < Constant(320 * K))
        self.SWITCH_TO("HeaterOff", Time()   > Constant(350 * s))

        self.STATE("HeaterOff")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.END_STN()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial14")
        self.m.Description = __doc__
                                
    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        # Set the state active at the beginning (the default is the first declared state; here 'Heating')
        self.m.stnRegulator.ActiveState = "Heating"

        self.m.T.SetInitialCondition(283 * K)


# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 0.5
    sim.TimeHorizon       = 500
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
    simulation.ReportingInterval = 0.5
    simulation.TimeHorizon = 500

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

    print('\n\nscipy.interp1d statistics:')
    print('  interp1d called %d times (cache value used %d times)' % (simulation.m.interp1d.counter, simulation.m.interp1d.cache_counter))

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
