#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial14.py
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
***********************************************************************************
"""
__doc__ = """
In this tutorial we introduce the external functions concept that can handle and execute
functions in external libraries. The daeScalarExternalFunction-derived external function
object is used to calculate the heat transferred and to interpolate a set of values
using the scipy.interpolate.interp1d object. In addition, functions defined in shared 
libraries (.so in GNU/Linux, .dll in Windows and .dylib in macOS) can be used via
ctypes Python library and daeCTypesExternalFunction class.

In this example we use the same model as in the tutorial 5 with few additional equations.

The simulation output should show the following messages at the end of simulation:

.. code-block:: none

   ...
   scipy.interp1d statistics:
     interp1d called 1703 times (cache value used 770 times)

The plot of the 'Heat_ext1' variable:

.. image:: _static/tutorial14-results.png
   :width: 500px

The plot of the 'Heat_ext2' variable:

.. image:: _static/tutorial14-results1.png
   :width: 500px

The plot of the 'Value_interp' variable:

.. image:: _static/tutorial14-results2.png
   :width: 500px
"""

import os, sys, platform, ctypes
import numpy, scipy.interpolate
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class extfnHeatTransferred(daeScalarExternalFunction):
    def __init__(self, Name, Model, units, m, cp, dT_dt):
        # Instantiate the scalar external function by specifying
        # the arguments dictionary {'name' : adouble-object}
        arguments = {}
        arguments["m"]     = m
        arguments["cp"]    = cp
        arguments["dT/dt"] = dT_dt

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
        m     = values["m"]
        cp    = values["cp"]
        dT_dt = values["dT/dt"]
        
        # 2. Always calculate the value of a function (derivative part is zero by default)
        res = adouble(m.Value * cp.Value * dT_dt.Value)
        
        # 3. If a function derivative per one of its arguments is requested,
        #    a derivative part of that argument will be non-zero.
        #    In that case, investigate which derivative is requested and calculate it
        #    using the chain rule: f'(x) = x' * df(x)/dx
        if m.Derivative != 0:
            # A derivative per 'm' was requested
            res.Derivative = m.Derivative * (cp.Value * dT_dt.Value)
        elif cp.Derivative != 0:
            # A derivative per 'cp' was requested
            res.Derivative = cp.Derivative * (m.Value * dT_dt.Value)
        elif dT_dt.Derivative != 0:
            # A derivative per 'dT_dt' was requested
            res.Derivative = dT_dt.Derivative * (m.Value * cp.Value)
        
        #print('Heat(m=(%f,%f), cp=(%f,%f), dT_dt=(%f,%f)) = (%f,%f)' % (m.Value,m.Derivative,
        #                                                             cp.Value,cp.Derivative,
        #                                                             dT_dt.Value,dT_dt.Derivative,
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
        
        self.Heat      = daeVariable("Heat",      power_t, self, "Heat transferred")
        self.Heat_ext1 = daeVariable("Heat_ext1", power_t, self, "Heat transferred calculated using an external function")
        self.Heat_ext2 = daeVariable("Heat_ext2", power_t, self, "Heat transferred calculated using an external function")

        self.Value        = daeVariable("Value",        time_t, self, "Simple value")
        self.Value_interp = daeVariable("Value_interp", time_t, self, "Simple value calculated using an external function that wraps scipy.interp1d")
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        eq = self.CreateEquation("Heat", "")
        eq.Residual = self.Heat() - self.m() * self.cp() * dt(self.T())

        #
        # Scalar external function #1
        #
        # Create external function
        # It has to be created in DeclareEquations since it accesses the params/vars values
        self.exfnHeat1 = extfnHeatTransferred("Heat", self, W, self.m(), self.cp(), dt(self.T()))

        eq = self.CreateEquation("Heat_ext1", "")
        eq.Residual = self.Heat_ext1() - self.exfnHeat1()

        #
        # Scalar external function #2
        #
        # Create ctypes external function
        # Use the function calculate from the shared library.
        plat = str(platform.system())
        if plat == 'Linux':
            lib_name = 'libheat_function.so'
        elif plat == 'Darwin':
            lib_name = 'libheat_function.dylib'
        elif plat == 'Windows':
            lib_name = 'heat_function.dll'
        else:
            lib_name = 'unknown'
        lib_dir  = os.path.realpath(os.path.dirname(__file__))
        lib_path = os.path.join(lib_dir, lib_name)
        # Load the shared library using ctypes.
        self.ext_lib = ctypes.CDLL(lib_path)
        
        # Arguments for the external function.
        arguments = {}
        arguments['m']     = self.m() 
        arguments['cp']    = self.cp()
        arguments['dT/dt'] = dt(self.T())
        
        # Function pointer, here we use 'calculate' function defined in the 'heat_function' shared library.
        function_ptr = self.ext_lib.calculate
        
        self.exfnHeat2 = daeCTypesExternalFunction("heat_function", self, W, function_ptr, arguments)

        eq = self.CreateEquation("Heat_ext2", "")
        eq.Residual = self.Heat_ext2() - self.exfnHeat2()

        #
        # Scalar external function #3
        #
        # Create scipy interp1d interpolation external function
        times  = numpy.arange(0.0, 1000.0)
        values = 2*times
        self.interp1d = extfn_interp1d("interp1d", self, s, times, values, Time())
        
        # Alternatively, C++ implementation of 1D linear interpolation in daeLinearInterpolationFunction can be used.
        #self.interp1d = daeLinearInterpolationFunction("daetools_interp1d", self, s, times.tolist(), values.tolist(), Time())

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
        self.ON_CONDITION(self.T() > Constant(340 * K), switchToStates = [ ('Regulator', 'Heating') ])
        self.ON_CONDITION(Time()   > Constant(350 * s), switchToStates = [ ('Regulator', 'HeaterOff') ])

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.ON_CONDITION(self.T() < Constant(320 * K), switchToStates = [ ('Regulator', 'Heating') ])
        self.ON_CONDITION(Time()   > Constant(350 * s), switchToStates = [ ('Regulator', 'HeaterOff') ])

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

def run(**kwargs):
    simulation = simTutorial()
   
    res = daeActivity.simulate(simulation, reportingInterval = 0.5, 
                                           timeHorizon       = 500,
                                           **kwargs)
    # Print some interp1d stats
    print('\n\nscipy.interp1d statistics:')
    print('  interp1d called %d times (cache value used %d times)' % (simulation.m.interp1d.counter, simulation.m.interp1d.cache_counter))
    
    return res

if __name__ == "__main__":
    guiRun = False if (len(sys.argv) > 1 and sys.argv[1] == 'console') else True
    run(guiRun = guiRun)
