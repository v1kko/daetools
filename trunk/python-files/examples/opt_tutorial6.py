#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             opt_tutorial6.py
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
This tutorial tests daeMinpackLeastSq module
"""

import sys
from numpy import *
from daetools.pyDAE import *
from daetools.solvers.daeMinpackLeastSq import *
from time import localtime, strftime
import matplotlib.pyplot as plt

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Model inputs
        self.x     = daeVariable("x", no_t, self)
        
        # Model outputs (measured quantities)
        self.y     = daeVariable("y", no_t, self)
        
        # Model parameters
        self.A     = daeVariable("A", no_t, self)
        self.k     = daeVariable("k", no_t, self)
        self.theta = daeVariable("&theta;", no_t, self)

    def DeclareEquations(self):
        eq = self.CreateEquation("y")
        eq.Residual = self.y() - self.A() * Sin(2 * pi * self.k() * self.x() + self.theta())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial6")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x.AssignValue(1)
        self.m.A.AssignValue(1)
        self.m.k.AssignValue(1)
        self.m.theta.AssignValue(1)

    def SetUpOptimization(self):
        # Obj. function must be a function of parameters
        self.ObjectiveFunctions[0].Residual = self.m.A() * Sin(2 * pi * self.m.k() * self.m.x() + self.m.theta())
        
        # Parameters must be defined as optimization variables (preferably continuous)
        self.A     = self.SetContinuousOptimizationVariable(self.m.A,     -10, 10, 0.7);
        self.k     = self.SetContinuousOptimizationVariable(self.m.k,     -10, 10, 0.8);
        self.theta = self.SetContinuousOptimizationVariable(self.m.theta, -10, 10, 1.9);

log          = daePythonStdOutLog()
daesolver    = daeIDAS()
datareporter = daeTCPIPDataReporter()
simulation   = simTutorial()

# Enable reporting of all variables
simulation.m.SetReportingOn(True)

# Set the time horizon and the reporting interval
simulation.ReportingInterval = 1
simulation.TimeHorizon = 5

# Connect data reporter
simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
if(datareporter.Connect("", simName) == False):
    sys.exit()

# After constructing simulation, daesolver, datareporter, log and 
# connecting the datareporter create daeMinpackLeastSq object
# It will call simulation.Initialize()
minpack = daeMinpackLeastSq(simulation, daesolver, datareporter, log)

# Save the model report and the runtime model report
simulation.m.SaveModelReport(simulation.m.Name + ".xml")
simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

# Measured variable (daeObjectiveFunction object)
y_measured_variable = simulation.ObjectiveFunction

# Model parameters (list of daeOptimizationVariable objects):
parameters = [simulation.A, simulation.k, simulation.theta]

# Model input variables (list of daeVariable objects; must be assigned variables):
x_input_variables = [simulation.m.x]

# Parameters values starting point: 
p0 = [8, 43.47826087, 1.047196667]

# Model inputs:
Ninputs = 1
Noutputs = 1
Nexperiments = 30

x_data = zeros((Ninputs, Nexperiments))
x_data[0] = arange(0, 0.06, 0.06 / Nexperiments)

# Model outputs (measured quantity):
y_data = array([ 5.95674236,  10.03610565,  10.14475642,   9.16722521,   8.52093929,
                 4.78842863,   2.87467755,  -3.93427325,  -6.13071010,  -9.26168083,
                -9.25272475, -10.42850414,  -4.71175587,  -3.60403013,  -0.11039750,
                 3.80372890,   8.51512082,   9.78232718,   9.91931747,   5.17108061,
                 6.47468360,   0.66528089,  -5.10344027,  -7.12668123,  -9.42080566,
                -8.23170543,  -6.56081590,  -6.28524014,  -2.30246340,  -0.79571452] )

try:
    # Initialize the MinpackLeastSq
    minpack.Initialize(parameters          = parameters,
                       x_input_variables   = x_input_variables,
                       y_measured_variable = y_measured_variable,
                       p0                  = p0,
                       x_data              = x_data,
                       y_data              = y_data)

    # Run
    minpack.Run()

    # Print the results and plot the comparison between the measured and fitted data
    print 'Status:', minpack.msg
    print 'Number of function evaluations =', minpack.infodict['nfev']
    print 'Root mean square deviation =', minpack.rmse
    print 'Estimated parameters values:', minpack.p_estimated

    y_fit = minpack.infodict['fvec'] + minpack.y_data
    plt.plot(minpack.x_data[0], y_fit, minpack.x_data[0], minpack.y_data, 'o')
    plt.title('Least-squares fit to experimental data')
    plt.legend(['Fit', 'Experimental'])
    plt.show()

except Exception, e:
    print str(e)
    
finally:
    minpack.Finalize()
   