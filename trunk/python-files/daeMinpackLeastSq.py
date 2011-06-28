#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             daeMinpackLeastSq.py
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

import sys
from numpy import *
from daetools.pyDAE import *
from time import localtime, strftime
from scipy.optimize import leastsq

class daeMinpackLeastSq:
    def __init__(self):
        self.simulation   = None
        self.daesolver    = None
        self.datareporter = None
        self.log          = None
        self.functions    = None
        self.parameters   = None
        self.x_inputs     = None
        self.y_outputs    = None
        self.x_data       = None
        self.y_data       = None
        self.Nparameters  = 0
        self.Nfunctions   = 0
        self.Ninputs      = 0
        self.Nexperiments = 0

    def Initialize(self, simulation, daesolver, datareporter, log, **kwargs):
        self.simulation   = simulation
        self.daesolver    = daesolver
        self.datareporter = datareporter
        self.log          = log
        
        self.functions    = kwargs.get('functions',  None)
        self.parameters   = kwargs.get('parameters', None)
        self.x_inputs     = kwargs.get('x_inputs',   None)
        self.y_outputs    = kwargs.get('y_outputs',  None)
        self.x_data       = kwargs.get('x_data',     None)
        self.y_data       = kwargs.get('y_data',     None)
        
        self.Nparameters  = len(parameters)
        self.Nfunctions   = len(functions)
        self.Ninputs      = len(x_inputs)
        self.Noutputs     = x_data.shape[0]
        
        if ( (self.functions == None) or (len(self.functions) == 0) ):
            raise RuntimeError('Argument [functions] has not been provided or the number of functions is zero') 
        if ( (self.parameters == None) or (len(self.parameters) == 0) ):
            raise RuntimeError('Argument [parameters] has not been provided or the number of parameters is zero') 
        if ( (self.x_inputs == None) or (len(self.x_inputs) == 0) ):
            raise RuntimeError('Argument [x_inputs] has not been provided or the number of x_inputs is zero') 
        if ( (self.y_outputs == None) or (len(self.y_outputs) == 0) ):
            raise RuntimeError('Argument [y_outputs] has not been provided or the number of y_outputs is zero') 
        if ( (self.x_data == None) or (len(self.x_data) == 0) ):
            raise RuntimeError('Argument [x_data] has not been provided or the number of x_data is zero') 
        if ( (self.y_data == None) or (len(self.y_data) == 0) ):
            raise RuntimeError('Argument [y_data] has not been provided or the number of y_data is zero') 

        if ( (self.x_data.ndim != 2) or (self.y_data.ndim != 2)):
            raise RuntimeError('Number of numpy array dimensions od x_data and y_data must be 2') 
        if ( (self.x_data.shape[0] != self.Noutputs) or (self.y_data.shape[0] != self.Noutputs) or (self.x_data.shape[0] != self.y_data.shape[0])):
            raise RuntimeError('The shapes of x_data and y_data do not match') 
        if ( (self.x_data.shape[1] != self.Ninputs) or (self.y_data.shape[1] != self.Nfunctions) ):
            raise RuntimeError('') 

        self.simulation.Initialize(self.daesolver, self.datareporter, self.log, CalculateSensitivities = True, NumberOfObjectiveFunctions = self.Nfunctions)

    def Run(self):
        pass
    
    def Finalize(self):
        pass
    
# Function to calculate either Residuals or Jacobian matrix, subject to the argument calc_values
def Function(p, simulation, xin, ymeas, calc_values):
    Nparams = len(p)
    Nexp    = len(xin)
    
    if(len(xin) != len(ymeas)):
        raise RuntimeError('The number of input data and the number of measurements must be equal') 
    
    values = zeros((Nexp))
    derivs = zeros((Nexp, Nparams))
    
    for e in range(0, Nexp):
        # Set initial conditions, initial guesses, initially active states etc
        # In this case it can be omitted; however, in general case it should be called
        simulation.SetUpVariables()
    
        # Assign the input data for the simulation
        simulation.m.x.ReAssignValue(xin[e])
        
        # Set the parameters values
        simulation.A.Value     = p[0]
        simulation.k.Value     = p[1]
        simulation.theta.Value = p[2]
        
        # Run the simulation
        simulation.Reset()
        simulation.SolveInitial()
        simulation.Run()
        
        # Get the results
        values[e]    = simulation.m.y.GetValue() - ymeas[e]
        derivs[e][:] = simulation.ObjectiveFunctions[0].Gradients
        
    print 'A =', simulation.A.Value, ', k =', simulation.k.Value, ', theta =', simulation.theta.Value
    if calc_values:
        print '  Residuals:'
        print values
    else:
        print '  Derivatives:'
        print derivs
    
    if calc_values:
        return values
    else:
        return derivs

# Function to calculate residuals R = ydata - f(xdata, params):
#   R[0], R[1], ..., R[n] 
def Residuals(p, simulation, xin, ymeas):
    return Function(p, simulation, xin, ymeas, True)
    
# Function to calculate a Jacobian for residuals: 
#   dR[0]/dp[0], dR[0]/dp[1], ..., dR[0]/dp[n] 
#   dR[1]/dp[0], dR[1]/dp[1], ..., dR[1]/dp[n] 
#   ...
#   dR[n]/dp[0], dR[n]/dp[1], ..., dR[n]/dp[n] 
def Derivatives(p, simulation, xin, ymeas):
    return Function(p, simulation, xin, ymeas, False)

# Function to calculate y  values for the estimated parameters
def peval(x, p):
    return p[0]*sin(2*pi*p[1]*x+p[2])



log          = daePythonStdOutLog()
daesolver    = daeIDAS()
datareporter = daeTCPIPDataReporter()
simulation   = simTutorial()

simulation.m.SetReportingOn(True)

simulation.ReportingInterval = 1
simulation.TimeHorizon = 5

simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
if(datareporter.Connect("", simName) == False):
    sys.exit()

simulation.Initialize(daesolver, datareporter, log, CalculateSensitivities = True, NumberOfObjectiveFunctions = 1)

simulation.m.SaveModelReport(simulation.m.Name + ".xml")
simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

# Exact values of the parameters
A, k, theta = [10, 33.33333333, 0.523598333]

# Starting point for parameters
p0 = [8, 43.47826087, 1.047196667]

# Input data for the model
x = arange(0, 0.06, 0.002)

# The values of y for given x and exact values of A, k, and theta
y_true = A * sin(2 * pi * k * x + theta)

# Measured values for y
y_meas = zeros_like(x)
y_meas = [ 5.95674236,  10.03610565,  10.14475642,   9.16722521,   8.52093929,
           4.78842863,   2.87467755,  -3.93427325,  -6.13071010,  -9.26168083,
          -9.25272475, -10.42850414,  -4.71175587,  -3.60403013,  -0.11039750,
           3.80372890,   8.51512082,   9.78232718,   9.91931747,   5.17108061,
           6.47468360,   0.66528089,  -5.10344027,  -7.12668123,  -9.42080566,
          -8.23170543,  -6.56081590,  -6.28524014,  -2.30246340,  -0.79571452]

# Call leastsq
p, cov_x, infodict, msg, ier = leastsq(Residuals, p0, Dfun=Derivatives, args=(simulation, x, y_meas), full_output=True)

# Print the results
print '------------------------------------------------------'
if ier in [1, 2, 3, 4]:
    print 'Solution found!'
else:
    print 'Least square method failed!'
print 'Status:', msg

print 'Number of function evaluations =', infodict['nfev']
chisq = (infodict['fvec']**2).sum()
dof = len(x) - len(p0)
rmse = sqrt(chisq / dof)
print 'Root mean square deviation =', rmse

A, k, theta = p
print 'Estimated parameters values:'
print '    A     =', A
print '    k     =', k
print '    theta =', theta
print '------------------------------------------------------'

# Plot the comparison between the exact values, measured and fitted data
plt.plot(x, peval(x, p), x, y_meas, 'o', x, y_true)
plt.title('Least-squares fit to experimental data')
plt.legend(['Fit', 'Experimental', 'Exact'])
plt.show()
