#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            opt_tutorial5.py
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
This tutorial shows the interoperability between DAE Tools and 3rd party optimization 
software (scipy.optimize) used to fit the simple function with experimental data.

DAE Tools simulation object is used to calculate the objective function and its gradients,
while scipy.optimize.leastsq function (a wrapper around MINPACK’s lmdif and lmder)
implementing Levenberg-Marquardt algorithm is used to estimate the parameters.
"""

import sys, numpy
from time import localtime, strftime
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from daetools.pyDAE import *

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x     = daeVariable("x", no_t, self)
        self.A     = daeVariable("A", no_t, self)
        self.k     = daeVariable("k", no_t, self)
        self.theta = daeVariable("&theta;", no_t, self)
        self.y     = daeVariable("y", no_t, self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("y")
        eq.Residual = self.y() - self.A() * Sin(2 * numpy.pi * self.k() * self.x() + self.theta())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial5")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x.AssignValue(1)
        self.m.A.AssignValue(1)
        self.m.k.AssignValue(1)
        self.m.theta.AssignValue(1)

    def SetUpSensitivityAnalysis(self):
        self.ObjectiveFunction.Residual = self.m.y()
        
        self.A     = self.SetContinuousOptimizationVariable(self.m.A,     -10, 10, 0.7);
        self.k     = self.SetContinuousOptimizationVariable(self.m.k,     -10, 10, 0.8);
        self.theta = self.SetContinuousOptimizationVariable(self.m.theta, -10, 10, 1.9);

# Function to calculate either Residuals or Jacobian matrix, subject to the argument calc_values
def Function(p, simulation, xin, ymeas, calc_values):
    Nparams = len(p)
    Nexp    = len(xin)
    
    if(len(xin) != len(ymeas)):
        raise RuntimeError('The number of input data and the number of measurements must be equal') 
    
    values = numpy.zeros((Nexp))
    derivs = numpy.zeros((Nexp, Nparams))
    
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
        simulation.Reinitialize()
        simulation.Run()
        
        # Get the results
        values[e]    = simulation.ObjectiveFunction.Value - ymeas[e]
        derivs[e][:] = simulation.ObjectiveFunction.Gradients
        
    print('A =', simulation.A.Value, ', k =', simulation.k.Value, ', theta =', simulation.theta.Value)
    if calc_values:
        print('  Residuals:')
        print(values)
    else:
        print('  Derivatives:')
        print(derivs)
    
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
    return p[0] * numpy.sin(2 * numpy.pi * p[1] * x + p[2])

def run():
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Do no print progress
    log.PrintProgress = False
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)
    simulation.SolveInitial()

    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Exact values of the parameters
    A, k, theta = [10, 33.33333333, 0.523598333]

    # Starting point for parameters
    p0 = [9.0, 43.0, 0.3]

    # Input data for the model
    x = numpy.arange(0, 0.06, 0.002)

    # The values of y for given x and exact values of A, k, and theta
    y_true = A * numpy.sin(2 * numpy.pi * k * x + theta)

    # Measured values for y
    y_meas = numpy.zeros_like(x)
    y_meas = [ 5.95674236,  10.03610565,  10.14475642,   9.16722521,   8.52093929,
               4.78842863,   2.87467755,  -3.93427325,  -6.13071010,  -9.26168083,
              -9.25272475, -10.42850414,  -4.71175587,  -3.60403013,  -0.11039750,
               3.80372890,   8.51512082,   9.78232718,   9.91931747,   5.17108061,
               6.47468360,   0.66528089,  -5.10344027,  -7.12668123,  -9.42080566,
              -8.23170543,  -6.56081590,  -6.28524014,  -2.30246340,  -0.79571452]

    # Call leastsq
    p, cov_x, infodict, msg, ier = leastsq(Residuals, 
                                           p0,
                                           Dfun=Derivatives,
                                           args=(simulation, x, y_meas),
                                           full_output=True)

    # Print the results
    print('------------------------------------------------------')
    if ier in [1, 2, 3, 4]:
        print('Solution found!')
    else:
        print('Least square method failed!')
    print('Status:', msg)

    print('Number of function evaluations =', infodict['nfev'])
    chisq = (infodict['fvec']**2).sum()
    dof = len(x) - len(p0)
    rmse = numpy.sqrt(chisq / dof)
    print('Root mean square deviation =', rmse)

    A, k, theta = p
    print('Estimated parameters values:')
    print('    A     =', A)
    print('    k     =', k)
    print('    theta =', theta)
    print('------------------------------------------------------')

    # Plot the comparison between the exact values, measured and fitted data
    plt.plot(x, peval(x, p), x, y_meas, 'o', x, y_true)
    plt.title('Least-squares fit to experimental data')
    plt.legend(['Fit', 'Experimental', 'Exact'])
    plt.show()

if __name__ == "__main__":
    run()
