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

"""
Levenberg–Marquardt algorithm (LMA) provides a numerical solution to the problem of 
minimizing a function, generally nonlinear, over a space of parameters of the function.
The main properties:
 - LMA is a blend of Gradient descent and Gauss-Newton iteration
 - LMA finds only a local minimum, not a global minimum
 - It only optimises for residual errors in the dependant variable (y);
   any errors in the independent variable are zero or considered to be negligible
 - Fobj = sum[ (yi - f(xi, p))^2 ]; i = [1, m]
 - Damping parameter λ (here given as argument called 'factor')
"""

# Function to calculate either Residuals or Jacobian matrix, subject to the argument calc_values
def Calculate(p, minpack, calc_values):
    # If col_deriv is True the shape of derivs is (Nparameters, Nexperiments)
    # If col_deriv is False the shape of derivs is (Nexperiments, Nparameters)    
    values = zeros( (minpack.Nmeasured_variables, minpack.Nexperiments, minpack.Ntime_points) )
    derivs = zeros( (minpack.Nparameters, minpack.Nmeasured_variables, minpack.Nexperiments, minpack.Ntime_points) )

    try:
        # Set the reporting times:
        if minpack.simulation.m.IsModelDynamic:
            times = minpack.experimental_data[0][0]
            minpack.simulation.ReportingTimes = times
            # print minpack.simulation.ReportingTimes
        
        for e in range(0, minpack.Nexperiments):
            # Set initial conditions, initial guesses, initially active states etc
            minpack.simulation.SetUpVariables()
        
            # Assign the input data for the simulation
            for i in range(0, minpack.Ninput_variables):
                minpack.x_input_variables[i].Value = minpack.experimental_data[e][1][i]
                # print minpack.x_input_variables[i].Name, minpack.x_input_variables[i].Value
            
            # Set the parameters values
            for i in range(0, minpack.Nparameters):
                minpack.parameters[i].Value = p[i]
                # print minpack.parameters[i].Name, minpack.parameters[i].Value
            
            # Run the simulation
            minpack.simulation.Reset()
            minpack.simulation.SolveInitial()
            
            ct = 0
            if minpack.simulation.m.IsModelDynamic:
                while minpack.simulation.CurrentTime < minpack.simulation.TimeHorizon:
                    t = minpack.simulation.NextReportingTime
                    # print 'Integrating from {0} to {1}...'.format(minpack.simulation.CurrentTime, t)
                    minpack.simulation.IntegrateUntilTime(t, eDoNotStopAtDiscontinuity)
                    minpack.simulation.ReportData()
                    for o in range(0, minpack.Nmeasured_variables):
                        values[o, e, ct]    = minpack.y_measured_variables[o].Value - minpack.experimental_data[e][2][o][ct]
                        derivs[:, o, e, ct] = minpack.y_measured_variables[o].Gradients
                        #print 'v =', values[o, e, ct]
                        #print 'd =', derivs[:, o, e, ct]
                    ct += 1
                
                if ct != minpack.Ntime_points:
                    raise RuntimeError('ct = {0} while Ntime_points = {1}'.format(ct, minpack.Ntime_points))
            else:
                for o in range(0, minpack.Nmeasured_variables):
                    values[o, e, 0]    = minpack.y_measured_variables[o].Value - minpack.experimental_data[e][2][o][0]
                    derivs[:, o, e, 0] = minpack.y_measured_variables[o].Gradients
                    #print 'v =', values[o, e, 0]
                    #print 'd =', derivs[:, o, e, 0]
            
    except Exception, e:
        print 'Exception in function Calculate, for parameters values: {0}'.format(p)
        print str(e)
    
    r_values = values.reshape( (minpack.Nmeasured_variables * minpack.Nexperiments * minpack.Ntime_points) )
    r_derivs = derivs.reshape( (minpack.Nparameters, minpack.Nmeasured_variables * minpack.Nexperiments * minpack.Ntime_points) )
    
    if minpack.PrintResidualsAndJacobian:
        print 'Parameters:', p
        if calc_values:
            print '  Residuals:'
            print r_values
        else:
            print '  Derivatives:'
            print r_derivs
        
    if calc_values:
        return r_values
    else:
        return r_derivs
        
# Function to calculate residuals R = f(x_data, parameters) - y_data:
#   R[0], R[1], ..., R[n] 
def Residuals(p, args):
    minpack = args[0]
    return Calculate(p, minpack, True)
    
# Function to calculate Jacobian matrix for residuals R: 
# | dR[0]/dp[0] dR[0]/dp[1] ... dR[0]/dp[n] |
# | dR[1]/dp[0] dR[1]/dp[1] ... dR[1]/dp[n] |
# | ...                                     |
# | dR[n]/dp[0] dR[n]/dp[1] ... dR[n]/dp[n] |
def Derivatives(p, args):
    minpack = args[0]
    return Calculate(p, minpack, False)
    
class daeMinpackLeastSq:
    def __init__(self):
        self.simulation                 = None
        self.daesolver                  = None
        self.datareporter               = None
        self.log                        = None
        self.parameters                 = []
        self.x_input_variables          = []
        self.y_measured_variables       = []
        self.p0                         = []
        self.experimental_data          = []
        self.rmse                       = 0.0
        self.Nparameters                = 0
        self.Ninput_variables           = 0
        self.Nmeasured_variables        = 0
        self.Nexperiments               = 0
        self.Ntime_points               = 0
        self.minpack_leastsq_arguments  = {}
        self.PrintResidualsAndJacobian  = False
        self.IsInitialized              = False
        
    def Initialize(self, simulation, daesolver, datareporter, log, **kwargs):
        # Get the inputs
        self.simulation   = simulation
        self.daesolver    = daesolver
        self.datareporter = datareporter
        self.log          = log
        if (self.simulation == None):
            raise RuntimeError('The simulation object cannot be None') 
        if (self.daesolver == None):
            raise RuntimeError('The daesolver object cannot be None') 
        if (self.datareporter == None):
            raise RuntimeError('The datareporter object cannot be None') 
        if (self.log == None):
            raise RuntimeError('The log object cannot be None') 
        
        self.experimental_data          = kwargs.get('experimental_data',           None)
        self.PrintResidualsAndJacobian  = kwargs.get('PrintResidualsAndJacobian',  False)
        self.minpack_leastsq_arguments  = kwargs.get('minpack_leastsq_arguments',     {})
        
        # Call simulation.Initialize (with eParameterEstimation mode)
        self.simulation.SimulationMode = eParameterEstimation
        self.simulation.Initialize(self.daesolver, self.datareporter, self.log)
        
        # Check the inputs
        if (self.simulation == None):
            raise RuntimeError('The simulation object cannot be None') 
        if (self.log == None):
            raise RuntimeError('The log object cannot be None') 
        if ( self.experimental_data == None ):
            raise RuntimeError('Argument [experimental_data] has not been provided') 
        
        self.parameters           = self.simulation.ModelParameters
        self.x_input_variables    = self.simulation.InputVariables
        self.y_measured_variables = self.simulation.MeasuredVariables
        self.Nparameters          = len(self.parameters)
        self.Ninput_variables     = len(self.x_input_variables)
        self.Nmeasured_variables  = len(self.y_measured_variables)
        if ( self.Nparameters == 0 ):
            raise RuntimeError('The number of parameters is equal to zero') 
        if ( self.Ninput_variables == 0 ):
            raise RuntimeError('The number of input variables is equal to zero') 
        if ( self.Nmeasured_variables == 0 ):
            raise RuntimeError('The number of measured variables is equal to zero') 
        
        self.Nexperiments = len(self.experimental_data)
        if ( len(self.experimental_data) != self.Nexperiments ):
            raise RuntimeError('[experimental_data] must contain Nexperiments ({0}) number of tuples: ([times], [input_variables], [measured_variables])'.format(self.Nexperiments))
        
        if ( len(self.experimental_data[0]) != 3 ):
            raise RuntimeError('Each experiment in [experimental_data] must contain tuple: ([time points], [input_variables], [measured_variables])')
        self.Ntime_points = len(self.experimental_data[0][0])
        
        for exp in self.experimental_data:
            if ( len(exp) != 3 ):
                raise RuntimeError('Each experiment in [experimental_data] must contain tuple: ([time points], [input_variables], [measured_variables])')
            times = exp[0]
            xin   = exp[1]
            yout  = exp[2]
            if ( len(times) != self.Ntime_points ):
                raise RuntimeError('Number of time points in each experiment ({0}) must be {1}'.format(len(times), self.Ntime_points))
            if ( len(xin) != self.Ninput_variables ):
                raise RuntimeError('Number of input variables in each experiment must be {0}'.format(self.Ninput_variables))
            if ( len(yout) != self.Nmeasured_variables ):
                raise RuntimeError('Number of measured variables in each experiment must be {0}'.format(self.Nmeasured_variables))
            for y in yout:
                if ( len(y) != self.Ntime_points ):
                    raise RuntimeError('Number of values in each measured variable must be equal to number of time intervals') 
        
        self.p0 = []
        for p in self.parameters:
            self.p0.append(p.StartingPoint)
            
        self.IsInitialized = True
        
    def Run(self):
        if self.IsInitialized == False:
            raise RuntimeError('daeMinpackLeastSq object has not been initialized') 
        
        self.minpack_leastsq_arguments['full_output'] = True
        self.minpack_leastsq_arguments['col_deriv']   = True
        print 'minpack.leastsq will proceed with the following arguments:', self.minpack_leastsq_arguments
        
        # Call scipy.optimize.leastsq (Minpack wrapper)
        self.p_estimated, self.cov_x, self.infodict, self.msg, self.ier = leastsq(Residuals, self.p0, Dfun = Derivatives, args = [self], 
                                                                                  **self.minpack_leastsq_arguments)
        
        # Check the info and calculate the least square statistics
        if self.ier not in [1, 2, 3, 4]:
            raise RuntimeError('MINPACK least square method failed ({0})'.format(self.msg))
        
        chisq = (self.infodict['fvec']**2).sum()
        dof = self.Nexperiments*self.Nmeasured_variables - self.Nparameters
        self.rmse = sqrt(chisq / dof)
        
    def Finalize(self):
        self.simulation.Finalize()
