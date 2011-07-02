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
    for exp in self.experimental_data:
        if ( len(exp) != 3 )
            raise RuntimeError('Each experiment in [experimental_data] must contain tuple: ([times], [input_variables], [measured_variables])'
        
        times = exp[0]
        xin   = exp[1]
        yout  = exp[2]
        if ( len(xin) != self.Ninput_variables ):
            raise RuntimeError('Number of input variables in each experiment must be {0}'.format(self.Ninput_variables))
        if ( len(yout) != self.Nmeasured_variables ):
            raise RuntimeError('Number of measured variables in each experiment must be {0}'.format(self.Nmeasured_variables))
        for y in yout:
            if ( len(y) != len(times) ):
                raise RuntimeError('Number of values in each measured variable must be equal to number of time intervals') 
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
            #print minpack.simulation.ReportingTimes
        
        for e in range(0, minpack.Nexperiments):
            # Set initial conditions, initial guesses, initially active states etc
            minpack.simulation.SetUpVariables()
        
            # Assign the input data for the simulation
            for i in range(0, minpack.Ninput_variables):
                minpack.x_input_variables[i].ReAssignValue(minpack.experimental_data[e][1][i])
                #print minpack.x_input_variables[i].Name, minpack.x_input_variables[i].GetValue()
            
            # Set the parameters values
            for i in range(0, minpack.Nparameters):
                minpack.parameters[i].Value = p[i]
                #print minpack.parameters[i].Name, minpack.parameters[i].Value
            
            # Run the simulation
            minpack.simulation.Reset()
            minpack.simulation.SolveInitial()
            
            ct = 0
            if minpack.simulation.m.IsModelDynamic:
                while minpack.simulation.CurrentTime < minpack.simulation.TimeHorizon:
                    t = minpack.simulation.NextReportingTime
                    #print 'Integrating from {0} to {1}...'.format(minpack.simulation.CurrentTime, t)
                    minpack.simulation.IntegrateUntilTime(t, eDoNotStopAtDiscontinuity)
                    minpack.simulation.ReportData()
                    for o in range(0, minpack.Nmeasured_variables):
                        #print 'e = {0}, o = {1}, ct = {2}'.format(e, o, ct)
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
    
    """
    print 'Parameters:', p
    if calc_values:
        print '  Residuals:'
        print values
    else:
        print '  Derivatives:'
        print derivs
    """
    
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
    def __init__(self, simulation, daesolver, datareporter, log, **kwargs):
        self.simulation   = simulation
        self.daesolver    = daesolver
        self.datareporter = datareporter
        self.log          = log
        
        self.parameters           = None
        self.x_input_variables    = None
        self.y_measured_variables = None
        self.p0                   = None
        self.experimental_data    = None
        self.PrintResidualsAndJacobian = False
        
        self.rmse                       = 0.0
        self.Nparameters                = kwargs.get('Nparameters',                    0)
        self.Ninput_variables           = kwargs.get('Ninput_variables',               0)
        self.Nmeasured_variables        = kwargs.get('Nmeasured_variables',            0)
        self.Nexperiments               = kwargs.get('Nexperiments',                   0)
        self.Ntime_points               = kwargs.get('Ntime_points',                   0)
        self.PrintResidualsAndJacobian  = kwargs.get('PrintResidualsAndJacobian',  False)
        self.ftol                       = kwargs.get('ftol',                        1E-8)
        self.xtol                       = kwargs.get('xtol',                        1E-8)
        self.factor                     = kwargs.get('factor',                     100.0)
        
        if ( self.simulation == None ):
            raise RuntimeError('The simulation object cannot be None') 
        if ( self.Nparameters == 0 ):
            raise RuntimeError('The number of parameters cannot be 0') 
        if ( self.Ninput_variables == 0 ):
            raise RuntimeError('The number of input variables cannot be 0') 
        if ( self.Nmeasured_variables == 0 ):
            raise RuntimeError('The number of measured variables cannot be 0') 
        if ( self.Nexperiments == 0 ):
            raise RuntimeError('The number of experiments cannot be 0') 
        if ( self.Ntime_points == 0 ):
            raise RuntimeError('The number of time points cannot be 0') 

        self.simulation.Initialize(self.daesolver, 
                                   self.datareporter, 
                                   self.log, 
                                   CalculateSensitivities = True,
                                   NumberOfObjectiveFunctions = self.Nmeasured_variables)
        
    def Initialize(self, **kwargs):
        # List of daeOptimizationVariable objects:
        self.parameters = kwargs.get('parameters', None)
        
        # List of parameters' starting values:
        self.p0 = kwargs.get('p0', None)

        # List of daeVariable/adouble objects (must be assigned variables):
        self.x_input_variables = kwargs.get('x_input_variables', None)
        
        # daeVariable/adouble object (must be state variable):
        self.y_measured_variables = kwargs.get('y_measured_variables', None)
        
        # 1-dimensional list of tuples (times, input_variables, measured_variables):
        self.experimental_data = kwargs.get('experimental_data', None)
        
        if ( (self.parameters == None) or (len(self.parameters) != self.Nparameters) ):
            raise RuntimeError('Argument [parameters] has not been provided or the number of parameters ({0}) is not equal to Nparameters ({1})'.format(len(self.parameters), self.Nparameters)) 
        if ( (self.p0 == None) or (len(self.p0) != self.Nparameters)):
            raise RuntimeError('Argument [p0] has not been provided or the number of p0 ({0}) is not equal to Nparameters ({1})'.format(len(self.p0), self.Nparameters)) 
        if ( (self.x_input_variables == None) or (len(self.x_input_variables) != self.Ninput_variables) ):
            raise RuntimeError('Argument [x_input_variables] has not been provided or the number of x_input_variables ({0}) is not equal to Ninput_variables ({1})'.format(len(self.x_input_variables), self.Ninput_variables)) 
        if ( (self.y_measured_variables == None) or (len(self.y_measured_variables) != self.Nmeasured_variables) ):
            raise RuntimeError('Argument [y_measured_variables] has not been provided or the number of y_measured_variables ({0}) is not equal to Nmeasured_variables ({1})'.format(len(self.y_measured_variables), self.Nmeasured_variables)) 
        if ( self.experimental_data == None ):
            raise RuntimeError('Argument [experimental_data] has not been provided') 

        if ( len(self.experimental_data) != self.Nexperiments ):
            raise RuntimeError('[experimental_data] must contain Nexperiments ({0}) number of tuples: ([times], [input_variables], [measured_variables])'.format(self.Nexperiments))
        
        for exp in self.experimental_data:
            if ( len(exp) != 3 ):
                raise RuntimeError('Each experiment in [experimental_data] must contain tuple: ([times], [input_variables], [measured_variables])')
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

    def Run(self):
        # Call scipy.optimize.leastsq (Minpack wrapper)
        self.p_estimated, self.cov_x, self.infodict, self.msg, self.ier = leastsq(Residuals, 
                                                                                  self.p0, 
                                                                                  Dfun        = Derivatives, 
                                                                                  args        = [self], 
                                                                                  full_output = True,
                                                                                  ftol        = self.ftol,
                                                                                  xtol        = self.xtol,
                                                                                  factor      = self.factor,
                                                                                  col_deriv   = True)
        
        if self.ier not in [1, 2, 3, 4]:
            raise RuntimeError('MINPACK least square method failed ({0})'.format(self.msg))
        
        chisq = (self.infodict['fvec']**2).sum()
        dof = self.Nexperiments*self.Nmeasured_variables - self.Nparameters
        self.rmse = sqrt(chisq / dof)
        
    def Finalize(self):
        self.simulation.Finalize()
 
