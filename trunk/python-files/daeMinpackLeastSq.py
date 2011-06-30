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

# Function to calculate either Residuals or Jacobian matrix, subject to the argument calc_values
def Calculate(p, minpack, calc_values):
    values = zeros((minpack.Nexperiments))
    # If col_deriv is True the shape of derivs is (Nparameters, Nexperiments)
    # If col_deriv is False the shape of derivs is (Nexperiments, Nparameters)
    derivs = zeros((minpack.Nparameters, minpack.Nexperiments))
    
    for e in range(0, minpack.Nexperiments):
        # Set initial conditions, initial guesses, initially active states etc
        minpack.simulation.SetUpVariables()
    
        # Assign the input data for the simulation
        for i in range(0, minpack.Ninputs):
            minpack.x_input_variables[i].ReAssignValue(minpack.x_data[i, e])
        
        # Set the parameters values
        for i in range(0, minpack.Nparameters):
            minpack.parameters[i].Value = p[i]
        
        # Run the simulation
        minpack.simulation.Reset()
        minpack.simulation.SolveInitial()
        minpack.simulation.Run()
        
        # Get the results
        values[e]    = minpack.y_measured_variable.Value - minpack.y_data[e]
        derivs[:, e] = minpack.y_measured_variable.Gradients
        
    print 'Parameters:', p
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

# Function to calculate residuals R = f(x_data, parameters) - y_data:
#   R[0], R[1], ..., R[n] 
def Residuals(p, args):
    minpack = args[0]
    return Calculate(p, minpack, True)
    
# Function to calculate Jacobian matrix for residuals R: 
#   dR[0]/dp[0], dR[0]/dp[1], ..., dR[0]/dp[n] 
#   dR[1]/dp[0], dR[1]/dp[1], ..., dR[1]/dp[n] 
#   ...
#   dR[n]/dp[0], dR[n]/dp[1], ..., dR[n]/dp[n] 
def Derivatives(p, args):
    minpack = args[0]
    return Calculate(p, minpack, False)
    
class daeMinpackLeastSq:
    def __init__(self, simulation, daesolver, datareporter, log):
        self.simulation   = simulation
        self.daesolver    = daesolver
        self.datareporter = datareporter
        self.log          = log
        
        self.parameters          = None
        self.x_input_variables   = None
        self.y_measured_variable = None
        self.p0                  = None
        self.x_data              = None
        self.y_data              = None
        
        self.Nparameters  = 0
        self.Ninputs      = 0
        self.Nexperiments = 0
        
        self.rmse         = 0.0
        
        self.simulation.Initialize(self.daesolver, 
                                   self.datareporter, 
                                   self.log, 
                                   CalculateSensitivities = True)
        
    def Initialize(self, **kwargs):
        # List of daeOptimizationVariable objects:
        self.parameters = kwargs.get('parameters', None)
        
        # List of parameters' starting values:
        self.p0 = kwargs.get('p0', None)

        # List of daeVariable/adouble objects (must be assigned variables):
        self.x_input_variables = kwargs.get('x_input_variables', None)
        
        # daeVariable/adouble object (must be state variable):
        self.y_measured_variable = kwargs.get('y_measured_variable', None)
        
        # 2-dimensional numpy array [Nexperiments][Ninputs]:
        self.x_data = kwargs.get('x_data', None)
        
        # 1-dimensional numpy array [Nexperiments]: 
        self.y_data = kwargs.get('y_data', None)
        
        self.Nparameters  = len(self.parameters)
        self.Ninputs      = len(self.x_input_variables)
        self.Nexperiments = len(self.y_data)
        
        if ( (self.parameters == None) or (len(self.parameters) == 0) ):
            raise RuntimeError('Argument [parameters] has not been provided or the number of parameters is zero') 
        if ( (self.p0 == None) or (len(self.p0) == 0)):
            raise RuntimeError('Argument [p_0] has not been provided or the number of P_0 is zero') 
        if ( (self.x_input_variables == None) or (len(self.x_input_variables) == 0) ):
            raise RuntimeError('Argument [x_input_variables] has not been provided or the number of x_input_variables is zero') 
        if ( self.y_measured_variable == None ):
            raise RuntimeError('Argument [y_measured_variable] has not been provided') 
        if ( self.x_data == None ):
            raise RuntimeError('Argument [x_data] has not been provided or the number of x_data is zero') 
        if ( self.y_data == None ):
            raise RuntimeError('Argument [y_data] has not been provided or the number of y_data is zero') 

        if ( (self.x_data.ndim != 2) or (self.y_data.ndim != 1) ):
            raise RuntimeError('Number of numpy array dimensions of x_data must be 2 and of y_data must be 1') 
        if ( (self.x_data.shape[1] != self.Nexperiments) or (self.y_data.shape[0] != self.Nexperiments) ):
            raise RuntimeError('Invalid shapes of x_data and y_data') 
        if ( self.x_data.shape[0] != self.Ninputs ):
            raise RuntimeError('The shapes of x_data and y_data do not match 2') 

    def Run(self):
        # Call scipy.optimize.leastsq (Minpack wrapper)
        self.p_estimated, self.cov_x, self.infodict, self.msg, self.ier = leastsq(Residuals, 
                                                                                  self.p0, 
                                                                                  Dfun        = Derivatives, 
                                                                                  args        = [self], 
                                                                                  full_output = True,
                                                                                  col_deriv   = True)
        
        if self.ier not in [1, 2, 3, 4]:
            raise RuntimeError('MINPACK least square method failed (%s)' % self.msg)
        
        chisq = (self.infodict['fvec']**2).sum()
        dof = self.Nexperiments - self.Nparameters
        self.rmse = sqrt(chisq / dof)
        
    def Finalize(self):
        self.simulation.Finalize()
 
