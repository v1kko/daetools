#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            opt_tutorial4.py
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
software (scipy.optimize) used to minimize the Rosenbrock function.

DAE Tools simulation is used to calculate the objective function and its gradients,
while scipy.optimize.fmin function (Nelder-Mead Simplex algorithm) to find the 
minimum of the Rosenbrock function.
"""

import sys
from time import localtime, strftime
from scipy.optimize import fmin, fmin_bfgs, fmin_l_bfgs_b
from daetools.pyDAE import *

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x1 = daeVariable("x1", no_t, self)
        self.x2 = daeVariable("x2", no_t, self)
        self.x3 = daeVariable("x3", no_t, self)
        self.x4 = daeVariable("x4", no_t, self)
        self.x5 = daeVariable("x5", no_t, self)

        self.dummy = daeVariable("dummy", no_t, self, "A dummy variable to satisfy the condition that there should be at least one-state variable and one equation in a model")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("Dummy")
        eq.Residual = self.dummy()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial4")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x1.AssignValue(1)
        self.m.x2.AssignValue(1)
        self.m.x3.AssignValue(1)
        self.m.x4.AssignValue(1)
        self.m.x5.AssignValue(1)

    def SetUpSensitivityAnalysis(self):
        self.SetNumberOfObjectiveFunctions(1)
        self.ObjectiveFunction.Residual = 100 * (self.m.x2() - self.m.x1()**2)**2 + (1 - self.m.x1())**2 + \
                                          100 * (self.m.x3() - self.m.x2()**2)**2 + (1 - self.m.x2())**2 + \
                                          100 * (self.m.x3() - self.m.x3()**2)**2 + (1 - self.m.x3())**2 + \
                                          100 * (self.m.x5() - self.m.x4()**2)**2 + (1 - self.m.x4())**2 
        
        self.ov1 = self.SetContinuousOptimizationVariable(self.m.x1, -10, 10, 1.3)
        self.ov2 = self.SetContinuousOptimizationVariable(self.m.x2, -10, 10, 0.7)
        self.ov3 = self.SetContinuousOptimizationVariable(self.m.x3, -10, 10, 0.8)
        self.ov4 = self.SetContinuousOptimizationVariable(self.m.x4, -10, 10, 1.9)
        self.ov5 = self.SetContinuousOptimizationVariable(self.m.x5, -10, 10, 1.2)

def ObjectiveFunction(x, *args):
    simulation = args[0]
    
    # This function will be called repeatedly to obtain the values of the objective function.
    # In order to call DAE Tools repedeatly the following sequence of calls is necessary:
    # 1. Set initial conditions, initial guesses, initially active states etc (function simulation.SetUpVariables)
    #    In general, variables values, active states, initial conditions etc can be saved in some arrays and
    #    later re-used. However, keeping the initialization data in SetUpVariables looks much better.
    simulation.SetUpVariables()
    
    # 2. Change values of optimization variables (this will call function daeVariable.ReAssignValue) by setting 
    #    the optimization variable's Value property. Optimization variables can be obtained in two ways:
    # 2a) Use OptimizationVariables attribute to get a list of optimization variables and then set their values:
    opt_vars = simulation.OptimizationVariables
    opt_vars[0].Value = x[0]
    opt_vars[1].Value = x[1]
    opt_vars[2].Value = x[2]
    opt_vars[3].Value = x[3]
    opt_vars[4].Value = x[4]
    # 2b) Use stored optimization variable objects in simulation object (ov1, ..., ov5):
    #simulation.ov1.Value = x[0]
    #simulation.ov2.Value = x[1]
    #simulation.ov3.Value = x[2]
    #simulation.ov4.Value = x[3]
    #simulation.ov5.Value = x[4]
    
    # 3. Call simulations's Reset (to reset simulation and DAE solver objects), SolveInitial (to re-initialize the system),
    #    and Run (to simulate the model and to calculate sensitivities) functions.
    simulation.Reset()
    simulation.SolveInitial()
    simulation.Run()
    
    # 4. Once finished with simulation, use ObjectiveFunction and Constraints properties of simulation object.
    #    Objective function and constraints have Value (float) and Gradients (numpy array) properties where
    #    their values and gradients in respect to optimization variables are stored. Here, as the example,
    #    the value and gradients of the objective function are printed (since no constraints are involved).
    print('Objective function inputs: ')
    print('   Inputs: {0}'.format(x))
    print('   Value     = {0}'.format(simulation.ObjectiveFunction.Value))
    print('   Gradients = {0}'.format(simulation.ObjectiveFunction.Gradients))
    print('')
    return simulation.ObjectiveFunction.Value

def run(**kwargs):
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Do no print progress
    log.PrintProgress = False
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 5

    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # ACHTUNG, ACHTUNG!!
    # To request simulation to calculate sensitivities use the keyword argument CalculateSensitivities:
    simulation.Initialize(daesolver, datareporter, log, calculateSensitivities = True)

    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Get the starting point from optimization variables
    x0 = [simulation.ov1.StartingPoint, 
          simulation.ov2.StartingPoint, 
          simulation.ov3.StartingPoint, 
          simulation.ov4.StartingPoint,
          simulation.ov5.StartingPoint]

    print(fmin(ObjectiveFunction, x0, args=(simulation,), xtol=1e-8))

if __name__ == "__main__":
    run()
