#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             opt_tutorial4.py
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
This tutorial shows the interoperability between DAE Tools and 3rd party optimization 
software (scipy.optimize) used to minimize the Rosenbrock function.
DAE Tools simulation is used to calculate the objective function and its gradients,
while scipy.optimize.fmin function (Nelder-Mead Simplex algorithm) to find the 
minimum of the Rosenbrock function.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime
from scipy.optimize import fmin, fmin_bfgs, fmin_l_bfgs_b

# Standard variable types are defined in daeVariableTypes.py

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
        eq = self.CreateEquation("Dummy")
        eq.Residual = self.dummy()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial4")
        self.m.Description = "This tutorial shows the interoperability between DAE Tools and 3rd party optimization " \
                             "software (scipy.optimize) used to minimize the Rosenbrock function. \n" \
                             "DAE Tools simulation is used to calculate the objective function and its gradients, " \
                             "while scipy.optimize.fmin function (Nelder-Mead Simplex algorithm) to find the " \
                             "minimum of the Rosenbrock function."

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x1.AssignValue(1)
        self.m.x2.AssignValue(1)
        self.m.x3.AssignValue(1)
        self.m.x4.AssignValue(1)
        self.m.x5.AssignValue(1)

    def SetUpOptimization(self):
        self.ObjectiveFunction.Residual = 100 * (self.m.x2() - self.m.x1()**2)**2 + (1 - self.m.x1())**2 + \
                                          100 * (self.m.x3() - self.m.x2()**2)**2 + (1 - self.m.x2())**2 + \
                                          100 * (self.m.x3() - self.m.x3()**2)**2 + (1 - self.m.x3())**2 + \
                                          100 * (self.m.x5() - self.m.x4()**2)**2 + (1 - self.m.x4())**2 
        
        self.ov1 = self.SetContinuousOptimizationVariable(self.m.x1, 1, 5, 2);
        self.ov2 = self.SetContinuousOptimizationVariable(self.m.x2, 1, 5, 2);
        self.ov3 = self.SetContinuousOptimizationVariable(self.m.x3, 1, 5, 2);
        self.ov4 = self.SetContinuousOptimizationVariable(self.m.x4, 1, 5, 2);
        self.ov5 = self.SetContinuousOptimizationVariable(self.m.x5, 1, 5, 2);

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    opt = daeOptimization()
    nlp = pyIPOPT.daeIPOPT()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 5
    simulator = daeSimulator(app, simulation = sim,
                                  optimization = opt,
                                  nlpsolver = nlp)
    simulator.exec_()

def Function(x, *params):
    simulation = params[0]
    
    simulation.SetUpVariables()
    # 1) Use OptimizationVariables attribute to get a list of optimization variables:
    opt_vars = simulation.OptimizationVariables
    opt_vars[0].Value = x[0]
    opt_vars[1].Value = x[1]
    opt_vars[2].Value = x[2]
    opt_vars[3].Value = x[3]
    opt_vars[4].Value = x[4]
    
    # 2) Or by using the stored daeOptimization variables (ov1, ..., ov5):
    #simulation.ov1.Value = x[0]
    #simulation.ov2.Value = x[1]
    #simulation.ov3.Value = x[2]
    #simulation.ov4.Value = x[3]
    #simulation.ov5.Value = x[4]
    simulation.Reset()
    simulation.SolveInitial()
    simulation.Run()

    return simulation.ObjectiveFunction.Value
    
def Function_der(x, *params):
    simulation = params[0]
    
    simulation.SetUpVariables()
    simulation.ov1.Value = x[0]
    simulation.ov2.Value = x[1]
    simulation.ov3.Value = x[2]
    simulation.ov4.Value = x[3]
    simulation.ov5.Value = x[4]
    simulation.Reset()
    simulation.SolveInitial()
    simulation.Run()

    return simulation.ObjectiveFunction.Gradients

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
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # ACHTUNG, ACHTUNG!!
    # To request simulation to calculate sensitivities use the function:
    #     simulation.Initialize(..., CalculateSensitivities = True)!!
    simulation.Initialize(daesolver, datareporter, log, CalculateSensitivities = True)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    #x0 = [simulation.ov1.StartingPoint, 
    #      simulation.ov2.StartingPoint, 
    #      simulation.ov3.StartingPoint, 
    #      simulation.ov4.StartingPoint]
    #xbounds = [(simulation.ov1.LowerBound, simulation.ov1.UpperBound), 
    #           (simulation.ov2.LowerBound, simulation.ov2.UpperBound),
    #           (simulation.ov3.LowerBound, simulation.ov3.UpperBound),
    #           (simulation.ov4.LowerBound, simulation.ov4.UpperBound)]
    #print fmin_l_bfgs_b(Function, x0, args=([simulation]), fprime=Function_der, bounds=xbounds, pgtol=1e-07)
    
    print fmin(Function, x0, args=([simulation]), xtol=1e-8)
    
if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
