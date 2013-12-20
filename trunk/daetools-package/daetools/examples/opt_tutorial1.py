#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            opt_tutorial1.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
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
This tutorial introduces IPOPT NLP solver, its setup and options.
"""

import sys
from daetools.pyDAE import *
from daetools.solvers.ipopt import pyIPOPT
from time import localtime, strftime

# Standard variable types are defined in variable_types.py

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x1 = daeVariable("x1", no_t, self)
        self.x2 = daeVariable("x2", no_t, self)
        self.x3 = daeVariable("x3", no_t, self)
        self.x4 = daeVariable("x4", no_t, self)

        self.dummy = daeVariable("dummy", no_t, self, "A dummy variable to satisfy the condition that there should be at least one-state variable and one equation in a model")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("Dummy")
        eq.Residual = self.dummy()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("opt_tutorial1")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        pass

    def SetUpVariables(self):
        self.m.x1.AssignValue(1)
        self.m.x2.AssignValue(5)
        self.m.x3.AssignValue(5)
        self.m.x4.AssignValue(1)

    def SetUpOptimization(self):
        # Set the objective function (minimization).
        # The objective function can be accessed by using ObjectiveFunction property which always returns the 1st obj. function,
        # for in general case more than one obj. function. can be defined, so ObjectiveFunctions[0] can be used as well:
        #       self.ObjectiveFunctions[0].Residual = ...
        # Obviously defining more than one obj. function has no meaning when using opt. software such as Ipopt, Bonmin or Nlopt
        # which cannot do the multi-objective optimization. The number of obj. functions can be defined in the function
        # optimization.Initialize by using the named argument NumberOfObjectiveFunctions (the default is 1).
        # Other obj. functions can be obtained by using ObjectiveFunctions[i] property.
        self.ObjectiveFunction.Residual = self.m.x1() * self.m.x4() * (self.m.x1() + self.m.x2() + self.m.x3()) + self.m.x3()

        # Set the constraints (inequality, equality)
        # Constraints are in the following form:
        #  - Inequality: g(i) <= 0
        #  - Equality: h(i) = 0
        self.c1 = self.CreateInequalityConstraint("Constraint 1") # g(x) >= 25:  25 - x1*x2*x3*x4 <= 0
        self.c1.Residual = 25 - self.m.x1() * self.m.x2() * self.m.x3() * self.m.x4()

        self.c2 = self.CreateEqualityConstraint("Constraint 2") # h(x) == 40
        self.c2.Residual = self.m.x1() * self.m.x1() + self.m.x2() * self.m.x2() + self.m.x3() * self.m.x3() + self.m.x4() * self.m.x4() - 40

        # Set the optimization variables, their lower/upper bounds and the starting point
        # The optimization variables can be stored and used later to get the optimization results or
        # to interact with some 3rd party software.
        self.x1 = self.SetContinuousOptimizationVariable(self.m.x1, 1, 5, 2);
        self.x2 = self.SetContinuousOptimizationVariable(self.m.x2, 1, 5, 2);
        self.x3 = self.SetContinuousOptimizationVariable(self.m.x3, 1, 5, 2);
        self.x4 = self.SetContinuousOptimizationVariable(self.m.x4, 1, 5, 2);

def setOptions(nlpsolver):
    # 1) Set the options manually
    try:
        nlpsolver.SetOption('print_level', 0)
        nlpsolver.SetOption('tol', 1e-7)
        nlpsolver.SetOption('mu_strategy', 'adaptive')

        # Print options loaded at pyIPOPT startup and the user set options:
        #nlpsolver.PrintOptions()
        #nlpsolver.PrintUserOptions()

        # ClearOptions can clear all options:
        #nlpsolver.ClearOptions()
    except Exception as e:
        print(str(e))
        
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
                                  nlpsolver = nlp,
                                  nlpsolver_setoptions_fn = setOptions)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    nlpsolver    = pyIPOPT.daeIPOPT()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    optimization = daeOptimization()

    # Do no print progress
    log.PrintProgress = False
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the optimization
    optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

    # Achtung! Achtung! NLP solver options can only be set after optimization.Initialize()
    # Otherwise seg. fault occurs for some reasons.
    setOptions(nlpsolver)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Run
    optimization.Run()
    optimization.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
