#!/usr/bin/env python

"""********************************************************************************
                             opt_tutorial1.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software 
Foundation; either version 3 of the License, or (at your option) any later version.
The DAE Tools is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, write to the Free Software Foundation, Inc., 
59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
********************************************************************************"""

"""
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone = daeVariableType("None", "-",  -1E20, 1E20,   1, 1e-6)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        
        self.x1 = daeVariable("x1", typeNone, self)
        self.x2 = daeVariable("x2", typeNone, self)
        self.x3 = daeVariable("x3", typeNone, self)
        self.x4 = daeVariable("x4", typeNone, self)
        
        self.dummy = daeVariable("dummy", typeNone, self, "A dummy variable to satisfy the condition that there should be at least one-state variable and one equation in a model")

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        eq.Residual = self.dummy()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("OptTutorial_1")
        self.m.Description = "This tutorial explains how to define a simple optimization problem. Here we use the steady-state test HS-71 ()"
          
    def SetUpParametersAndDomains(self):
        pass
    
    def SetUpVariables(self):
        self.m.x1.AssignValue(1)
        self.m.x2.AssignValue(5)
        self.m.x3.AssignValue(5)
        self.m.x4.AssignValue(1)
        
    def SetUpOptimization(self):
        # Set the objective function (min)
        self.ObjectiveFunction.Residual = self.m.x1() * self.m.x4() * (self.m.x1() + self.m.x2() + self.m.x3()) + self.m.x3()
        
        # Set the constraints (inequality, equality)
        c1 = self.CreateInequalityConstraint(25, 2E19, "Constraint 1")
        c1.Residual = self.m.x1() * self.m.x2() * self.m.x3() * self.m.x4()
        
        c2 = self.CreateEqualityConstraint(40, "Constraint 2")
        c2.Residual = self.m.x1() * self.m.x1() + self.m.x2() * self.m.x2() + self.m.x3() * self.m.x3() + self.m.x4() * self.m.x4()
                
        # Set the optimization variables and their lower and upper bounds
        self.SetOptimizationVariable(self.m.x1, 1, 5, 1);
        self.SetOptimizationVariable(self.m.x2, 1, 5, 5);
        self.SetOptimizationVariable(self.m.x3, 1, 5, 5);
        self.SetOptimizationVariable(self.m.x4, 1, 5, 1);

# Use daeSimulator class
def guiRun():
    from PyQt4 import QtCore, QtGui
    app = QtGui.QApplication(sys.argv)
    simulation   = simTutorial()
    opt = daeIPOPT()
    simu.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 5
    simulator  = daeSimulator(app, simulation=sim, optimization=opt)
    simulator.show()
    app.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    solver       = daeIDASolver()
    nlpsolver    = None
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()
    optimization = daeIPOPT()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 5

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.InitOptimization(solver, datareporter, log)
    optimization.Initialize(simulation, nlpsolver, solver, datareporter, log)

    # Save the model report and the runtime model report 
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Run
    optimization.Run()
    optimization.Finalize()
    
    print "dae version:", daeVersion()

if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        guiRun()
    else:
        consoleRun()
