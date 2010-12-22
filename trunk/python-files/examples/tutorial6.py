#!/usr/bin/env python

"""********************************************************************************
                             tutorial6.py
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
This is the simple port demo.
Here we introduce:
 - Ports
 - Port connections
 - Units (instances of other models)
 
A simple port type 'portSimple' is defined which contains only one variable 't'.
Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'.
The wrapper model 'modTutorial' instantiate these two models as its units and connects them
by connecting their ports.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-",      0, 1E10,   0, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",    100, 1000, 300, 1e-5)
typeConductivity = daeVariableType("Conductivity", "W/mK",   0, 1E10, 100, 1e-5)
typeDensity      = daeVariableType("Density",      "kg/m3",  0, 1E10, 100, 1e-5)
typeHeatCapacity = daeVariableType("HeatCapacity", "J/KgK",  0, 1E10, 100, 1e-5)

# Ports, like models, consist of domains, parameters and variables. Parameters and variables
# can be distributed as well. Here we define a very simple port, with only one variable.
# The process of defining ports is analogous to defining models. Domains, parameters and
# variables are declared in the constructor __init__ and their constructor accepts ports as
# the 'Parent' argument. 
class portSimple(daePort):
    def __init__(self, Name, PortType, Model, Description = ""):
        daePort.__init__(self, Name, PortType, Model, Description)
        
        self.t = daeVariable("t", typeNone, self, "Time elapsed in the process, s")

# Here we define two models, 'modPortIn' and 'modPortOut' each having one port of type portSimple.
# The model 'modPortIn' contains inlet port Pin while the model 'modPortOut' contains outlet port Pout.
class modPortIn(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Pin = portSimple("P_in", eInletPort, self, "The simple port")
    
    def DeclareEquations(self):
        pass

class modPortOut(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Pout = portSimple("P_out", eOutletPort, self, "The simple port")
        self.time = daeVariable("Time", typeNone, self, "Time elapsed in the process, s")
    
    def DeclareEquations(self):
        eq = self.CreateEquation("time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = self.time.dt() - 1.0

        eq = self.CreateEquation("Port_t", "")
        eq.Residual = self.Pout.t() - self.time()

# Model 'modTutorial' declares two units mpin of type 'modPortIn' and 'mpout' of type 'modPortOut'.
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.mpin  = modPortIn("Port_In", self)
        self.mpout = modPortOut("Port_Out", self)

    def DeclareEquations(self):
        # Ports can be connected by using the function ConnectPorts from daeModel class. Apparently, 
        # ports dont have to be of the same type but must contain the same number of parameters and variables.
        self.ConnectPorts(self.mpout.Pout, self.mpin.Pin)

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("Tutorial_6")
        self.m.Description = "This tutorial explains how to define and connect ports. \n" \
                             "A simple port type 'portSimple' is defined which contains only one variable 't'. " \
                             "Two models 'modPortIn' and 'modPortOut' are defined, each having one port of type 'portSimple'. " \
                             "The wrapper model 'modTutorial' instantiate these two models as its units and connects them " \
                             "by connecting their ports. "
                             
    def SetUpParametersAndDomains(self):
        pass
    
    def SetUpVariables(self):
        self.m.mpout.time.SetInitialCondition(0)
    
# Use daeSimulator class
def guiRun():
    from PyQt4 import QtCore, QtGui
    app = QtGui.QApplication(sys.argv)
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation)
    simulator.show()
    app.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    solver       = daeIDASolver()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 100

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.InitSimulation(solver, datareporter, log)
    
    # Save the model report and the runtime model report 
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        guiRun()
    else:
        consoleRun()
