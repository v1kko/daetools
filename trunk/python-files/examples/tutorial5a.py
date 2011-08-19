#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             tutorial5a.py
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
In this example we use the same problem as in the tutorial 5.
Here we introduce:
 - The event ports
 - Advanced actions executed during state transitions
 - Actions executed when an event is triggered
 - User defined action
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in daeVariableTypes.py

# User defined action executed in OnEvent handler
class simpleUserAction(daeAction):
    def __init__(self):
        daeAction.__init__(self)

    def Execute(self):
        print 'simpleUserAction executed'
        
class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       eReal, self, "Mass of the copper plate, kg")
        self.cp    = daeParameter("c_p",     eReal, self, "Specific heat capacity of the plate, J/kgK")
        self.alpha = daeParameter("&alpha;", eReal, self, "Heat transfer coefficient, W/m2K")
        self.A     = daeParameter("A",       eReal, self, "Area of the plate, m2")
        self.Tsurr = daeParameter("T_surr",  eReal, self, "Temperature of the surroundings, K")

        self.Qin   = daeVariable("Q_in",  power_t,       self, "Power of the heater, W")
        self.T     = daeVariable("T",     temperature_t, self, "Temperature of the plate, K")
        self.dummy = daeVariable("dummy", no_t,          self, "dummy")

        # Here we create two event ports (inlet and outlet) and connect them
        # (it makes no sense in reality, but this is example just to show the possibilities).
        # Event ports can be connected/disconnected at any time.
        self.epIn  = daeEventPort("epIn",  eInletPort,  self, "Inlet event port")
        self.epOut = daeEventPort("epOut", eOutletPort, self, "Outlet event port")
        self.ConnectEventPorts(self.epIn, self.epOut)

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        self.stnRegulator = self.STN("Regulator")

        self.STATE("Heating")

        eq = self.CreateEquation("Q_in", "The heater is on")
        eq.Residual = self.Qin() - 1500

        # ON_CONDITION function
        # Arguments:
        # - Condition that triggers the actions
        # - switchTo is the name of the next active state
        # - triggerEvents is a list of python tuples (outlet-event-port, expression)
        # - setVariableValues is a list of python tuples (variable, expression)
        # - userDefinedActions is a list of daeAction derived classes
        self.ON_CONDITION(self.T()    > 340, switchTo           = 'Cooling',
                                             triggerEvents      = [],
                                             setVariableValues  = [ (self.dummy, 1) ],
                                             userDefinedActions = [] )

        self.ON_CONDITION(self.time() > 350, switchTo           = 'HeaterOff',
                                             triggerEvents      = [ (self.epOut, self.T() + 5.0) ],
                                             setVariableValues  = [],
                                             userDefinedActions = [] )

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.ON_CONDITION(self.T()    < 320, switchTo          = 'Heating',
                                             triggerEvents     = [],
                                             setVariableValues = [ (self.dummy, 0) ],
                                             userDefinedActions = [] )

        self.ON_CONDITION(self.time() > 350, switchTo          = 'HeaterOff',
                                             triggerEvents     = [ (self.epOut, self.T() + 6.0) ],
                                             setVariableValues = [],
                                             userDefinedActions = [] )

        self.STATE("HeaterOff")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.END_STN()

        # Users are responsible for creating/deleting of user actions and have to ensure
        # that they still exist until the end of simulation.
        self.action = simpleUserAction()

        # ON_EVENT function
        # The actions executed when the event on the inlet epIn event port is received.
        # OnEvent handlers can be also specified as a part of the state definition
        # and then they are active only when that particular state is active.
        # Arguments:
        # - Inlet event port
        # - switchToStates is a list of python tuples (STN-name, State-name)
        # - triggerEvents is a list of python tuples (outlet-event-port, expression)
        # - setVariableValues is a list of python tuples (variable, expression)
        # - userDefinedActions is a list of daeAction derived classes
        self.ON_EVENT(self.epIn, switchToStates     = [ ('Regulator', 'HeaterOff')],
                                 triggerEvents      = [],
                                 setVariableValues  = [ (self.dummy, self.epIn()) ],
                                 userDefinedActions = [self.action])

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial5a")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385)
        self.m.m.SetValue(1)
        self.m.alpha.SetValue(200)
        self.m.A.SetValue(0.1)
        self.m.Tsurr.SetValue(283)

    def SetUpVariables(self):
        # Set the state active at the beginning (the default is the first declared state; here 'Heating')
        self.m.stnRegulator.ActiveState = "Heating"

        self.m.T.SetInitialCondition(283)
        self.m.dummy.AssignValue(0.0)


# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 2
    sim.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

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
    simulation.ReportingInterval = 2
    simulation.TimeHorizon = 500

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
