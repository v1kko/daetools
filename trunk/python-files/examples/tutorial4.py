#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             tutorial4.py
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
In this example we use the same conduction problem as in the tutorial 1.
Here we introduce:
 - Discontinuous equations (symmetrical state transition networks: daeIF statements)

Here we have a very simple heat balance:
    ro * cp * dT/dt - Qin = h * A * (T - Tsurr)

A piece of copper (a plate) is at one side exposed to the source of heat and at the
other to the surroundings. The process starts at the temperature of the metal of 283K.
The metal is allowed to warm up for 200 seconds and then the heat source is
removed and the metal cools down slowly again to the ambient temperature.

"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in daeVariableTypes.py

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       eReal, self, "Mass of the copper plate, kg")
        self.cp    = daeParameter("c_p",     eReal, self, "Specific heat capacity of the plate, J/kgK")
        self.alpha = daeParameter("&alpha;", eReal, self, "Heat transfer coefficient, W/m2K")
        self.A     = daeParameter("A",       eReal, self, "Area of the plate, m2")
        self.Tsurr = daeParameter("T_surr",  eReal, self, "Temperature of the surroundings, K")

        self.Qin  = daeVariable("Q_in",  power_t,       self, "Power of the heater, W")
        self.time = daeVariable("&tau;", no_t,          self, "Time elapsed in the process, s")
        self.T    = daeVariable("T",     temperature_t, self, "Temperature of the plate, K")

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        eq = self.CreateEquation("Time", "Differential equation to calculate the time elapsed in the process.")
        eq.Residual = self.time.dt() - 1.0

        # Symmetrical STNs in DAE Tools can be created by using IF/ELSE_IF/ELSE/END_IF statements.
        # These statements are more or less used as normal if/else if/else blocks in all programming languages.
        # An important rule is that all states MUST contain the SAME NUMBER OF EQUATIONS.
        # First start with the call to IF( condition ) function from daeModel class.
        # After that call, write equations that will be active if 'condition' is satisfied.
        # If there are only two states call the function ELSE() and write equations that will be active
        # if 'condition' is not satisfied.
        # If there are more than two states, start a new state by calling the function ELSE_IF (condition2)
        # and write the equations that will be active if 'condition2' is satisfied. And so on...
        # Finally call the function END_IF() to finalize the state transition network.
        # There is an optional argument EventTolerance of functions IF and ELSE_IF. It is used by the solver
        # to control the process of discovering the discontinuities.
        # Details about the EventTolerance purpose will be given for the condition time < 200, given below.
        # Conditions like time < 200 will be internally transformed into the following equations:
        #        time - 200 - EventTolerance = 0
        #        time - 200 = 0
        #        time - 200 + EventTolerance = 0
        # where EventTolerance is used to control how far will solver go after/before discovering a discontinuity.
        # The default value is 1E-7. Therefore, the above expressions will transform into:
        #        time - 199.9999999 = 0
        #        time - 200         = 0
        #        time - 200.0000001 = 0
        # For example, if the variable 'time' is increasing from 0 and is approaching the value of 200,
        # the equation 'Q_on' will be active. As the simulation goes on, the variable 'time' will reach the value
        # of 199.9999999 and the solver will discover that the expression 'time - 199.9999999' became equal to zero.
        # Then it will check if the condition 'time < 200' is satisfied. It is, and no state change will occur.
        # The solver will continue, the variable 'time' will increase to 200 and the solver will discover that
        # the expression 'time - 200' became equal to zero. It will again check the condition 'time < 200' and
        # find out that it is not satisfied. Now the state ELSE becomes active, and the solver will use equations
        # from that state (in this example equation 'Q_off').
        # But, if we have 'time > 200' condition instead, we can see that when the variable 'time' reaches 200
        # the expression 'time - 200' becomes equal to zero. The solver will check the condition 'time > 200'
        # and will find out that it is not satisfied and no state change will occur. However, once the variable
        # 'time' reaches the value of 200.0000001 the expression 'time - 200.0000001' becomes equal to zero.
        # The solver will check the condition 'time > 200' and will find out that it is satisfied and it will
        # go to the state ELSE.
        # In this example, input power of the heater will be 1500 Watts if the time is less than 200.
        # Once we reach 200 seconds the heater is switched off (power is 0 W) and the sytem starts to cool down.
        self.IF(self.time() < 200, EventTolerance = 1E-5)

        eq = self.CreateEquation("Q_on", "The heater is on")
        eq.Residual = self.Qin() - 1500

        self.ELSE()

        eq = self.CreateEquation("Q_off", "The heater is off")
        eq.Residual = self.Qin()

        self.END_IF()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial4")
        self.m.Description = "This tutorial explains how to define and use discontinuous equations: symmetric state transition networks (daeIF). \n" \
                              "A piece of copper (a plate) is at one side exposed to the source of heat and at the " \
                              "other to the surroundings. The process starts at the temperature of the metal of 283K. " \
                              "The metal is allowed to warm up for 200 seconds and then the heat source is " \
                              "removed and the metal cools down slowly again to the ambient temperature."

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385)
        self.m.m.SetValue(1)
        self.m.alpha.SetValue(200)
        self.m.A.SetValue(0.1)
        self.m.Tsurr.SetValue(283)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(283)
        self.m.time.SetInitialCondition(0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
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
    simulation.ReportingInterval = 10
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
