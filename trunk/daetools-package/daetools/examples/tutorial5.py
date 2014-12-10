#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial5.py
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
In this example we use the same conduction problem as in the tutorial 4.

Here we introduce:

- Discontinuous equations (non-symmetrical state transition networks: daeSTN statements)

Again we have a piece of copper (a plate) is at one side exposed to the source of heat
and at the other to the surroundings. The process starts at the temperature of 283K.
The metal is allowed to warm up, and then its temperature is kept in the interval
[320 - 340] for at 350 seconds. After 350s the heat source is removed and the metal
cools down slowly again to the ambient temperature.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin   = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T     = daeVariable("T",     temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        """
        Non-symmetrical STNs in DAE Tools can be created by using STN/STATE/END_STN statements.
        Again, states MUST contain the SAME NUMBER OF EQUATIONS.
        First start with the call to STN("STN_Name") function from daeModel class.
        If you need to change active states in operating procedure in function Run()
        store the stn reference (here in the stnRegulator object).
        After that call, define your states by calling the function STATE("State1") and write
        equations that will be active if this state (called 'State1') is active.

        Models and states can contain OnConditionActions and OnEventActions.
        OnConditionActions are trigerred when a specified logical condition is satisfied.
        OnEventActions are trigerred when a specified EventPort receives an event (more info in tutorial13.py).
        There are different types of actions that can be executed by a logical condition.
        In this example, we want to change the active states subject to given conditions and
        ON_CONDITION(logical_condition, switchToStates = [list_of_actions]) function will be used.
        Again, there is an optional argument eventTolerance, as explained in tutorial 4.
        
        Repeat this procedure for all states in the state transition network.
        Finally call the function END_STN() to finalize the state transition network.
        """
        self.stnRegulator = self.STN("Regulator")

        self.STATE("Heating")

        eq = self.CreateEquation("Q_in", "The heater is on")
        eq.Residual = self.Qin() - Constant(1500 * W)

        """
        ON_CONDITION() function
        Arguments:
          - Condition that triggers the actions
          - 'switchToStates' is a list of python tuples (STN-name-relative-to-model, State-name) that will be set active
             when the condition is satisified
          - 'triggerEvents' is a list of python tuples (outlet-event-port, expression),
             where the first part is the event-port object and the second a value to be sent when the event is trigerred
          - 'setVariableValues' is a list of python tuples (variable, expression); if the variable is differential it
             will be reinitialized (using ReInitialize() function), otherwise it will be reassigned (using ReAssign() function)
          - 'userDefinedActions' is a list of user defined daeAction-derived objects (see tutorial13.py for more information)
        """
        # Here the built-in Time() function is used to get the current time (time elapsed) in the simulation
        self.ON_CONDITION(self.T() > Constant(340*K), switchToStates     = [ ('Regulator', 'Cooling') ],
                                                      setVariableValues  = [],
                                                      triggerEvents      = [],
                                                      userDefinedActions = [] )
        self.ON_CONDITION(Time() > Constant(350*s), switchToStates     = [ ('Regulator', 'HeaterOff') ],
                                                    setVariableValues  = [],
                                                    triggerEvents      = [],
                                                    userDefinedActions = [] )

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.ON_CONDITION(self.T() < Constant(320*K), switchToStates     = [ ('Regulator', 'Heating') ],
                                                      setVariableValues  = [],
                                                      triggerEvents      = [],
                                                      userDefinedActions = [] )
        self.ON_CONDITION(Time() > Constant(350*s), switchToStates     = [ ('Regulator', 'HeaterOff') ],
                                                    setVariableValues  = [],
                                                    triggerEvents      = [],
                                                    userDefinedActions = [] )

        self.STATE("HeaterOff")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.END_STN()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial5")
        self.m.Description = __doc__
        
    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        # Set the state active at the beginning (the default is the first declared state; here 'Heating')
        self.m.stnRegulator.ActiveState = "Heating"

        self.m.T.SetInitialCondition(283 * K)


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
