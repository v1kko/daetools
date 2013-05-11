#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
..
 ***********************************************************************************
                             tutorial13.py
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
 ************************************************************************************

In this example we use the same problem as in the tutorial 5.

Here we introduce:

- The event ports
- ON_CONDITION() function showing the new types of actions that can be executed
  during state transitions
- ON_EVENT() function showing the new types of actions that can be executed
  when an event is triggered
- User defined actions
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

# User defined action executed in OnEvent handler
# The __init__ function accepts an event port as an argument that is used 
# to retrieve the value of the event data 
class simpleUserAction(daeAction):
    def __init__(self, eventPort):
        daeAction.__init__(self)
        self.eventPort = eventPort
        print self.Name

    def Execute(self):
        # The floating point value of the data sent with the event can be retrieved 
        # using daeEventPort property EventData 
        msg = 'simpleUserAction executed; input data = {0}'.format(self.eventPort.EventData)
        
        # Try to show a message box if there is application instance already defined
        # that is we are in the 'gui' mode
        try:
            from PyQt4 import QtCore, QtGui
            if QtCore.QCoreApplication.instance():
                QtGui.QMessageBox.warning(None, 'tutorial13', msg)
        except:
            print msg
        
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
        self.event = daeVariable("event", no_t,          self, "Variable which value is set in ON_EVENT function")

        # Here we create two event ports (inlet and outlet) and connect them.
        # It makes no sense in reality, but this is example is just a show case - in the real application
        # they would be defined in separate models.
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
        eq.Residual = self.Qin() - Constant(1500 * W)

        """
                                          ON_CONDITION() function
        Arguments:
          - Condition that triggers the actions
          - 'switchTo' is the name of the state (in the current state transition network) that will be set active 
             when the condition is satisified
          - 'triggerEvents' is a list of python tuples (outlet-event-port, expression), 
             where the first part is the event-port object and the second a value to be sent when the event is trigerred
          - 'setVariableValues' is a list of python tuples (variable, expression); if the variable is differential it
             will be reinitialized (using ReInitialize() function), otherwise it will be reassigned (using ReAssign() function)
          - 'userDefinedActions' is a list of user defined daeAction-derived objects
        """
        self.ON_CONDITION(self.T() > Constant(340*K),    switchTo           = 'Cooling',
                                                         setVariableValues  = [ (self.event, 100) ], # event variable is dimensionless
                                                         triggerEvents      = [],
                                                         userDefinedActions = [] )

        self.ON_CONDITION(Time() > Constant(350*s), switchTo           = 'HeaterOff',
                                                    setVariableValues  = [],
                                                    triggerEvents      = [ (self.epOut, self.T() + Constant(5.0*K)) ],
                                                    userDefinedActions = [] )

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.ON_CONDITION(self.T() < Constant(320*K),    switchTo           = 'Heating',
                                                         setVariableValues  = [ (self.event, 200) ], # event variable is dimensionless
                                                         triggerEvents      = [],
                                                         userDefinedActions = [] )

        self.ON_CONDITION(Time() > Constant(350*s), switchTo           = 'HeaterOff',
                                                    setVariableValues  = [],
                                                    triggerEvents      = [ (self.epOut, self.T() + Constant(6.0*K)) ],
                                                    userDefinedActions = [] )

        self.STATE("HeaterOff")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.END_STN()

        # Users are responsible for creating/deleting of user actions and have to ensure
        # that they still exist until the end of simulation.
        self.action = simpleUserAction(self.epIn)

        """
                                   ON_EVENT() function
        The actions executed when the event on the inlet epIn event port is received.
        OnEvent handlers can be also specified as a part of the state definition
        and then they are active only when that particular state is active.
        Arguments:
          - Event port (could be either inlet or outlet)
          - 'switchToStates' is a list of python tuples (STN-name, State-name)
          - 'triggerEvents' is a list of python tuples (outlet-event-port, expression)
          - 'setVariableValues' is a list of python tuples (variable, expression)
          - 'userDefinedActions' is a list of daeAction derived classes
        daeEventPort defines the operator() which returns adouble object that can be evaluated at the moment
        when the action is executed to get the value of the event data (ie. to set a new value of a variable).
        """
        self.ON_EVENT(self.epIn, switchToStates     = [ ('Regulator', 'HeaterOff')],
                                 setVariableValues  = [ (self.event, self.epIn()) ],
                                 triggerEvents      = [],
                                 userDefinedActions = [self.action])

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial13")
        self.m.Description = "In this example we use the same problem as in the tutorial 5. \n" \
                             "Here we introduce: \n" \
                             "  - The event ports \n" \
                             "  - ON_CONDITION() function showing the new types of actions that can be executed " \
                             "during state transitions \n" \
                             "  - ON_EVENT() function showing the new types of actions that can be executed " \
                             "when an event is triggered \n" \
                             "  - User defined actions" \

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
        self.m.event.AssignValue(0.0)


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

    # Print the list of events
    log.Message('Events occured at the {0} event port:'.format(simulation.m.epOut.CanonicalName), 0)
    log.Message(str(simulation.m.epOut.Events), 0)

    log.Message('Events occured at the {0} event port:'.format(simulation.m.epIn.CanonicalName), 0)
    log.Message(str(simulation.m.epIn.Events), 0)
    
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
