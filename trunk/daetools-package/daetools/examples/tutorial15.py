#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
..
 ***********************************************************************************
                             tutorial15.py
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

In this example we use the same problem as in the tutorial 4.

Here we introduce:

- Nested state transitions

"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kW

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin  = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        # IF-1
        self.IF(Time() < Constant(200*s)) # --------------------------------------------------> 1

        # IF-1.1
        self.IF((Time() >= Constant(0*s)) & (Time() < Constant(100*s))) # ------------> 1.1

        # IF-1.1.1
        self.IF((Time() >= Constant(0*s)) & (Time() < Constant(50*s))) # ---> 1.1.1
        eq = self.CreateEquation("Q_111", "The heater is on")
        eq.Residual = self.Qin() - Constant(1600 * W)

        self.ELSE() # ----------------------------------------------------> 1.1.1
        eq = self.CreateEquation("Q_112", "The heater is on")
        eq.Residual = self.Qin() - Constant(1500 * W)

        self.END_IF() # --------------------------------------------------> 1.1.1

        self.ELSE() # --------------------------------------------------------------> 1.1
        eq = self.CreateEquation("Q_12", "The heater is on")
        eq.Residual = self.Qin() - Constant(1400 * W)

        self.END_IF() # ------------------------------------------------------------>1.1

        self.ELSE_IF((Time() >= Constant(200*s)) & (Time() < Constant(300*s))) # ------------->1

        eq = self.CreateEquation("Q_2", "The heater is on")
        eq.Residual = self.Qin() - Constant(1300 * W)

        self.ELSE() # ------------------------------------------------------------------------>1

        eq = self.CreateEquation("Q_3", "The heater is off")
        eq.Residual = self.Qin()

        self.END_IF() # ---------------------------------------------------------------------->1

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial15")
        self.m.Description = "This tutorial introdces nested state transitions."

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(283 * K)

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
