#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                             parser_test.py
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

import sys, tempfile
from daetools.pyDAE.parser import ExpressionParser
from daetools.pyDAE.parserDictionaries import getParserDictionaries
from daetools.pyDAE import *
from daetools.pyDAE.daeDataReporters import *
from time import localtime, strftime

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x1  = daeParameter("x1", eReal, self, "Power of the heater, W")
        self.x2  = daeParameter("x2", eReal, self, "Power of the heater, W")
        self.x3  = daeParameter("x3", eReal, self, "Power of the heater, W")
        self.x4  = daeParameter("x4", eReal, self, "Power of the heater, W")

        self.y = daeVariable("y", temperature_t, self, "Temperature of the plate, K")

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation.")

        dictIdentifiers, dictFunctions = getParserDictionaries(self)
        print 'Identifiers:\n', dictIdentifiers
        print '\n'
        print 'Functions:\n', dictFunctions
        print '\n'

        parser  = ExpressionParser(dictIdentifiers, dictFunctions)
        parser.parse('y - x1 + x2 * x3 / (2 * x4 - 12)')
        residual = parser.evaluate()
        print 'parse_result =', str(parser.parseResult)
        eq.Residual = residual

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("parser_test")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.x1.SetValue(10)
        self.m.x2.SetValue(1.01)
        self.m.x3.SetValue(5.2)
        self.m.x4.SetValue(500)

    def SetUpVariables(self):
        pass


# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    dr  = setupDataReporters(sim)
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 100
    simulator  = daeSimulator(app, simulation=sim, datareporter=dr)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = daeTCPIPDataReporter()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 10

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
