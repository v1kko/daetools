#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial20.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2014
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
This is a simple model to test the support for:
 - Scilab/GNU Octave/Matlab MEX functions
 - Simulink S-functions
 - FMI

The model is very simple: it has two inlet and two outlet ports.
The values of the outlet port variables are: 
    out1 = in1
    out2 = in2
The ports in1 and out1 are scalar (width = 1).
The ports in2 and out2 are vectors (width = 1).

Note:
  1. Inlet ports must be DOFs (have their values asssigned) for they can't be connected
     outside of daetools simulation.
  2. Only scalar output ports are supported at the moment.
"""

import sys, numpy
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class portScalar(daePort):
    def __init__(self, Name, PortType, Model, Description = ""):
        daePort.__init__(self, Name, PortType, Model, Description)

        self.y = daeVariable("y", no_t, self, "")

class portVector(daePort):
    def __init__(self, Name, PortType, Model, Description, width):
        daePort.__init__(self, Name, PortType, Model, Description)

        self.y = daeVariable("y", no_t, self, "", [width])

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.w = daeDomain("w", self, m, "Ports width")

        self.y1 = daeVariable("y1", no_t, self, "State variable 1")
        self.y2 = daeVariable("y2", no_t, self, "State variable 2", [self.w])

        self.in1  = portScalar("in_1",  eInletPort,  self, "The multiplier 1")
        self.out1 = portScalar("out_1", eOutletPort, self, "The result 1")

        self.in2  = portVector("in_2",  eInletPort,  self, "The multiplier 2", self.w)
        self.out2 = portVector("out_2", eOutletPort, self, "The result 2",     self.w)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        nw = self.w.NumberOfPoints

        # State variables
        eq = self.CreateEquation("y1", "Equation to calculate the state variable y1.")
        eq.Residual = self.y1.dt() - self.in1.y() * Constant(1/s)

        for w in range(nw):
            eq = self.CreateEquation("y2(%d)" % w, "Equation to calculate the state variable y2(%d)" % w)
            eq.Residual = self.y2.dt(w) - self.in2.y(w) * Constant(1/s)

        # Set the outlet port values
        eq = self.CreateEquation("out_1", "Equation to calculate out1.y")
        eq.Residual = self.out1.y() - self.y1()

        for w in range(nw):
            eq = self.CreateEquation("out_2(%d)" % w, "Equation to calculate out2.y(%d)" % w)
            eq.Residual = self.out2.y(w) - self.y2(w)
   
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial20")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.w.CreateArray(1)

    def SetUpVariables(self):
        nw = self.m.w.NumberOfPoints

        self.m.in1.y.AssignValue(1)
        self.m.in2.y.AssignValues(numpy.ones(nw) * 2)

        self.m.y1.SetInitialCondition(0)
        self.m.y2.SetInitialConditions(numpy.zeros(nw))

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 100
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
    simulation.TimeHorizon = 100

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
