#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial21.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2016
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
__doc__ = """Another example of using daetools and numpy.
This problem is from the Sundials ARKODE (ark_analytic_sys.cpp):
The following is a simple example problem with analytical solution,
    dy/dt = A*y
where A = V*D*Vi,
    V = [1 -1 1; -1 2 1; 0 -1 2];
    Vi = 0.25*[5 1 -3; 2 2 -2; 1 1 1];
    D = [-0.5 0 0; 0 -0.1 0; 0 0 lam];
where lam is a large negative number. The analytical solution to this problem is
    Y(t) = V*exp(D*t)*Vi*Y0
for t in the interval [0.0, 0.05], with initial condition:
    y(0) = [1,1,1]'.

The stiffness of the problem is directly proportional to the value of "lamda".
The value of lamda should be negative to result in a well-posed ODE;
for values with magnitude larger than 100 the problem becomes quite stiff.

In this example, we choose lamda = -100.

Solution:
    lamda = -100
   reltol = 1e-06
   abstol = 1e-10

      t        y0        y1        y2
   --------------------------------------
    0.0050   0.70327   0.70627   0.41004
    0.0100   0.52267   0.52865   0.05231
    0.0150   0.41249   0.42145  -0.16456
    0.0200   0.34504   0.35696  -0.29600
    0.0250   0.30349   0.31838  -0.37563
    0.0300   0.27767   0.29551  -0.42383
    0.0350   0.26138   0.28216  -0.45296
    0.0400   0.25088   0.27459  -0.47053
    0.0450   0.24389   0.27053  -0.48109
    0.0500   0.23903   0.26858  -0.48740
   --------------------------------------
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

typeNone = daeVariableType("typeNone", unit(), 0, 1E10,   0, 1e-10)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x = daeDomain("x", self, unit(), "")
        self.y = daeVariable("y", typeNone, self, "", [self.x])

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Input data:
        lamda = -100
        V = numpy.array([[1, -1, 1], [-1, 2, 1], [0, -1, 2]])
        Vi = 0.25 * numpy.array([[5, 1, -3], [2, 2, -2], [1, 1, 1]])
        D = numpy.array([[-0.5, 0, 0], [0, -0.1, 0], [0, 0, lamda]])

        # Create a vector of y's:
        y = numpy.empty(3, dtype=object)
        y[:] = [self.y(i) for i in range(3)]

        # Use dot product (numpy arrays don't behave as matrices)
        # or use numpt.matrix where the operator * performss the dot product.
        dydt = V.dot(D).dot(Vi).dot(y)
        print(dydt)
        for i in range(3):
            eq = self.CreateEquation("y(%d)" % i, "")
            eq.Residual = self.y.dt(i) - dydt[i]
            eq.CheckUnitsConsistency = False
   
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial21")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.x.CreateArray(3)

    def SetUpVariables(self):
        y0 = numpy.array([1.0, 1.0, 1.0])
        self.m.y.SetInitialConditions(y0)

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 0.005
    sim.TimeHorizon       = 0.05
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    daesolver.RelativeTolerance = 1E-6
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 0.005
    simulation.TimeHorizon = 0.05

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
