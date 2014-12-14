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
This is a simple example to test daetools support for:
 - Scilab/GNU_Octave/Matlab MEX functions
 - Simulink S-functions
 - Modelica code-generator
 - gPROMS code-generator
 - FMI code-generator (for Co-Simulation)

The model has two inlet and two outlet ports.
The values of the outlets are equal to inputs multiplied by a multiplier:
    out1.y   = m1   x in1.y
    out2.y[] = m2[] x in2.y[]

where multipliers m1 and m2[] are:
  STN(Multipliers):
    when 'Variable':
        dm1/dt   = p1
        dm2[]/dt = p2
    when 'Constant':
        dm1/dt   = 0
        dm2[]/dt = 0
that is can be constant or variable.

The ports in1 and out1 are scalar (width = 1).
The ports in2 and out2 are vectors (width = 1).

Achtung, Achtung!!
Notate bene:
  1. Inlet ports must be DOFs (that is to have their values asssigned),
     for they can't be connected when the model is simulated outside of daetools context.
  2. Only scalar output ports are supported at the moment!! (Simulink issue)
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

        self.w = daeDomain("w", self, unit(), "Ports width")

        self.p1 = daeParameter("p1", s**(-1), self, "Parameter multiplier 1 (fixed)")
        self.p2 = daeParameter("p2", s**(-1), self, "Parameter multiplier 2 (fixed)")

        self.m1 = daeVariable("m1", no_t, self, "Multiplier 1")
        self.m2 = daeVariable("m2", no_t, self, "Multiplier 2", [self.w])

        self.in1  = portScalar("in_1",  eInletPort,  self, "Input 1")
        self.out1 = portScalar("out_1", eOutletPort, self, "Output 1 = p1 x m1")

        self.in2  = portVector("in_2",  eInletPort,  self, "Input 2",              self.w)
        self.out2 = portVector("out_2", eOutletPort, self, "Output 2 = p2 x m2[]", self.w)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        nw = self.w.NumberOfPoints

        # Set the outlet port values
        eq = self.CreateEquation("out_1", "out_1.y = m1 x in1.y")
        eq.Residual = self.out1.y() - self.m1() * self.in1.y()

        for w in range(nw):
            eq = self.CreateEquation("out_2(%d)" % w, "out_2.y[%d] = m2[%d] * in2.y[%d]" % (w, w, w))
            eq.Residual = self.out2.y(w) - self.m2(w) * self.in2.y(w)

        # STN Multipliers
        self.stnMultipliers = self.STN("Multipliers")

        self.STATE("variableMultipliers") # Variable multipliers

        eq = self.CreateEquation("m1", "Multiplier 1 (Variable)")
        eq.Residual = self.m1.dt() - self.p1()

        for w in range(nw):
            eq = self.CreateEquation("m2(%d)" % w, "Multiplier 2 (Variable)")
            eq.Residual = self.m2.dt(w) - self.p2()

        self.STATE("constantMultipliers") # Constant multipliers

        eq = self.CreateEquation("m1", "Multiplier 1 (Constant)")
        eq.Residual = self.m1.dt()

        for w in range(nw):
            eq = self.CreateEquation("m2(%d)" % w, "Multiplier 2 (Constant)")
            eq.Residual = self.m2.dt(w)

        self.END_STN()
   
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial20")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.w.CreateArray(1)

        self.m.p1.SetValue(10)
        self.m.p2.SetValues(20)

    def SetUpVariables(self):
        nw = self.m.w.NumberOfPoints

        self.m.stnMultipliers.ActiveState = "variableMultipliers"

        self.m.in1.y.AssignValue(1)
        self.m.in2.y.AssignValues(numpy.ones(nw) * 2)

        self.m.m1.SetInitialCondition(1)
        self.m.m2.SetInitialConditions(numpy.ones(nw))

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
    if(datareporter.Connect("127.0.0.1:50000", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()
    
    # Code generators
    import tempfile
    tmp_folder = tempfile.mkdtemp(prefix = 'daetools-code_generator-fmi-')
    print('Generated code will be located in: %s' % tmp_folder)

    # Modelica:
    from daetools.code_generators.modelica import daeCodeGenerator_Modelica
    cg = daeCodeGenerator_Modelica()
    cg.generateSimulation(simulation, tmp_folder)

    # gPROMS:
    from daetools.code_generators.gproms import daeCodeGenerator_gPROMS
    cg = daeCodeGenerator_gPROMS()
    cg.generateSimulation(simulation, tmp_folder)

    # Functional Mock-up Interface for co-simulation
    from daetools.code_generators.fmi import daeCodeGenerator_FMI
    cg = daeCodeGenerator_FMI()
    cg.generateSimulation(simulation, tmp_folder, __file__, 'create_simulation', '', [])

    # Run
    simulation.Run()
    simulation.Finalize()

def create_simulation():
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
    if(datareporter.Connect("127.0.0.1:50000", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Nota bene: store the objects since they will be destroyed when they go out of scope
    simulation.__rt_objects__ = [daesolver, datareporter, log]

    return simulation
    
if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
