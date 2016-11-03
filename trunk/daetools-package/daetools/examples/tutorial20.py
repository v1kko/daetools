#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial20.py
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
__doc__ = """
This is a simple example to test daetools support for:

- Scilab/GNU_Octave/Matlab MEX functions
- Simulink S-functions
- Modelica code-generator
- gPROMS code-generator
- FMI code-generator (for Co-Simulation)

The model has two inlet and two outlet ports.
The values of the outlets are equal to inputs multiplied by a multiplier "m":

.. code-block:: none

    out1.y   = m1   x in1.y
    out2.y[] = m2[] x in2.y[]

where multipliers m1 and m2[] are:

.. code-block:: none

  STN('Multipliers'):
      when 'variableMultipliers':
          dm1/dt   = p1
          dm2[]/dt = p2
      when 'constantMultipliers':
          dm1/dt   = 0
          dm2[]/dt = 0
        
(that is the multipliers can be constant or variable).

The ports in1 and out1 are scalar (width = 1).
The ports in2 and out2 are vectors (width = 1).

Achtung, Achtung!!
Notate bene:

1. Inlet ports must be DOFs (that is to have their values asssigned),
   for they can't be connected when the model is simulated outside of daetools context.
2. Only scalar output ports are supported at the moment!! (Simulink issue)

FMI Cross-Check results:

1. Start the DAEPlotter (or change the data reporter below).

2. Execute:

   .. code-block:: none

      ./fmuCheck.linux64 -n 10 tutorial20.fmu

3. Results:

   .. code-block:: none

    [INFO][FMUCHK] FMI compliance checker 2.0 [FMILibrary: 2.0] build date: Aug 22 2014
    [INFO][FMUCHK] Will process FMU tutorial20.fmu
    [INFO][FMILIB] XML specifies FMI standard version 2.0
    [INFO][FMUCHK] Model name: tutorial20
    [INFO][FMUCHK] Model GUID: e9654532-0998-11e6-957b-9cb70d5dfdfc
    [INFO][FMUCHK] Model version:
    [INFO][FMUCHK] FMU kind: CoSimulation
    [INFO][FMUCHK] The FMU contains:
    0 constants
    3 parameters
    0 discrete variables
    4 continuous variables
    2 inputs
    2 outputs
    0 local variables
    0 independent variables
    0 calculated parameters
    6 real variables
    0 integer variables
    0 enumeration variables
    0 boolean variables
    1 string variables

    [INFO][FMUCHK] Printing output file header
    time,out_1.y,out_2.y
    [INFO][FMUCHK] Model identifier for CoSimulation: tutorial20
    [INFO][FMILIB] Loading 'linux64' binary with 'default' platform types
    [INFO][FMUCHK] Version returned from CS FMU:   2.0
    ***********************************************************************
                                     Version:   1.5.0
                                     Copyright: Dragan Nikolic, 2016
                                     Homepage:  http://www.daetools.com
           @                       @
           @   @@@@@     @@@@@   @@@@@    @@@@@    @@@@@   @      @@@@@
      @@@@@@        @   @     @    @     @     @  @     @  @     @
     @     @   @@@@@@   @@@@@@     @     @     @  @     @  @      @@@@@
     @     @  @     @   @          @     @     @  @     @  @           @
      @@@@@@   @@@@@@    @@@@@      @@@   @@@@@    @@@@@    @@@@  @@@@@
    ***********************************************************************
    DAE Tools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3
    as published by the Free Software Foundation.
    DAE Tools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses>.
    ***********************************************************************
    Creating the system...
    The system created successfully in: 0.002 s
    Starting the initialization of the system... Done.
    [INFO][FMUCHK] Initialized FMU for simulation starting at time 0
    0,1.0000000010000001E+00,2.0000000039999999E+00
    10,1.0100000000000001E+02,4.0200000000000006E+02
    20,2.0099999999999994E+02,8.0199999999999977E+02
    30,3.0100000000000000E+02,1.2020000000000000E+03
    40,4.0100000000000000E+02,1.6020000000000000E+03
    50,5.0099999999999989E+02,2.0019999999999995E+03
    60,6.0100000000000000E+02,2.4020000000000000E+03
    70,7.0100000000000000E+02,2.8020000000000000E+03
    80,8.0100000000000000E+02,3.2020000000000000E+03
    90,9.0099999999999977E+02,3.6019999999999991E+03
    100,1.0009999999999998E+03,4.0019999999999991E+03
    [INFO][FMUCHK] Simulation finished successfully at time 100
    FMU check summary:
    FMU reported:
            0 warning(s) and error(s)
    Checker reported:
            0 Warning(s)
            0 Error(s)
"""

import sys, numpy
from time import localtime, strftime
from daetools.pyDAE import *

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

def run_code_generators(simulation, log):
    # Demonstration of daetools code-generators:
    import tempfile
    tmp_folder = tempfile.mkdtemp(prefix = 'daetools-code_generator-fmi-')
    msg = 'Generated code (Modelica, gPROMS and FMU) \nwill be located in: \n%s' % tmp_folder
    log.Message(msg, 0)
    
    try:
        from PyQt4 import QtCore, QtGui
        if not QtGui.QApplication.instance():
            app_ = QtGui.QApplication(sys.argv)
        QtGui.QMessageBox.warning(None, "tutorial20", msg)
    except Exception as e:
        log.Message(str(e), 0)

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

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 100
    simulator  = daeSimulator(app, simulation=sim, run_before_simulation_begin_fn = run_code_generators)
    simulator.exec_()

# This function is used by daetools_mex, daetools_s and daetools_fmi_cs to load a simulation.
# It can have any number of arguments, but must return an initialized daeSimulation object.
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
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Nota bene: store the objects since they will be destroyed when they go out of scope
    simulation.__rt_objects__ = [daesolver, datareporter, log]

    return simulation
    
# Setup everything manually and run in a console
def consoleRun():
    # Create simulation
    simulation = create_simulation()

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run code-generators
    run_code_generators(simulation, simulation.Log)
    
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
