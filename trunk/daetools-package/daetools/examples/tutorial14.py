#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial14.py
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
***********************************************************************************
"""
__doc__ = """
In this example we use the same conduction problem as in the tutorial 5.

Here we introduce the external functions concept that can handle and evaluate
functions in external libraries. Here we use daeScalarExternalFunction class
derived external function object to calculate the power.

A support for external functions is still experimental and the goal is
to support certain software components such as thermodynamic property packages etc.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

class extfnPower(daeScalarExternalFunction):
    def __init__(self, Name, Model, units, m, cp, dT):
        arguments = {}
        arguments["m"]  = m
        arguments["cp"] = cp
        arguments["dT"] = dT
        
        daeScalarExternalFunction.__init__(self, Name, Model, units, arguments)
    
    def Calculate(self, values):
        """
        """
        m  = values["m"]
        cp = values["cp"]
        dT = values["dT"]
        
        res = m * cp * dT
        
        #print('Power(m = {0}, cp = {1}, dT = {2}) = {3}'.format(m, cp, dT, res))
        return res

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
        
        self.Power     = daeVariable("Power",      power_t,       self, "Power")
        self.Power_ext = daeVariable("Power_ext",  power_t,       self, "Power")

        self.extFun_ = None
        
    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Create external function
        # It has to be created in DeclareEquations since it accesses the params/vars values
        self.Pext = extfnPower("Power", self, W, self.m(), self.cp(), self.T.dt())

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

        eq = self.CreateEquation("Power", "")
        eq.Residual = self.Power() - self.m() * self.cp() * self.T.dt()

        eq = self.CreateEquation("Power_ext", "")
        eq.Residual = self.Power_ext() - self.Pext()
        self.extFun_ = eq

        self.stnRegulator = self.STN("Regulator")

        self.STATE("Heating")

        eq = self.CreateEquation("Q_in", "The heater is on")
        eq.Residual = self.Qin() - Constant(1500 * W)

        # Here the Time() function is used to get the current time (time elapsed) in the simulation
        self.SWITCH_TO("Cooling",   self.T() > Constant(340 * K))
        self.SWITCH_TO("HeaterOff", Time()   > Constant(350 * s))

        self.STATE("Cooling")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.SWITCH_TO("Heating",   self.T() < Constant(320 * K))
        self.SWITCH_TO("HeaterOff", Time()   > Constant(350 * s))

        self.STATE("HeaterOff")

        eq = self.CreateEquation("Q_in", "The heater is off")
        eq.Residual = self.Qin()

        self.END_STN()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial14")
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
    sim.ReportingInterval = 0.5
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
    print(simulation.m.extFun_.EquationExecutionInfos[0].VariableIndexes)

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
