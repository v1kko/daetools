#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                      stirred_reactor_Van_de_Vusse.py
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
Reactions:
 A -> B -> 2C
  \-> 2D
"""

import sys
from daetools.pyDAE import *
from daetools.solvers import pyIPOPT
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-",      0, 1E10,   1, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",    100, 1000, 300, 1e-5)
typeConductivity = daeVariableType("Conductivity", "W/mK",   0, 1E10, 100, 1e-5)
typeDensity      = daeVariableType("Density",      "kg/m3",  0, 1E10, 100, 1e-5)
typeHeatCapacity = daeVariableType("HeatCapacity", "J/KgK",  0, 1E10, 100, 1e-5)

class modSTR(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        # Parameters
        self.F      = daeParameter("F",             eReal, self, "Feed rate, l/h")
        self.Qk     = daeParameter("Q_k",           eReal, self, "Jacket cooling rate kJ/h")
        self.Ca0    = daeParameter("Ca_0",          eReal, self, "Inlet feed concentration, mol/m3")
        self.T0     = daeParameter("T_0",           eReal, self, "Inlet feed temperature, C")

        self.k10    = daeParameter("k_10",          eReal, self, "A->B Pre-exponential factor, 1/h")
        self.k20    = daeParameter("k_20",          eReal, self, "B->C Pre-exponential factor, 1/h")
        self.k30    = daeParameter("k_30",          eReal, self, "A->2D Pre-exponential factor, 1/h")

        self.E1     = daeParameter("E_1",          eReal, self, "A->B Activation energy, K")
        self.E2     = daeParameter("E_2",          eReal, self, "B->C Activation energy, K")
        self.E3     = daeParameter("E_3",          eReal, self, "A->2D Activation energy, K")

        self.dHrAB  = daeParameter("&Delta;Hr_AB", eReal, self, "A->B Heat of reaction, kJ/mol")
        self.dHrBC  = daeParameter("&Delta;Hr_BC", eReal, self, "B->C Heat of reaction, kJ/mol")
        self.dHrAD  = daeParameter("&Delta;Hr_AD", eReal, self, "A->2D Heat of reaction, kJ/mol")

        self.rho    = daeParameter("&rho;",        eReal, self, "Density, kg/lit")
        self.cp     = daeParameter("c_p",          eReal, self, "Heat capacity of reactants, kJ/(kgK)")
        self.kw     = daeParameter("k_w",          eReal, self, "Heat transfer coefficient, kJ/(Khm2)")

        self.AR     = daeParameter("Q_r",          eReal, self, "Area of jacket cooling, m2")
        self.VR     = daeParameter("V_r",          eReal, self, "Reactor volume, lit")
        self.mK     = daeParameter("m_K",          eReal, self, "Mass of cooling, kg")
        self.CpK    = daeParameter("c_pk",         eReal, self, "Heat capacity of cooling, kJ/(kgK)")

        # Variables
        self.Ca = daeVariable("Ca",   typeNone, self, "Concentration of A, mol/lit")
        self.Cb = daeVariable("Cb",   typeNone, self, "Concentration of B, mol/lit")
        self.Cc = daeVariable("Cc",   typeNone, self, "Concentration of C, mol/lit")
        self.Cd = daeVariable("Cd",   typeNone, self, "Concentration of D, mol/lit")

        self.T  = daeVariable("T",    typeNone, self, "Temperature, C")
        self.Tk = daeVariable("T_k",  typeNone, self, "Cooling jacket temperature, C")

        self.k1 = daeVariable("k_1",  typeNone, self, "")
        self.k2 = daeVariable("k_2",  typeNone, self, "")
        self.k3 = daeVariable("k_3",  typeNone, self, "")

    def DeclareEquations(self):
        eq = self.CreateEquation("k1", "")
        eq.Residual = self.k1() - self.k10() * Exp(-self.E1() / (self.T() + 273.15))

        eq = self.CreateEquation("k2", "")
        eq.Residual = self.k2() - self.k20() * Exp(-self.E2() / (self.T() + 273.15))

        eq = self.CreateEquation("k3", "")
        eq.Residual = self.k3() - self.k30() * Exp(-self.E3() / (self.T() + 273.15))

        eq = self.CreateEquation("Ca", "")
        eq.Residual = self.VR() * self.Ca.dt() - ( -self.k1() * self.VR() * self.Ca() - self.k3() * self.VR() * self.Ca()**2 + self.F() * (self.Ca0() - self.Ca()) )

        eq = self.CreateEquation("Cb", "")
        eq.Residual = self.VR() * self.Cb.dt() - ( self.k1() * self.VR() * self.Ca() - self.k2() * self.VR() * self.Cb()- self.F() * self.Cb() )

        eq = self.CreateEquation("Cc", "")
        eq.Residual = self.VR() * self.Cc.dt() - ( self.k2() * self.VR() * self.Cb() - self.F() * self.Cc() )

        eq = self.CreateEquation("Cd", "")
        eq.Residual = self.VR() * self.Cd.dt() - ( self.k3() * self.VR() * self.Ca()**2 - self.F() * self.Cd() )

        # Energy balance - reactor
        eq = self.CreateEquation("EnergyBalanceReactor", "")
        eq.Residual = self.rho() * self.cp() * self.VR() * self.T.dt() - \
                        ( \
                            self.F() * self.rho() * self.cp() * (self.T0() - self.T()) - \
                            self.VR() * (
                                         self.k1() * self.Ca() * self.dHrAB() + \
                                         self.k2() * self.Cb() * self.dHrBC() + \
                                         self.k3() * Pow(self.Ca(), 2) * self.dHrAD() \
                                        ) + \
                            self.kw() * self.AR() * (self.Tk() - self.T()) \
                        )

        # Energy balance - cooling
        eq = self.CreateEquation("EnergyBalanceCooling", "")
        eq.Residual = self.mK() * self.CpK() * self.Tk.dt() - ( self.Qk() + self.kw() * self.AR() * (self.T() - self.Tk()) )

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("str_Van_de_Vusse")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.F.SetValue(14.19)
        self.m.Qk.SetValue(-1579.5)
        self.m.Ca0.SetValue(5.1)
        self.m.T0.SetValue(104.9)
        self.m.k10.SetValue(1.287e12)
        self.m.k20.SetValue(1.287e12)
        self.m.k30.SetValue(9.043e9)
        self.m.E1.SetValue(9758.3)
        self.m.E2.SetValue(9758.3)
        self.m.E3.SetValue(8560)
        self.m.dHrAB.SetValue(4.2)
        self.m.dHrBC.SetValue(-11)
        self.m.dHrAD.SetValue(-41.85)
        self.m.rho.SetValue(0.9342)
        self.m.cp.SetValue(3.01)
        self.m.kw.SetValue(4032)
        self.m.AR.SetValue(0.215)
        self.m.VR.SetValue(10.0)
        self.m.mK.SetValue(5)
        self.m.CpK.SetValue(2)

    def SetUpVariables(self):
        self.m.Ca.SetInitialCondition(2.2291)
        self.m.Cb.SetInitialCondition(1.0417)
        self.m.Cc.SetInitialCondition(0.91397)
        self.m.Cd.SetInitialCondition(0.91520)
        self.m.T.SetInitialCondition(79.591)
        self.m.Tk.SetInitialCondition(77.69)

def setOptions(nlpsolver):
    # 1) Set the options manually
    nlpsolver.SetOption('print_level', 5)
    nlpsolver.SetOption('tol', 1e-7)
    nlpsolver.SetOption('mu_strategy', 'adaptive')

    # Print options loaded at pyIPOPT startup and the user set options:
    nlpsolver.PrintOptions()
    nlpsolver.PrintUserOptions()

    # ClearOptions can clear all options:
    #nlpsolver.ClearOptions()

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    opt = daeOptimization()
    nlp = pyIPOPT.daeIPOPT()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 5
    sim.TimeHorizon       = 1000
    simulator = daeSimulator(app, simulation = sim,
                                  optimization = opt,
                                  nlpsolver = nlp,
                                  nlpsolver_setoptions_fn = setOptions)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    doOptimization = False

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    # Here the time is given in HOURS!
    simulation.ReportingInterval = 1./60 # 1 min
    simulation.TimeHorizon       = 6.    # 5 h

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    if doOptimization:
        nlpsolver    = pyIPOPT.daeIPOPT()
        optimization = daeOptimization()

        # Initialize the simulation
        optimization.Initialize(simulation, nlpsolver, daesolver, datareporter, log)

        # Achtung! Achtung! NLP solver options can be only set after optimization.Initialize()
        # Otherwise seg. fault occurs for some reasons.
        setOptions(nlpsolver)

        # Save the model report and the runtime model report
        simulation.m.SaveModelReport(simulation.m.Name + ".xml")
        simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

        # Run
        optimization.Run()
        optimization.Finalize()
    else:
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
