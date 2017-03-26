#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial19.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
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
This tutorial introduces the thermo physical property packages.

In this example we use the same model as in the tutorial4.

.. image:: _static/tutorial19-results.png
   :width: 500px
"""

import sys
from time import localtime, strftime
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W

def test_single_phase():
    # calcXXX functions perform actual calculations (return float values)
    # XXX functions create adouble with adThermoPhysicalPropertyPackageNode
    ctpp = daeCapeOpenThermoPhysicalPropertyPackage("ChemSepCapeOpenTPP", None, "")
    ctpp.LoadPackage("ChemSep Property Package Manager", # packageManager name (case sensitive)
                    "SMROG",                             # package name (case sensitive)
                    ["Hydrogen", "Carbon monoxide",
                     "Methane", "Carbon dioxide"],       # compound IDs in the mixture
                    [],                                  # compund CAS numbers (optional)
                    {"Vapor":eVapor})                    # dictionary {'phaseLabel' : stateOfAggregation}
                                                         # where stateOfAggregation can be:
                                                         # eVapor, eLiquid, eSolid or etppPhaseUnknown

    P = 1e5
    T = 300
    x = [0.7557, 0.04, 0.035, 0.1693]

    print("*****************************************************************")
    print("                         Single phase tests                      ")
    print("*****************************************************************")

    try:
        result = ctpp.calcPureCompoundConstantProperty("heatOfFusionAtNormalFreezingPoint", "Hydrogen")
        print("Test 1. heatOfFusionAtNormalFreezingPoint = %f J/mol" % result)
    except Exception as e:
        print("Test 1. Calculation failed")
        print(str(e))

    try:
        result = ctpp.calcPureCompoundTDProperty("idealGasEnthalpy", T, "Methane")
        print("Test 2. idealGasEnthalpy = %f J/mol" % result)
    except Exception as e:
        print("Test 2. Calculation failed")
        print(str(e))

    try:
        result = ctpp.calcPureCompoundPDProperty("boilingPointTemperature", P, "Hydrogen")
        print("Test 3. boilingPointTemperature = %f K" % result)
    except Exception as e:
        print("Test 3. Calculation failed")
        print(str(e))

    try:
        result = ctpp.calcSinglePhaseScalarProperty("density",  P, T, x, "Vapor", eMole)
        print("Test 4. density = %f mol/m^3" % result)
    except Exception as e:
        print("Test 3. Calculation failed")
        print(str(e))

    try:
        results = ctpp.calcSinglePhaseVectorProperty("fugacity",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 5. fugacity = %s" % results)
    except Exception as e:
        print("Test 5. Calculation failed")
        print(str(e))

def test_two_phase():
    # calcXXX functions perform actual calculations (return float values)
    ctpp = daeCapeOpenThermoPhysicalPropertyPackage("ChemSepCapeOpenTPP", None, "")
    ctpp.LoadPackage("ChemSep Property Package Manager",  # packageManager name (case sensitive)
                     "Water+Ethanol",                     # package name (case sensitive)
                     ["Water", "Ethanol"],                # compound IDs in the mixture
                     [],                                  # compund CAS numbers (optional)
                     {"Vapor"  : eVapor,
                      "Liquid" : eLiquid})                # dictionary {'phaseLabel' : stateOfAggregation}

    P1 = 1e5
    T1 = 300
    x1 = [0.60, 0.40]
    P2 = 1e5
    T2 = 300
    x2 = [0.60, 0.40]

    print("*****************************************************************")
    print("                         Two phase tests                         ")
    print("*****************************************************************")

    try:
        result = ctpp.calcTwoPhaseScalarProperty("surfaceTension",
                                                 P1, T1, x1, "Liquid",
                                                 P2, T2, x2, "Vapor",
                                                 eUndefinedBasis)
        print("Test 1. Water+Ethanol mixture surfaceTension = %f N/m" % result)
    except Exception as e:
        print("Test 1. Calculation failed")
        print(str(e))

    try:
        results = ctpp.calcTwoPhaseVectorProperty("kvalue",
                                                  P1, T1, x1, "Liquid",
                                                  P2, T2, x2, "Vapor",
                                                  eUndefinedBasis)
        print("Test 2. Water+Ethanol mixture kvalues = %s" % results)
    except Exception as e:
        print("Test 2. Calculation failed")
        print(str(e))

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        test_single_phase()
        test_two_phase()

        self.tpp = daeCapeOpenThermoPhysicalPropertyPackage("ChemSepCapeOpenTPP", self, "")
        self.tpp.LoadPackage("ChemSep Property Package Manager", # packageManager name (case sensitive)
                             "Water+Ethanol",                    # package name (case sensitive)
                             ["Water", "Ethanol"],               # compound IDs in the mixture
                             [],                                 # compund CAS numbers (optional)
                             {'Liquid':eLiquid})                 # dictionary {'phaseLabel' : stateOfAggregation}
                                                                 # where stateOfAggregation can be:
                                                                 # eVapor, eLiquid, eSolid or etppPhaseUnknown

        self.m    = daeParameter("m",       kg,           self, "Mass of water")

        self.cp   = daeVariable("c_p",   specific_heat_capacity_t, self, "Specific heat capacity of the liquid")
        self.Qin  = daeVariable("Q_in",  power_t,                  self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t,            self, "Temperature of the liquid")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("C_p", "Equation to calculate the specific heat capacity of water as a function of the temperature.")
        tpp_cp = self.tpp.SinglePhaseScalarProperty("heatCapacityCp", 1e5, self.T(), [0.60, 0.40], 'Liquid', eMole)
        # Calculating molecularWeight does not work for some reasons
        #tpp_MW = self.tpp.calcSinglePhaseScalarProperty("molecularWeight", 1e5, 283, [0.60, 0.40], 'Liquid', eMole)
        eq.Residual = self.cp() - tpp_cp/0.018 # TPP package returns cp in J/(mol.K)
        eq.CheckUnitsConsistency = False

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.BuildJacobianExpressions = True
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial19")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.m.SetValue(1 * kg)

    def SetUpVariables(self):
        self.m.Qin.AssignValue(500 * W)
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
    #simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    #simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
