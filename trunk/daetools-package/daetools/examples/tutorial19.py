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

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # calcXXX functions perform actual calculations (return float values)
        # XXX functions create adouble with adThermoPhysicalPropertyPackageNode
        #self.tpp = daeCapeOpenThermoPhysicalPropertyPackage("ChemSepCapeOpenTPP", self, "")
        #self.tpp.LoadPackage("ChemSep Property Package Manager", "SMROG", ["Hydrogen", "Carbon monoxide", "Methane", "Carbon dioxide"])
        #P = 1e5
        #T = 300
        #x = [0.7557, 0.04, 0.035, 0.1693]
        #density = self.tpp.calcSinglePhaseScalarProperty(daeeThermoPhysicalProperty.density, P, T, x, eVapor, eMole)
        #print("density = %f" % density)
        #density = self.tpp.SinglePhaseScalarProperty(daeeThermoPhysicalProperty.density, P, T, x, eVapor, eMole)
        #print("density = %s" % density.NodeAsLatex())

        self.tpp = daeCapeOpenThermoPhysicalPropertyPackage("ChemSepCapeOpenTPP", self, "")
        self.tpp.LoadPackage("ChemSep Property Package Manager", "Water+Ethanol", ["Water", "Ethanol"])

        self.m    = daeParameter("m",       kg,           self, "Mass of water")

        self.cp   = daeVariable("c_p",   specific_heat_capacity_t, self, "Specific heat capacity of water")
        self.Qin  = daeVariable("Q_in",  power_t,                  self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t,            self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("C_p", "Equation to calculate the specific heat capacity of water as a function of the temperature.")
        tpp_cp = self.tpp.SinglePhaseScalarProperty(daeeThermoPhysicalProperty.heatCapacityCp, 1e5, self.T(), [0.60, 0.40], eLiquid, eMole)
        # Calculating molecularWeight does not work for some reasons
        #tpp_MW = self.tpp.calcSinglePhaseScalarProperty(daeeThermoPhysicalProperty.molecularWeight, 1e5, 283, [0.60, 0.40], eLiquid, eMole)
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

    #import pprint
    #for eq in simulation.m.Equations:
    #    print(eq.CanonicalName, ':')
    #    for eei in eq.EquationExecutionInfos:
    #        print('    %s:' % eei.Name)
    #        # dictionary {overall_index : (block_index,derivative_node)}
    #        for oi, (bi,node) in eei.JacobianExpressions.items():
    #            print('        %d : %s' % (bi, node))
            
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
