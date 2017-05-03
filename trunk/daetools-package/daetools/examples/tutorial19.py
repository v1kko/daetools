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

Since there are many thermo packages with a very different API the CapeOpen
standard has been adopted in daetools. This way, all thermo packages implementing
the CapeOpen thermo interfaces are automatically vailable to daetools. Those which
do not are wrapped by the class with the CapeOpen conforming API.
At the moment, two types of thermophysical property packages are implemented:
    
- Any CapeOpen v1.1 thermo package (available only in Windows)
- CoolProp thermo package (available for all platforms) wrapped in
  the class with the CapeOpen interface.
  
The central point is the daeThermoPhysicalPropertyPackage class. It can load any
COM component that implements CapeOpen 1.1 ICapeThermoPropertyPackageManager interface
or the CoolProp thermo package.

The framework provides low-level functions (specified in the CapeOpen standard) in the
daeThermoPhysicalPropertyPackage class and the higher-level functions in the auxiliary
daeThermoPackage class defined in the daetools/pyDAE/thermo_packages.py file.
The low-level functions are defined in the ICapeThermoCoumpounds and ICapeThermoPropertyRoutine
CapeOpen interfaces. These functions come in two flavours:
    
(a) The ordinary functions return adouble/adouble_array objects and can only be used to specify equations:
    
    - GetCompoundConstant (from ICapeThermoCoumpounds interface)
    - GetTDependentProperty (from ICapeThermoCoumpounds interface)
    - GetPDependentProperty (from ICapeThermoCoumpounds interface)
    - CalcSinglePhaseScalarProperty (from ICapeThermoPropertyRoutine interface: scalar version)
    - CalcSinglePhaseVectorProperty (from ICapeThermoPropertyRoutine interface: vector version)
    - CalcTwoPhaseScalarProperty (from ICapeThermoPropertyRoutine interface: scalar version)
    - CalcTwoPhaseVectorProperty (from ICapeThermoPropertyRoutine interface: vector version)
    
(b) The functions starting with the underscores can be used for calculations (they use and return float values):
    
    - _GetCompoundConstant
    - _GetTDependentProperty
    - _GetPDependentProperty
    - _CalcSinglePhaseScalarProperty
    - _CalcSinglePhaseVectorProperty
    - _CalcTwoPhaseScalarProperty
    - _CalcTwoPhaseVectorProperty

The daeThermoPackage auxiliary class offers functions to calculate specified properties, for instance:
    
- Transport properties:
      
  - cp, kappa, mu, Dab (heat capacity, thermal conductivity, dynamic viscosity, diffusion coefficient)
    
- Thermodynamic properties:
      
  - rho
  - h, s, G, H, I (enthalpy, entropy, gibbs/helmholtz/internal energy)
  - h_E, s_E, G_E, H_E, I_E, V_E (excess enthalpy, entropy, gibbs/helmholtz/internal energy, volume)
  - f and phi (fugacity and coefficient of fugacity)
  - a and gamma (activity and the coefficient of activity)
  - z (compressibility factor)
  - K, surfaceTension (ratio of fugacity coefficients and the surface tension)

Nota bene:
  Some of the above functions return scalars while the others return arrays of values.
  Check the thermo_packages.py file for details.

All functions return properties in the SI units (as specified in the CapeOpen 1.1 standard).

Known issues:
    
- Many properties from the CapeOpen standard are not supported by all thermo packages.
- CalcEquilibrium from the ICapeThermoEquilibriumRoutine is not supported.
- CoolProp does not provide transport models for many compounds.
- The function calls are NOT thread safe.
- The code generation will NOT work for models using the thermo packages.
- Some CapeOpen thermo packags refuse to return properties for mass basis (i.e. density).

In this tutorial, we use a very simple model: a quantity of liquid (water + ethanol mixture) 
is heated using the constant input power. The model uses a thermo package to calculate the commonly
used transport properties such as specific heat capacity, thermal conductivity, dynamic viscosity
and binary diffusion coefficients. First, the low-level functions are tested for CapeOpen and 
CoolProp packages in the test_single_phase, test_two_phase, test_coolprop_single_phase functions. 
The results depend on the options selected in the CapeOpen package (equation of state, etc.).
Then, the model that uses a thermo package is simulated.

The plot of the specific heat capacity as a function of temperature:
    
.. image:: _static/tutorial19-results.png
   :width: 500px
   
Nota bene:
  There is a difference between results in Windows and other platforms since 
  the CapeOpen thermo packages are available only in Windows.
"""

import sys
from time import localtime, strftime
import daetools
from daetools.pyDAE import *

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, Pa, mol, J, W

def test_single_phase():
    """
    Test the low-level functions offered by the CapeOpen standard (functions starting with underscores):
      - _GetCompoundConstant
      - _GetTDependentProperty
      - _GetPDependentProperty
      - _CalcSinglePhaseScalarProperty
      - _CalcSinglePhaseVectorProperty
    Here we use the ChemSep property package.
    In general, any package implementing thermo 1.1 standard can be used.
    The package "H2+CH4+CO2" must be created and configured using the property package manager.
    It contains three components (H2, CH4 and CO2).
    """
    ctpp = daeThermoPackage("ChemSepCapeOpenTPP", None, "")
    ctpp.LoadCapeOpen("ChemSep Property Package Manager",  # packageManager name (case sensitive)
                      "H2+CH4+CO2",                        # package name (case sensitive)
                      ["Hydrogen", "Methane",
                       "Carbon dioxide"],                  # compound IDs in the mixture
                      [],                                  # compund CAS numbers (optional)
                      {"Vapor":eVapor},                    # dictionary {'phaseLabel' : stateOfAggregation}
                                                           #   where stateOfAggregation can be:
                                                           #   eVapor, eLiquid, eSolid or etppPhaseUnknown
                      eMole,                               # optional: default basis (eMole  by default)
                      {})                                  # optional: options dictionary (empty by default)

    P = 2e5
    T = 300
    x = [0.755, 0.075, 0.170]

    print("*****************************************************************")
    print("                 Cape-Open single phase tests                    ")
    print("*****************************************************************")

    # Pure compound properties from the ICapeThermoCompounds interface.
    try:
        result = ctpp._GetCompoundConstant("heatOfFusionAtNormalFreezingPoint", "Hydrogen")
        print("Test 1. heatOfFusionAtNormalFreezingPoint = %f J/mol" % result)
    except Exception as e:
        print("Test 1. Calculation failed")
        print(str(e))

    try:
        result = ctpp._GetTDependentProperty("idealGasEnthalpy", T, "Methane")
        print("Test 2. idealGasEnthalpy = %f J/mol" % result)
    except Exception as e:
        print("Test 2. Calculation failed")
        print(str(e))

    try:
        result = ctpp._GetPDependentProperty("boilingPointTemperature", P, "Hydrogen")
        print("Test 3. boilingPointTemperature = %f K" % result)
    except Exception as e:
        print("Test 3. Calculation failed")
        print(str(e))

    # Mixture properties from the ICapeThermoPropertyRoutine interface
    try:
        result = ctpp._CalcSinglePhaseScalarProperty("density",  P, T, x, "Vapor", eMole)
        print("Test 4. density = %f mol/m^3" % result)
    except Exception as e:
        print("Test 3. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("heatCapacityCp",  P, T, x, "Vapor", eMole)
        print("Test 5. heatCapacityCp = %s J/(mol.K)" % results)
    except Exception as e:
        print("Test 5. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("thermalConductivity",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 6. thermalConductivity = %s W/(m.K)" % results)
    except Exception as e:
        print("Test 6. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("viscosity",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 7. viscosity = %s Pa.s" % results)
    except Exception as e:
        print("Test 7. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseVectorProperty("diffusionCoefficient",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 8. diffusionCoefficient = %s m^2/s" % results)
    except Exception as e:
        print("Test 8. Calculation failed")
        print(str(e))

def test_two_phase():
    """
    Test the two phase low-level functions.
      - _CalcTwoPhaseScalarProperty
      - _CalcTwoPhaseVectorProperty
    Here we use the ChemSep property package again.
    The package "Water+Ethanol" must be created and configured using the property package manager.
    It contains two components (water and ethanol).
    """
    ctpp = daeThermoPackage("ChemSepCapeOpenTPP", None, "")
    ctpp.LoadCapeOpen("ChemSep Property Package Manager",  # packageManager name (case sensitive)
                      "Water+Ethanol",                     # package name (case sensitive)
                      ["Water", "Ethanol"],                # compound IDs in the mixture
                      [],                                  # compund CAS numbers (optional)
                      {"Vapor"  : eVapor,
                       "Liquid" : eLiquid},                # dictionary {'phaseLabel' : stateOfAggregation}
                      eMole,                               # optional: default basis (eMole by default)
                      {})                                  # optional: options dictionary (empty by default)
    P1 = 1e5
    T1 = 300
    x1 = [0.60, 0.40]
    P2 = 1e5
    T2 = 300
    x2 = [0.60, 0.40]

    print("*****************************************************************")
    print("                   Cape-Open two phase tests                     ")
    print("*****************************************************************")

    # Two-pahase mixture properties from the ICapeThermoPropertyRoutine interface
    try:
        result = ctpp._CalcTwoPhaseScalarProperty("surfaceTension",
                                                 P1, T1, x1, "Liquid",
                                                 P2, T2, x2, "Vapor",
                                                 eUndefinedBasis)
        print("Test 1. Water+Ethanol mixture surfaceTension = %f N/m" % result)
    except Exception as e:
        print("Test 1. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcTwoPhaseVectorProperty("kvalue",
                                                  P1, T1, x1, "Liquid",
                                                  P2, T2, x2, "Vapor",
                                                  eUndefinedBasis)
        print("Test 2. Water+Ethanol mixture kvalues = %s" % results)
    except Exception as e:
        print("Test 2. Calculation failed")
        print(str(e))

def test_coolprop_single_phase():
    """
    Test the CoolProp library.
    Default backend is HEOS and the default reference state is DEF.
    Different backends/ref.states can be specified in the options dictionary.
    Nota bene:
      CoolProp does not have transport properties models defined for some compounds (i.e. CarbonMonoxide).
    """
    ctpp = daeThermoPackage("CoolPropTPP", None, "")
    ctpp.LoadCoolProp(["Hydrogen", "Methane",
                       "CarbonDioxide"],                # compound IDs in the mixture
                       [],                              # unused (compund CAS numbers)
                       {"Vapor"  : eVapor},             # dictionary {'phaseLabel' : stateOfAggregation}
                       eMole,                           # optional: default basis (eMole by default)
                       {"backend"        : "HEOS",
                        "referenceState" : "DEF",
                        "debugLevel"     : "0"})        # optional: options dictionary (empty by default)


    P = 2.0e5 # Must be in Pascals
    T = 300   # Must be in Kelvins
    x = [0.755, 0.075, 0.170]

    print("*****************************************************************")
    print("                  CoolProp single phase tests                    ")
    print("*****************************************************************")

    # CoolProp does not implement functions for pure compound properties from the ICapeThermoCompounds interface.
    # Transport mixture properties from the ICapeThermoPropertyRoutine interface
    try:
        result = ctpp._CalcSinglePhaseScalarProperty("density",  P, T, x, "Vapor", eMole)
        print("Test 1. density = %f mol/m^3" % result)
    except Exception as e:
        print("Test 1. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("heatCapacityCp",  P, T, x, "Vapor", eMole)
        print("Test 2. heatCapacityCp = %s J/(mol.K)" % results)
    except Exception as e:
        print("Test 2. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("thermalConductivity",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 3. thermalConductivity = %s W/(m.K)" % results)
    except Exception as e:
        print("Test 3. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseScalarProperty("viscosity",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 4. viscosity = %s Pa.s" % results)
    except Exception as e:
        print("Test 4. Calculation failed")
        print(str(e))

    try:
        results = ctpp._CalcSinglePhaseVectorProperty("diffusionCoefficient",  P, T, x, "Vapor", eUndefinedBasis)
        print("Test 5. diffusionCoefficient = %s m^2/s" % results)
    except Exception as e:
        print("Test 5. Calculation failed")
        print(str(e))

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.tpp = daeThermoPackage("TPP", self, "")
        # CapeOpen thermo packages are available only in Windows.
        # For other platforms use CoolProp thermo package.
        if daetools.daetools_system == 'Windows':
            self.tpp.LoadCapeOpen("ChemSep Property Package Manager", # packageManager name (case sensitive)
                                  "Water+Ethanol",                    # package name (case sensitive)
                                  ["Water", "Ethanol"],               # compound IDs in the mixture
                                  [],                                 # compund CAS numbers (optional)
                                  {'Liquid':eLiquid},                 # dictionary {'phaseLabel' : stateOfAggregation}
                                  eMole,                              # default basis is eMole (other options are eMass or eUndefinedBasis)
                                  {})                                 # options dictionary (defaut is empty)
        else:
            self.tpp.LoadCoolProp(["Water", "Ethanol"],               # compound IDs in the mixture
                                  [],                                 # compund CAS numbers (optional)
                                  {'Liquid':eLiquid},                 # dictionary {'phaseLabel' : stateOfAggregation}
                                  eMole,                              # default basis is eMole (other options are eMass or eUndefinedBasis)
                                  {})                                 # options dictionary (defaut is empty)

        self.m    = daeParameter("m", kg, self, "Mass of water")

        self.cp    = daeVariable("c_p",   specific_heat_capacity_t, self, "Specific heat capacity of the liquid")
        self.kappa = daeVariable("k",     thermal_conductivity_t,   self, "Thermal conductivity of the liquid")
        self.mu    = daeVariable("mu",    dynamic_viscosity_t,      self, "Viscosity of the liquid")
        self.Qin   = daeVariable("Q_in",  power_t,                  self, "Power of the heater")
        self.T     = daeVariable("T",     temperature_t,            self, "Temperature of the liquid")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # P, T, x arguments can be floats or adouble objects/arrays as illustrated here.
        P = 1e5 * Pa
        T = self.T()
        x = [0.60, 0.40]

        # daeThermoPackage uses the default basis to calculate the properties.
        # If a property for a different basis is required it should be specified as a keyword argument 'basis'=eMole|eMass|eUndefinedBasis.
        # Also, it needs the phase information. If there is only one phase it will be used as a default.
        # Otherwise, it must be specified as a keyword argument 'phase'='phaseLabel'.
        # For instance:
        #    self.tpp.kappa(P,T,x, phase='Vapor', basis=eUndefinedBasis)

        # Calculate transport properties:
        # 1. Specific heat capacity (cp) in J/(kg*K)
        #    TPP package returns cp in J/(mol.K) (mole basis) - for some reasons it refuses to calculate it for the mass basis.
        #    Therefore, divide it by the molar mass of the mixture.
        eq = self.CreateEquation("C_p", "Specific heat capacity as a function of the temperature.")
        eq.Residual = self.cp() - self.tpp.cp(P,T,x) / self.tpp.M(P,T,x)

        # 2. Thermal conductivity (kappa) in W/(m*K)
        eq = self.CreateEquation("kappa", "Thermal conductivity as a function of the temperature.")
        eq.Residual = self.kappa() - self.tpp.kappa(P,T,x)

        # 3. Dynamic viscosity (mu) in Pa*s
        eq = self.CreateEquation("mu", "Viscosity as a function of the temperature.")
        eq.Residual = self.mu() - self.tpp.mu(P,T,x)

        # Simple integral heat balance for the liquid
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
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
    # Test low-level calculation routines.
    # CapeOpen thermo packages are available only on Windows.
    if daetools.daetools_system == 'Windows':
        test_single_phase()
        test_two_phase()
    test_coolprop_single_phase()

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
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
