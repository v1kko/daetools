#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_che_7.py
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
********************************************************************************"""
__doc__ = """
Steady-state Plug Flow Reactor with energy balance and first order reaction:

.. code-block:: none

    A -> B

The problem is example 9.4.3 from the section 9.4 Nonisothermal Plug Flow Reactor
from the following book:

- Davis M.E., Davis R.J. (2003) Fundamentals of Chemical Reaction Engineering.
  McGraw Hill, New York, US. ISBN 007245007X.

The dimensionless concentration plot:

.. image:: _static/tutorial_che_7-results.png
   :width: 500px

The dimensionless temperature plot (adiabatic and nonisothermal cases):

.. image:: _static/tutorial_che_7-results2.png
   :width: 500px
"""

import sys, numpy
from daetools.pyDAE import *
from time import localtime, strftime
# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, kmol

K_t  = daeVariableType("k",  s**(-1), 0, 1E20, 0, 1e-8)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        # Domains
        self.z = daeDomain("z", self, m, "Axial domain")

        # Parameters
        self.T0    = daeParameter("T_0",     K,         self, "Inlet temperature")
        self.Ca0   = daeParameter("Ca0",     mol/m**3,  self, "Inlet concentration of A")
        self.u     = daeParameter("u",       m/s,       self, "Velocity")
        self.A     = daeParameter("A",       s**(-1),   self, "A->B pre-exponential factor")
        self.Ea    = daeParameter("Ea",      J/mol,     self, "A->B activation energy")
        self.dHra  = daeParameter("dHra",    J/mol,     self, "Heat of reaction A->B")
        self.rho   = daeParameter("&rho;",   kg/(m**3), self, "Density")
        self.cp    = daeParameter("c_p",     J/(kg*K),  self, "Heat capacity")
        self.Tj    = daeParameter("Tj",      K,         self, "Cooling jacket temperature")
        self.aj    = daeParameter("aj",      m**(-1),   self, "Cooling jacket specific surface area (area/volume)")
        self.U     = daeParameter("U",       W/(m**2*K),self, "Cooling jacket heat transfer coefficient")

        # Variables
        self.tau_r = daeVariable("tau_r",   time_t,                self, "Residence time")
        self.ka    = daeVariable("ka",      K_t,                   self, "Reaction A->B rate constant",      [self.z])
        self.Ca    = daeVariable("Ca",      molar_concentration_t, self, "Concentration of A",               [self.z])
        self.ra    = daeVariable("ra",      molar_reaction_rate_t, self, "Reaction A->B rate",               [self.z])
        self.T     = daeVariable("T",       temperature_t,         self, "Temperature in the reactor",       [self.z])
        self.xa    = daeVariable("xa",      fraction_t,            self, "Conversion of A",                  [self.z])
        self.ya    = daeVariable("ya",      no_t,                  self, "Dimensionless concentration of A", [self.z])
        self.theta = daeVariable("theta",   no_t,                  self, "Dimensionless temperature",        [self.z])

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        Rg   = Constant(8.314 * J/(mol*K))
        L    = Constant(self.z.UpperBound * self.z.Units) # Reactor Length
        u    = self.u()
        Ca0  = self.Ca0()
        T0   = self.T0()
        cp   = self.cp()
        rho  = self.rho()
        dHra = self.dHra()
        Tj   = self.Tj()
        U    = self.U()
        aj   = self.aj()
        Ea   = self.Ea()
        A    = self.A()

        # Variables
        tau_r = self.tau_r()
        # Define functions (lambdas) to make equations more readable
        ka      = lambda z: self.ka(z)
        ra      = lambda z: self.ra(z)
        xa      = lambda z: self.xa(z)
        dxa_dt  = lambda z: dt(self.xa(z))
        dxa_dz  = lambda z: d(self.xa(z), self.z)
        ya      = lambda z: self.ya(z)
        theta   = lambda z: self.theta(z)
        Ca      = lambda z: self.Ca(z)
        dCa_dt  = lambda z: dt(self.Ca(z))
        dCa_dz  = lambda z: d(self.Ca(z), self.z)
        d2Ca_dz2= lambda z: d2(self.Ca(z), self.z)
        T       = lambda z: self.T(z)
        dT_dt   = lambda z: dt(self.T(z))
        dT_dz   = lambda z: d(self.T(z), self.z)
        d2T_dz2 = lambda z: d2(self.T(z), self.z)

        # Reaction rate constants
        eq = self.CreateEquation("k1", "")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = ka(z) - A * Exp(-Ea/(Rg*T(z)))

        # Reaction rate
        eq = self.CreateEquation("ra", "Reaction A->B rate")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = ra(z) - ka(z)*Ca(z)

        # Conversion
        eq = self.CreateEquation("xa", "Conversion of A")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = xa(z)*Ca0 - (Ca0 - Ca(z))

        # Dimensionless concentration of A
        eq = self.CreateEquation("ya", "Dimensionless concentration of A")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = ya(z) * Ca0 - Ca(z)

        # Conversion
        eq = self.CreateEquation("theta", "Dimensionless temperature")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = theta(z) * T0 - T(z)

        # Mass balance for the reactor
        eq = self.CreateEquation("Ca", "Mass balance for the reactor")
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = u*dCa_dz(z) + ra(z)

        # Mass balance (step change inlet BCs)
        eq = self.CreateEquation("Ca_inlet", "Inlet boundary conditions for Ca")
        z = eq.DistributeOnDomain(self.z, eLowerBound)
        eq.Residual = Ca(z) - Ca0

        # Mass balance (outlet BCs)
        eq = self.CreateEquation("Ca_outlet", "Outlet boundary conditions for Ca")
        z = eq.DistributeOnDomain(self.z, eUpperBound)
        eq.Residual = dCa_dz(z)

        # Energy balance for the reactor
        eq = self.CreateEquation("T", "Energy balance for the reactor")
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = u*dT_dz(z) + ra(z)*dHra/(rho*cp) + U*aj*(T(z)-Tj)/(rho*cp)

        # Heat balance (step change inlet BCs)
        eq = self.CreateEquation("T_inlet", "Inlet boundary conditions for T")
        z = eq.DistributeOnDomain(self.z, eLowerBound)
        eq.Residual = T(z) - T0

        # Heat balance (outlet BCs)
        eq = self.CreateEquation("T_outlet", "Outlet boundary conditions for T")
        z = eq.DistributeOnDomain(self.z, eUpperBound)
        eq.Residual = dT_dz(z)

        # Residence time
        eq = self.CreateEquation("ResidenceTime", "Residence time of the reactor")
        eq.Residual = tau_r * u - L

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_7")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.z.CreateStructuredGrid(50, 0.0, 2.0)

        # Notate bene:
        #   Ea is 100 kJ/mol in the book but ka is 1E-7 then (too small).
        #   dHra is 1E4 kJ/kmol in the book but the heat effect is too small.
        #   U is 70 W/(m**2*K) in the book but the heat transferred is too low.
        # The values below reproduces the results from the book.
        # Definitely there are some typos in the values/units in the book.
        self.m.u.SetValue(3 * m/s)
        self.m.Ca0.SetValue(300 * mol/m**3)
        self.m.A.SetValue(5 * 1/s)
        self.m.Ea.SetValue(100 * J/mol) # Nota Bene
        self.m.T0.SetValue(700 * K)
        self.m.rho.SetValue(1200 * kg/m**3)
        self.m.cp.SetValue(1000 * J/(kg*K))
        self.m.dHra.SetValue(-1000 * kJ/mol) # Nota Bene
        self.m.Tj.SetValue(700 * K)
        self.m.aj.SetValue(4/0.2 * 1/m)
        self.m.U.SetValue(70000 * W/(m**2*K)) # Nota Bene

    def SetUpVariables(self):
        pass

# Use daeSimulator class
def guiRun(app):
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 1
    simulator = daeSimulator(app, simulation = simulation)
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
    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 1

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
