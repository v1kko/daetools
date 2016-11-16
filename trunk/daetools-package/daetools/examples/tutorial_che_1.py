#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_che_1.py
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
__doc__ = """
Continuously Stirred Tank Reactor with energy balance and Van de Vusse reactions:

.. code-block:: none

    A -> B -> C
   2A -> D

Reference: G.A. Ridlehoover, R.C. Seagrave. Optimization of Van de Vusse Reaction Kinetics
Using Semibatch Reactor Operation, Ind. Eng. Chem. Fundamen. 1973;12(4):444-447.
`doi:10.1021/i160048a700 <https://doi.org//10.1021/i160048a700>`_

The concentrations plot:

.. image:: _static/tutorial_che_1-results.png
   :width: 500px

The temperatures plot:

.. image:: _static/tutorial_che_1-results2.png
   :width: 500px
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime
# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l

K_t  = daeVariableType("k",  s**(-1),        0, 1E20,   0, 1e-5)
K2_t = daeVariableType("k2", m**3/(mol*s),   0, 1E20,   0, 1e-5)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        # Parameters
        self.F     = daeParameter("F",           (m**3)/s, self, "Feed flowrate")
        self.Qk    = daeParameter("Q_k",              J/s, self, "Jacket cooling rate")
        self.Ca0   = daeParameter("Ca_0",      mol/(m**3), self, "Inlet feed concentration")
        self.T0    = daeParameter("T_0",                K, self, "Inlet feed temperature")
        self.k10   = daeParameter("k_10",         s**(-1), self, "A->B pre-exponential factor")
        self.k20   = daeParameter("k_20",         s**(-1), self, "B->C pre-exponential factor")
        self.k30   = daeParameter("k_30",    m**3/(mol*s), self, "2A->D pre-exponential factor")
        self.E1    = daeParameter("E_1",                K, self, "A->B activation energy")
        self.E2    = daeParameter("E_2",                K, self, "B->C activation energy")
        self.E3    = daeParameter("E_3",                K, self, "2A->D activation energy")
        self.dHr1  = daeParameter("dHr1",           J/mol, self, "A->B heat of reaction")
        self.dHr2  = daeParameter("dHr2",           J/mol, self, "B->C heat of reaction")
        self.dHr3  = daeParameter("dHr3",           J/mol, self, "2A->D heat of reaction")
        self.rho   = daeParameter("&rho;",      kg/(m**3), self, "Density")
        self.cp    = daeParameter("c_p",         J/(kg*K), self, "Heat capacity of reactants")
        self.kw    = daeParameter("k_w",   J/(K*s*(m**2)), self, "Heat transfer coefficient")
        self.AR    = daeParameter("A_r",             m**2, self, "Area of jacket cooling")
        self.VR    = daeParameter("V_r",             m**3, self, "Reactor volume")
        self.mK    = daeParameter("m_K",               kg, self, "Mass of the cooling fluid")
        self.cpK   = daeParameter("c_pk",        J/(kg*K), self, "Heat capacity of the cooling fluid")

        # Variables
        self.Ca = daeVariable("Ca",   molar_concentration_t, self, "Concentration of A")
        self.Cb = daeVariable("Cb",   molar_concentration_t, self, "Concentration of B")
        self.Cc = daeVariable("Cc",   molar_concentration_t, self, "Concentration of C")
        self.Cd = daeVariable("Cd",   molar_concentration_t, self, "Concentration of D")

        self.T  = daeVariable("T",    temperature_t, self, "Temperature in the reactor")
        self.Tk = daeVariable("T_k",  temperature_t, self, "Temperature of the cooling jacket")

        self.k1 = daeVariable("k_1",  K_t,  self, "Reaction A->B rate constant")
        self.k2 = daeVariable("k_2",  K_t,  self, "Reaction B->C rate constant")
        self.k3 = daeVariable("k_3",  K2_t, self, "Reaction 2A->D rate constant")

    def DeclareEquations(self):
        # Create adouble objects to make equations more readable
        rho = self.rho()
        cp  = self.cp()
        kw  = self.kw()
        AR  = self.AR()
        mK  = self.mK()
        cpK = self.cpK()
        Qk  = self.Qk()
        dHr1 = self.dHr1()
        dHr2 = self.dHr2()
        dHr3 = self.dHr3()
        k10 = self.k10()
        k20 = self.k20()
        k30 = self.k30()
        E1  = self.E1()
        E2  = self.E2()
        E3  = self.E3()
        F   = self.F()
        VR  = self.VR()
        T0  = self.T0()
        Ca0 = self.Ca0()
        # Variables
        k1  = self.k1()
        k2  = self.k2()
        k3  = self.k3()
        T   = self.T()
        Tk  = self.Tk()
        Ca  = self.Ca()
        Cb  = self.Cb()
        Cc  = self.Cc()
        Cd  = self.Cd()
        # Derivatives
        dCa_dt = self.Ca.dt()
        dCb_dt = self.Cb.dt()
        dCc_dt = self.Cc.dt()
        dCd_dt = self.Cd.dt()
        dT_dt  = self.T.dt()
        dTk_dt = self.Tk.dt()

        # Intermediates
        r1 = k1 * VR * Ca
        r2 = k2 * VR * Cb
        r3 = k3 * VR * (Ca**2)

        ra = -r1 - 2*r3 + F*(Ca0-Ca)
        rb =  r1   - r2 - F*Cb
        rc =  r2        - F*Cc
        rd =  r3        - F*Cd

        # Reaction rate constants
        eq = self.CreateEquation("k1", "")
        eq.Residual = k1 - k10 * Exp(-E1 / T)

        eq = self.CreateEquation("k2", "")
        eq.Residual = k2 - k20 * Exp(-E2 / T)

        eq = self.CreateEquation("k3", "")
        eq.Residual = k3 - k30 * Exp(-E3 / T)

        # Mass balance
        eq = self.CreateEquation("Ca", "")
        eq.Residual = VR * dCa_dt - ra

        eq = self.CreateEquation("Cb", "")
        eq.Residual = VR * dCb_dt - rb

        eq = self.CreateEquation("Cc", "")
        eq.Residual = VR * dCc_dt - rc

        eq = self.CreateEquation("Cd", "")
        eq.Residual = VR * dCd_dt - rd

        # Energy balance - reactor
        eq = self.CreateEquation("EnergyBalanceReactor", "")
        eq.Residual = rho * cp * VR * dT_dt - (  F * rho * cp * (T0 - T) \
                                               - r1 * dHr1 \
                                               - r2 * dHr2 \
                                               - r3 * dHr3 \
                                               + kw * AR * (Tk - T) )

        # Energy balance - cooling fluid
        eq = self.CreateEquation("EnergyBalanceCooling", "")
        eq.Residual = mK * cpK * dTk_dt - (Qk + kw * AR * (T - Tk))

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_1")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        # Nota bene:
        #   All values are in the units from the original model
        self.m.F.SetValue(14.19 * l/hour)
        self.m.Qk.SetValue(-1579.5 * kJ/hour)
        self.m.Ca0.SetValue(5.1 * mol/l)
        self.m.T0.SetValue((273.15 + 104.9) * K)
        self.m.k10.SetValue(1.287e10 * 1/hour)
        self.m.k20.SetValue(1.287e10 * 1/hour)
        self.m.k30.SetValue(9.043e9 * l/(mol*hour))
        self.m.E1.SetValue(9758.3 * K)
        self.m.E2.SetValue(9758.3 * K)
        self.m.E3.SetValue(8560 * K)
        self.m.dHr1.SetValue(4.2 * kJ/mol)
        self.m.dHr2.SetValue(-11 * kJ/mol)
        self.m.dHr3.SetValue(-41.85 * kJ/mol)
        self.m.rho.SetValue(0.9342 * kg/l)
        self.m.cp.SetValue(3.01 * kJ/(kg*K))
        self.m.kw.SetValue(4032 * kJ/(K*hour*(m**2)))
        self.m.AR.SetValue(0.215 * m**2)
        self.m.VR.SetValue(10.0 * l)
        self.m.mK.SetValue(5 * kg)
        self.m.cpK.SetValue(2 * kJ/(kg*K))

    def SetUpVariables(self):
        # Nota bene:
        #   All values are in the units from the original model
        self.m.Ca.SetInitialCondition(2.2291 * mol/l)
        self.m.Cb.SetInitialCondition(1.0417 * mol/l)
        self.m.Cc.SetInitialCondition(0.91397 * mol/l)
        self.m.Cd.SetInitialCondition(0.91520 * mol/l)
        self.m.T.SetInitialCondition((273.15 + 79.591) * K)
        self.m.Tk.SetInitialCondition((273.15 + 77.69) * K)

# Use daeSimulator class
def guiRun(app):
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 600     # 10 min
    simulation.TimeHorizon       = 3*60*60 # 3 h
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
    # Here the time is given in HOURS!
    simulation.ReportingInterval = 600     # 10 min
    simulation.TimeHorizon       = 3*60*60 # 3 h

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
