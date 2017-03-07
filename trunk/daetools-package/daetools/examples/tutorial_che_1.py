#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_che_1.py
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
Continuously Stirred Tank Reactor with energy balance and Van de Vusse reactions:

.. code-block:: none

    A -> B -> C
   2A -> D

Reference: G.A. Ridlehoover, R.C. Seagrave. Optimization of Van de Vusse Reaction Kinetics
Using Semibatch Reactor Operation, Ind. Eng. Chem. Fundamen. 1973;12(4):444-447.
`doi:10.1021/i160048a700 <https://doi.org/10.1021/i160048a700>`_

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

K_t  = daeVariableType("K_t",  s**(-1),        0, 1E20,   0, 1e-5)
K2_t = daeVariableType("K2_t", m**3/(mol*s),   0, 1E20,   0, 1e-5)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        # Parameters
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
        self.mK    = daeParameter("m_K",               kg, self, "Mass of the cooling fluid")
        self.cpK   = daeParameter("c_pk",        J/(kg*K), self, "Heat capacity of the cooling fluid")

        # Degrees of freedom (for optimisation)
        self.VR    = daeVariable("V_r",  volume_t,              self, "Reactor volume")
        self.Qk    = daeVariable("Q_k",  power_t,               self, "Jacket cooling rate")
        self.Qf    = daeVariable("Qf",   volume_flowrate_t,     self, "Feed flowrate")
        self.Caf   = daeVariable("Caf",  molar_concentration_t, self, "Feed concentration")
        self.Tf    = daeVariable("Tf",   temperature_t,         self, "Feed temperature")

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

        self.V0      = daeVariable("V0",        volume_t,   self, "Minimum reactor volume")
        self.Vm      = daeVariable("Vm",        volume_t,   self, "Maximum reactor volume")
        self.V0_star = daeVariable("V0_star",   no_t,       self, "Dimensionless minimum volume")
        self.Ca_star = daeVariable("Ca_star",   no_t,       self, "Dimensionless concentration of A")
        self.Cb_star = daeVariable("Cb_star",   no_t,       self, "Dimensionless concentration of B")
        self.P1      = daeVariable("P1",        no_t,       self, "Dimensionless A->B reaction rate constant")
        self.P2      = daeVariable("P2",        no_t,       self, "Dimensionless B->C reaction rate constant")
        self.P3      = daeVariable("P3",        no_t,       self, "Dimensionless 2A->D reaction rate constant")
        self.theta   = daeVariable("theta",     no_t,       self, "Dimensionless time")
        #self.tau_ref = daeVariable("tau_ref",   time_t,     self, "Mean residence time of the reference reactor")
        self.sigma_f = daeVariable("sigma_f",   no_t,       self, "The proportion of the semibatch cycle occupied by filling")
        self.sigma_b = daeVariable("sigma_b",   no_t,       self, "The proportion of the semibatch cycle occupied by batch")
        self.sigma_e = daeVariable("sigma_e",   no_t,       self, "The proportion of the semibatch cycle occupied by emptying")
        self.time_f  = daeVariable("time_f",    time_t,     self, "The time duration of filling phase")
        self.time_b  = daeVariable("time_b",    time_t,     self, "The time duration of batch phase")
        self.time_e  = daeVariable("time_e",    time_t,     self, "The time duration of emptying phase")

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
        Qf   = self.Qf()
        VR  = self.VR()
        Tf  = self.Tf()
        Caf = self.Caf()
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
        V0      = self.V0()
        Vm      = self.Vm()
        V0_star = self.V0_star()
        Ca_star = self.Ca_star()
        Cb_star = self.Cb_star()
        P1      = self.P1()
        P2      = self.P2()
        P3      = self.P3()
        theta   = self.theta()
        #tau_ref = self.tau_ref()
        sigma_f = self.sigma_f()
        sigma_b = self.sigma_b()
        sigma_e = self.sigma_e()
        time_f  = self.time_f()
        time_b  = self.time_b()
        time_e  = self.time_e()
        # Derivatives
        dVr_dt   = dt(self.VR())
        dVrCa_dt = dt(self.VR() * self.Ca())
        dVrCb_dt = dt(self.VR() * self.Cb())
        dVrCc_dt = dt(self.VR() * self.Cc())
        dVrCd_dt = dt(self.VR() * self.Cd())
        dVrT_dt  = dt(self.VR() * self.T())
        dTk_dt   = dt(self.Tk())

        # Intermediates
        r1 = k1 * VR * Ca
        r2 = k2 * VR * Cb
        r3 = k3 * VR * (Ca**2)

        ra = -r1 - 2*r3 + Qf*(Caf-Ca)
        rb =  r1   - r2 - Qf*Cb
        rc =  r2        - Qf*Cc
        rd =  r3        - Qf*Cd

        # Volume (constant in this case)
        eq = self.CreateEquation("k1", "")
        eq.Residual = dVr_dt #- Qf

        # Reaction rate constants
        eq = self.CreateEquation("k1", "")
        eq.Residual = k1 - k10 * Exp(-E1 / T)

        eq = self.CreateEquation("k2", "")
        eq.Residual = k2 - k20 * Exp(-E2 / T)

        eq = self.CreateEquation("k3", "")
        eq.Residual = k3 - k30 * Exp(-E3 / T)

        # Mass balance
        eq = self.CreateEquation("Ca", "")
        eq.Residual = dVrCa_dt - ra

        eq = self.CreateEquation("Cb", "")
        eq.Residual = dVrCb_dt - rb

        eq = self.CreateEquation("Cc", "")
        eq.Residual = dVrCc_dt - rc

        eq = self.CreateEquation("Cd", "")
        eq.Residual = dVrCd_dt - rd

        # Energy balance - reactor
        eq = self.CreateEquation("EnergyBalanceReactor", "")
        eq.Residual = rho * cp * dVrT_dt - (  Qf * rho * cp * (Tf - T) \
                                            - r1 * dHr1 \
                                            - r2 * dHr2 \
                                            - r3 * dHr3 \
                                            + kw * AR * (Tk - T)
                                           )

        # Energy balance - cooling fluid
        eq = self.CreateEquation("EnergyBalanceCooling", "")
        eq.Residual = mK * cpK * dTk_dt - (Qk + kw * AR * (T - Tk))

        # V0*
        eq = self.CreateEquation("V0_star", "")
        eq.Residual = V0_star - V0/Vm
        # Ca*
        eq = self.CreateEquation("Ca_star", "")
        eq.Residual = Ca_star * Caf - Ca
        # Cb*
        eq = self.CreateEquation("Cb_star", "")
        eq.Residual = Cb_star * Caf - Cb
        # P1
        eq = self.CreateEquation("P1", "")
        eq.Residual = P1 * Qf - k1*Vm
        eq.CheckUnitsConsistency = False
        # P2
        eq = self.CreateEquation("P2", "")
        eq.Residual = P2 * Qf - k2*Vm
        eq.CheckUnitsConsistency = False
        # P3
        eq = self.CreateEquation("P3", "")
        eq.Residual = P3 * Qf - k3*Vm
        eq.CheckUnitsConsistency = False
        # theta
        eq = self.CreateEquation("theta", "")
        eq.Residual = theta - Time() * Qf / Vm
        # tau_ref
        #eq = self.CreateEquation("tau_ref", "")
        #eq.Residual = tau_ref * (sigma_f * Qf) - Vm
        # sigma_f
        eq = self.CreateEquation("sigma_f", "")
        eq.Residual = sigma_f * (time_f + time_b + time_e) - time_f
        # sigma_b
        eq = self.CreateEquation("sigma_b", "")
        eq.Residual = sigma_b * (time_f + time_b + time_e) - time_b
        # sigma_e
        eq = self.CreateEquation("sigma_e", "")
        eq.Residual = sigma_e * (time_f + time_b + time_e) - time_e

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_1")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
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
        self.m.mK.SetValue(5 * kg)
        self.m.cpK.SetValue(2 * kJ/(kg*K))

    def SetUpVariables(self):
        self.m.time_f.AssignValue( 1.0 / 16.0)
        self.m.time_b.AssignValue(14.0 / 16.0)
        self.m.time_e.AssignValue( 1.0 / 16.0)
        self.m.V0.AssignValue(1 * l)
        self.m.Vm.AssignValue(100 * l)
        self.m.Qf.AssignValue(14.19 * l/hour)
        self.m.Qk.AssignValue(-1579.5 * kJ/hour)
        self.m.Caf.AssignValue(5.1 * mol/l)
        self.m.Tf.AssignValue((273.15 + 104.9) * K)

        self.m.VR.SetInitialCondition(10.0 * l)
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
        app = daeCreateQtApplication(sys.argv)
        guiRun(app)
