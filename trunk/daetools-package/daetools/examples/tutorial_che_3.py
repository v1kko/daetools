#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                           tutorial_che_3.py
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
Batch reactor seeded crystallisation using the method of moments.

References (model equations and input parameters):

- Nikolic D.D., Frawley P.J. (2016) Application of the Lagrangian Meshfree Approach
  to Modelling of Batch Crystallisation: Part I – Modelling of Stirred Tank Hydrodynamics.
  Chemical Engineering Science 145:317–328.
  `doi:10.1016/j.ces.2015.08.052 <http://dx.doi.org/10.1016/j.ces.2015.08.052>`_
- Mitchell N.A., O’Ciardha C.T., Frawley P.J. (2011) Estimation of the growth kinetics
  for the cooling crystallisation of paracetamol and ethanol solutions.
  Journal of Crystal Growth 328:39–49.
  `doi:10.1016/j.jcrysgro.2011.06.016 <http://dx.doi.org/10.1016/j.jcrysgro.2011.06.016>`_

The main assumptions:

- Seeded crystallisation
- Ideal mixing
- Fixed cooling rate
- Size independent growth

Solubility of Paracetamol in ethanol:

.. code-block:: none

   ---------------------------------------------------------
   Temperature       Solubility          Solubility 
        C        kg Parac./kg EtOH   mol Parac./m3 EtOH
   ---------------------------------------------------------
         0           0.11362                593.0387
        10           0.14128                737.4215
        20           0.17568                916.9562
        30           0.21845                1140.2008
        40           0.27163                1417.7972
        50           0.33777                1762.9779
        60           0.42000                2192.1973
   ---------------------------------------------------------

The supersaturation plot:

.. image:: _static/tutorial_che_3-results.png
   :width: 500px

The concentration plot:

.. image:: _static/tutorial_che_3-results2.png
   :width: 500px

The recovery plot:

.. image:: _static/tutorial_che_3-results3.png
   :width: 500px

The yield plot:

.. image:: _static/tutorial_che_3-results4.png
   :width: 500px

The total number of crystals plot:

.. image:: _static/tutorial_che_3-results5.png
   :width: 500px
"""

import sys, numpy
from daetools.pyDAE import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, g, kg, s, K, mol, kmol, J, um

pbm_growth_rate_t                = daeVariableType("pbm_growth_rate_t",                m/s,                    0.0, 1.0e+20,  0.1, 1e-05)
pbm_birth_rate_t                 = daeVariableType("pbm_birth_rate_t",                 (s**(-1)) * (m**(-3)),  0.0, 1.0e+20,  0.0, 1e-05)
pbm_death_rate_t                 = daeVariableType("pbm_death_rate_t",                 (s**(-1)) * (m**(-3)),  0.0, 1.0e+20,  0.0, 1e-05)
pbm_solution_concentration_t     = daeVariableType("pbm_solution_concentration_t",     kg/kg,                  0.0, 1.0e+20,  0.1, 1e-05)
pbm_solution_concentration_vol_t = daeVariableType("pbm_solution_concentration_vol_t", kmol/(m**3),            0.0, 1.0e+20,  0.1, 1e-05)

pbm_number_t  = daeVariableType("pbm_number_t",  m**(-3),           0.0, 1.0e+20,  0.0, 1e-05)
pbm_length_t  = daeVariableType("pbm_length_t",  m/(m**3),          0.0, 1.0e+20,  0.0, 1e-06)
pbm_area_t    = daeVariableType("pbm_area_t",    (m**2)/(m**3),     0.0, 1.0e+20,  0.0, 1e-08)
pbm_volume_t  = daeVariableType("pbm_volume_t",  (m**3)/(m**3),     0.0, 1.0e+20,  0.0, 1e-10)
pbm_moment4_t = daeVariableType("pbm_moment4_t", (m**4)/(m**3),     0.0, 1.0e+20,  0.0, 1e-11)
pbm_moment5_t = daeVariableType("pbm_moment5_t", (m**5)/(m**3),     0.0, 1.0e+20,  0.0, 1e-12)

pbm_diameter_t = daeVariableType("pbm_diameter_t",  m, 0.0, 1.0e+20,  0.0, 1e-10)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.n = 2.276 # Nucleation rate exponent
        self.g = 1.602 # Growth rate exponent
        kg_units = (m/s)*(((m**3)/kmol)**1.602)
        kn_units = (s**(-1))*(m**(-3)) * ((kg/kg)**2.276)

        self.Nm = daeDomain("Nm", self, m, "Number of moments")
        
        self.R      = daeParameter("R",     J/(mol*K), self, "Universal gas constant")
        self.MW     = daeParameter("MW",    kg/kmol,   self, "Molecular weight of paracetamol")
        self.V      = daeParameter("V",     m**3,      self, "Volume of the suspension in the reactor")
        self.kg     = daeParameter("kg",    kg_units,  self, "Growth rate constant")
        self.Ea     = daeParameter("Ea",    J/mol,     self, "Activation energy")
        self.kn     = daeParameter("kn",    kn_units,  self, "Nucleation rate constant")
        self.Kv     = daeParameter("Kv",    unit(),    self, "Volume shape factor")
        self.Ka     = daeParameter("Ka",    unit(),    self, "Surface area shape factor")
        self.rho_s  = daeParameter("rho_s", kg/(m**3), self, "Density of solution")
        self.cp_s   = daeParameter("cp_s",  J/(kg*K),  self, "Specific heat capacity of solution")
        self.rho_c  = daeParameter("rho_c", kg/(m**3), self, "Density of crystals")
        self.Qr     = daeParameter("Qr",    W,         self, "Cooling rate")
        self.L0     = daeParameter("L_0",   m,         self, "Initial size of the nucleus")

        self.m0     = daeVariable("m_0",    pbm_number_t,                       self, "Distribution moment 0")
        self.m1     = daeVariable("m_1",    pbm_length_t,                       self, "Distribution moment 1")
        self.m2     = daeVariable("m_2",    pbm_area_t,                         self, "Distribution moment 2")
        self.m3     = daeVariable("m_3",    pbm_volume_t,                       self, "Distribution moment 3")
        self.m4     = daeVariable("m_4",    pbm_moment4_t,                      self, "Distribution moment 4")
        self.m5     = daeVariable("m_5",    pbm_moment5_t,                      self, "Distribution moment 5")
        self.m_norm = daeVariable("m_norm", no_t,                               self, "Normalized distribution moments", [self.Nm])
        self.T      = daeVariable("T",      temperature_t,                      self, "Solution temperature")
        self.G      = daeVariable("G",      pbm_growth_rate_t,                  self, "Growth rate")
        self.B      = daeVariable("B",      pbm_birth_rate_t,                   self, "Birth rate")
        self.c      = daeVariable("c",      pbm_solution_concentration_t,       self, "Solution concentration")
        self.c_star = daeVariable("c_star", pbm_solution_concentration_t,       self, "Solution equlibrium concentration (solubility)")
        self.delta_c= daeVariable("delta_c",pbm_solution_concentration_vol_t,   self, "Concentrations driving force")
        self.S      = daeVariable("S",      no_t,                               self, "Supersaturation")
        
        self.Yield      = daeVariable("Yield",      mass_t,         self, "Crystals yield")
        self.Yield_max  = daeVariable("Yield_max",  mass_t,         self, "Maximal crystals yield")
        self.Recovery   = daeVariable("Recovery",   no_t,           self, "Product recovery")
        self.Ntotal     = daeVariable("Ntotal",     pbm_number_t,   self, "Total number of crystals")
        self.Ltotal     = daeVariable("Ltotal",     pbm_length_t,   self, "Total length of crystals")
        self.Atotal     = daeVariable("Atotal",     pbm_area_t,     self, "Total surface area of crystals")
        self.Vtotal     = daeVariable("Vtotal",     pbm_volume_t,   self, "Total volume of crystals")
        self.D10        = daeVariable("D10",        pbm_diameter_t, self, "Mean size of crystals 1")
        self.D43        = daeVariable("D43",        pbm_diameter_t, self, "Mean size of crystals 2")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        # Heat balance
        eq = self.CreateEquation("T", "")
        eq.Residual = self.rho_s() * self.V() * self.cp_s() * self.T.dt() - self.Qr() * (self.T() - Constant(273.14 * K))/Constant(1 * K)

        # Solution concentration
        eq = self.CreateEquation("c", "")
        eq.Residual = self.c.dt() + 3 * self.rho_c() * self.Kv() * self.G() * self.m2() / self.rho_s()

        # Solubility (in kg/kg, as given in the paper)
        eq = self.CreateEquation("c_star", "")
        eq.Residual = self.c_star() - 2.955e-4 * Exp(Constant(2.179e-2 * 1/K) * self.T())

        eq = self.CreateEquation("S", "")
        eq.Residual = self.S() - self.c() / (self.c_star() + Constant(1e-20 * kg/kg))

        # Nucleation kinetics
        eq = self.CreateEquation("B", "")
        eq.Residual = self.B() - self.kn() * (self.delta_c()**self.n)
        eq.CheckUnitsConsistency = False

        # Nucleation kinetics
        eq = self.CreateEquation("delta_c", "")
        eq.Residual = self.delta_c() - (self.c() - self.c_star()) * self.rho_s() / self.MW()
        
        # Growth kinetics
        # A safe-guard: prevent the negative concentration by setting the growth rate to zero
        # if the supersaturation drops below one (that is no driving force)
        self.IF(self.S() > 1)
        eq = self.CreateEquation("G", "")
        eq.Residual = self.G() - self.kg() * Exp(-self.Ea() / (self.R() * self.T())) * (self.delta_c()**self.g)
        eq.CheckUnitsConsistency = False

        self.ELSE()
        eq = self.CreateEquation("G", "")
        eq.Residual = self.G()
        self.END_IF()
        
        # Moments
        eq = self.CreateEquation("Moment(0)", "")
        eq.Residual = self.m0.dt() - self.B()

        eq = self.CreateEquation("Moment(1)", "")
        eq.Residual = self.m1.dt() - 1 * self.G() * self.m0() - self.B() * (self.L0()**1)

        eq = self.CreateEquation("Moment(2)", "")
        eq.Residual = self.m2.dt() - 2 * self.G() * self.m1() - self.B() * (self.L0()**2)
        eq.Scaling = 1e5

        eq = self.CreateEquation("Moment(3)", "")
        eq.Residual = self.m3.dt() - 3 * self.G() * self.m2() - self.B() * (self.L0()**3)
        eq.Scaling = 1e8

        eq = self.CreateEquation("Moment(4)", "")
        eq.Residual = self.m4.dt() - 4 * self.G() * self.m3() - self.B() * (self.L0()**4)
        eq.Scaling = 1e10

        eq = self.CreateEquation("Moment(5)", "")
        eq.Residual = self.m5.dt() - 5 * self.G() * self.m4() - self.B() * (self.L0()**5)
        eq.Scaling = 1e12

        # Normalized moments 0-5
        moments = [self.m0(), self.m1(), self.m2(), self.m3(), self.m4(), self.m5()]
        for k in range(0, self.Nm.NumberOfPoints):
            eq = self.CreateEquation("NormalizedMoment(%d)" % k, "")
            eq.Residual = self.m_norm(k) * self.m0() - moments[k]
            eq.CheckUnitsConsistency = False

        # Performance indicators
        # Crystals concentration (crystals yield)
        eq = self.CreateEquation("Yield", "")
        eq.Residual = self.Yield() - self.rho_c() * self.Kv() * self.m3() * self.V()
            
        eq = self.CreateEquation("Yield_max", "")
        eq.Residual = self.Yield_max() - self.rho_c() * self.Kv() * self.m3() * self.V() - self.c()*self.rho_s()*self.V()

        eq = self.CreateEquation("Recovery", "")
        eq.Residual = self.Recovery() - self.Yield() / (self.Yield_max() + Constant(1E-20*kg))

        eq = self.CreateEquation("Ntotal", "")
        eq.Residual = self.Ntotal() - self.m0()

        eq = self.CreateEquation("Ltotal", "")
        eq.Residual = self.Ltotal() - self.m1()

        eq = self.CreateEquation("Atotal", "")
        eq.Residual = self.Atotal() - self.Ka() * self.m2()

        eq = self.CreateEquation("Vtotal", "")
        eq.Residual = self.Vtotal() - self.Kv() * self.m3()

        eq = self.CreateEquation("D10", "")
        eq.Residual = self.D10() * self.m0() - self.m1()

        eq = self.CreateEquation("D43", "")
        eq.Residual = self.D43() * self.m3() - self.m4()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_3")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.Nm.CreateArray(6)

        self.m.V.SetValue(0.5e-3 * m**3)
        
        self.m.R.SetValue(8.3144 * J/(mol*K))
        self.m.MW.SetValue(151.163 * g/mol)
        
        self.m.kg.SetValue(9.979)
        self.m.Ea.SetValue(40560 * J/mol)

        self.m.kn.SetValue(1.597E10/60.0 * ((s**(-1))*(m**(-3))))

        self.m.Ka.SetValue(5.196)               # Surface area shape factor
        self.m.Kv.SetValue(0.866)               # Volume shape factor
        self.m.rho_c.SetValue(1332 * kg/(m**3)) # Paracetamol
        self.m.rho_s.SetValue(789 * kg/(m**3))  # Ethanol
        self.m.cp_s.SetValue(112.4 * J/(kg*K))  # cp of Ethanol
        self.m.Qr.SetValue(0.0 * W)             # Cooling rate in Watts
        self.m.L0.SetValue(1e-6 * m)            # 1um

    def SetUpVariables(self):
        # Initial conditions
        self.m.T.SetInitialCondition(293.15 * K)
        self.m.c.SetInitialCondition(0.275395968 * kg/kg)
        
        # Actung, Achtung!!
        # Initial moments for 1.0081g of 125-250um seed (as in Niall's paper)
        # The meaning of the above: the values are given in m**k per gram of paracetamol,
        # and need to be transformed into the values in m**k/m**3
        m_seed = numpy.array([514967.2105 * (m**0)/g,
                              42.56573654 * (m**1)/g,
                              0.005234078 * (m**2)/g,
                              8.84516e-07 * (m**3)/g,
                              1.90332e-10 * (m**4)/g,
                              5.00159e-14 * (m**5)/g])
        moments = [self.m.m0, self.m.m1, self.m.m2, self.m.m3, self.m.m4, self.m.m5]
        # a) Change to moments per kg of ethanol: m_seed /= rho_EtOH*V
        #    (in the Niall's paper moments are given in m**k/kg, therefore: m_seed /= 0.394 * g/kg)
        # b) Change to moments per m3 of ethanol: m_seed *= V (we use moments standardly given in m**k/m**3)
        # Also, the above numbers are given per one gram of seed - multiply the moments by actual weight of seed in grams
        m_seed *= (7.0109*g) / self.m.V.GetQuantity()
        #print m_seed
        for k in range(0, self.m.Nm.NumberOfPoints):
            moments[k].SetInitialCondition(m_seed[k])

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 60
    sim.TimeHorizon       = 3000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Relative tolerance
    daesolver.RelativeTolerance = 1e-07
    
    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 60
    simulation.TimeHorizon       = 3000

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
