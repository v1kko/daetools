#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                           adsorption_column.py
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
http://www.jobs.ac.uk/cgi-bin/search.cgi?category=0800&keywords=&x=30&y=15&category=0807&category=0802&category=0812&category=0806&salary_from=&salary_to=&sort=gl&s=51|http://pyo.oulu.fi/index.php?979|http://www.academictransfer.com/search_results/?q=simulation&sort=publ_start_date%20desc,%20score%20desc,%20id%20desc&page=3|http://ec.europa.eu/euraxess/index.cfm/jobs/jvSearch/page/28|http://cc.oulu.fi/%7Epokemwww/staff/karhu_uusi.htm|http://ec.europa.eu/euraxess/index.cfm/jobs/jobDetails/33673057|https://www.abo.fi/student/en/kt_forskning|http://www.academictransfer.com/employer/UU/vacancy/8790/lang/en/|http://www.phys.tue.nl/MTP/|http://jobs.tue.nl/wd/plsql/wd_portal.search_results?p_web_site_id=3085&p_category_id=3713&p_show_results=Y&p_form_type=CHECKBOX&p_no_days=999&p1=6047&p1_val=Any&p2=6048&p2_val=Any&p_text=&p_save_search=N|http://jobs.tue.nl/wd/plsql/wd_portal.search_results?p_web_site_id=3085&p_category_id=3713&p_show_results=Y&p_form_type=CHECKBOX&p_no_days=999&p1=6047&p1_val=Any&p2=6048&p2_val=Any&p_text=Nanofluidics|http://www.eng.bham.ac.uk/chemical/research/energy.shtml|http://www.chalmers.se/insidan/EN/news/vacancies|http://www.nature.com/naturejobs/science/jobs/187915|http://www.jobs.ac.uk/job/ACM026/research-associate-fellow-in-process-chemical-engineering/
"""

import sys
from daetools.pyDAE import *
from daetools.solvers import pySuperLU
from time import localtime, strftime

class modAdsorptionColumn(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Rc = daeParameter("R_c", eReal, self, "Universal gas constant")

        # Domains
        self.Nc = daeDomain("N_c", self, "Number of components")
        self.z  = daeDomain("z",   self, "Axial domain")

        # Bed parameters
        self.Rb     = daeParameter("R_b",          eReal, self, "Bed radius, m")
        self.L      = daeParameter("L",            eReal, self, "Bed length, m")
        self.eb     = daeParameter("&epsilon;_b",  eReal, self, "Bed porosity, -")

        # Particle parameters
        self.ep     = daeParameter("&epsilon;_p",  eReal, self, "Particle porosity, -")
        self.Rp     = daeParameter("R_p",          eReal, self, "Particle radius, m")

        self.Deff   = daeParameter("D_eff",        eReal, self, "Deff, m2/s")
        self.Deff.DistributeOnDomain(self.Nc)

        self.A1     = daeParameter("A_1",          eReal, self, "Langmuir, ")
        self.A1.DistributeOnDomain(self.Nc)
        self.A2     = daeParameter("A_2",          eReal, self, "Langmuir, ")
        self.A2.DistributeOnDomain(self.Nc)
        self.B0     = daeParameter("B_0",          eReal, self, "Langmuir, ")
        self.B0.DistributeOnDomain(self.Nc)
        self.B1     = daeParameter("B_1",          eReal, self, "Langmuir, ")
        self.B1.DistributeOnDomain(self.Nc)

        self.P0 = daeParameter("P_0", eReal, self, "")
        self.T0 = daeParameter("T_0", eReal, self, "")
        self.X0 = daeParameter("X_0", eReal, self, "")
        self.X0.DistributeOnDomain(self.Nc)

        # Particle thermo-physical properties
        self.rho_p  = daeVariable("&rho;_p",      density_t,                    self, "Particle density")
        self.cp_p   = daeVariable("cp_p",         specific_heat_capacity_t,     self, "Particle heat capacity")
        self.k_p    = daeVariable("&lambda;_p",   specific_heat_conductivity_t, self, "Particle heat conductivity")

        # Gas mixture thermo-physical properties
        self.rho    = daeVariable("&rho;",        density_t,                    self, "Gas mixture density")
        self.cp     = daeVariable("cp",           specific_heat_conductivity_t, self, "Gas mixture heat capacity")
        self.mu     = daeVariable("&mu;",         dynamic_viscosity_t,          self, "Gas mixture dynamic viscosity")

        # Gas mixture axial dispersion coefficients
        self.Dz     = daeVariable("D_z",          diffusivity_t,                self, "Gas mixture axial dispersion")
        self.kz     = daeVariable("&lambda;_z",   specific_heat_conductivity_t, self, "Gas mixture heat axial dispersion")

        # Feed properties
        self.Pf = daeVariable("P_f", pressure_t,    self, "")
        self.uf = daeVariable("u_f", velocity_t,    self, "")
        self.Tf = daeVariable("T_f", temperature_t, self, "")
        self.Xf = daeVariable("X_f", fraction_t,    self, "")
        self.Xf.DistributeOnDomain(self.Nc)

        # State variables
        self.C = daeVariable("C", molar_concentration_t, self, "")
        self.C.DistributeOnDomain(self.Nc)
        self.C.DistributeOnDomain(self.z)

        self.Q = daeVariable("Q", amount_adsorbed_t, self, "")
        self.Q.DistributeOnDomain(self.Nc)
        self.Q.DistributeOnDomain(self.z)

        self.Qeq = daeVariable("Q_eq", amount_adsorbed_t, self, "")
        self.Qeq.DistributeOnDomain(self.Nc)
        self.Qeq.DistributeOnDomain(self.z)

        self.Qm = daeVariable("Q_m", amount_adsorbed_t, self, "Langmuir Qm")
        self.Qm.DistributeOnDomain(self.Nc)
        self.Qm.DistributeOnDomain(self.z)

        self.B = daeVariable("B", no_t, self, "Langmuir B")
        self.B.DistributeOnDomain(self.Nc)
        self.B.DistributeOnDomain(self.z)

        self.u = daeVariable("u", velocity_t, self, "")
        self.u.DistributeOnDomain(self.z)

        self.P = daeVariable("P", pressure_t, self, "")
        self.P.DistributeOnDomain(self.z)

        self.T = daeVariable("T", temperature_t, self, "")
        self.T.DistributeOnDomain(self.z)

        self.Qg = daeVariable("Q_g", no_t, self, "")
        self.Qg.DistributeOnDomain(self.z)

        self.X = daeVariable("X", fraction_t, self, "")
        self.X.DistributeOnDomain(self.Nc)
        self.X.DistributeOnDomain(self.z)

        self.Ng = daeVariable("Ng", no_t, self, "")
        self.Ng.DistributeOnDomain(self.Nc)
        self.Ng.DistributeOnDomain(self.z)

    def DeclareEquations(self):
        #
        # Mass balance
        #
        eq = self.CreateEquation("MassBalance", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = self.u(z) * self.C.d(self.z, i, z) + \
                      self.C.dt(i, z) + \
                      ((1 - self.eb()) / self.eb()) * self.Ng(i, z) - \
                      self.Dz() * self.C.d2(self.z, i, z)

        #
        # Heat balance
        #
        eq = self.CreateEquation("HeatBalance", "")
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = self.T(z) - self.Tf()
        #eq.Residual = self.rho() * self.cp() * self.u(z) * self.T.d(self.z, z) + \
        #              self.rho_p() * self.cp_p() * self.T.dt(z) - \
        #              ((1 - self.eb()) / self.eb()) * self.Qg(z) - \
        #              self.k_p() * self.T.d2(self.z, z)

        #
        # Momentum balance
        #
        eq = self.CreateEquation("MomentumBalance", "")
        z = eq.DistributeOnDomain(self.z, eOpenOpen)
        eq.Residual = -self.P.d(self.z, z) - 180 * ((1 - self.eb())**2 / self.eb()**3) * self.mu() * self.u(z) / (2 * self.Rp())

        #
        # Equation of state
        #
        eq = self.CreateEquation("EquationOfState", "")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        nc = daeIndexRange(self.Nc)
        eq.Residual = self.P(z) - self.sum(self.C.array(nc,z) * self.Rc() * self.T(z))

        #
        # Particle mass balance (Linear Driving Force equation)
        #
        eq = self.CreateEquation("LDF", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = self.Q.dt(i, z) - (15 * self.Deff(i) / self.Rp() ** 2) *  (self.Qeq(i, z) - self.Q(i, z))

        #
        # Adsorption isotherm (Extended Langmuir)
        #
        eq = self.CreateEquation("AdsorptionIsotherm", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        nc = daeIndexRange(self.Nc)
        eq.Residual = self.Qeq(i, z) - self.Qm(i, z) * self.B(i, z) * self.X(i, z) * self.P(z) / (1 + self.sum(self.B.array(nc, z) * self.X.array(nc, z) * self.P(z)))

        eq = self.CreateEquation("B", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = self.B(i, z) - self.B0(i) * Exp(self.B1(i) / self.T(z))

        eq = self.CreateEquation("Qm", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = self.Qm(i, z) - 1e3 * (self.A1(i) + self.A2(i) / self.T(z))

        #
        # Molar fraction
        #
        eq = self.CreateEquation("X", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        nc = daeIndexRange(self.Nc)
        eq.Residual = self.X(i, z) * self.sum(self.C.array(nc, z)) - self.C(i, z)

        #
        # Accumulation terms in mass and heat balances
        #
        eq = self.CreateEquation("Ng", "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = self.Ng(i, z) - self.rho_p() * self.Q.dt(i, z)

        eq = self.CreateEquation("Qg", "")
        z = eq.DistributeOnDomain(self.z, eClosedClosed)
        eq.Residual = self.Qg(z)

        #
        # Operating modes (boundary conditions)
        #
        self.stnBC = self.STN("OperatingMode")

        #
        # State 1: Adsorption
        #
        self.STATE("Adsorption")

        self.bcInlet(self.uf(), eLowerBound, 'inlet')
        self.bcOutlet(-0.5 * self.uf(), eUpperBound, 'outlet')

        #
        # State n: Closed
        #
        self.STATE("Closed")

        self.bcClosedEnd(eLowerBound, 'inlet')
        self.bcClosedEnd(eUpperBound, 'outlet')

        self.END_STN()


        # TPP properties
        eq = self.CreateEquation("Cp_p", "")
        eq.Residual = self.cp_p() - 1046.0

        eq = self.CreateEquation("&lambda;_p", "")
        eq.Residual = self.k_p() - 0.1

        eq = self.CreateEquation("&rho;_p", "")
        eq.Residual = self.rho_p() - 850.0

        eq = self.CreateEquation("&mu;", "")
        eq.Residual = self.mu() - 1.0E-5

        eq = self.CreateEquation("C_p", "")
        eq.Residual = self.cp() - 3000.0

        eq = self.CreateEquation("&rho;", "")
        eq.Residual = self.rho() - 82 * self.P(0) / 1.0E5

        eq = self.CreateEquation("D_z", "")
        eq.Residual = self.Dz() - 1.0E-3

        eq = self.CreateEquation("k_z", "")
        eq.Residual = self.kz() - 1.0

    def bcClosedEnd(self, columnEnd, namePostfix):
        eq = self.CreateEquation("C" + namePostfix, "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.C.d(self.z, i, z)

        eq = self.CreateEquation("T" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.T.d(self.z, z)

        eq = self.CreateEquation("U" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.u.d(self.z, z)

    def bcOutlet(self, Uout, columnEnd, namePostfix):
        eq = self.CreateEquation("C" + namePostfix, "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.C.d(self.z, i, z)

        eq = self.CreateEquation("T" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.T.d(self.z, z)

        eq = self.CreateEquation("U" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.u(z) # - Uout

    def bcInlet(self, Uin, columnEnd, namePostfix):
        eq = self.CreateEquation("C" + namePostfix, "")
        i = eq.DistributeOnDomain(self.Nc, eClosedClosed)
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.u(z) * (self.Xf(i) * self.Pf() / (self.Rc() * self.Tf()) - self.C(i, z)) + self.Dz() * self.C.d(self.z, i, z)

        eq = self.CreateEquation("T" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.rho() * self.cp() * self.u(z) * (self.Tf() - self.T(z)) + self.kz() * self.T.d(self.z, z)

        eq = self.CreateEquation("U" + namePostfix, "")
        z = eq.DistributeOnDomain(self.z, columnEnd)
        eq.Residual = self.u(z) - Uin

class simAdsorptionColumn(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modAdsorptionColumn("AdsorptionColumn")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.Rc.SetValue(8.314)
        self.m.Rb.SetValue(0.1)
        self.m.L.SetValue(0.5)
        self.m.eb.SetValue(0.36)
        self.m.Rp.SetValue(0.003)
        self.m.ep.SetValue(0.6)

        self.m.Nc.CreateArray(4)
        self.m.z.CreateDistributed(eCFDM, 2, 50, 0, self.m.L.GetValue())

        self.m.T0.SetValue(300)
        self.m.P0.SetValue(1E5)
        self.m.X0.SetValue(0, 0.7557)
        self.m.X0.SetValue(1, 0.0400)
        self.m.X0.SetValue(2, 0.0350)
        self.m.X0.SetValue(3, 0.1693)

        self.m.A1.SetValue(0,   4.32e-3) # mol/g
        self.m.A1.SetValue(1,   0.92e-3)
        self.m.A1.SetValue(2,  -1.78e-3)
        self.m.A1.SetValue(3, -14.20e-3)

        self.m.A2.SetValue(0, 0.00) # K
        self.m.A2.SetValue(1, 0.52)
        self.m.A2.SetValue(2, 1.98)
        self.m.A2.SetValue(3, 6.63)

        self.m.B0.SetValue(0,  6.72e-7 / 133.322) # 1/Pa
        self.m.B0.SetValue(1,  7.86e-7 / 133.322)
        self.m.B0.SetValue(2, 26.60e-7 / 133.322)
        self.m.B0.SetValue(3, 33.03e-7 / 133.322)

        self.m.B1.SetValue(0,  850.5) # K
        self.m.B1.SetValue(1, 1730.9)
        self.m.B1.SetValue(2, 1446.7)
        self.m.B1.SetValue(3, 1496.6)

        self.m.Deff.SetValue(0, 1.0 * 3e-3 ** 2 / 15)
        self.m.Deff.SetValue(1, 0.3 * 3e-3 ** 2 / 15)
        self.m.Deff.SetValue(2, 0.4 * 3e-3 ** 2 / 15)
        self.m.Deff.SetValue(3, 0.1 * 3e-3 ** 2 / 15)

    def SetUpVariables(self):
        self.m.stnBC.ActiveState = 'Closed'

        self.m.Tf.AssignValue(300)
        self.m.uf.AssignValue(0.001)
        self.m.Pf.AssignValue(1E5)
        self.m.Xf.AssignValue(0, 0.7557)
        self.m.Xf.AssignValue(1, 0.0400)
        self.m.Xf.AssignValue(2, 0.0350)
        self.m.Xf.AssignValue(3, 0.1693)

        #self.m.InitialConditionMode = eDifferentialValuesProvided
        for i in range(0, self.m.Nc.NumberOfPoints):
            for z in range(1, self.m.z.NumberOfPoints - 1):
                self.m.C.SetInitialCondition(i, z, self.m.X0.GetValue(i) * self.m.P0.GetValue() / (self.m.Rc.GetValue() * self.m.T0.GetValue()))
                #self.m.C.SetInitialCondition(i, z, 0.0)
            for z in range(0, self.m.z.NumberOfPoints):
                self.m.Q.SetInitialCondition(i, z, 0.1)

        #for z in range(1, self.m.z.NumberOfPoints - 1):
            #self.m.T.SetInitialCondition(z, self.m.T0.GetValue())
            #self.m.T.SetInitialCondition(z, 0)

    def Run(self):
        self.m.InitialConditionMode = eAlgebraicValuesProvided
        self.m.stnBC.ActiveState = 'Adsorption'
        self.Reinitialize()
        self.ReportData()

        time = self.ReportingInterval
        while time < self.TimeHorizon:
            if time > self.TimeHorizon:
                time = self.TimeHorizon
            self.Log.Message("Integrating to " + str(time) + " ... ", 0)
            self.IntegrateUntilTime(time, eDoNotStopAtDiscontinuity)
            self.ReportData()
            time += self.ReportingInterval

# Use daeSimulator class
def guiRun(app):
    sim = simAdsorptionColumn()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 1
    sim.TimeHorizon       = 100
    simulator = daeSimulator(app, simulation = sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = simAdsorptionColumn()
    lasolver     = pySuperLU.daeCreateSuperLUSolver()
    daesolver.SetLASolver(lasolver)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    # Here the time is given in HOURS!
    simulation.ReportingInterval = 1
    simulation.TimeHorizon       = 100

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
