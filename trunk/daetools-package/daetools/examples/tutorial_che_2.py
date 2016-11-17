#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                            tutorial_che_2.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
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
Binary distillation column model.

Reference: J. Hahn, T.F. Edgar. An improved method for nonlinear model reduction using
balancing of empirical gramians. Computers and Chemical Engineering 2002; 26:1379-1397.
`doi:10.1016/S0098-1354(02)00120-5 <http://dx.doi.org/10.1016/S0098-1354(02)00120-5>`_

The liquid fraction after 120 min (x(reboiler)=0.935420, x(condenser)=0.064581):

.. image:: _static/tutorial_che_2-results.png
   :width: 500px

The liquid fraction in the reboiler (tray 1) and in the condenser (tray 32):

.. image:: _static/tutorial_che_2-results2.png
   :width: 500px
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime
# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kJ, hour, l, min

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        # Domains
        self.Nt = daeDomain("N_t", self, unit(), "Number of trays")
        
        # Parameters
        self.Ntrays   = daeParameter("N_trays",       unit(), self, "Number of trays")
        self.FeedTray = daeParameter("FeedTray",      unit(), self, "Feed tray")
        self.RR       = daeParameter("RR",            unit(), self, "Reflux ratio")
        self.F        = daeParameter("F",             mol/s,  self, "Feed flowrate")
        self.xf       = daeParameter("xf",            unit(), self, "Mole fraction of feed")
        self.alpha    = daeParameter("&alpha;",       unit(), self, "Relative volatility")
        self.Atray    = daeParameter("Atray",         mol,    self, "Total molar holdup on each tray")
        self.Acond    = daeParameter("Acond",         mol,    self, "Total molar holdup in the condenser")
        self.Areb     = daeParameter("Areb",          mol,    self, "Total molar holdup in the reboiler")

        # Variables
        self.x  = daeVariable("x", no_t, self, "Liquid fraction of component A")
        self.x.DistributeOnDomain(self.Nt)
        
        self.y  = daeVariable("y", no_t, self, "Vapour fraction of component A")
        self.y.DistributeOnDomain(self.Nt)
        
        self.D  = daeVariable("D",   molar_flowrate_t, self, "Distillate flowrate")
        self.L1 = daeVariable("L1",  molar_flowrate_t, self, "Liquid flowrate in the rectification section")
        self.V  = daeVariable("V",   molar_flowrate_t, self, "Vapour flowrate")
        self.L2 = daeVariable("L2",  molar_flowrate_t, self, "Liquid flowrate in the stripping section")

    def DeclareEquations(self):
        FeedTray = int(self.FeedTray.GetValue())
        Ntrays   = self.Nt.NumberOfPoints
        print(FeedTray, Ntrays)
        
        eq = self.CreateEquation("Condenser", "")
        eq.Residual = self.Acond() * self.x.dt(0) - ( self.V() * (self.y(1) - self.x(0)) )
        
        eq = self.CreateEquation("RectificationSection", "")
        tr = list(range(1, FeedTray)) # [1, 2, ..., FeedTray-1]
        t = eq.DistributeOnDomain(self.Nt, tr)
        eq.Residual = self.Atray() * self.x.dt(t) - (  self.L1() * (self.x(t-1) - self.x(t)) \
                                                     - self.V()  * (self.y(t) - self.y(t+1)) )

        eq = self.CreateEquation("FeedTray", "")
        eq.Residual = self.Atray() * self.x.dt(FeedTray) - ( \
                                                            self.F()  * self.xf() + \
                                                            self.L1() * self.x(FeedTray-1) - \
                                                            self.L2() * self.x(FeedTray) - \
                                                            self.V()  * (self.y(FeedTray) - self.y(FeedTray+1)) \
                                                           )

        eq = self.CreateEquation("StrippingSection", "")
        tr = list(range(FeedTray+1, Ntrays-1)) # [FeedTray, FeedTray+1, ..., Ntrays]
        t = eq.DistributeOnDomain(self.Nt, tr)
        eq.Residual = self.Atray() * self.x.dt(t) - ( self.L2() * (self.x(t-1) - self.x(t)) - \
                                                      self.V()  * (self.y(t)   - self.y(t+1)) )

        eq = self.CreateEquation("Reboiler", "")
        eq.Residual = self.Areb() * self.x.dt(Ntrays-1) - ( self.L2()            * self.x(Ntrays-2) - \
                                                           (self.F() - self.D()) * self.x(Ntrays-1) - \
                                                            self.V()             * self.y(Ntrays-1) )

        eq = self.CreateEquation("D", "")
        eq.Residual = self.D() - 0.5 * self.F()

        eq = self.CreateEquation("L1", "")
        eq.Residual = self.L1() - self.RR() * self.D()

        eq = self.CreateEquation("V", "")
        eq.Residual = self.V() - (self.L1() + self.D())

        eq = self.CreateEquation("L2", "")
        eq.Residual = self.L2() - (self.F() + self.L1())

        eq = self.CreateEquation("y", "")
        t = eq.DistributeOnDomain(self.Nt, eClosedClosed)
        eq.Residual = self.y(t) - ( self.x(t) * self.alpha() / (1 + (self.alpha() - 1) * self.x(t)) )

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial_che_2")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.FeedTray.SetValue(16)
        self.m.Ntrays.SetValue(32)
        self.m.Nt.CreateArray(int(self.m.Ntrays.GetValue()))
        self.m.RR.SetValue(3.0)
        self.m.F.SetValue(2.0 * mol/min)
        self.m.xf.SetValue(0.5)
        self.m.alpha.SetValue(1.6)
        self.m.Atray.SetValue(0.25 * mol)
        self.m.Acond.SetValue(0.5 * mol)
        self.m.Areb.SetValue(0.1 * mol)

    def SetUpVariables(self):
        for t in range(0, self.m.Nt.NumberOfPoints):
            self.m.x.SetInitialCondition(t, 0.3)

# Use daeSimulator class
def guiRun(app):
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval =   2*60 #   1 min
    simulation.TimeHorizon       = 120*60 # 120 min
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
    simulation.ReportingInterval =   2*60 #  1 min
    simulation.TimeHorizon       = 120*60 # 30 min

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
