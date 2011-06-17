#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""********************************************************************************
                      binary_destilation_column.py
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

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)
        # Domains
        self.Nt = daeDomain("N_t", self, "Number of trays")
        
        # Parameters
        self.Ntrays   = daeParameter("N_trays",       eInteger, self, "Number of trays")
        self.FeedTray = daeParameter("FeedTray",      eInteger, self, "Feed tray")
        self.R        = daeParameter("R",             eReal,    self, "Reflux ratio, -")
        self.Nfeed    = daeParameter("Nfeed",         eReal,    self, "Feed flowrate, mol/min")
        self.xf       = daeParameter("xf",            eReal,    self, "Mole fraction of feed")
        self.alpha    = daeParameter("&alpha;",       eReal,    self, "Relative volatility, -")
        self.atray    = daeParameter("atray",         eReal,    self, "Total molar holdup on each tray")
        self.acond    = daeParameter("acond",         eReal,    self, "Total molar holdup in the condenser")
        self.areb     = daeParameter("areb",          eReal,    self, "Total molar holdup in the reboiler")

        # Variables
        self.x  = daeVariable("x",   typeNone, self, "Mole fraction of component 1")
        self.x.DistributeOnDomain(self.Nt)
        
        self.y  = daeVariable("y",   typeNone, self, "Mole fraction of component 2")
        self.y.DistributeOnDomain(self.Nt)
        
        self.D  = daeVariable("D",   typeNone, self, "")
        self.L  = daeVariable("L",   typeNone, self, "")
        self.V  = daeVariable("V",   typeNone, self, "")
        self.FL = daeVariable("FL",  typeNone, self, "")

    def DeclareEquations(self):
        FeedTray = int(self.FeedTray.GetValue())
        Ntrays   = self.Nt.NumberOfPoints
        print FeedTray,  Ntrays
        
        eq = self.CreateEquation("Condenser", "")
        eq.Residual = self.acond() * self.x.dt(0) - ( self.V() * (self.y(1) - self.x(0)) )
        
        eq = self.CreateEquation("RectificationSection", "")
        tr = range(1, FeedTray) # [1, 2, ..., FeedTray-1]
        t = eq.DistributeOnDomain(self.Nt, tr)
        eq.Residual = self.atray() * self.x.dt(t) - ( self.L() * (self.x(t-1) - self.x(t)) - self.V() * (self.y(t) - self.y(t+1)) )
        
        eq = self.CreateEquation("FeedTray", "")
        eq.Residual = self.atray() * self.x.dt(FeedTray) - ( \
                                                            self.Nfeed() * self.xf() + self.L() * self.x(FeedTray-1) - self.FL() * self.x(FeedTray) - \
                                                            self.V() * (self.y(FeedTray) - self.y(FeedTray+1)) \
                                                           )

        eq = self.CreateEquation("StrippingSection", "")
        tr = range(FeedTray+1, Ntrays-1) # [FeedTray, FeedTray+1, ..., Ntrays]
        t = eq.DistributeOnDomain(self.Nt, tr)
        eq.Residual = self.atray() * self.x.dt(t) - ( self.FL() * (self.x(t-1) - self.x(t)) - self.V() * (self.y(t) - self.y(t+1)) )
        
        eq = self.CreateEquation("Reboiler", "")
        eq.Residual = self.areb() * self.x.dt(Ntrays-1) - ( self.FL() * self.x(Ntrays-2) - (self.Nfeed() - self.D()) * self.x(Ntrays-1) - self.V() * self.y(Ntrays-1) )
        
        eq = self.CreateEquation("D", "")
        eq.Residual = self.D() - 0.5 * self.Nfeed()
        
        eq = self.CreateEquation("L", "")
        eq.Residual = self.L() - self.R() * self.D()

        eq = self.CreateEquation("V", "")
        eq.Residual = self.V() - self.L() - self.D()
        
        eq = self.CreateEquation("FL", "")
        eq.Residual = self.FL() - self.Nfeed() - self.L()

        eq = self.CreateEquation("y", "")
        t = eq.DistributeOnDomain(self.Nt, eClosedClosed)
        eq.Residual = self.y(t) - ( self.x(t) * self.alpha() / (1 + (self.alpha() - 1) * self.x(t)) )

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("binary_destilation_column")
        self.m.Description = ""

    def SetUpParametersAndDomains(self):
        self.m.FeedTray.SetValue(16)
        self.m.Ntrays.SetValue(32)
        
        self.m.Nt.CreateArray(int(self.m.Ntrays.GetValue()))
        self.m.R.SetValue(0.7)
        self.m.Nfeed.SetValue(2.0)
        self.m.xf.SetValue(0.5)
        self.m.alpha.SetValue(1.6)
        self.m.atray.SetValue(0.25)
        self.m.acond.SetValue(0.5)
        self.m.areb.SetValue(0.1)

    def SetUpVariables(self):
        for t in range(0, self.m.Nt.NumberOfPoints):
            self.m.x.SetInitialCondition(t, 0.3)

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
    simulation.ReportingInterval =   1 # 1 min
    simulation.TimeHorizon       = 120 # 30 min

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
