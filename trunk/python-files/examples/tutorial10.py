#!/usr/bin/env python

"""********************************************************************************
                             tutorial10.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License as published by the Free Software 
Foundation; either version 3 of the License, or (at your option) any later version.
The DAE Tools is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, write to the Free Software Foundation, Inc., 
59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
********************************************************************************"""

"""
In this example we use the same conduction problem as in the tutorial 1.
Here we introduce:
 - daeModel functions d() and dt() which calculate time- or partial-derivative of an expression
 - Initialization files
 - Mathematical operators which operate on both adouble and adouble_array
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-",          0, 1E10,   0, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",        100, 1000, 300, 1e-5)
typeConductivity = daeVariableType("Conductivity", "W/mK",       0, 1E10, 100, 1e-5)
typeDensity      = daeVariableType("Density",      "kg/m3",      0, 1E10, 100, 1e-5)
typeHeatCapacity = daeVariableType("HeatCapacity", "J/KgK",      0, 1E10, 100, 1e-5)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, "X axis domain")
        self.y  = daeDomain("y", self, "Y axis domain")

        self.Qb = daeParameter("Q_b", eReal, self, "Heat flux at the bottom edge of the plate, W/m2")
        self.Qt = daeParameter("Q_t", eReal, self, "Heat flux at the top edge of the plate, W/m2")

        self.ro = daeParameter("&rho;", eReal, self, "Density of the plate, kg/m3")
        self.cp = daeParameter("c_p", eReal, self, "Specific heat capacity of the plate, J/kgK")
        self.k  = daeParameter("&lambda;",  eReal, self, "Thermal conductivity of the plate, W/mK")
 
        self.Q_int = daeVariable("Q_int", typeTemperature, self, "The heat input per unit of length, W/m")

        self.T = daeVariable("T", typeTemperature, self, "Temperature of the plate, K")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        # All equations are written so that they use only functions d() and dt() from daeModel
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.dt( self.ro() * self.cp() * self.T(x, y) ) - \
                      self.k() * ( self.d ( self.d ( self.T(x, y), self.x ), self.x ) + \
                                   self.d ( self.d ( self.T(x, y), self.y ), self.y ) )

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.d( self.k() * self.T(x, y), self.y) - self.Qb()

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = - self.d ( self.k() * self.T(x, y), self.y) - self.Qt()

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.d( self.T(x, y), self.x )

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.d( self.T(x, y), self.x )

        xr = daeIndexRange(self.x)
        yr = daeIndexRange(self.y)

        # Here we have a complex expression in integral function 
        eq = self.CreateEquation("Q_int", "Integral of the heat flux per x domain; just an example of the integral function")
        eq.Residual = self.Q_int() - self.integral( -self.k() * self.T.d_array(self.y, xr, 0) )
              
class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial10")
        self.m.Description = "This tutorial explains how to use daeModel functions d() and dt() " \
                             "that calculate time- and partial-derivative of an expression (not of a single variable), " \
                             "how to use initialization files (.init) and how to use mathematical operators " \
                             "which operate on both adouble and adouble_array."
          
    def SetUpParametersAndDomains(self):
        n = 10
        
        self.m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        
        self.m.ro.SetValue(8960)
        self.m.cp.SetValue(385)
        self.m.k.SetValue(401)

        self.m.Qb.SetValue(1e6)
        self.m.Qt.SetValue(0)

    def SetUpVariables(self):
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300)
        
        # Load initialization file previously saved after the successful initialization phase (see below)
        #self.LoadInitializationValues("tutorial10.init")

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    solver       = daeIDASolver()
    datareporter = daeTCPIPDataReporter()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 1000

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(solver, datareporter, log)

    # Save the model report and the runtime model report 
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()
    # Save the initialization file that can be used during later initialization
    simulation.StoreInitializationValues("tutorial10.init")

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
    else:
        consoleRun()
