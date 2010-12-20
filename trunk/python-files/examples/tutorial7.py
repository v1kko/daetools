#!/usr/bin/env python

"""********************************************************************************
                             tutorial7.py
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
 - Custom operating procedures
 - Resetting of degrees of freedom
 - Resetting of initial conditions 
 
Here the heat flux at the bottom edge is defined as a variable. In the simulation its 
value will be fixed and manipulated in the custom operating procedure.
"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-",          0, 1E10,   0, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",        100, 1000, 300, 1e-5)
typeConductivity = daeVariableType("Conductivity", "W/mK",       0, 1E10, 100, 1e-5)
typeDensity      = daeVariableType("Density",      "kg/m3",      0, 1E10, 100, 1e-5)
typeHeatCapacity = daeVariableType("HeatCapacity", "J/KgK",      0, 1E10, 100, 1e-5)
typeHeatFlux     = daeVariableType("HeatFlux",     "W/m2" ,  -1E10, 1E10,   0, 1e-5)

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.x  = daeDomain("x", self, "X axis domain")
        self.y  = daeDomain("y", self, "Y axis domain")

        self.Qb = daeVariable("Q_b", typeHeatFlux, self, "Heat flux at the bottom edge of the plate, W/m2")
        self.Qt = daeParameter("Q_t", eReal, self, "Heat flux at the top edge of the plate, W/m2")

        self.ro = daeParameter("&rho;", eReal, self, "Density of the plate, kg/m3")
        self.cp = daeParameter("c_p", eReal, self, "Specific heat capacity of the plate, J/kgK")
        self.k  = daeParameter("&lambda;",  eReal, self, "Thermal conductivity of the plate, W/mK")
 
        self.T = daeVariable("T", typeTemperature, self, "Temperature of the plate, K")
        self.T.DistributeOnDomain(self.x)
        self.T.DistributeOnDomain(self.y)

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains")
        x = eq.DistributeOnDomain(self.x, eOpenOpen)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.ro() * self.cp() * self.T.dt(x, y) - self.k() * \
                     (self.T.d2(self.x, x, y) + self.T.d2(self.y, x, y))

        eq = self.CreateEquation("BC_bottom", "Boundary conditions for the bottom edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eLowerBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qb()

        eq = self.CreateEquation("BC_top", "Boundary conditions for the top edge")
        x = eq.DistributeOnDomain(self.x, eClosedClosed)
        y = eq.DistributeOnDomain(self.y, eUpperBound)
        eq.Residual = - self.k() * self.T.d(self.y, x, y) - self.Qt()

        eq = self.CreateEquation("BC_left", "Boundary conditions at the left edge")
        x = eq.DistributeOnDomain(self.x, eLowerBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

        eq = self.CreateEquation("BC_right", "Boundary conditions for the right edge")
        x = eq.DistributeOnDomain(self.x, eUpperBound)
        y = eq.DistributeOnDomain(self.y, eOpenOpen)
        eq.Residual = self.T.d(self.x, x, y)

class simTutorial(daeDynamicSimulation):
    def __init__(self):
        daeDynamicSimulation.__init__(self)
        self.m = modTutorial("Tutorial_7")
        self.m.Description = "This tutorial explains how to create custom operating procedures, how to re-set the values of " \
                             "assigned variables and how to re-set the initial conditions. "
          
    def SetUpParametersAndDomains(self):
        n = 25
        
        self.m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1)
        self.m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1)

        self.m.k.SetValue(401)
        self.m.cp.SetValue(385)
        self.m.ro.SetValue(8960)

        self.m.Qt.SetValue(0)

    def SetUpVariables(self):
        self.m.Qb.AssignValue(1e6)        
        for x in range(1, self.m.x.NumberOfPoints - 1):
            for y in range(1, self.m.y.NumberOfPoints - 1):
                self.m.T.SetInitialCondition(x, y, 300)
    
    # daeDynamicSimulation class defines the function Run which is called after successful initialization
    # to run the simulation. By default it runs for time period defined by the TimeHorizon property,
    # stopping after each period of time defined by the ReportInterval property to send the data to
    # the data reporter. However, default behaviour can be changed by implementing a user defined
    # function Run. The functions Integrate, IntegrateUntilTime, and IntegrateForTimeInterval from 
    # daeDynamicSimulation class can be used to advance in time, while functions ReAssignValue and 
    # ReSetInitialCondition from daeVariable class can be used to alter the values of variables.
    # In this example we first assign the value of Qb to 1E6 and then use the function IntegrateForTimeInterval
    # to run for 100 seconds. After that we re-assign the variable Qb to a new value (2E6). Note that after 
    # you finished with re-assigning or re-setting the initial conditions you have to call the function
    # Reinitialize from daeDynamicSimulation class. The function Reinitialize reinitializes the DAE solver
    # and clears all previous data accumulated in the solver. Also, you can call the function ReportData
    # at any point to send the results to the data reporter. After re-assigning and subsequent reinitialization 
    # we run the simulation until 200 seconds are reached (by using the function IntegrateUntilTime) and 
    # then we again report the data. After that, we again change the value of Qb and also re-set the initial 
    # conditions for the variable T (again to 300K) and then run until the TimeHorizon is reached 
    # (by using the function Integrate). 
    def Run(self):
        self.Log.Message("OP: Integrating for 100 seconds ... ", 0)
        time = self.IntegrateForTimeInterval(100)
        self.ReportData()
        
        self.m.Qb.ReAssignValue(2E6)
        self.Reinitialize()
        self.Log.Message("OP: Integrating until time = 200 seconds ... ", 0)
        time = self.IntegrateUntilTime(200, eDoNotStopAtDiscontinuity)
        self.ReportData()

        self.m.Qb.ReAssignValue(1.5E6)
        #self.m.Qt.SetValue(2E6)
        for x in range(1, self.m.x.NumberOfPoints-1):
            for y in range(1, self.m.y.NumberOfPoints-1):
                self.m.T.ReSetInitialCondition(x, y, 300)
        self.Reinitialize()
        self.ReportData()

        self.Log.Message("OP: Integrating from " + str(time) + " to the time horizon (" + str(self.TimeHorizon) + ") ... ", 0)
        time = self.Integrate(eDoNotStopAtDiscontinuity)
        self.ReportData()
        self.Log.Message("OP: Finished", 0)

# Use daeSimulator class
def guiRun():
    from PyQt4 import QtCore, QtGui
    app = QtGui.QApplication(sys.argv)
    simulation = simTutorial()
    simulation.m.SetReportingOn(True)
    simulation.ReportingInterval = 10
    simulation.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation)
    simulator.show()
    app.exec_()

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
    simulation.TimeHorizon = 500

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

    # Run
    simulation.Run()
    simulation.Finalize()

if __name__ == "__main__":
    runInGUI = True
    if len(sys.argv) > 1:
        if(sys.argv[1] == 'console'):
            runInGUI = False
    if runInGUI:
        guiRun()
    else:
        consoleRun()
