#!/usr/bin/env python

"""********************************************************************************
                             tutorial8.py
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
In this example we use a similar problem as in the tutorial 5.
Here we introduce:
 - Custom data reporters

Some time it is not enough to send the result to daePlotter but it is desirable to 
export them in certain format for use in other programs. Here we show how the custom
data reporter can be created. In this example the data reporter simply, after the simulation
is finished, save the results into a plain text file. Obviously, the data can be exported to
any format. Also some numpy functions that operate on numpy arrays can be used as well.
In addition, a new type of data reporters (daeDelegateDataReporter) is presented. It
has the same interface and the functionality like all data reporters. However, it does not do
any data processing by itself but calls the corresponding functions of data reporters which 
are added to it by using the function AddDataReporter. This way it is possible, at the same 
time, to send the results to the daePlotter and save them into a file (or process them in
some other ways). 
"""

import sys, tempfile
from daetools.pyDAE import *
from time import localtime, strftime

typeNone         = daeVariableType("None",         "-",      0, 1E10,   0, 1e-5)
typeTemperature  = daeVariableType("Temperature",  "K",    100, 1000, 300, 1e-5)
typeConductivity = daeVariableType("Conductivity", "W/mK",   0, 1E10, 100, 1e-5)
typeDensity      = daeVariableType("Density",      "kg/m3",  0, 1E10, 100, 1e-5)
typeHeatCapacity = daeVariableType("HeatCapacity", "J/KgK",  0, 1E10, 100, 1e-5)

# The best starting point in creating custom data reporters that can export the results 
# into a file is daeDataReporterLocal class. It internally does all the processing
# and offers a user the Process property of type daeDataReporterProcess which contains
# all domains and variables sent by simulation. The following functions have to be implemented:
#  - Connect
#    It is used to connect the data reporter. In the case when the local data reporter is used
#    it may contain a file name, for instance.
#  - Disconnect
#    Disconnects the data reporter.
#  - IsConnected
#    Check if the data reporter is connected or not.
# In this example we use the first argument of the function Connect as a file name to open 
# a text file in the TMP folder (/tmp or c:\temp) and implement a new function Write to write 
# the data into the file. In the function MakeString we iterate over all variables and write 
# their values into a string which will be saved in the function Write.
# The content of the file (/tmp/tutorial8.out) will also be printed to the console.
class MyDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)
        self.ProcessName = ""

    def Connect(self, ConnectionString, ProcessName):
        self.ProcessName = ProcessName
        try:
            self.f = open(ConnectionString, "w")            
        except IOError:
            return False       
        return True
        
    def Disconnect(self):
        self.Write()
        return True

    def MakeString(self):
        s = "Process name: " + self.ProcessName + "\n"
        variables = self.Process.Variables
        for var in variables:
            values  = var.Values
            domains = var.Domains
            times   = var.TimeValues
            s += " - Variable: " + var.Name + "\n"
            s += "    - Domains:" + "\n"
            for domain in domains:
                s += "       - " + domain.Name + "\n"
            s += "    - Values:" + "\n"
            for i in range(len(times)):
                s += "      - Time: " + str(times[i]) + "\n"
                s += "        " + str(values[i, ...]) + "\n"

        return s
        
    def Write(self):
        try:
            content = self.MakeString()
            print content
            self.f.write(content)
            self.f.close()

        except IOError:
            self.f.close()
            return False

    def IsConnected(self):
        return True


class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.Qin  = daeParameter("Q_in", eReal, self, "Power of the heater, W")

        self.m  = daeParameter("m",   eReal, self, "Mass of the plate, kg")
        self.cp = daeParameter("c_p", eReal, self, "Specific heat capacity of the plate, J/kgK")
 
        self.T = daeVariable("T", typeTemperature, self, "Temperature of the plate, K")

    def DeclareEquations(self):
        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation.")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial8")
        self.m.Description = "This tutorial explains how to create custom data reporters and how to " \
                             "create a composite data reporter which delegates " \
                             "the data processing to other data reporters. "
          
    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385)
        self.m.m.SetValue(1)
        self.m.Qin.SetValue(500)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(300)

# This function does not work!!
# It seems the class daeDelegateDataReporter cant iterate over
# data reporters!!???
def setupDataReporters(simulationName):
    # Create daeDelegateDataReporter and the add 2 data reporters:
    # - MyDataReporterLocal (to write data to the file 'tutorial8.out')
    # - daeTCPIPDataReporter (to send data to the daePlotter)
    datareporter = daeDelegateDataReporter()
    dr1 = MyDataReporter()
    dr2 = daeTCPIPDataReporter()
    datareporter.AddDataReporter(dr1)
    datareporter.AddDataReporter(dr2)

    # Connect data reporter
    simName = simulationName + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    filename = tempfile.gettempdir() + "/tutorial8.out"
    if(dr1.Connect(filename, simName) == False):
        sys.exit()
    if(dr2.Connect("", simName) == False):
        sys.exit()
    
    return datareporter
   
# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    dr  = MyDataReporter()
    filename = tempfile.gettempdir() + "/tutorial8.out"
    if(dr.Connect(filename, sim.m.Name) == False):
        sys.exit()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 100
    simulator  = daeSimulator(app, simulation=sim, datareporter=dr)
    simulator.exec_()
    
# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    solver       = daeIDASolver()
    simulation   = simTutorial()
    datareporter = MyDataReporter()
    
    filename = tempfile.gettempdir() + "/tutorial8.out"
    if(datareporter.Connect(filename, simulation.m.Name) == False):
        sys.exit()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 100

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
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
    else:
        consoleRun()
