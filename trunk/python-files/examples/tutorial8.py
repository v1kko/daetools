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

import sys
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
# a text file and implement a new function Write to write the data into the file.
# In the function Write we iterate over all variables and write their values into a file.
class MyDataReporter(daeDataReporterLocal):
    def __init__(self):
        daeDataReporterLocal.__init__(self)

    def Connect(self, ConnectionString, ProcessName):
        self.ProcessName = ProcessName
        try:
            self.f = open(ConnectionString, "w")            
        except IOError:
            return False       
        return True
        
    def Disconnect(self):
        return True

    def Write(self):
        try:
            self.f.write("Process name: " + self.ProcessName + "\n")
            variables = self.Process.Variables
            for var in variables:
                values  = var.Values
                domains = var.Domains
                times   = var.TimeValues
                self.f.write(" - Variable: " + var.Name + "\n")
                self.f.write("    - Domains:" + "\n")
                for domain in domains:
                    self.f.write("       - " + domain.Name + "\n")
                self.f.write("    - Values:" + "\n")
                for i in range(len(times)):
                    self.f.write("      - Time: " + str(times[i]) + "\n")
                    self.f.write("        " + str(values[i, ...]) + "\n")

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

class simTutorial(daeDynamicSimulation):
    def __init__(self):
        daeDynamicSimulation.__init__(self)
        self.m = modTutorial("Tutorial_8")
        self.m.Description = "This tutorial explains how to create custom data reporters and how to create a composite data reporter which delegates " \
                             "the data processing to other data reporters. "
          
    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385)
        self.m.m.SetValue(1)
        self.m.Qin.SetValue(500)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(300)

if __name__ == "__main__":
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    solver       = daeIDASolver()
    simulation   = simTutorial()

    # Create daeDelegateDataReporter and the add 2 data reporters:
    # - MyDataReporterLocal (to write data to the file 'tutorial8.out')
    # - daeTCPIPDataReporter (to send data to the daePlotter)
    datareporter = daeDelegateDataReporter()
    dr1 = MyDataReporter()
    dr2 = daeTCPIPDataReporter()
    datareporter.AddDataReporter(dr1)
    datareporter.AddDataReporter(dr2)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 100

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(dr1.Connect("tutorial8.out", simName) == False):
        sys.exit()
    if(dr2.Connect("", simName) == False):
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

    # Finally, write the data to a file
    dr1.Write()
