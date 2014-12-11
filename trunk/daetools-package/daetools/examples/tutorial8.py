#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial8.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic, 2013
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
In this example we use a similar problem as in the tutorial 5.

Here we introduce a data reporter concept and show how to export results to several formats:
 - Matlab MAT file (needs python-scipy package)
 - MS Excel .xls file (needs python-xlwt package)
 - JSON file (no third party dependencies)
 - XML file (needs python-lxml package)
 - HDF5 file (needs python-h5py package)
 - Pandas dataset (needs python-pandas package)
and how to write custom data reporters. Some time it is not enough to send the results
to daePlotter but it is desirable to export them in certain format for use in other programs.
Here we show how the custom data reporter can be created.
In this example the data reporter simply, after the simulation is finished, save the results
into a plain text file. Obviously, the data can be exported to any format.

In addition, a new type of data reporters (daeDelegateDataReporter) is presented. It
has the same interface and the functionality like all data reporters. However, it does not do
any data processing itself but calls the corresponding functions of data reporters which
are added to it by using the function AddDataReporter. This way it is possible, at the same
time, to send the results to the daePlotter and save them into a file (or process them in
some other ways). In this example we will use it to process results in eight different ways
at the same time.
"""

import sys, tempfile
from daetools.pyDAE import *
from daetools.pyDAE.data_reporters import *
from time import localtime, strftime

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, J, W

# The best starting point in creating custom data reporters that can export the results
# into a file is daeDataReporterLocal class. It internally does all the processing
# and offers to users the Process property (daeDataReceiverProcess object) which contains
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
            print(content)
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

        self.Qin  = daeParameter("Q_in", W, self, "Power of the heater")

        self.m  = daeParameter("m",   kg, self, "Mass of the plate")
        self.cp = daeParameter("c_p", J/(kg*K), self, "Specific heat capacity of the plate")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation.")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin()

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial8")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.Qin.SetValue(500 * W)

    def SetUpVariables(self):
        self.m.T.SetInitialCondition(300 * K)

# All data reporters should be members of let's say simulation object, otherwise
# the objects are destroyed at the exit of setupDataReporters function and
# we get dangling daeDataReporter_t pointers stored in the daeDelegateDataReporter list and
# receive the 'pure virtual method called' error
def setupDataReporters(simulation):
    """
    Create daeDelegateDataReporter and add 9 data reporters:
     - MyDataReporterLocal
       User-defined data reporter to write data to the file 'tutorial8.out'
     - daeTCPIPDataReporter
       Standard data reporter that sends data to the daePlotter
     - daeMatlabMATFileDataReporter
       Exports the results into the Matlab .mat file format
     - daePlotDataReporter
       Plots selected variables using Matplotlib
     - daeExcelFileDataReporter
       Exports the results into MS Excel file format
     - daeJSONFileDataReporter
     - daeXMLFileDataReporter
     - daeHDF5FileDataReporter
     - daePandasDataReporter
    """
    datareporter = daeDelegateDataReporter()

    simulation.dr1 = MyDataReporter()
    simulation.dr2 = daeTCPIPDataReporter()
    simulation.dr3 = daeMatlabMATFileDataReporter()
    simulation.dr4 = daePlotDataReporter()
    simulation.dr5 = daeExcelFileDataReporter()
    simulation.dr6 = daeJSONFileDataReporter()
    simulation.dr7 = daeXMLFileDataReporter()
    simulation.dr8 = daeHDF5FileDataReporter()
    simulation.dr9 = daePandasDataReporter()

    datareporter.AddDataReporter(simulation.dr1)
    datareporter.AddDataReporter(simulation.dr2)
    datareporter.AddDataReporter(simulation.dr3)
    datareporter.AddDataReporter(simulation.dr4)
    datareporter.AddDataReporter(simulation.dr5)
    datareporter.AddDataReporter(simulation.dr6)
    datareporter.AddDataReporter(simulation.dr7)
    datareporter.AddDataReporter(simulation.dr8)
    datareporter.AddDataReporter(simulation.dr9)

    # Connect data reporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    filename     = tempfile.gettempdir() + "/tutorial8.out"
    matfilename  = tempfile.gettempdir() + "/tutorial8.mat"
    txtfilename  = tempfile.gettempdir() + "/tutorial8.txt"
    xlsfilename  = tempfile.gettempdir() + "/tutorial8.xls"
    jsonfilename = tempfile.gettempdir() + "/tutorial8.json"
    xmlfilename  = tempfile.gettempdir() + "/tutorial8.xml"
    hdf5filename = tempfile.gettempdir() + "/tutorial8.hdf5"

    if not simulation.dr1.Connect(filename, simName):
        sys.exit()
    if not simulation.dr2.Connect("", simName):
        sys.exit()
    if not simulation.dr3.Connect(matfilename, simName):
        sys.exit()
    if not simulation.dr4.Connect(txtfilename, simName):
        sys.exit()
    if not simulation.dr5.Connect(xlsfilename, simName):
        sys.exit()
    if not simulation.dr6.Connect(jsonfilename, simName):
        sys.exit()
    if not simulation.dr7.Connect(xmlfilename, simName):
        sys.exit()
    if not simulation.dr8.Connect(hdf5filename, simName):
        sys.exit()
    if not simulation.dr9.Connect("", simName):
        sys.exit()

    return datareporter

# Use daeSimulator class
def guiRun(app):
    sim = simTutorial()
    dr  = setupDataReporters(sim)
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 100
    simulator  = daeSimulator(app, simulation=sim, datareporter=dr)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    simulation   = simTutorial()
    datareporter = setupDataReporters(simulation)

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 10
    simulation.TimeHorizon = 100

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
    simulation.dr4.Plot(
        simulation.m.T,                       # Subplot 1
        [simulation.m.T, simulation.m.T]      # Subplot 2 (2 sets)
        )

    # All data reporters derived from daeDataReporterLocal and daeTCPIPDataReporter
    # classes have Process property (daeDataReceiverProcess object). The daeDataReceiverProcess class
    # contains dictVariableValues property which represents a dictionary 'variable_name':(ndarr_times, ndarr_values)
    # First print the contents of the abovementioned dictionary:
    import pprint
    pprint.pprint(simulation.dr1.Process.dictVariableValues)

    # Get the dictionary
    dvals = simulation.dr1.Process.dictVariableValues
    # Plot some variables
    values,times,domains = dvals['tutorial8.T']
    import matplotlib
    matplotlib.pyplot.plot(times,values)
    matplotlib.pyplot.show()

    print(simulation.dr9.data_frame)
    print(simulation.dr9.data_frame['Value']['tutorial8.T'])

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
