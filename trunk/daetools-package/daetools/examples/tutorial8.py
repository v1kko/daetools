#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial8.py
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
************************************************************************************
"""
__doc__ = """
This tutorial introduces the following concepts:

- Data reporters and exporting results into the following file formats:

  - Matlab MAT file (requires python-scipy package)
  - MS Excel .xls file (requires python-xlwt package)
  - JSON file (no third party dependencies)
  - VTK file (requires pyevtk and vtk packages)
  - XML file (requires python-lxml package)
  - HDF5 file (requires python-h5py package)
  - Pandas dataset (requires python-pandas package)

- Implementation of user-defined data reporters
- daeDelegateDataReporter

Some time it is not enough to send the results to the DAE Plotter but it is desirable to
export them into a specified file format (i.e. for use in other programs). For that purpose,
daetools provide a range of data reporters that save the simulation results in various formats.
In adddition, daetools allow implementation of custom, user-defined data reporters.
As an example, a user-defined data reporter is developed to save the results
into a plain text file (after the simulation is finished). Obviously, the data can be
processed in any other fashion.
Moreover, in certain situation it is required to process the results in more than one way.
The daeDelegateDataReporter can be used in those cases. It has the same interface and the
functionality like all data reporters. However, it does not do any data processing itself
but calls the corresponding functions of data reporters which are added to it using the
function AddDataReporter. This way it is possible, at the same time, to send the results
to the DAE Plotter and save them into a file (or process the data in some other ways).
In this example the results are processed in 10 different ways at the same time.

The model used in this example is very similar to the model in the tutorials 4 and 5.
"""

import sys, tempfile
from time import localtime, strftime
from daetools.pyDAE import *
from daetools.pyDAE.data_reporters import *

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

    def Connect(self, ConnectString, ProcessName):
        self.ProcessName = ProcessName
        try:
            self.f = open(ConnectString, "w")
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
            s += "    - Units: " + var.Units + "\n"
            s += "    - Domains:" + "\n"
            for domain in domains:
                s += "       - " + domain.Name + "\n"
            s += "    - Values:" + "\n"
            for i in range(len(times)):
                s += "      - Time: " + str(times[i]) + "s\n"
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

        self.Qin = daeParameter("Q_in",       W, self, "Power of the heater")
        self.m   = daeParameter("m",         kg, self, "Mass of the plate")
        self.cp  = daeParameter("c_p", J/(kg*K), self, "Specific heat capacity of the plate")

        self.T = daeVariable("T", temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation.")
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin()

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

def setupDataReporters(simulation):
    """
    Create daeDelegateDataReporter and add 10 data reporters:
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
     - daeVTKDataReporter

    The daeDelegateDataReporter does not process the data but simply delegate all calls
    to the contained data reporters.
    """
    datareporter = daeDelegateDataReporter()

    dr1  = MyDataReporter()
    dr2  = daeTCPIPDataReporter()
    dr3  = daeMatlabMATFileDataReporter()
    dr4  = daePlotDataReporter()
    dr5  = daeExcelFileDataReporter()
    dr6  = daeJSONFileDataReporter()
    dr7  = daeXMLFileDataReporter()
    dr8  = daeHDF5FileDataReporter()
    dr9  = daePandasDataReporter()
    dr10 = daeVTKDataReporter()

    # Add all data reporters to a list and store the list in the simulation object.
    # The reason is that the data reporter objects are destroyed at the exit of the
    # setupDataReporters function resulting in the dangling pointers and either the
    # segmentation fault or the 'pure virtual method called' errors.
    simulation._data_reporters_ = [dr1, dr2, dr3, dr4, dr5, dr6, dr7, dr8, dr9, dr10]

    datareporter.AddDataReporter(dr1)
    datareporter.AddDataReporter(dr2)
    datareporter.AddDataReporter(dr3)
    datareporter.AddDataReporter(dr4)
    datareporter.AddDataReporter(dr5)
    datareporter.AddDataReporter(dr6)
    datareporter.AddDataReporter(dr7)
    datareporter.AddDataReporter(dr8)
    datareporter.AddDataReporter(dr9)
    datareporter.AddDataReporter(dr10)

    # Connect data reporters
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    filename      = tempfile.gettempdir() + "/tutorial8.out"
    mat_filename  = tempfile.gettempdir() + "/tutorial8.mat"
    txt_filename  = tempfile.gettempdir() + "/tutorial8.txt"
    xls_filename  = tempfile.gettempdir() + "/tutorial8.xls"
    json_filename = tempfile.gettempdir() + "/tutorial8.json"
    xml_filename  = tempfile.gettempdir() + "/tutorial8.xml"
    hdf5_filename = tempfile.gettempdir() + "/tutorial8.hdf5"
    vtk_filename  = tempfile.gettempdir() + "/tutorial8.vtk"

    dr1.Connect(filename, simName)
    dr2.Connect("", simName)
    dr3.Connect(mat_filename, simName)
    dr4.Connect(txt_filename, simName)
    dr5.Connect(xls_filename, simName)
    dr6.Connect(json_filename, simName)
    dr7.Connect(xml_filename, simName)
    dr8.Connect(hdf5_filename, simName)
    dr9.Connect("", simName)
    dr10.Connect(vtk_filename, simName)

    # Print the connection status for all data reporters
    for dr in simulation._data_reporters_:
        print('%s (%s): %s' % (dr.__class__.__name__, dr.ConnectString, 'connected' if dr.IsConnected() else 'NOT connected'))

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
    simulation._data_reporters_[3].Plot(
        simulation.m.T,                       # Subplot 1
        [simulation.m.T, simulation.m.T]      # Subplot 2 (2 sets)
        )

    try:
        # All data reporters derived from daeDataReporterLocal and daeTCPIPDataReporter
        # classes have Process property (daeDataReceiverProcess object). The daeDataReceiverProcess class
        # contains dictVariableValues property which represents a dictionary
        # 'variable_name':(ndarr_times, ndarr_values, list_domains, string_units)
        # First print the contents of the abovementioned dictionary:
        import pprint
        pprint.pprint(simulation._data_reporters_[0].Process.dictVariableValues)

        # Get the dictionary
        dvals = simulation._data_reporters_[0].Process.dictVariableValues
        # Plot some variables
        values,times,domains,units = dvals['tutorial8.T']
        import matplotlib
        matplotlib.pyplot.plot(times,values)
        matplotlib.pyplot.title('tutorial8.T (%s)' % units)
        matplotlib.pyplot.show()

        # Pandas dataset
        print('pandas dataset')
        print(simulation._data_reporters_[8].data_frame)
    except Exception as e:
        print(str(e))
    
if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtCore, QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
