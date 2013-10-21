#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial16.py
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
In this example we use the same conduction problem as in the tutorial 4.

Here we introduce:

- Interactive operating procedures

"""

import sys
from daetools.pyDAE import *
from time import localtime, strftime, sleep
from os.path import join, realpath, dirname
from PyQt4 import QtCore, QtGui
from tutorial16_ui import Ui_InteractiveRunDialog
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

# Standard variable types are defined in variable_types.py
from pyUnits import m, kg, s, K, Pa, mol, J, W, kW

class modTutorial(daeModel):
    def __init__(self, Name, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.m     = daeParameter("m",       kg,           self, "Mass of the copper plate")
        self.cp    = daeParameter("c_p",     J/(kg*K),     self, "Specific heat capacity of the plate")
        self.alpha = daeParameter("&alpha;", W/((m**2)*K), self, "Heat transfer coefficient")
        self.A     = daeParameter("A",       m**2,         self, "Area of the plate")
        self.Tsurr = daeParameter("T_surr",  K,            self, "Temperature of the surroundings")

        self.Qin  = daeVariable("Q_in",  power_t,       self, "Power of the heater")
        self.T    = daeVariable("T",     temperature_t, self, "Temperature of the plate")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)

        eq = self.CreateEquation("HeatBalance", "Integral heat balance equation")
        eq.Residual = self.m() * self.cp() * self.T.dt() - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial16")
        self.m.Description = __doc__

    def SetUpParametersAndDomains(self):
        self.m.cp.SetValue(385 * J/(kg*K))
        self.m.m.SetValue(1 * kg)
        self.m.alpha.SetValue(200 * W/((m**2)*K))
        self.m.A.SetValue(0.1 * m**2)
        self.m.Tsurr.SetValue(283 * K)

    def SetUpVariables(self):
        self.m.Qin.AssignValue(1500 * W)
        self.m.T.SetInitialCondition(283 * K)

    def Run(self):
        opDlg = InteractiveOP(self)
        opDlg.exec_()
        
class InteractiveOP(QtGui.QDialog):
    def __init__(self, simulation):
        QtGui.QDialog.__init__(self)

        self.ui = Ui_InteractiveRunDialog()
        self.ui.setupUi(self)

        self.setWindowIcon(QtGui.QIcon(join(dirname(__file__), 'daetools-48x48.png')))
        self.setWindowTitle("Tutorial16 - An Interactive Operating Procedure")

        self.simulation = simulation
        self.ui.powerSpinBox.setValue(self.simulation.m.Qin.GetValue())
        self.ui.reportingIntervalSpinBox.setValue(self.simulation.ReportingInterval)

        self.figure = Figure((5.0, 4.0), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.ui.frame)
        self.canvas.axes = self.figure.add_subplot(111)

        # Add an empty curve
        self.line, = self.canvas.axes.plot([], [])

        self.fp9  = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='normal', size=9)
        self.fp10 = matplotlib.font_manager.FontProperties(family='sans-serif', style='normal', variant='normal', weight='bold', size=10)

        self.canvas.axes.set_xlabel('Time, s',        fontproperties=self.fp10)
        self.canvas.axes.set_ylabel('Temperature, K', fontproperties=self.fp10)

        for xlabel in self.canvas.axes.get_xticklabels():
            xlabel.set_fontproperties(self.fp9)
        for ylabel in self.canvas.axes.get_yticklabels():
            ylabel.set_fontproperties(self.fp9)

        self.canvas.axes.grid(True)

        self.connect(self.ui.runButton, QtCore.SIGNAL('clicked()'), self.integrate)

    def integrate(self):
        try:
            # Get the data from the GUI
            Qin      = float(self.ui.powerSpinBox.value())
            interval = float(self.ui.intervalSpinBox.value())
            self.simulation.ReportingInterval = float(self.ui.reportingIntervalSpinBox.value())
            self.simulation.TimeHorizon       = self.simulation.CurrentTime + interval

            if self.simulation.ReportingInterval > interval:
                QtGui.QMessageBox.warning(self, "tutorial16", 'Reporting interval must be lower than the integration interval')
                return

            # Disable the input boxes and buttons
            self.ui.powerSpinBox.setEnabled(False)
            self.ui.intervalSpinBox.setEnabled(False)
            self.ui.reportingIntervalSpinBox.setEnabled(False)
            self.ui.runButton.setEnabled(False)

            # Reassign the new Qin value, reinitialize the simulation and report the data
            self.simulation.m.Qin.ReAssignValue(Qin)
            self.simulation.Reinitialize()
            self.simulation.ReportData(self.simulation.CurrentTime)

            # Integrate for ReportingInterval until the TimeHorizon is reached
            # After each integration call update the plot with the new data
            # Sleep for 0.1 seconds after each integration to give some real-time impression
            time = self.simulation.IntegrateForTimeInterval(self.simulation.ReportingInterval)
            self.simulation.ReportData(self.simulation.CurrentTime)
            self._updatePlot()
            sleep(0.1)
            
            while time < self.simulation.TimeHorizon:
                if time + self.simulation.ReportingInterval > self.simulation.TimeHorizon:
                    interval = self.simulation.TimeHorizon - time
                else:
                    interval = self.simulation.ReportingInterval

                time = self.simulation.IntegrateForTimeInterval(interval)
                self.simulation.ReportData(self.simulation.CurrentTime)
                self._updatePlot()
                sleep(0.1)

        except Exception as e:
            QtGui.QMessageBox.warning(self, "tutorial16", 'Error: %s' % str(e))
            
        finally:
            # Enable the input boxes and buttons again
            self.ui.powerSpinBox.setEnabled(True)
            self.ui.intervalSpinBox.setEnabled(True)
            self.ui.reportingIntervalSpinBox.setEnabled(True)
            self.ui.runButton.setEnabled(True)

    def _updatePlot(self):
        temperature = self.simulation.DataReporter.dictVariables['tutorial16.T']
        x = temperature.TimeValues
        y = temperature.Values
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.canvas.axes.relim()
        self.canvas.axes.autoscale_view()
        self.canvas.draw()
        self.ui.currentTimeEdit.setText(str(self.simulation.CurrentTime) + ' s')
        QtGui.QApplication.processEvents()

def guiRun(app):
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDataReporterLocal()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 10000

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) == False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()
    simulation.Finalize()

def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeDataReporterLocal()
    simulation   = simTutorial()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = 1
    simulation.TimeHorizon = 10000

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
        app = QtGui.QApplication(sys.argv)
        consoleRun()
    else:
        app = QtGui.QApplication(sys.argv)
        guiRun(app)
