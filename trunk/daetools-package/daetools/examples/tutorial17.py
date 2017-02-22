#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
***********************************************************************************
                            tutorial17.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
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

- TCPIP Log and TCPIPLogServer

In this example we use the same heat transfer problem as in the tutorial 7.

The screenshot of the TCP/IP log server:

.. image:: _static/tutorial17-screenshot.png
   :width: 500px

The temperature plot:

.. image:: _static/tutorial17-results.png
   :width: 500px
"""

import os, sys, threading
from PyQt4 import QtCore, QtGui
from time import localtime, strftime, sleep
from os.path import join, realpath, dirname
from subprocess import Popen, call
from daetools.pyDAE import *
try:
    from .tutorial17_ui import Ui_tcpipLogServerMainWindow
except Exception as e:
    from tutorial17_ui import Ui_tcpipLogServerMainWindow

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
        eq.Residual = self.m() * self.cp() * dt(self.T()) - self.Qin() + self.alpha() * self.A() * (self.T() - self.Tsurr())

class simTutorial(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        self.m = modTutorial("tutorial17")
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
        # The default Run() function is re-implemented here (just the basic version)
        # to allow simulation to wait for certain period between time intervals
        while self.CurrentTime < self.TimeHorizon:
            self.Log.Message('Integrating from %f to %f ...' % (self.CurrentTime, self.CurrentTime+self.ReportingInterval), 0)
            self.IntegrateForTimeInterval(self.ReportingInterval, eDoNotStopAtDiscontinuity)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100.0 * self.CurrentTime/self.TimeHorizon))
            sleep(0.2)

        self.Log.Message('The simulation has finished succesfully!', 0)

class tcpipLogServer(daeTCPIPLogServer):
    def __init__(self, port, app, textEdit):
        if port <= 0:
            cfg  = daeGetConfig()
            port = cfg.GetInteger("daetools.logging.tcpipLogPort", 51000)

        daeTCPIPLogServer.__init__(self, port)

        self.app      = app
        self.textEdit = textEdit

    def MessageReceived(self, message):
        self.textEdit.append(message)
        if self.textEdit.isVisible() == True:
            self.textEdit.update()
        self.textEdit.verticalScrollBar().setSliderPosition(self.textEdit.verticalScrollBar().maximum())
        self.app.processEvents()

class tcpipLogServerMainWindow(QtGui.QMainWindow):
    def __init__(self, app):
        QtGui.QMainWindow.__init__(self)

        self.ui = Ui_tcpipLogServerMainWindow()
        self.ui.setupUi(self)
        #self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowIcon(QtGui.QIcon(join(dirname(__file__), 'daetools-48x48.png')))
        self.setGeometry(0, 100, 600, 200) # Position window
        self.resize(600, 300)  # Resize window

        # Create TCPIP log server
        self.logServer = tcpipLogServer(0, app, self.ui.messagesEdit)
        # Start TCPIP log server
        self.logServer.Start()

class simulationThread(threading.Thread):
    def __init__(self, saveReports):
        threading.Thread.__init__(self)

        self.saveReports = saveReports

    def run(self):
        # Create a delegate log and add two logs: stdout and tcpip logs
        log, log1, log2  = setupLog()

        # Create Solver, DataReporter and Simulation object
        daesolver    = daeIDAS()
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
        simulation.Initialize(daesolver, datareporter, log)
        
        # Save the model report and the runtime model report
        if self.saveReports:
            simulation.m.SaveModelReport(simulation.m.Name + ".xml")
            simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

        # Solve at time=0 (initialization)
        simulation.SolveInitial()

        simulation.Run()
        simulation.Finalize()

def setupLog():
    log  = daeDelegateLog()
    log1 = daePythonStdOutLog()
    log2 = daeTCPIPLog()

    # Connect TCPIP log
    if(log2.Connect("", 0) == False):
        sys.exit()

    log.AddLog(log1)
    log.AddLog(log2)

    # Return all of them for we have to keep referencem to them
    # and prevent them going out of scope
    return log, log1, log2

# Use daeSimulator class
def guiRun(app):
    # Start TCP/IP log server
    log_server = tcpipLogServerMainWindow(app)
    log_server.show()
    # Give it some time to start the TCP/IP server
    sleep(1)

    # Create a delegate log and add two logs: stdout and tcpip logs
    log, log1, log2  = setupLog()

    sim = simTutorial()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 500
    simulator  = daeSimulator(app, simulation=sim, log = log)
    simulator.exec_()
    app.exec_()

# Setup everything manually and run in a console
def consoleRun(app):
    # Start TCP/IP log server
    log_server = tcpipLogServerMainWindow(app)
    log_server.show()
    # Give it some time to start the TCP/IP server
    sleep(1)

    # Create and start a thread with the simulation
    st = simulationThread(True)
    st.start()

    app.exec_()
    st.join()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun(app)
    else:
        guiRun(app)
