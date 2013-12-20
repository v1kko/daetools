#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                               simulator.py
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
***********************************************************************************
"""
import os, platform, sys, tempfile, traceback, webbrowser, math
from os.path import join, realpath, dirname
from daetools.pyDAE import *
from time import ctime, time, localtime, strftime, struct_time
from PyQt4 import QtCore, QtGui
from simulator_ui import Ui_SimulatorDialog
import aux

try:
    from daetools.pyDAE.web_view_dialog import daeWebView
except Exception as e:
    print('Cannot load web_view_dialog module\n Error: ', str(e))

images_dir = join(dirname(__file__), 'images')

class daeTextEditLog(daeBaseLog):
    def __init__(self, TextEdit, ProgressBar, ProgressLabel, App):
        daeBaseLog.__init__(self)
        self.TextEdit         = TextEdit
        self.ProgressBar      = ProgressBar
        self.ProgressLabel    = ProgressLabel
        self.App              = App
        self.time             = time()

    def SetProgress(self, progress):
        daeBaseLog.SetProgress(self, progress)
        self.ProgressBar.setValue(self.Progress)
        self.ProgressLabel.setText(self.ETA)
        #self.App.processEvents()
    
    def Message(self, message, severity):
        self.TextEdit.append(self.IndentString + message)
        if self.TextEdit.isVisible() == True:
            self.TextEdit.update()
        self.App.processEvents()

class daeSimulator(QtGui.QDialog):
    def __init__(self, app, **kwargs):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulatorDialog()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        font = QtGui.QFont()
        if platform.system() == 'Linux':
            font.setFamily("Monospace")
            font.setPointSize(9)
        elif platform.system() == 'Darwin':
            font.setFamily("Monaco")
            font.setPointSize(10)
        else:
            font.setFamily("Courier New")
            font.setPointSize(9)

        self.ui.textEdit.setFont(font)
        self.setWindowTitle("DAE Tools Simulator v" + daeVersion(True))

        self.available_la_solvers  = aux.getAvailableLASolvers()
        self.available_nlp_solvers = aux.getAvailableNLPSolvers()

        self.ui.DAESolverComboBox.addItem("Sundials IDAS")
        for la in self.available_la_solvers:
            self.ui.LASolverComboBox.addItem(la[0], userData = QtCore.QVariant(la[1]))
        
        for nlp in self.available_nlp_solvers:
            self.ui.MINLPSolverComboBox.addItem(nlp[0], userData = QtCore.QVariant(nlp[1]))

        self.connect(self.ui.RunButton,    QtCore.SIGNAL('clicked()'), self.slotRun)
        self.connect(self.ui.ResumeButton, QtCore.SIGNAL('clicked()'), self.slotResume)
        self.connect(self.ui.PauseButton,  QtCore.SIGNAL('clicked()'), self.slotPause)
        self.connect(self.ui.MatrixButton, QtCore.SIGNAL('clicked()'), self.slotOpenSparseMatrixImage)
        self.connect(self.ui.ExportButton, QtCore.SIGNAL('clicked()'), self.slotExportSparseMatrixAsMatrixMarketFormat)

        self.app                         = app
        self.simulation                  = kwargs.get('simulation',                 None)
        self.optimization                = kwargs.get('optimization',               None)
        self.datareporter                = kwargs.get('datareporter',               None)
        self.log                         = kwargs.get('log',                        None)
        self.daesolver                   = kwargs.get('daesolver',                  None)
        self.lasolver                    = kwargs.get('lasolver',                   None)
        self.nlpsolver                   = kwargs.get('nlpsolver',                  None)
        self.nlpsolver_setoptions_fn     = kwargs.get('nlpsolver_setoptions_fn',    None)
        self.lasolver_setoptions_fn      = kwargs.get('lasolver_setoptions_fn',     None)
        self.run_after_simulation_end_fn = kwargs.get('run_after_simulation_end_fn',None)

        if self.app == None:
            if not QtCore.QCoreApplication.instance():
                self.app = QtGui.QApplication(sys.argv)
        if self.simulation == None:
            raise RuntimeError('daeSimulator: simulation object must not be None')

        self.ui.DAESolverComboBox.setEnabled(False)

        if self.lasolver == None:
            self.ui.LASolverComboBox.setEnabled(True)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self.ui.LASolverComboBox.clear()
            self.ui.LASolverComboBox.addItem(self.lasolver.Name)
            self.ui.LASolverComboBox.setEnabled(False)

        if self.optimization == None:
            # If we are simulating then clear and disable MINLPSolver combo box
            self.ui.simulationLabel.setText('Simulation')
            self.ui.MINLPSolverComboBox.clear()
            self.ui.MINLPSolverComboBox.setEnabled(False)
        else:
            self.ui.simulationLabel.setText('Optimization')
            if(self.nlpsolver == None):
                self.ui.MINLPSolverComboBox.setEnabled(True)
            else:
                # If nlpsolver has been sent then clear and disable MINLPSolver combo box
                self.ui.MINLPSolverComboBox.clear()
                self.ui.MINLPSolverComboBox.addItem(self.nlpsolver.Name)
                self.ui.MINLPSolverComboBox.setEnabled(False)
        self.ui.SimulationLineEdit.insert(self.simulation.m.Name)

        self.ui.ReportingIntervalDoubleSpinBox.setValue(self.simulation.ReportingInterval)
        self.ui.TimeHorizonDoubleSpinBox.setValue(self.simulation.TimeHorizon)

        cfg = daeGetConfig()
        tcpip = cfg.GetString("daetools.datareporting.tcpipDataReceiverAddress", "127.0.0.1")
        port  = cfg.GetInteger("daetools.datareporting.tcpipDataReceiverPort", 50000)
        self.ui.DataReporterTCPIPAddressLineEdit.setText( tcpip + ':' + str(port) )

    def done(self, status):
        #print('daeSimulator.done = {0}'.format(status))
        if self.simulation:
            self.simulation.Pause()
        elif self.optimization:
            pass
        
        return QtGui.QDialog.done(self, status)
        
    def __del__(self):
        # Calling Finalize is not mandatory since it will be called
        # in a simulation/optimization destructor
        if self.simulation:
            self.simulation.Finalize()
        elif self.optimization:
            self.optimization.Finalize()

    #@QtCore.pyqtSlot()
    def slotResume(self):
        self.simulation.Resume()

    #@QtCore.pyqtSlot()
    def slotPause(self):
        self.simulation.Pause()

    #@QtCore.pyqtSlot()
    def slotRun(self):
        try:
            self.ui.textEdit.clear()

            tcpipaddress                      = str(self.ui.DataReporterTCPIPAddressLineEdit.text())
            self.simulation.ReportingInterval = float(self.ui.ReportingIntervalDoubleSpinBox.value())
            self.simulation.TimeHorizon       = float(self.ui.TimeHorizonDoubleSpinBox.value())

            if self.datareporter == None:
                self.datareporter = daeTCPIPDataReporter()
                simName = self.simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if(self.datareporter.Connect(str(tcpipaddress), simName) == False):
                    self.showMessage("Cannot connect data reporter!\nDid you forget to start daePlotter?")
                    raise RuntimeError("Cannot connect daeTCPIPDataReporter")
                    return

            if self.log == None:
                self.log = daeTextEditLog(self.ui.textEdit, self.ui.progressBar, self.ui.progressLabel, self.app)
            else:
                log1 = self.log
                log2 = daeTextEditLog(self.ui.textEdit, self.ui.progressBar, self.ui.progressLabel, self.app)
                self.log = daeDelegateLog()
                self.log.AddLog(log1)
                self.log.AddLog(log2)
                self._logs = [log1, log2]

            if self.daesolver == None:
                self.daesolver = daeIDAS()

            lasolverIndex    = -1
            minlpsolverIndex = -1

            # If nlpsolver is not sent then choose it from the selection
            if self.nlpsolver == None and len(self.available_nlp_solvers) > 0:
                minlpsolverIndex = self.ui.MINLPSolverComboBox.itemData(self.ui.MINLPSolverComboBox.currentIndex()).toInt()[0]

            # If lasolver is not sent then create it based on the selection
            if self.lasolver == None and len(self.available_la_solvers) > 0:
                lasolverIndex = self.ui.LASolverComboBox.itemData(self.ui.LASolverComboBox.currentIndex()).toInt()[0]
                self.lasolver = aux.createLASolver(lasolverIndex)

            self.ui.RunButton.setEnabled(False)
            self.ui.MINLPSolverComboBox.setEnabled(False)
            self.ui.DAESolverComboBox.setEnabled(False)
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)
            self.ui.MatrixButton.setEnabled(False)
            self.ui.ExportButton.setEnabled(False)
            if lasolverIndex in [aux.laAmesos_Klu,
                                 aux.laAmesos_Superlu,
                                 aux.laAmesos_Umfpack,
                                 aux.laAztecOO,
                                 aux.laIntelPardiso,
                                 aux.laSuperLU,
                                 aux.laSuperLU_MT,
                                 aux.laSuperLU_CUDA,
                                 aux.laCUSP]:
                self.ui.MatrixButton.setEnabled(True)
                self.ui.ExportButton.setEnabled(True)

            if self.lasolver:
                self.daesolver.SetLASolver(self.lasolver)

            if self.optimization == None:
                self.simulation.Initialize(self.daesolver, self.datareporter, self.log)
                if(self.lasolver_setoptions_fn):
                    self.lasolver_setoptions_fn(self.lasolver)
                self.simulation.SolveInitial()
                self.simulation.Run()
                if self.run_after_simulation_end_fn:
                    self.run_after_simulation_end_fn(self.simulation, self.log)

            else:
                # If nlpsolver is not sent then create it based on the selection
                if(self.nlpsolver == None):
                    self.nlpsolver = aux.createNLPSolver(minlpsolverIndex)

                self.optimization.Initialize(self.simulation, self.nlpsolver, self.daesolver, self.datareporter, self.log)

                if(self.nlpsolver_setoptions_fn):
                    self.nlpsolver_setoptions_fn(self.nlpsolver)

                self.optimization.Run()

                if self.run_after_simulation_end_fn:
                    self.run_after_simulation_end_fn(self.simulation, self.log)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            messages = traceback.format_tb(exc_traceback)
            self.ui.textEdit.append('\n'.join(messages))
            self.ui.textEdit.append(str(e))
            if self.ui.textEdit.isVisible() == True:
                self.ui.textEdit.update()
            self.app.processEvents()

    def showMessage(self, msg):
        QtGui.QMessageBox.warning(self, "daeSimulator", str(msg))
        
    #@QtCore.pyqtSlot()
    def slotOpenSparseMatrixImage(self):
        if(self.lasolver != None):
            tmpdir = tempfile.gettempdir() + os.sep
            matName = tmpdir + self.simulation.m.Name + ".xpm"
            self.lasolver.SaveAsXPM(matName)
            url = QtCore.QUrl(matName)
            try:
                wv = daeWebView(url)
                wv.resize(400, 400)
                wv.setWindowTitle("Sparse matrix: " + matName)
                wv.exec_()
            except Exception as e:
                webbrowser.open_new_tab(matName)

    #@QtCore.pyqtSlot()
    def slotExportSparseMatrixAsMatrixMarketFormat(self):
        if(self.lasolver != None):
            fileName = QtGui.QFileDialog.getSaveFileName(self, "Save File", self.simulation.m.Name +".mtx", "Matrix Market Format Files (*.mtx)")
            if(str(fileName) != ""):
                self.lasolver.SaveAsMatrixMarketFile(str(fileName), self.simulation.m.Name + " matrix", self.simulation.m.Name + " description")

