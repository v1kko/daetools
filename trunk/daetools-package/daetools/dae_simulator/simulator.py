"""
***********************************************************************************
                               simulator.py
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
***********************************************************************************
"""
import os, platform, sys, tempfile, traceback, webbrowser, math
from os.path import join, realpath, dirname
from daetools.pyDAE import *
from time import ctime, time, localtime, strftime, struct_time
from PyQt4 import QtCore, QtGui

from .simulator_ui import Ui_SimulatorDialog
from . import auxiliary
from . import simulation_explorer

python_major = sys.version_info[0]
python_minor = sys.version_info[1]
python_build = sys.version_info[2]

try:
    from daetools.pyDAE.web_view_dialog import daeWebView
except Exception as e:
    print(('Cannot load web_view_dialog module\n Error: ', str(e)))

images_dir = join(dirname(__file__), 'images')

class daeTextEditLog(daeBaseLog):
    def __init__(self, TextEdit, ProgressBar, ProgressLabel, App):
        daeBaseLog.__init__(self)
        self.TextEdit         = TextEdit
        self.ProgressBar      = ProgressBar
        self.ProgressLabel    = ProgressLabel
        self.App              = App
        self.time             = time()

    @property
    def Name(self):
        return 'daeTextEditLog'
        
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
        self.setWindowTitle("DAE Tools Simulator v%s [py%d.%d]" % (daeVersion(True), python_major, python_minor))

        self.available_la_solvers  = auxiliary.getAvailableLASolvers()
        self.available_nlp_solvers = auxiliary.getAvailableNLPSolvers()

        self.ui.DAESolverComboBox.addItem("Sundials IDAS")
        for i, la in enumerate(self.available_la_solvers):
            self.ui.LASolverComboBox.addItem(la[0], userData = la[1])
            self.ui.LASolverComboBox.setItemData(i, la[2], QtCore.Qt.ToolTipRole)
        
        for i, nlp in enumerate(self.available_nlp_solvers):
            self.ui.MINLPSolverComboBox.addItem(nlp[0], userData = nlp[1])
            self.ui.MINLPSolverComboBox.setItemData(i, nlp[2], QtCore.Qt.ToolTipRole)

        menuRun = QtGui.QMenu()
        actionRun                = QtGui.QAction('Run', self)
        actionShowExplorerAndRun = QtGui.QAction('Show simulation explorer and run', self)
        menuRun.addAction(actionRun)
        menuRun.addAction(actionShowExplorerAndRun)
        self.ui.RunButton.setMenu(menuRun)

        self.connect(self.ui.ResumeButton,      QtCore.SIGNAL('clicked()'),   self.slotResume)
        self.connect(self.ui.PauseButton,       QtCore.SIGNAL('clicked()'),   self.slotPause)
        self.connect(self.ui.MatrixButton,      QtCore.SIGNAL('clicked()'),   self.slotOpenSparseMatrixImage)
        self.connect(self.ui.ExportButton,      QtCore.SIGNAL('clicked()'),   self.slotExportSparseMatrixAsMatrixMarketFormat)
        self.connect(actionRun,                 QtCore.SIGNAL('triggered()'), self.slotRun)
        self.connect(actionShowExplorerAndRun,  QtCore.SIGNAL('triggered()'), self.slotShowExplorerAndRun)

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
        self.run_before_simulation_begin_fn = kwargs.get('run_before_simulation_begin_fn',None)
        self.run_after_simulation_end_fn    = kwargs.get('run_after_simulation_end_fn',None)

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

    """
    def __del__(self):
        # Calling Finalize is not mandatory since it will be called
        # in a simulation/optimization destructor
        if self.simulation:
            self.simulation.Finalize()
        elif self.optimization:
            self.optimization.Finalize()
    """
    
    #@QtCore.pyqtSlot()
    def slotResume(self):
        self.simulation.Resume()

    #@QtCore.pyqtSlot()
    def slotPause(self):
        self.simulation.Pause()

    #@QtCore.pyqtSlot()
    def slotRun(self):
        self.run(False)
        
    #@QtCore.pyqtSlot()
    def slotShowExplorerAndRun(self):
        self.run(True)
        
    def run(self, showExplorer):
        try:
            self.ui.textEdit.clear()
            
            tcpipaddress                      = str(self.ui.DataReporterTCPIPAddressLineEdit.text())
            self.simulation.ReportingInterval = float(self.ui.ReportingIntervalDoubleSpinBox.value())
            self.simulation.TimeHorizon       = float(self.ui.TimeHorizonDoubleSpinBox.value())

            if self.datareporter == None:
                self.datareporter = daeTCPIPDataReporter()
                simName = self.simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
                if(self.datareporter.Connect(str(tcpipaddress), simName) == False):
                    self.showMessage("Cannot connect data reporter!\nDid you forget to start DAE Tools Plotter?")
                    self.datareporter = None
                    raise RuntimeError("Cannot connect daeTCPIPDataReporter")

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
                _index = self.ui.MINLPSolverComboBox.itemData(self.ui.MINLPSolverComboBox.currentIndex())
                if isinstance(_index, QtCore.QVariant):
                    minlpsolverIndex = _index.toInt()[0]
                else:
                    if _index != None and _index >= 0:
                        minlpsolverIndex = int(_index)

            # If lasolver is not sent then create it based on the selection
            if self.lasolver == None and len(self.available_la_solvers) > 0:
                _index = self.ui.LASolverComboBox.itemData(self.ui.LASolverComboBox.currentIndex())
                if isinstance(_index, QtCore.QVariant):
                    lasolverIndex = _index.toInt()[0]
                else:
                    if _index != None and _index >= 0:
                        lasolverIndex = int(_index)
                self.lasolver = auxiliary.createLASolver(lasolverIndex)

            self.ui.RunButton.setEnabled(False)
            self.ui.MINLPSolverComboBox.setEnabled(False)
            self.ui.DAESolverComboBox.setEnabled(False)
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)
            self.ui.MatrixButton.setEnabled(False)
            self.ui.ExportButton.setEnabled(False)
            if lasolverIndex in [auxiliary.laAmesos_Klu,
                                 auxiliary.laAmesos_Superlu,
                                 auxiliary.laAmesos_Umfpack,
                                 auxiliary.laAztecOO,
                                 auxiliary.laIntelPardiso,
                                 auxiliary.laSuperLU,
                                 auxiliary.laSuperLU_MT,
                                 auxiliary.laSuperLU_CUDA,
                                 auxiliary.laCUSP]:
                self.ui.MatrixButton.setEnabled(True)
                self.ui.ExportButton.setEnabled(True)

            if self.lasolver:
                self.daesolver.SetLASolver(self.lasolver)

            if self.optimization == None:
                self.simulation.Initialize(self.daesolver, self.datareporter, self.log)
                if(self.lasolver_setoptions_fn):
                    self.lasolver_setoptions_fn(self.lasolver)

                self.simulation.SolveInitial()

                if self.run_before_simulation_begin_fn:
                    self.run_before_simulation_begin_fn(self.simulation, self.log)

                if showExplorer:
                    explorer = simulation_explorer.daeSimulationExplorer(self.app, simulation = self.simulation)
                    explorer.exec_()

                self.simulation.Run()

                if self.run_after_simulation_end_fn:
                    self.run_after_simulation_end_fn(self.simulation, self.log)

                self.simulation.Finalize()

            else:
                # If nlpsolver is not sent then create it based on the selection
                if(self.nlpsolver == None):
                    self.nlpsolver = auxiliary.createNLPSolver(minlpsolverIndex)

                self.optimization.Initialize(self.simulation, self.nlpsolver, self.daesolver, self.datareporter, self.log)

                if(self.nlpsolver_setoptions_fn):
                    self.nlpsolver_setoptions_fn(self.nlpsolver)

                self.optimization.Run()

                if self.run_after_simulation_end_fn:
                    self.run_after_simulation_end_fn(self.simulation, self.log)

                self.optimization.Finalize()

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

