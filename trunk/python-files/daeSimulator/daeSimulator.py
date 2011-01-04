import os, sys, tempfile
from daetools.pyDAE import *
from time import localtime, strftime
from PyQt4 import QtCore, QtGui
from Simulator_ui import Ui_SimulatorDialog
from daetools.pyDAE.WebViewDialog import WebView

class daeTextEditLog(daeBaseLog):
    def __init__(self, TextEdit, App):
        daeBaseLog.__init__(self)
        self.TextEdit = TextEdit
        self.App      = App

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
        
        self.setWindowTitle("DAE Tools Simulator v" + daeVersion(True))

        self.connect(self.ui.RunButton,    QtCore.SIGNAL('clicked()'), self.slotRun)
        self.connect(self.ui.ResumeButton, QtCore.SIGNAL('clicked()'), self.slotResume)
        self.connect(self.ui.PauseButton,  QtCore.SIGNAL('clicked()'), self.slotPause)
        self.connect(self.ui.MatrixButton, QtCore.SIGNAL('clicked()'), self.slotOpenSparseMatrixImage)
        
        self.app          = app
        self.simulation   = kwargs.get('simulation',   None)
        self.optimization = kwargs.get('optimization', None)
        self.datareporter = kwargs.get('datareporter', None)
        self.log          = kwargs.get('log',          None)
        self.daesolver    = kwargs.get('daesolver',    None)
        self.lasolver     = kwargs.get('lasolver',     None)
        self.nlpsolver    = kwargs.get('nlpsolver',    None)
        
        if self.app == None:
            raise RuntimeError('daeSimulator: app object must not be None')
        if self.simulation == None:
            raise RuntimeError('daeSimulator: simulation object must not be None')

        if self.optimization == None:
            self.ui.simulationLabel.setText('Simulation')
            self.ui.MINLPSolverComboBox.setEnabled(False)
        else:
            self.ui.simulationLabel.setText('Optimization')
            self.ui.MINLPSolverComboBox.setEnabled(True)
        self.ui.SimulationLineEdit.insert(self.simulation.m.Name)
            
        self.ui.ReportingIntervalDoubleSpinBox.setValue(self.simulation.ReportingInterval)
        self.ui.TimeHorizonDoubleSpinBox.setValue(self.simulation.TimeHorizon)
        
        cfg = daeGetConfig()
        tcpip = cfg.GetString("daetools.datareporting.tcpipDataReceiverAddress", "127.0.0.1")
        port  = cfg.GetInteger("daetools.datareporting.tcpipDataReceiverPort", 50000)
        self.ui.DataReporterTCPIPAddressLineEdit.setText( tcpip + ':' + str(port) )

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
                    QtGui.QMessageBox.warning(None, "DAE Tools Simulator", "Cannot connect data reporter!\nDid you forget to start daePlotter?")
                    raise RuntimeError("Cannot connect daeTCPIPDataReporter")
                    return
            if self.log == None:
                self.log = daeTextEditLog(self.ui.textEdit, self.app)
            if self.daesolver == None:
                self.daesolver = daeIDAS()
            
            self.lasolver = None
            lasolverIndex = self.ui.LASolverComboBox.currentIndex()
            if lasolverIndex == 0:
                pass
            
            elif lasolverIndex == 1:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Klu")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 2:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Superlu")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 3:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Umfpack")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 4:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Lapack")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 5:
                try:
                    import daetools.pyIntelPardiso as pyIntelPardiso
                    self.lasolver = pyIntelPardiso.daeCreateIntelPardisoSolver()
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create IntelPardiso LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 6:
                try:
                    import daetools.pyIntelMKL as pyIntelMKL
                    self.lasolver = pyIntelMKL.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create IntelMKL LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 7:
                try:
                    import daetools.pyAmdACML as pyAmdACML
                    self.lasolver = pyAmdACML.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create AmdACML LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 8:
                try:
                    import daetools.pyLapack as pyLapack
                    self.lasolver = pyLapack.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create Lapack LA solver\nError: " + str(e))
                    return
                
            else:
                raise RuntimeError("Unsupported LA Solver selected")
            
            self.ui.RunButton.setEnabled(False)
            if(lasolverIndex in [1, 2, 3, 5]):
                self.ui.MatrixButton.setEnabled(True)
            self.ui.MINLPSolverComboBox.setEnabled(False)
            self.ui.DAESolverComboBox.setEnabled(False)
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)

            if self.optimization == None:
                self.simulation.Initialize(self.daesolver, self.datareporter, self.log)
                self.simulation.SolveInitial()
                self.simulation.Run()
                self.simulation.Finalize()
            else:
                if self.nlpsolver == None:
                    self.nlpsolver = daeBONMIN()
                self.optimization.Initialize(self.simulation, self.nlpsolver, self.daesolver, self.datareporter, self.log)
                self.optimization.Run()
                self.optimization.Finalize()                
            
        except Exception, error:
            self.ui.textEdit.append(str(error))
            if self.ui.textEdit.isVisible() == True:
                self.ui.textEdit.update()
            self.app.processEvents()

    #@QtCore.pyqtSlot()
    def slotOpenSparseMatrixImage(self):
        if(self.lasolver != None):
            tmpdir = tempfile.gettempdir() + os.sep
            matName = tmpdir + self.simulation.m.Name + ".xpm"
            self.lasolver.SaveAsXPM(matName)
            url = QtCore.QUrl(matName)
            wv = WebView(url)
            wv.resize(400, 400)
            wv.setWindowTitle("Sparse matrix: " + matName)
            wv.exec_()
