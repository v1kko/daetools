import os, sys, tempfile
from daetools.pyDAE import *
from time import localtime, strftime
from PyQt4 import QtCore, QtGui
from Simulator_ui import Ui_SimulatorDialog

#from ImageViewer_ui import Ui_ImageViewerDialog
from daetools.pyDAE.WebViewDialog import WebView

class daeTextEditLog(daeStdOutLog):
    def __init__(self, TextEdit, App):
        daeStdOutLog.__init__(self)
        self.TextEdit = TextEdit
        self.App      = App

    def Message(self, message, severity):
        self.TextEdit.append(message)
        if self.TextEdit.isVisible() == True:
            self.TextEdit.update()
        self.App.processEvents()

class daeImageViewer(QtGui.QDialog):
    def __init__(self, fileName):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ImageViewerDialog()
        self.ui.setupUi(self)
        
        #filename = str(QtGui.QFileDialog.getSaveFileName(None, "Insert the file name for text data reporter", "tutorial8.out", "Text files (*.txt)"))
        if fileName == "":
            return
            
        self.image = QtGui.QImage(fileName);
        if(self.image.isNull()):
            QtGui.QMessageBox.warning(None, "DAE Tools Simulator", "Cannot load image ")
            return
        
        self.ui.label.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.ui.label.adjustSize()

class daeSimulator(QtGui.QDialog):
    def __init__(self, app, simulation, datareporter = None, log = None, daesolver = None):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulatorDialog()
        self.ui.setupUi(self)

        self.app          = app
        self.simulation   = simulation
        self.datareporter = datareporter
        self.log          = log
        self.daesolver    = daesolver
        self.lasolver     = None
       
        self.ui.SimulationLineEdit.insert(simulation.m.Name)
        self.ui.ReportingIntervalDoubleSpinBox.setValue(self.simulation.ReportingInterval)
        self.ui.TimeHorizonDoubleSpinBox.setValue(self.simulation.TimeHorizon)
        
        self.connect(self.ui.RunButton,    QtCore.SIGNAL('clicked()'), self.slotRun)
        self.connect(self.ui.ResumeButton, QtCore.SIGNAL('clicked()'), self.slotResume)
        self.connect(self.ui.PauseButton,  QtCore.SIGNAL('clicked()'), self.slotPause)
        self.connect(self.ui.MatrixButton, QtCore.SIGNAL('clicked()'), self.slotOpenSparseMatrixImage)
        
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
            if self.log == None:
                self.log = daeTextEditLog(self.ui.textEdit, self.app)
            if self.daesolver == None:
                self.daesolver = daeIDASolver()
            
            simName = self.simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
            if(self.datareporter.Connect(str(tcpipaddress), simName) == False):
                QtGui.QMessageBox.warning(None, "DAE Tools Simulator", "Cannot connect data reporter!\nDid you forget to start daePlotter?")
                raise RuntimeError("Cannot connect daeTCPIPDataReporter")
                return
            
            self.lasolver = None
            lasolverIndex = self.ui.LASolverComboBox.currentIndex()
            if lasolverIndex == 0:
                pass
            
            elif lasolverIndex == 1:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Klu")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 2:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Superlu")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 3:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    self.lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Umfpack")
                    self.daesolver.SetLASolver(self.lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 4:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
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
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)

            self.simulation.InitSimulation(self.daesolver, self.datareporter, self.log)
            self.simulation.SolveInitial()
            self.simulation.Run()
            self.simulation.Finalize()
            
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
