import sys
from daetools.pyDAE import *
from time import localtime, strftime
from PyQt4 import QtCore, QtGui
from Simulator_ui import Ui_SimulatorDialog

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
        
        self.ui.SimulationLineEdit.insert(simulation.m.Name)
        self.ui.ReportingIntervalDoubleSpinBox.setValue(self.simulation.ReportingInterval)
        self.ui.TimeHorizonDoubleSpinBox.setValue(self.simulation.TimeHorizon)
        
        self.connect(self.ui.RunButton,    QtCore.SIGNAL('clicked()'), self.slotRun)
        self.connect(self.ui.ResumeButton, QtCore.SIGNAL('clicked()'), self.slotResume)
        self.connect(self.ui.PauseButton,  QtCore.SIGNAL('clicked()'), self.slotPause)
        
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
            
            lasolverIndex = self.ui.LASolverComboBox.currentIndex()
            if lasolverIndex == 0:
                pass
            
            elif lasolverIndex == 1:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Klu")
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 2:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Superlu")
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 3:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Umfpack")
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 4:
                try:
                    import daetools.pyTrilinosAmesos as pyTrilinosAmesos
                    self.log.Message("Supported TrilinosAmesos 3rd party LA solvers: " + str(pyTrilinosAmesos.daeTrilinosAmesosSupportedSolvers()), 0)
                    lasolver = pyTrilinosAmesos.daeCreateTrilinosAmesosSolver("Amesos_Lapack")
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 5:
                try:
                    import daetools.pyIntelPardiso as pyIntelPardiso
                    lasolver = pyIntelPardiso.daeCreateIntelPardisoSolver()
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create IntelPardiso LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 6:
                try:
                    import daetools.pyIntelMKL as pyIntelMKL
                    lasolver = pyIntelMKL.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create IntelMKL LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 7:
                try:
                    import daetools.pyAmdACML as pyAmdACML
                    lasolver = pyAmdACML.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create AmdACML LA solver\nError: " + str(e))
                    return
                
            elif lasolverIndex == 8:
                try:
                    import daetools.pyLapack as pyLapack
                    lasolver = pyLapack.daeCreateLapackSolver()
                    self.daesolver.SetLASolver(lasolver)
                except Exception, e:
                    QtGui.QMessageBox.warning(None, "daeSimulator", "Cannot create Lapack LA solver\nError: " + str(e))
                    return
                
            else:
                raise RuntimeError("Unsupported LA Solver selected")
            
            self.ui.RunButton.setEnabled(False)
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)

            self.simulation.Initialize(self.daesolver, self.datareporter, self.log)
            self.simulation.SolveInitial()
            self.simulation.Run()
            
        except Exception, error:
            self.ui.textEdit.append(str(error))
            if self.ui.textEdit.isVisible() == True:
                self.ui.textEdit.update()
            self.app.processEvents()
