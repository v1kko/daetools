"""********************************************************************************
                             daeSimulator.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""
import os, platform, sys, tempfile, traceback, webbrowser, math
from daetools.pyDAE import *
from time import ctime, time, localtime, strftime, struct_time
from PyQt4 import QtCore, QtGui
from Simulator_ui import Ui_SimulatorDialog

try:
    from daetools.pyDAE.WebViewDialog import WebView
except Exception as e:
    print 'Cannot load WebView module\n Error: ', str(e)

class daeTextEditLog(daeBaseLog):
    def __init__(self, TextEdit, ProgressBar, ProgressLabel, App):
        daeBaseLog.__init__(self)
        self.TextEdit         = TextEdit
        self.ProgressBar      = ProgressBar
        self.ProgressLabel    = ProgressLabel
        self.App              = App
        self.time             = time()

    def SetProgress(self, progress):
        value = int(progress)
        if progress > 100:
            value = 100
        elif progress < 0:
            value = 0
        self.ProgressBar.setValue(value)
        now = time()
        if value < 100 and value > 0:
            s = 100.0 * (now - self.time) / (value + 1E-7) - (now - self.time)
        else:
            s = 0.0
        
        days  = int(math.floor(s / 86400))
        left  = s - days * 86400
        hours = int(math.floor(left / 3600))
        left  = left - hours * 3600
        mins  = int(math.floor(left / 60))
        secs  = float(left - mins * 60)
        eta = 'ETA: [{0:0>2d}d {1:0>2d}h {2:0>2d}m {3:0>4.1f}s]'.format(days, hours, mins, secs)
        self.ProgressLabel.setText(eta)
        #self.App.processEvents()
    
    def Message(self, message, severity):
        self.TextEdit.append(self.IndentString + message)
        if self.TextEdit.isVisible() == True:
            self.TextEdit.update()
        self.App.processEvents()

class daeSimulator(QtGui.QDialog):
    laSundialsLU        = 0
    laAmesos_Klu        = 1
    laAmesos_Superlu    = 2
    laAmesos_Umfpack    = 3
    laAmesos_Lapack     = 4
    laAztecOO           = 5
    laIntelPardiso      = 6
    laIntelMKL          = 7
    laAmdACML           = 8
    laLapack            = 9
    laMagmaLapack       = 10
    laSuperLU           = 11
    laSuperLU_MT        = 12
    laSuperLU_CUDA      = 13
    laCUSP              = 14

    nlpIPOPT            = 0
    nlpNLOPT            = 1
    nlpBONMIN           = 2

    def __init__(self, app, **kwargs):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulatorDialog()
        self.ui.setupUi(self)

        font = QtGui.QFont()
        font.setPointSize(9)
        if platform.system() == 'Linux':
            font.setFamily("Monospace")
        else:
            font.setFamily("Courier New")
        self.ui.textEdit.setFont(font)
        self.setWindowTitle("DAE Tools Simulator v" + daeVersion(True))

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
            if self.daesolver == None:
                self.daesolver = daeIDAS()

            lasolverIndex    = -1
            minlpsolverIndex = -1

            # If nlpsolver is not sent then choose it from the selection
            if (self.nlpsolver == None):
                minlpsolverIndex = self.ui.MINLPSolverComboBox.currentIndex()

            # If lasolver is not sent then create it based on the selection
            if (self.lasolver == None):
                lasolverIndex = self.ui.LASolverComboBox.currentIndex()

                if lasolverIndex == self.laSundialsLU:
                    pass

                elif lasolverIndex == self.laAmesos_Klu:
                    try:
                        from daetools.solvers import pyTrilinos
                        self.lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Klu", "")
                    except Exception as e:
                        self.showMessage("Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laAmesos_Superlu:
                    try:
                        from daetools.solvers import pyTrilinos
                        self.lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Superlu", "")
                    except Exception as e:
                        self.showMessage("Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laAmesos_Umfpack:
                    try:
                        from daetools.solvers import pyTrilinos
                        self.lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Umfpack", "")
                    except Exception as e:
                        self.showMessage("Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laAmesos_Lapack:
                    try:
                        from daetools.solvers import pyTrilinos
                        self.lasolver = pyTrilinos.daeCreateTrilinosSolver("Amesos_Lapack", "")
                    except Exception as e:
                        self.showMessage("Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laAztecOO:
                    try:
                        from daetools.solvers import pyTrilinos
                        self.lasolver = pyTrilinos.daeCreateTrilinosSolver("AztecOO", "ILUT")
                    except Exception as e:
                        self.showMessage("Cannot create TrilinosAmesos LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laIntelPardiso:
                    try:
                        from daetools.solvers import pyIntelPardiso
                        self.lasolver = pyIntelPardiso.daeCreateIntelPardisoSolver()
                    except Exception as e:
                        self.showMessage("Cannot create IntelPardiso LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laIntelMKL:
                    try:
                        from daetools.solvers import pyIntelMKL
                        self.lasolver = pyIntelMKL.daeCreateLapackSolver()
                    except Exception as e:
                        self.showMessage("Cannot create IntelMKL LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laAmdACML:
                    try:
                        from daetools.solvers import pyAmdACML
                        self.lasolver = pyAmdACML.daeCreateLapackSolver()
                    except Exception as e:
                        self.showMessage("Cannot create AmdACML LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laLapack:
                    try:
                        from daetools.solvers import pyLapack
                        self.lasolver = pyLapack.daeCreateLapackSolver()
                    except Exception as e:
                        self.showMessage("Cannot create Lapack LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laMagmaLapack:
                    try:
                        from daetools.solvers import pyMagma
                        self.lasolver = pyMagma.daeCreateLapackSolver()
                    except Exception as e:
                        self.showMessage("Cannot create Magma Lapack LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laSuperLU:
                    try:
                        from daetools.solvers import pySuperLU
                        self.lasolver = pySuperLU.daeCreateSuperLUSolver()
                    except Exception as e:
                        self.showMessage("Cannot create SuperLU LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laSuperLU_MT:
                    try:
                        from daetools.solvers import pySuperLU_MT
                        self.lasolver = pySuperLU_MT.daeCreateSuperLUSolver()
                    except Exception as e:
                        self.showMessage("Cannot create SuperLU_MT LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laSuperLU_CUDA:
                    try:
                        from daetools.solvers import pySuperLU_CUDA
                        self.lasolver = pySuperLU_CUDA.daeCreateSuperLUSolver()
                    except Exception as e:
                        self.showMessage("Cannot create SuperLU_CUDA LA solver\nError: " + str(e))
                        return

                elif lasolverIndex == self.laCUSP:
                    try:
                        from daetools.solvers import pyCUSP
                        self.lasolver = pyCUSP.daeCreateCUSPSolver()
                    except Exception as e:
                        self.showMessage("Cannot create CUSP LA solver\nError: " + str(e))
                        return

                else:
                    raise RuntimeError("Unsupported LA Solver selected")

            self.ui.RunButton.setEnabled(False)
            self.ui.MINLPSolverComboBox.setEnabled(False)
            self.ui.DAESolverComboBox.setEnabled(False)
            self.ui.LASolverComboBox.setEnabled(False)
            self.ui.DataReporterTCPIPAddressLineEdit.setEnabled(False)
            self.ui.ReportingIntervalDoubleSpinBox.setEnabled(False)
            self.ui.TimeHorizonDoubleSpinBox.setEnabled(False)
            self.ui.MatrixButton.setEnabled(False)
            self.ui.ExportButton.setEnabled(False)
            if lasolverIndex in [self.laAmesos_Klu,
                                 self.laAmesos_Superlu,
                                 self.laAmesos_Umfpack,
                                 self.laAztecOO,
                                 self.laIntelPardiso,
                                 self.laSuperLU,
                                 self.laSuperLU_MT,
                                 self.laSuperLU_CUDA,
                                 self.laCUSP]:
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
                    self.run_after_simulation_end_fn(simulation, log)

            else:
                # If nlpsolver is not sent then create it based on the selection
                if(self.nlpsolver == None):
                    if minlpsolverIndex == self.nlpIPOPT:
                        try:
                            from daetools.solvers import pyIPOPT
                            self.nlpsolver = pyIPOPT.daeIPOPT()
                        except Exception as e:
                            self.showMessage("Cannot create IPOPT NLP solver\nError: " + str(e))
                            return

                    elif minlpsolverIndex == self.nlpNLOPT:
                        try:
                            from daetools.solvers import pyNLOPT
                            from PyQt4 import QtCore, QtGui
                            algorithms = ['NLOPT_GN_DIRECT','NLOPT_GN_DIRECT_L','NLOPT_GN_DIRECT_L_RAND','NLOPT_GN_DIRECT_NOSCAL','NLOPT_GN_DIRECT_L_NOSCAL',
                                        'NLOPT_GN_DIRECT_L_RAND_NOSCAL','NLOPT_GN_ORIG_DIRECT','NLOPT_GN_ORIG_DIRECT_L','NLOPT_GD_STOGO','NLOPT_GD_STOGO_RAND',
                                        'NLOPT_LD_LBFGS_NOCEDAL','NLOPT_LD_LBFGS','NLOPT_LN_PRAXIS','NLOPT_LD_VAR1','NLOPT_LD_VAR2','NLOPT_LD_TNEWTON',
                                        'NLOPT_LD_TNEWTON_RESTART','NLOPT_LD_TNEWTON_PRECOND','NLOPT_LD_TNEWTON_PRECOND_RESTART','NLOPT_GN_CRS2_LM',
                                        'NLOPT_GN_MLSL','NLOPT_GD_MLSL','NLOPT_GN_MLSL_LDS','NLOPT_GD_MLSL_LDS','NLOPT_LD_MMA','NLOPT_LN_COBYLA',
                                        'NLOPT_LN_NEWUOA','NLOPT_LN_NEWUOA_BOUND','NLOPT_LN_NELDERMEAD','NLOPT_LN_SBPLX','NLOPT_LN_AUGLAG','NLOPT_LD_AUGLAG',
                                        'NLOPT_LN_AUGLAG_EQ','NLOPT_LD_AUGLAG_EQ','NLOPT_LN_BOBYQA','NLOPT_GN_ISRES',
                                        'NLOPT_AUGLAG','NLOPT_AUGLAG_EQ','NLOPT_G_MLSL','NLOPT_G_MLSL_LDS','NLOPT_LD_SLSQP']
                            # Show the input box to choose the algorithm (the default is len(algorithms)-1 that is: NLOPT_LD_SLSQP)
                            algorithm, ok = QtGui.QInputDialog.getItem(None, "NLOPT Algorithm", "Choose the NLOPT algorithm:", algorithms, len(algorithms)-1, False)
                            if not ok:
                                algorithm = 'NLOPT_LD_SLSQP'
                            self.nlpsolver = pyNLOPT.daeNLOPT(str(algorithm))
                        except Exception as e:
                            self.showMessage("Cannot create NLOPT NLP solver\nError: " + str(e))
                            return

                    elif minlpsolverIndex == self.nlpBONMIN:
                        try:
                            from daetools.solvers import pyBONMIN
                            self.nlpsolver = pyBONMIN.daeBONMIN()
                        except Exception as e:
                            self.showMessage("Cannot create BONMIN MINLP solver\nError: " + str(e))
                            return

                    else:
                        raise RuntimeError("Unsupported (MI)NLP Solver selected")

                self.optimization.Initialize(self.simulation, self.nlpsolver, self.daesolver, self.datareporter, self.log)
                if(self.nlpsolver_setoptions_fn):
                    self.nlpsolver_setoptions_fn(self.nlpsolver)
                self.optimization.Run()
                if self.run_after_simulation_end_fn:
                    self.run_after_simulation_end_fn(simulation, log)

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
                wv = WebView(url)
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
