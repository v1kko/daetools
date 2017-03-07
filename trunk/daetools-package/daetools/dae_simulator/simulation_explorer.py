"""
***********************************************************************************
                          simulation_explorer.py
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
***********************************************************************************
"""
import sys, tempfile, numpy, json, traceback
from time import localtime, strftime
from os.path import join, dirname
from daetools.pyDAE import *
from PyQt5 import QtCore, QtGui, QtWidgets

from .simulation_explorer_ui import Ui_SimulationExplorer
from .simulation_inspector import daeSimulationInspector
from .exception_dlg_ui import Ui_ExceptionDialog
from .tree_item import *
from . import auxiliary

images_dir = join(dirname(__file__), 'images')

def simulate(simulation, **kwargs):
    qt_app = kwargs.get('qt_app', None)
    if not qt_app:
        qt_app = QtWidgets.QApplication(sys.argv)
    explorer = daeSimulationExplorer(qt_app, simulation = simulation, **kwargs)
    explorer.exec_()
    return explorer

def optimize(optimization, **kwargs):
    qt_app = kwargs.get('qt_app', None)
    if not qt_app:
        qt_app = QtWidgets.QApplication(sys.argv)
    explorer = daeSimulationExplorer(qt_app, simulation = simulation, optimization = optimization, **kwargs)
    explorer.exec_()
    return explorer

class daeExceptionDialog(QtWidgets.QDialog):
    def __init__(self, parent, e, tb):
        QtWidgets.QDialog.__init__(self, parent)
        self._ui = Ui_ExceptionDialog()
        self._ui.setupUi(self)
        
        messages = traceback.format_tb(tb)
        msg = '\n'.join(messages)
        self._ui.exceptionLabel.setText(str(e))
        self._ui.tracebackEdit.setPlainText(msg)
        
        self.setWindowTitle('Exception raised')
        
class daeSimulationExplorer(QtWidgets.QDialog):
    def __init__(self, qt_app, simulation, **kwargs):
        QtWidgets.QDialog.__init__(self)
        self._ui = Ui_SimulationExplorer()
        self._ui.setupUi(self)

        """
        #self._simulation                  = kwargs.get('simulation',                 None)
        self._optimization                = kwargs.get('optimization',               None)
        self._datareporter                = kwargs.get('datareporter',               None)
        self._lasolver                    = kwargs.get('lasolver',                   None)
        self._nlpsolver                   = kwargs.get('nlpsolver',                  None)
        self._log                         = kwargs.get('log',                        None)
        #self._nlpsolver_setoptions_fn     = kwargs.get('nlpsolver_setoptions_fn',    None)
        #self._lasolver_setoptions_fn      = kwargs.get('lasolver_setoptions_fn',     None)
        #self._run_after_simulation_end_fn = kwargs.get('run_after_simulation_end_fn',None)
        """
        
        if not qt_app:
            raise RuntimeError('qt_app object must not be None')
        if not simulation:
            raise RuntimeError('simulation object must not be None')

        self.setWindowTitle("DAE Tools Simulation Explorer v%s - [%s]" % (daeVersion(True), simulation.m.GetStrippedName()))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self._qt_app = qt_app
        self._initialize(simulation)

        # First populate trees and set values. Only after that connect signals to slots (otherwise the event handlers might be triggered in the process)
        self._ui.buttonCancel.clicked.connect(self._slotCancel)
        self._ui.buttonUpdateSimulationAndClose.clicked.connect(self._slotUpdateSimulationAndClose)
        self._ui.buttonGenerateCode.clicked.connect(self._slotGenerateCode)
        self._ui.buttonSaveRuntimeSettingsAsJSON.clicked.connect(self._slotSaveRuntimeSettingsAsJSON)        
        self._ui.treeParameters.itemSelectionChanged.connect(self._slotParameterTreeItemSelectionChanged)
        self._ui.treeOutputVariables.itemChanged.connect(self._slotOutputVariablesTreeItemChanged)
        self._ui.treeSTNs.itemChanged.connect(self._slotSTNsTreeItemChanged)
        self._ui.treeDomains.itemSelectionChanged.connect(self._slotDomainsTreeItemChanged)
        self._ui.treeDOFs.itemSelectionChanged.connect(self._slotDOFsTreeItemChanged)
        self._ui.treeInitialConditions.itemSelectionChanged.connect(self._slotInitialConditionsTreeItemChanged)
        #self._ui.treeSTNs.itemSelectionChanged.connect(self._slotSTNsTreeItemChanged)

    def updateSimulation(self, verbose = False):
        # Update runtime data
        self._simulation.TimeHorizon                  = float(self._ui.timeHorizonEdit.text())
        self._simulation.ReportingInterval            = float(self._ui.reportingIntervalEdit.text())
        self._simulation.DAESolver.RelativeTolerance  = float(self._ui.relativeToleranceEdit.text())
        
        # Update Domains
        if verbose:
            print('Update Domains...')
        for canonicalName, (domain, item) in list(self._inspector.domains.items()):
            val_ = item.getValue()
            
            if item.type == eStructuredGrid:                
                points = item.getValue()
                if verbose:
                    print(('    Updating domain %s points ...' % canonicalName))
                    print(('        from: %s' % domain.Points))
                    print(('          to: %s' % points))
                domain.Points = points
            
            elif item.type == eArray:
                pass
            
            elif item.type == eUnstructuredGrid:
                pass
                
        # Update Parameters
        if verbose:
            print('Update Parameters...')
        for canonicalName, (parameter, item) in list(self._inspector.parameters.items()):
            val_, units_ = item.getValue()
            if isinstance(val_, list):
                q = numpy.array(val_, dtype = object)
                q = q * units_
                if verbose:
                    print(('    Updating parameter %s ...' % canonicalName))
                    print(('        from: %s %s' % (parameter.npyValues, parameter.Units)))
                    print(('          to: %s' % q))
                parameter.SetValues(q)
            else:
                q = val_ * units_
                if verbose:
                    print(('    Updating parameter %s ...' % canonicalName))
                    print(('        from: %s' % parameter.GetQuantity()))
                    print(('          to: %s' % q))
                parameter.SetValue(q)
                
        # Update DegreesOfFreedom
        if verbose:
            print('Update DegreesOfFreedom...')
        for canonicalName, (variable, item) in list(self._inspector.dofs.items()):
            val_, units_ = item.getValue()
            if isinstance(val_, list):
                q = numpy.array(val_, dtype = object)
                # Some items may be None and the operator * will not work, therefore first create a flat view 
                # and then multiply each non-null item by units to get a quantity
                c = q.view()
                c.shape = q.size
                for i in range(c.size):
                    if c[i] != None:
                        c[i] *= units_
                if verbose:
                    print(('    Reassigning %s ...' % canonicalName))
                    print(('        from: %s %s' % (variable.npyValues, variable.VariableType.Units)))
                    print(('          to: %s' % q))
                variable.ReAssignValues(q)
            else:
                q = val_ * units_
                if verbose:
                    print(('    Reassigning %s ...' % canonicalName))
                    print(('        from: %s' % variable.GetQuantity()))
                    print(('          to: %s' % q))
                variable.ReAssignValue(q)
                
        # Update InitialConditions
        if verbose:
            print('Update InitialConditions...')
        for canonicalName, (variable, item) in list(self._inspector.initial_conditions.items()):
            val_, units_ = item.getValue()
            if isinstance(val_, list):
                q = numpy.array(val_, dtype = object)
                # Some items may be None and the operator * will not work, therefore first create a flat view 
                # and then multiply each non-null item by units to get a quantity
                c = q.view()
                c.shape = q.size
                for i in range(c.size):
                    if c[i] != None:
                        c[i] *= units_
                if verbose:
                    print(('    Resetting initial conditions for %s ...' % canonicalName))
                    print(('        from: %s %s' % (variable.npyValues, variable.VariableType.Units)))
                    print(('          to: %s' % q))
                variable.ReSetInitialConditions(q)
            else:
                q = val_ * units_
                if verbose:
                    print(('    Resetting initial conditions for %s ...' % canonicalName))
                    print(('        from: %s' % variable.GetQuantity()))
                    print(('          to: %s' % q))
                variable.ReSetInitialCondition(q)
            
        # Update Outputs
        if verbose:
            print('Update Outputs...')
        for canonicalName, (variable, item) in list(self._inspector.output_variables.items()):
            if verbose:
                print(('    Updating the ReportingOn flag for %s ...' % canonicalName))
                print(('        from: %s' % variable.ReportingOn))
                print(('          to: %s' % item.getValue()))
            if item.getValue():
                variable.ReportingOn = True
            else:
                variable.ReportingOn = False
                
        # Update STNs
        if verbose:
            print('Update STNs...')
        for canonicalName, (stn, lstates) in list(self._inspector.stns.items()):
            # Iterate over states and detect which of them is checked
            for state, item in lstates:
                if item.getValue():
                    activeState = state.Name
                    if verbose:
                        print(('    Changing the active state for %s:' % canonicalName))
                        print(('        from: %s' % stn.ActiveState))
                        print(('          to: %s' % activeState))
                    stn.ActiveState = activeState
                    break
        
        if verbose:
            print('Reinitializing the simulation...')
        self._simulation.Reinitialize()
        self._simulation.ReportData(self._simulation.CurrentTime)
        if verbose:
            print('Done!')
        
    def generateCode(self, language):
        if not language in ['c99', 'c++ (MPI)', 'Modelica', 'gPROMS', 'FMI (Co-Simulation)']:
            return
        
        options = QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        
        if language == 'c99':
            from daetools.code_generators.c99 import daeCodeGenerator_c99
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Code generator: %s" % language, 
                                                                       '', 
                                                                       options)
            if not directory:
                return
            cg = daeCodeGenerator_c99()
            cg.generateSimulation(self._simulation, str(directory))
        
        elif language == 'c++ (MPI)':
            nproc, ok = QtWidgets.QInputDialog.getInt(self, "Code generator", "Set the # of MPI processes:", 4, min=2)
            if not ok:
                return

            from daetools.code_generators.cxx_mpi import daeCodeGenerator_cxx_mpi
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Code generator: %s" % language,
                                                                       '',
                                                                       options)
            if not directory:
                return
            cg = daeCodeGenerator_cxx_mpi()
            cg.generateSimulation(self._simulation, str(directory), nproc)

        elif language == 'Modelica':
            from daetools.code_generators.modelica import daeCodeGenerator_Modelica
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Code generator: %s" % language,
                                                                       '',
                                                                       options)
            if not directory:
                return
            cg = daeCodeGenerator_Modelica()
            cg.generateSimulation(self._simulation, str(directory))
        
        elif language == 'gPROMS':
            from daetools.code_generators.gproms import daeCodeGenerator_gPROMS
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Code generator: %s" % language,
                                                                   '',
                                                                   options)
            if not directory:
                return
            cg = daeCodeGenerator_gPROMS()
            cg.generateSimulation(self._simulation, str(directory))

        elif language == 'FMI (Co-Simulation)':
            from daetools.code_generators.fmi import daeCodeGenerator_FMI
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Code generator: %s" % language, 
                                                                   "",
                                                                   options)
            if not directory:
                return

            qt_file, ok = QtWidgets.QFileDialog.getOpenFileName(self, "Select python file to pack in the .FMU file",
                                                                "",
                                                                "Python files (*.py *.*)")
            if not ok:
                return
            py_file = str(qt_file)

            qt_callable_name, ok = QtWidgets.QInputDialog.getText(self, "FMU Code generator", "Insert the name of python callable object:")
            if not ok:
                return
            callable_name = str(qt_callable_name)

            qt_arguments, ok = QtWidgets.QInputDialog.getText(self, "FMU Code generator", "Insert the arguments for the python callable object:")
            if not ok:
                return
            arguments = str(qt_arguments)

            cg = daeCodeGenerator_FMI()
            cg.generateSimulation(self._simulation, str(directory), py_file, callable_name, arguments, [])
            
        QtWidgets.QMessageBox.information(self, "Code generator: %s" % language, 'Code generated successfuly!')

    @staticmethod
    def generateHTMLForm(simulation, htmlOutputFile, find_files_dir = '.'):
        # Create application in case there is not one already created
        if not QtWidgets.QApplication.instance():
            app_ = QtWidgets.QApplication(sys.argv)

        _inspector = daeSimulationInspector(simulation)

        from .html_form import css_styles, html_template
        of = open(htmlOutputFile, 'w')
        html_content = _inspector._generateHTMLForm()
        #logo = open(os.path.join(find_files_dir, 'logo.png'), 'rb').read()
        #logo_data = logo.encode("base64")
        dictOptions = {'css_style' : css_styles,
                       'name' : simulation.m.GetStrippedName(),
                       'content' : html_content
                      }
        html = html_template % dictOptions
        of.write(html)
        of.close()
            
    @property
    def jsonRuntimeSettings(self):
        return json.dumps(self.runtimeSettings, indent = 4, sort_keys = True)
    
    @property
    def runtimeSettings(self):
        self._processInputs()
        return self._runtimeSettings
    
    @staticmethod
    def saveJSONSettings(simulation, filename):
        jsonSettings = daeSimulationExplorer.generateJSONSettings(simulation)
        f = open(filename, 'w')
        f.write(jsonSettings)
        f.close()
        
    @staticmethod
    def generateJSONSettings(simulation):
        try:
            # Create application in case there is not one already created
            if not QtWidgets.QApplication.instance():
                app_ = QtWidgets.QApplication(sys.argv)

            _inspector = daeSimulationInspector(simulation)

            _runtimeSettings = {}
            _runtimeSettings['Name']                  = simulation.m.Name
            _runtimeSettings['TimeHorizon']           = simulation.TimeHorizon
            _runtimeSettings['ReportingInterval']     = simulation.ReportingInterval
            _runtimeSettings['RelativeTolerance']     = simulation.DAESolver.RelativeTolerance
            _runtimeSettings['DAESolver']             = {'Name' : None}
            _runtimeSettings['LASolver']              = {'Name' : None}
            _runtimeSettings['DataReporter']          = {'Name' : None}
            _runtimeSettings['Log']                   = {'Name' : None}
            if simulation.DAESolver:
                _runtimeSettings['DAESolver']    = {'Name'    : simulation.DAESolver.Name,
                                                    'Options' : {}}
            if simulation.DAESolver.LASolver:
                _runtimeSettings['LASolver']     = {'Name'    : simulation.DAESolver.LASolver.Name,
                                                    'Options' : {}}
            if simulation.DataReporter:
                _runtimeSettings['DataReporter'] = {'Name'          : simulation.DataReporter.Name,
                                                    'ConnectString' : simulation.DataReporter.ConnectString,
                                                    'ProcessName'   : simulation.DataReporter.ProcessName,
                                                    'Options'       : {}}
            if simulation.Log:
                _runtimeSettings['Log']          = {'Name'    : simulation.Log.Name,
                                                    'Options' : {}}

            _runtimeSettings['QuazySteadyState']       = True if simulation.InitialConditionMode == eQuasySteadyState else False
            _runtimeSettings['CalculateSensitivities'] = simulation.CalculateSensitivities

            _runtimeSettings['Parameters']             = _inspector.treeParameters.toDictionary()
            _runtimeSettings['Domains']                = _inspector.treeDomains.toDictionary()
            if not _runtimeSettings['QuazySteadyState']:
                _runtimeSettings['InitialConditions']  = _inspector.treeInitialConditions.toDictionary()
            _runtimeSettings['DOFs']                   = _inspector.treeDOFs.toDictionary()
            _runtimeSettings['STNs']                   = _inspector.treeStates.toDictionary()
            _runtimeSettings['Outputs']                = _inspector.treeOutputVariables.toDictionary()

            return json.dumps(_runtimeSettings, indent = 4, sort_keys = True)

        except Exception as e:
            print('Exception in generateJSONSettings: %s' % str(e))
            
    ############################################################################
    #                   Implementation (private methods)
    ############################################################################
    def _initialize(self, simulation):
        self._simulation = simulation
        if not self._simulation:
            raise RuntimeError('simulation object must not be None')

        self._inspector          = daeSimulationInspector(self._simulation)
        self._runtimeSettings    = None
        self._simulationName     = daeGetStrippedName(self._simulation.m.Name) + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        self._timeHorizon        = self._simulation.TimeHorizon
        self._reportingInterval  = self._simulation.ReportingInterval
        self._relativeTolerance  = self._simulation.DAESolver.RelativeTolerance
        if self._simulation.InitialConditionMode == eQuasySteadyState:
            self._quazySteadyState = True
        else:
            self._quazySteadyState = False

        self._daesolver    = self._simulation.DAESolver
        self._lasolver     = self._simulation.DAESolver.LASolver
        self._datareporter = self._simulation.DataReporter
        self._log          = self._simulation.Log

        d_validator = QtGui.QDoubleValidator(self)

        self._ui.simulationNameEdit.setText(self._simulationName)
        self._ui.timeHorizonEdit.setValidator(d_validator)
        self._ui.timeHorizonEdit.setText(str(self._timeHorizon))
        self._ui.reportingIntervalEdit.setValidator(d_validator)
        self._ui.reportingIntervalEdit.setText(str(self._reportingInterval))
        self._ui.relativeToleranceEdit.setValidator(d_validator)
        self._ui.relativeToleranceEdit.setText(str(self._relativeTolerance))
        if self._quazySteadyState:
            self._ui.quazySteadyStateCheckBox.setCheckState(Qt.Checked)
            self._ui.tab_InitialConditions.setEnabled(False)
        else:
            self._ui.quazySteadyStateCheckBox.setCheckState(Qt.Unchecked)
            self._ui.tab_InitialConditions.setEnabled(True)

        self._available_la_solvers     = auxiliary.getAvailableLASolvers()
        self._available_nlp_solvers    = auxiliary.getAvailableNLPSolvers()
        self._available_data_reporters = auxiliary.getAvailableDataReporters()
        self._available_logs           = auxiliary.getAvailableLogs()

        self._ui.daesolverComboBox.addItem("Sundials IDAS")
        for i, la in enumerate(self._available_la_solvers):
            self._ui.lasolverComboBox.addItem(la[0], userData = la[1]) #QtCore.QVariant(la[1]))
            self._ui.lasolverComboBox.setItemData(i, la[2], QtCore.Qt.ToolTipRole)
        for i, dr in enumerate(self._available_data_reporters):
            self._ui.datareporterComboBox.addItem(dr[0], userData = dr[1]) #QtCore.QVariant(dr[1]))
            self._ui.datareporterComboBox.setItemData(i, dr[2], QtCore.Qt.ToolTipRole)
        for i, log in enumerate(self._available_logs):
            self._ui.logComboBox.addItem(log[0], userData = log[1]) #QtCore.QVariant(log[1]))
            self._ui.logComboBox.setItemData(i, dr[2], QtCore.Qt.ToolTipRole)

        # Nota bene:
        # DAE/LA solver, datareporter and log are disabled, since they can't be changed anyway
        
        # DAE Solvers
        self._ui.daesolverComboBox.setEnabled(False)

        # LA Solvers
        if not self._lasolver:
            self._ui.lasolverComboBox.setEnabled(False)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self._ui.lasolverComboBox.clear()
            self._ui.lasolverComboBox.addItem(self._lasolver.Name)
            self._ui.lasolverComboBox.setEnabled(False)

        # Logs
        if not self._log:
            self._ui.logComboBox.setEnabled(False)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self._ui.logComboBox.clear()
            self._ui.logComboBox.addItem(self._log.Name)
            self._ui.logComboBox.setEnabled(False)

        # DataReporters
        if not self._datareporter:
            self._ui.datareporterComboBox.setEnabled(False)
        else:
            # If data reporter has been sent then clear and disable DataReporters combo box
            self._ui.datareporterComboBox.clear()
            self._ui.datareporterComboBox.addItem(self._datareporter.Name)
            self._ui.datareporterComboBox.setEnabled(False)

        self._currentParameterItem        = None
        self._currentStateTransitionItem  = None
        self._currentDomainItem           = None
        self._currentDOFItem              = None
        self._currentInitialConditionItem = None

        # treeStates-related data
        self._allowUnchecking                 = False
        self._treeStates_itemChanged_disabled = False

        # Clear all trees
        self._ui.treeParameters.clear()
        self._ui.treeOutputVariables.clear()
        self._ui.treeSTNs.clear()
        self._ui.treeDomains.clear()
        self._ui.treeDOFs.clear()
        self._ui.treeInitialConditions.clear()

        # Populate the trees
        addItemsToTree(self._ui.treeParameters,        self._ui.treeParameters,        self._inspector.treeParameters)
        addItemsToTree(self._ui.treeOutputVariables,   self._ui.treeOutputVariables,   self._inspector.treeOutputVariables)
        addItemsToTree(self._ui.treeSTNs,              self._ui.treeSTNs,              self._inspector.treeStates)
        addItemsToTree(self._ui.treeDomains,           self._ui.treeDomains,           self._inspector.treeDomains)
        addItemsToTree(self._ui.treeDOFs,              self._ui.treeDOFs,              self._inspector.treeDOFs)
        addItemsToTree(self._ui.treeInitialConditions, self._ui.treeInitialConditions, self._inspector.treeInitialConditions)

        self._ui.treeDomains.expandAll()
        self._ui.treeDomains.resizeColumnToContents(0)
        self._ui.treeDomains.setSortingEnabled(True)
        self._ui.treeDomains.sortItems(0, QtCore.Qt.AscendingOrder)

        self._ui.treeParameters.expandAll()
        self._ui.treeParameters.resizeColumnToContents(0)
        self._ui.treeParameters.setSortingEnabled(True)
        self._ui.treeParameters.sortItems(0, QtCore.Qt.AscendingOrder)

        self._ui.treeInitialConditions.expandAll()
        self._ui.treeInitialConditions.resizeColumnToContents(0)
        self._ui.treeInitialConditions.setSortingEnabled(True)
        self._ui.treeInitialConditions.sortItems(0, QtCore.Qt.AscendingOrder)

        self._ui.treeDOFs.expandAll()
        self._ui.treeDOFs.resizeColumnToContents(0)
        self._ui.treeDOFs.setSortingEnabled(True)
        self._ui.treeDOFs.sortItems(0, QtCore.Qt.AscendingOrder)

        self._ui.treeSTNs.expandAll()
        self._ui.treeSTNs.resizeColumnToContents(0)
        self._ui.treeSTNs.setSortingEnabled(True)
        self._ui.treeSTNs.sortItems(0, QtCore.Qt.AscendingOrder)

        self._ui.treeOutputVariables.expandAll()
        self._ui.treeOutputVariables.resizeColumnToContents(0)
        self._ui.treeOutputVariables.setSortingEnabled(True)
        self._ui.treeOutputVariables.sortItems(0, QtCore.Qt.AscendingOrder)

    def _processInputs(self):
        self._simulationName    = str(self._ui.simulationNameEdit.text())
        self._timeHorizon       = float(self._ui.timeHorizonEdit.text())
        self._reportingInterval = float(self._ui.reportingIntervalEdit.text())
        self._relativeTolerance = float(self._ui.relativeToleranceEdit.text())
        
        if not self._daesolver:
            self._daesolver = daeIDAS()

        # If lasolver is not sent then create it based on the selection
        if self._lasolver == None and len(self._available_la_solvers) > 0:
            _index = self._ui.lasolverComboBox.itemData(self._ui.lasolverComboBox.currentIndex())
            if isinstance(_index, QtCore.QVariant):
                lasolverIndex = _index.toInt()[0]
            else:
                if _index != None and _index >= 0:
                    lasolverIndex = int(_index)
            self._lasolver = auxiliary.createLASolver(lasolverIndex)
            if self._lasolver: # Can be None if SundialsLU has been selected
                self._daesolver.SetLASolver(self._lasolver)

        if not self._datareporter:
            _index  = self._ui.datareporterComboBox.itemData(self._ui.datareporterComboBox.currentIndex())
            drIndex = -1
            if isinstance(_index, QtCore.QVariant):
                drIndex = _index.toInt()[0]
            else:
                if _index != None and _index >= 0:
                    drIndex = int(_index)
            self._datareporter = auxiliary.createDataReporter(drIndex)
            
        if not self._log:
            _index   = self._ui.logComboBox.itemData(self._ui.logComboBox.currentIndex())
            logIndex = -1
            if isinstance(_index, QtCore.QVariant):
                logIndex = _index.toInt()[0]
            else:
                if _index != None and _index >= 0:
                    logIndex = int(_index)
            self._log = auxiliary.createLog(logIndex)
        
        self._runtimeSettings = {}
        self._runtimeSettings['Name']                  = self._simulationName
        self._runtimeSettings['TimeHorizon']           = self._timeHorizon
        self._runtimeSettings['ReportingInterval']     = self._reportingInterval
        self._runtimeSettings['RelativeTolerance']     = self._relativeTolerance
        self._runtimeSettings['DAESolver']             = {'Name' : None}
        self._runtimeSettings['LASolver']              = {'Name' : None}
        self._runtimeSettings['DataReporter']          = {'Name' : None}
        self._runtimeSettings['Log']                   = {'Name' : None}
        if self._simulation.DAESolver:
            self._runtimeSettings['DAESolver']    = {'Name'    : self._simulation.DAESolver.Name,
                                                     'Options' : {}}
        if self._simulation.DAESolver.LASolver:
            self._runtimeSettings['LASolver']     = {'Name'    : self._simulation.DAESolver.LASolver.Name,
                                                     'Options' : {}}
        if self._simulation.DataReporter:
            self._runtimeSettings['DataReporter'] = {'Name'          : self._simulation.DataReporter.Name,
                                                     'ConnectString' : self._simulation.DataReporter.ConnectString,
                                                     'ProcessName'   : self._simulation.DataReporter.ProcessName,
                                                     'Options'       : {}}
        if self._simulation.Log:
            self._runtimeSettings['Log']          = {'Name'    : self._simulation.Log.Name,
                                                     'Options' : {}}

        self._runtimeSettings['QuazySteadyState']      = self._quazySteadyState
        self._runtimeSettings['CalculateSensitivities']= self._simulation.CalculateSensitivities
        
        self._runtimeSettings['Parameters']            = self._inspector.treeParameters.toDictionary()
        self._runtimeSettings['Domains']               = self._inspector.treeDomains.toDictionary()
        if not self._quazySteadyState:
            self._runtimeSettings['InitialConditions'] = self._inspector.treeInitialConditions.toDictionary()
        self._runtimeSettings['DOFs']                  = self._inspector.treeDOFs.toDictionary()
        self._runtimeSettings['STNs']                  = self._inspector.treeStates.toDictionary()
        self._runtimeSettings['Outputs']               = self._inspector.treeOutputVariables.toDictionary()

    def _slotCancel(self):
        self.done(QtWidgets.QDialog.Rejected)

    def reject(self):
        QtWidgets.QDialog.reject(self)
        
    def _slotUpdateSimulationAndClose(self):
        try:
            cfg = daeGetConfig()
            printInfo = cfg.GetBoolean('daetools.core.printInfo', False)
            self.updateSimulation(verbose = True)
            self.done(QtWidgets.QDialog.Accepted)
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            dlg = daeExceptionDialog(self, e, exc_traceback)
            dlg.exec_()

    def _slotSaveRuntimeSettingsAsJSON(self):
        try:
            name = self._simulation.m.GetStrippedName() + ".json"
            jsonFilename, ok = QtWidgets.QFileDialog.getSaveFileName(self, "Save Runtime Settings As JSON File", name, "JSON Files (*.json)")
            if not ok:
                return
            
            self._processInputs()
            f = open(str(jsonFilename), 'w')
            f.write(self.jsonRuntimeSettings)
            f.close()
            
            QtWidgets.QMessageBox.information(self, "Save Runtime Settings As JSON File", 'JSON file successfuly saved!')
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            dlg = daeExceptionDialog(self, e, exc_traceback)
            dlg.exec_()

    """
    def _slotLoadRuntimeSettingsAsJSON(self):
        try:
            jsonFilename, ok = QtWidgets.QFileDialog.getOpenFileName(self, "Load Runtime Settings As JSON File", "", "JSON Files (*.json)")
            if not ok:
                return

            f = open(str(jsonFilename), 'r')
            json_str = f.read()
            f.close()

            jsonSettings = json.loads(json_str)
            self.updateSimulation(verbose = False)
            self._initialize(self._simulation)

            QtWidgets.QMessageBox.information(self, "Load Runtime Settings As JSON File", 'JSON file successfuly loaded!')

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            dlg = daeExceptionDialog(self, e, exc_traceback)
            dlg.exec_()
    """
    
    def _slotGenerateCode(self):
        #try:
        languages = ['Modelica', 'gPROMS', 'c99', 'c++ (MPI)', 'FMI (Co-Simulation)']
        language, ok = QtWidgets.QInputDialog.getItem(self, "Code generator", "Choose the target language:", languages, 0, False)
        if not ok:
            return
        
        self.generateCode(language)
            
        #except Exception as e:
            #exc_type, exc_value, exc_traceback = sys.exc_info()
            #dlg = daeExceptionDialog(self, e, exc_traceback)
            #dlg.exec_()

    def _slotParameterTreeItemSelectionChanged(self):
        currentItem = self._ui.treeParameters.selectedItems()[0]
        data = currentItem.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data

        if self._currentParameterItem:
            self._currentParameterItem.hide()
        
        if item.editor:
            self._currentParameterItem = item
            self._currentParameterItem.show(self._ui.frameParameters)

    def _slotOutputVariablesTreeItemChanged(self, treeWidgetItem, column):
        if column == 0:
            data = treeWidgetItem.data(0, QtCore.Qt.UserRole)
            if not data:
                return
            if isinstance(data, QtCore.QVariant):
                item = data.toPyObject()
            else:
                item = data
                
            if item.itemType == treeItem.typeOutputVariable:
                if treeWidgetItem.checkState(0) == Qt.Checked:
                    item.setValue(True)
                else:
                    item.setValue(False)

    def _slotSTNsTreeItemChanged(self, treeWidgetItem, column):
        """
        currentItem = self._ui.treeSTNs.selectedItems()[0]
        data = currentItem.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data

        if self._currentStateTransitionItem:
            self._currentStateTransitionItem.hide()
        
        if item.editor:
            self._currentStateTransitionItem = item
            self._currentStateTransitionItem.show(self._ui.frameStateTransitions)
        """
        if column == 0:
            if self._treeStates_itemChanged_disabled:
                return
                
            data = treeWidgetItem.data(0, QtCore.Qt.UserRole)
            if not data:
                return
            if isinstance(data, QtCore.QVariant):
                item = data.toPyObject()
            else:
                item = data
            if item.itemType == treeItem.typeState:
                if treeWidgetItem.checkState(0) == Qt.Unchecked:
                    # Do not allow item to be unchecked unless the flag _allowUnchecking is set
                    if self._allowUnchecking:
                        item.setValue(False)
                        #print 'Item: %s unchecked' % item.canonicalName
                    
                    else:
                        # If unchecking is not allowed check it again
                        self._treeStates_itemChanged_disabled = True
                        treeWidgetItem.setCheckState(0, Qt.Checked)
                        self._treeStates_itemChanged_disabled = False
                
                else: 
                    # Item has just been checked, either through GUI or from a function
                    try:
                        # 1. Set the value to True
                        item.setValue(True)
                        #print 'Item: %s checked' % item.canonicalName
                        
                        # Prevent itemChanged event to be handled
                        self._allowUnchecking                 = True
                        self._treeStates_itemChanged_disabled = True
                        
                        # 2. Uncheck all siblings and their children (except the current item)
                        parent = treeWidgetItem.parent()
                        self._uncheckSiblings(parent, treeWidgetItem)
                        
                        # 3. Check first children of the current item 
                        self._checkFirstChildren(treeWidgetItem)
                    
                        # 4. Check all parents of the current item
                        self._checkAllParents(treeWidgetItem)
                        
                    except:
                        pass
                    finally:
                        # Restore flags back
                        self._allowUnchecking                 = False
                        self._treeStates_itemChanged_disabled = False
                    
    def _checkAllParents(self, treeWidgetItem):
        # The item is checked and all its parents must be checked as well, 
        # while the siblings of its parents (and their children) must be unchecked
        parent = treeWidgetItem.parent()
        if not parent:
            return
        data = parent.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data
        if item and item.itemType == treeItem.typeState:
            item.setValue(True)
            parent.setCheckState(0, Qt.Checked)
            #print 'Item: %s checked' % item.canonicalName
            
            grandpa = parent.parent()
            self._uncheckSiblings(grandpa, parent)
        
        self._checkAllParents(parent)
        
    def _uncheckSiblings(self, parent, currentTreeWidgetItem):
        # The unchecked item is checked so its siblings (and all their children) must be unchecked.
        # This means that when a state becomes active all other states (and their nested STNS) within 
        # the same STN must be inactive.
        if not parent:
            return
        
        for i  in range(parent.childCount()):
            child = parent.child(i)
            # Do not uncheck the item currently being set ()
            if child != currentTreeWidgetItem:
                data = child.data(0, QtCore.Qt.UserRole)
                if isinstance(data, QtCore.QVariant):
                    item = data.toPyObject()
                else:
                    item = data
                if item and item.itemType == treeItem.typeState:
                    item.setValue(False)
                    child.setCheckState(0, Qt.Unchecked)
                    #print 'Item: %s unchecked' % item.canonicalName
                
                # Now uncheck all down through the hierarchy of children 
                # (since thier parent is unchecked they should be too) 
                self._uncheckSiblings(child, None)
    
    def _checkFirstChildren(self, parent):
        # Once the unchecked item is checked some defaults must be provided for its children.
        # Here we chose to check the first items (this means that the first states will be initially active).
        if parent.childCount() > 0:
            child = parent.child(0)
            data = child.data(0, QtCore.Qt.UserRole)
            if isinstance(data, QtCore.QVariant):
                item = data.toPyObject()
            else:
                item = data
            if item and item.itemType == treeItem.typeState:
                item.setValue(True)
                child.setCheckState(0, Qt.Checked)
                #print 'Child item: %s checked' % item.canonicalName
            
            self._checkFirstChildren(child)
        
    def _slotDomainsTreeItemChanged(self):
        currentItem = self._ui.treeDomains.selectedItems()[0]
        data = currentItem.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data

        if self._currentDomainItem:
            self._currentDomainItem.hide()
        
        if item.editor:
            self._currentDomainItem = item
            self._currentDomainItem.show(self._ui.frameDomains)

    def _slotDOFsTreeItemChanged(self):
        currentItem = self._ui.treeDOFs.selectedItems()[0]
        data = currentItem.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data

        if self._currentDOFItem:
            self._currentDOFItem.hide()
        
        if item.editor:
            self._currentDOFItem = item
            self._currentDOFItem.show(self._ui.frameDOFs)

    def _slotInitialConditionsTreeItemChanged(self):
        currentItem = self._ui.treeInitialConditions.selectedItems()[0]
        data = currentItem.data(0, QtCore.Qt.UserRole)
        if isinstance(data, QtCore.QVariant):
            item = data.toPyObject()
        else:
            item = data
            
        if self._currentInitialConditionItem:
            self._currentInitialConditionItem.hide()
        
        if item.editor:
            self._currentInitialConditionItem = item
            self._currentInitialConditionItem.show(self._ui.frameInitialConditions)
            
