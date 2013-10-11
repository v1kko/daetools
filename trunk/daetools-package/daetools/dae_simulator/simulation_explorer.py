#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                          simulation_explorer.py
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

import sys, tempfile, numpy
from time import localtime, strftime
from os.path import join, dirname
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from simulation_explorer_ui import Ui_SimulationExplorer
from simulation_inspector import daeSimulationInspector
from tree_item import *
import aux

images_dir = join(dirname(__file__), 'images')

def simulate(simulation, **kwargs):
    qt_app = kwargs.get('qt_app', None)
    if not qt_app:
        qt_app = QtGui.QApplication(sys.argv)
    explorer = daeSimulationExplorer(qt_app, simulation = simulation, **kwargs)
    explorer.exec_()

def optimize(optimization, **kwargs):
    qt_app = kwargs.get('qt_app', None)
    if not qt_app:
        qt_app = QtGui.QApplication(sys.argv)
    explorer = daeSimulationExplorer(qt_app, simulation = simulation, optimization = optimization, **kwargs)
    explorer.exec_()

class daeSimulationExplorer(QtGui.QDialog):
    def __init__(self, qt_app, **kwargs):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulationExplorer()
        self.ui.setupUi(self)
        
        self.setWindowTitle("DAE Tools Simulation Explorer v" + daeVersion(True))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.connect(self.ui.buttonOk,                  QtCore.SIGNAL('clicked()'),                          self.slotOK)
        self.connect(self.ui.buttonCancel,              QtCore.SIGNAL('clicked()'),                          self.slotCancel)
        self.connect(self.ui.treeParameters,            QtCore.SIGNAL("itemSelectionChanged()"),             self.slotParameterTreeItemSelectionChanged)
        self.connect(self.ui.treeOutputVariables,       QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"), self.slotOutputVariablesTreeItemChanged)
        self.connect(self.ui.treeStateTransitions,      QtCore.SIGNAL("itemSelectionChanged()"),             self.slotStateTransitionsTreeItemChanged)
        self.connect(self.ui.treeDomains,               QtCore.SIGNAL("itemSelectionChanged()"),             self.slotDomainsTreeItemChanged)
        self.connect(self.ui.treeDOFs,                  QtCore.SIGNAL("itemSelectionChanged()"),             self.slotDOFsTreeItemChanged)
        self.connect(self.ui.treeInitialConditions,     QtCore.SIGNAL("itemSelectionChanged()"),             self.slotInitialConditionsTreeItemChanged)

        self.qt_app                      = qt_app
        self.simulation                  = kwargs.get('simulation',                 None)
        self.optimization                = kwargs.get('optimization',               None)
        self.datareporter                = kwargs.get('datareporter',               None)
        #self.log                         = kwargs.get('log',                        None)
        self.lasolver                    = kwargs.get('lasolver',                   None)
        self.nlpsolver                   = kwargs.get('nlpsolver',                  None)
        #self.nlpsolver_setoptions_fn     = kwargs.get('nlpsolver_setoptions_fn',    None)
        #self.lasolver_setoptions_fn      = kwargs.get('lasolver_setoptions_fn',     None)
        #self.run_after_simulation_end_fn = kwargs.get('run_after_simulation_end_fn',None)

        if not self.qt_app:
            raise RuntimeError('qt_app object must not be None')
        if not self.simulation:
            raise RuntimeError('simulation object must not be None')
        
        self.inspector = daeSimulationInspector(self.simulation)
        
        self.simulationName     = self.simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        self.timeHorizon        = self.simulation.TimeHorizon
        self.reportingInterval  = self.simulation.ReportingInterval
        self.relativeTolerance  = self.simulation.DAESolver.RelativeTolerance
        if self.simulation.InitialConditionMode == eQuasySteadyState:
            self.quazySteadyState = True
        else:
            self.quazySteadyState = False
        
        self.daesolver = self.simulation.DAESolver
        self.log       = self.simulation.Log
        
        d_validator = QtGui.QDoubleValidator(self)

        self.ui.simulationNameEdit.setText(self.simulationName)
        self.ui.timeHorizonEdit.setValidator(d_validator)
        self.ui.timeHorizonEdit.setText(str(self.timeHorizon))
        self.ui.reportingIntervalEdit.setValidator(d_validator)
        self.ui.reportingIntervalEdit.setText(str(self.reportingInterval))
        self.ui.relativeToleranceEdit.setValidator(d_validator)
        self.ui.relativeToleranceEdit.setText(str(self.relativeTolerance))
        if self.quazySteadyState:
            self.ui.quazySteadyStateCheckBox.setCheckState(Qt.Checked)
            self.ui.tab_InitialConditions.setEnabled(False)
        else:
            self.ui.quazySteadyStateCheckBox.setCheckState(Qt.Unchecked)
            self.ui.tab_InitialConditions.setEnabled(True)

        self.available_la_solvers     = aux.getAvailableLASolvers()
        self.available_nlp_solvers    = aux.getAvailableNLPSolvers()
        self.available_data_reporters = aux.getAvailableDataReporters()
        self.available_logs           = aux.getAvailableLogs()

        self.ui.daesolverComboBox.addItem("Sundials IDAS")
        for la in self.available_la_solvers:
            self.ui.lasolverComboBox.addItem(la[0], userData = QtCore.QVariant(la[1]))
        for nlp in self.available_nlp_solvers:
            self.ui.minlpsolverComboBox.addItem(nlp[0], userData = QtCore.QVariant(nlp[1]))
        for dr in self.available_data_reporters:
            self.ui.datareporterComboBox.addItem(dr[0], userData = QtCore.QVariant(dr[1]))
        for log in self.available_logs:
            self.ui.logComboBox.addItem(log[0], userData = QtCore.QVariant(log[1]))
        
        # DAE Solvers
        self.ui.daesolverComboBox.setEnabled(False)

        # LA Solvers
        if not self.lasolver:
            self.ui.lasolverComboBox.setEnabled(True)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self.ui.lasolverComboBox.clear()
            self.ui.lasolverComboBox.addItem(self.lasolver.Name)
            self.ui.lasolverComboBox.setEnabled(False)

        # MINLP Solvers
        if not self.optimization:
            # If we are simulating then clear and disable MINLPSolver combo box
            #self.ui.simulationLabel.setText('Simulation')
            self.ui.minlpsolverComboBox.clear()
            self.ui.minlpsolverComboBox.setEnabled(False)
        else:
            #self.ui.simulationLabel.setText('Optimization')
            if not self.nlpsolver:
                self.ui.minlpsolverComboBox.setEnabled(True)
            else:
                # If nlpsolver has been sent then clear and disable MINLPSolver combo box
                self.ui.minlpsolverComboBox.clear()
                self.ui.minlpsolverComboBox.addItem(self.nlpsolver.Name)
                self.ui.minlpsolverComboBox.setEnabled(False)
                
        # Logs
        if not self.log:
            self.ui.logComboBox.setEnabled(True)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self.ui.logComboBox.clear()
            self.ui.logComboBox.addItem(self.log.Name)
            self.ui.logComboBox.setEnabled(False)
                
        # DataReporters
        if not self.datareporter:
            self.ui.datareporterComboBox.setEnabled(True)
        else:
            # If LA solver has been sent then clear and disable LASolver combo box
            self.ui.datareporterComboBox.clear()
            self.ui.datareporterComboBox.addItem(self.log.Name)
            self.ui.datareporterComboBox.setEnabled(False)
        
        self.currentParameterItem        = None
        self.currentStateTransitionItem  = None
        self.currentDomainItem           = None
        self.currentDOFItem              = None
        self.currentInitialConditionItem = None
        
        addItemsToTree(self.ui.treeParameters,        self.ui.treeParameters,        self.inspector.treeParameters)
        addItemsToTree(self.ui.treeOutputVariables,   self.ui.treeOutputVariables,   self.inspector.treeOutputVariables)
        addItemsToTree(self.ui.treeStateTransitions,  self.ui.treeStateTransitions,  self.inspector.treeStateTransitions)
        addItemsToTree(self.ui.treeDomains,           self.ui.treeDomains,           self.inspector.treeDomains)
        addItemsToTree(self.ui.treeDOFs,              self.ui.treeDOFs,              self.inspector.treeDOFs)
        addItemsToTree(self.ui.treeInitialConditions, self.ui.treeInitialConditions, self.inspector.treeInitialConditions)
        
        self.ui.treeDomains.expandAll()
        self.ui.treeDomains.resizeColumnToContents(0)
        
        self.ui.treeParameters.expandAll()
        self.ui.treeParameters.resizeColumnToContents(0)
        
        self.ui.treeInitialConditions.expandAll()
        self.ui.treeInitialConditions.resizeColumnToContents(0)
        
        self.ui.treeDOFs.expandAll()
        self.ui.treeDOFs.resizeColumnToContents(0)
        
        self.ui.treeStateTransitions.expandAll()
        self.ui.treeStateTransitions.resizeColumnToContents(0)
        
        self.ui.treeOutputVariables.expandAll()
        self.ui.treeOutputVariables.resizeColumnToContents(0)

    def processInputs(self):
        self.simulationName    = self.ui.simulationNameEdit.text()
        self.timeHorizon       = float(self.ui.timeHorizonEdit.text())
        self.reportingInterval = float(self.ui.reportingIntervalEdit.text())
        self.relativeTolerance = float(self.ui.relativeToleranceEdit.text())
        
        if not self.daesolver:
            self.daesolver = daeIDAS()
        
        # If in optimization mode and nlpsolver is not sent then choose it from the selection
        if self.optimization and not self.nlpsolver and len(self.available_nlp_solvers) > 0:
            minlpsolverIndex = self.ui.minlpsolverComboBox.itemData(self.ui.minlpsolverComboBox.currentIndex()).toInt()[0]
            self.nlpsolver = aux.createNLPSolver(minlpsolverIndex)

        # If lasolver is not sent then create it based on the selection
        if self.lasolver == None and len(self.available_la_solvers) > 0:
            lasolverIndex = self.ui.lasolverComboBox.itemData(self.ui.lasolverComboBox.currentIndex()).toInt()[0]
            self.lasolver = aux.createLASolver(lasolverIndex)
            self.daesolver.SetLASolver(self.lasolver)

        if not self.datareporter:
            drIndex = self.ui.datareporterComboBox.itemData(self.ui.datareporterComboBox.currentIndex()).toInt()[0]
            self.datareporter = aux.createDataReporter(drIndex)
            
        if not self.log:
            logIndex = self.ui.logComboBox.itemData(self.ui.logComboBox.currentIndex()).toInt()[0]
            self.log = aux.createLog(logIndex)
            
    def slotOK(self):
        self.done(QtGui.QDialog.Accepted)

    def slotCancel(self):
        self.done(QtGui.QDialog.Rejected)

    def slotParameterTreeItemSelectionChanged(self):
        currentItem = self.ui.treeParameters.selectedItems()[0]
        data = currentItem.data(1, QtCore.Qt.UserRole)
        item = data.toPyObject()

        if self.currentParameterItem:
            self.currentParameterItem.hide()
        
        if item.editor:
            self.currentParameterItem = item
            self.currentParameterItem.show(self.ui.frameParameters)

    def slotOutputVariablesTreeItemChanged(self, treeWidgetItem, column):
        if column == 0:
            data = treeWidgetItem.data(1, QtCore.Qt.UserRole)
            if not data:
                return
            item = data.toPyObject()
            if item.itemType == treeItem.typeOutputVariable:
                if treeWidgetItem.checkState(0) == Qt.Checked:
                    item.setValue(True)
                else:
                    item.setValue(False)

    def slotStateTransitionsTreeItemChanged(self):
        currentItem = self.ui.treeStateTransitions.selectedItems()[0]
        data = currentItem.data(1, QtCore.Qt.UserRole)
        item = data.toPyObject()

        if self.currentStateTransitionItem:
            self.currentStateTransitionItem.hide()
        
        if item.editor:
            self.currentStateTransitionItem = item
            self.currentStateTransitionItem.show(self.ui.frameStateTransitions)


    def slotDomainsTreeItemChanged(self):
        currentItem = self.ui.treeDomains.selectedItems()[0]
        data = currentItem.data(1, QtCore.Qt.UserRole)
        item = data.toPyObject()

        if self.currentDomainItem:
            self.currentDomainItem.hide()
        
        if item.editor:
            self.currentDomainItem = item
            self.currentDomainItem.show(self.ui.frameDomains)

    def slotDOFsTreeItemChanged(self):
        currentItem = self.ui.treeDOFs.selectedItems()[0]
        data = currentItem.data(1, QtCore.Qt.UserRole)
        item = data.toPyObject()

        if self.currentDOFItem:
            self.currentDOFItem.hide()
        
        if item.editor:
            self.currentDOFItem = item
            self.currentDOFItem.show(self.ui.frameDOFs)

    def slotInitialConditionsTreeItemChanged(self):
        currentItem = self.ui.treeInitialConditions.selectedItems()[0]
        data = currentItem.data(1, QtCore.Qt.UserRole)
        item = data.toPyObject()

        if self.currentInitialConditionItem:
            self.currentInitialConditionItem.hide()
        
        if item.editor:
            self.currentInitialConditionItem = item
            self.currentInitialConditionItem.show(self.ui.frameInitialConditions)
            
