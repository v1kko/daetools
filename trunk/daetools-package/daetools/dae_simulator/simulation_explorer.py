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

class daeSimulationExplorer(QtGui.QDialog):
    def __init__(self, simulation):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulationExplorer()
        self.ui.setupUi(self)
        
        self.setWindowTitle("DAE Tools Simulation Explorer v" + daeVersion(True))
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.simulation = simulation
        self.inspector  = daeSimulationInspector(simulation)
        
        self.simulationName     = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
        self.timeHorizon        = simulation.TimeHorizon
        self.reportingInterval  = simulation.ReportingInterval
        self.relativeTolerance  = simulation.DAESolver.RelativeTolerance
        if simulation.InitialConditionMode == eQuasySteadyState:
            self.quazySteadyState = True
        else:
            self.quazySteadyState = False
        self.DAESolver          = simulation.DAESolver
        self.LASolver           = simulation.DAESolver.LASolver
        self.dataReporter       = simulation.DataReporter
        self.log                = simulation.Log
        
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
        else:
            self.ui.quazySteadyStateCheckBox.setCheckState(Qt.Unchecked)

        self.available_la_solvers     = aux.getAvailableLASolvers()
        self.available_data_reporters = aux.getAvailableDataReporters()
        self.available_logs           = aux.getAvailableLogs()

        self.ui.daesolverComboBox.addItem("Sundials IDAS")
        for la in self.available_la_solvers:
            self.ui.lasolverComboBox.addItem(la[0], userData = QtCore.QVariant(la[1]))
        for dr in self.available_data_reporters:
            self.ui.datareporterComboBox.addItem(dr[0], userData = QtCore.QVariant(dr[1]))
        for log in self.available_logs:
            self.ui.logComboBox.addItem(log[0], userData = QtCore.QVariant(log[1]))
        
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
        
        self.connect(self.ui.buttonOk,                  QtCore.SIGNAL('clicked()'),                          self.slotOK)
        self.connect(self.ui.buttonCancel,              QtCore.SIGNAL('clicked()'),                          self.slotCancel)
        self.connect(self.ui.treeParameters,            QtCore.SIGNAL("itemSelectionChanged()"),             self.slotParameterTreeItemSelectionChanged)
        self.connect(self.ui.treeOutputVariables,       QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"), self.slotOutputVariablesTreeItemChanged)
        self.connect(self.ui.treeStateTransitions,      QtCore.SIGNAL("itemSelectionChanged()"),             self.slotStateTransitionsTreeItemChanged)
        self.connect(self.ui.treeDomains,               QtCore.SIGNAL("itemSelectionChanged()"),             self.slotDomainsTreeItemChanged)
        self.connect(self.ui.treeDOFs,                  QtCore.SIGNAL("itemSelectionChanged()"),             self.slotDOFsTreeItemChanged)
        self.connect(self.ui.treeInitialConditions,     QtCore.SIGNAL("itemSelectionChanged()"),             self.slotInitialConditionsTreeItemChanged)
        self.connect(self.ui.quazySteadyStateCheckBox,  QtCore.SIGNAL("stateChanged(int)"),                  self.slotQuazySteadyStateCheckBoxStateChanged)

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

    def slotOK(self):
        #self.simulationName = self.ui.lowerBoundEdit.text()
        #self.timeHorizon = 
        #self.reportingInterval = 
        #self.relativeTolerance = 
        #self.quazySteadyStateInitialConditions = 
        #self.DAESolver = 
        #self.LASolver = 
        #self.dataReporter = 
        #self.log = 
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
            
    def slotQuazySteadyStateCheckBoxStateChanged(self, state):
        if state == Qt.Checked:
            self.ui.tab_InitialConditions.setEnabled(False)
        else:
            self.ui.tab_InitialConditions.setEnabled(True)
