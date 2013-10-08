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
from os.path import join, dirname
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from simulation_explorer_ui import Ui_SimulationExplorer
from simulation_inspector import daeSimulationInspector
from tree_item import *

images_dir = join(dirname(__file__), 'images')

class daeSimulationExplorer(QtGui.QDialog):
    def __init__(self, simulation):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_SimulationExplorer()
        self.ui.setupUi(self)

        self.simulation = simulation
        self.inspector  = daeSimulationInspector(simulation)
        
        self.currentParameterItem       = None
        self.currentStateTransitionItem = None
        
        addItemsToTree(self.ui.treeParameters,       self.ui.treeParameters,       self.inspector.treeParameters)
        addItemsToTree(self.ui.treeOutputVariables,  self.ui.treeOutputVariables,  self.inspector.treeOutputVariables)
        addItemsToTree(self.ui.treeStateTransitions, self.ui.treeStateTransitions, self.inspector.treeStateTransitions)
        
        self.connect(self.ui.treeParameters,        QtCore.SIGNAL("itemSelectionChanged()"),             self.slotParameterTreeItemSelectionChanged)
        self.connect(self.ui.treeOutputVariables,   QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"), self.slotOutputVariablesTreeItemChanged)
        self.connect(self.ui.treeStateTransitions,  QtCore.SIGNAL("itemSelectionChanged()"),             self.slotStateTransitionsTreeItemChanged)
        
        self.setWindowIcon(QtGui.QIcon(join(images_dir, 'daetools-48x48.png')))

        self.ui.treeDomains.expandAll()
        self.ui.treeDomains.resizeColumnToContents(0)
        
        self.ui.treeParameters.expandAll()
        self.ui.treeParameters.resizeColumnToContents(0)
        ##self.ui.treeParameters.resizeColumnToContents(1)
        
        self.ui.treeInitialConditions.expandAll()
        self.ui.treeInitialConditions.resizeColumnToContents(0)
        
        self.ui.treeDOFs.expandAll()
        self.ui.treeDOFs.resizeColumnToContents(0)
        
        self.ui.treeStateTransitions.expandAll()
        self.ui.treeStateTransitions.resizeColumnToContents(0)
        
        self.ui.treeOutputVariables.expandAll()
        self.ui.treeOutputVariables.resizeColumnToContents(0)

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
