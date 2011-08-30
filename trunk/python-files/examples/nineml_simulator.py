#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys, subprocess
from time import localtime, strftime, time
from daetools.pyDAE.parser import ExpressionParser
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_simulator_ui import Ui_ninemlSimulator


def printDictionary(dictionary):
    sortedDictionary = sorted(dictionary.iteritems())
    for key, value in sortedDictionary:
        print key + ' : ' + repr(value)
    print '\n'

def printList(l):
    for value in l:
        print repr(value)
    print '\n'

def collectParameters(root, component, parameters, initialValues = {}):
    rootName = root
    if rootName != '':
        rootName += '.'
    for obj in component.parameters:
        objName = rootName + obj.name
        if initialValues.has_key(objName):
            parameters[objName] = initialValues[objName]
        else:
            parameters[objName] = 0.0

    for name, subcomponent in component.subnodes.items():
        parameters = collectParameters(rootName + subcomponent.name, subcomponent, parameters, initialValues)

    return parameters

def collectStateVariables(root, component, stateVariables, initialValues = {}):
    rootName = root
    if rootName != '':
        rootName += '.'
    for obj in component.state_variables:
        objName = rootName + obj.name
        if initialValues.has_key(objName):
            stateVariables[objName] = initialValues[objName]
        else:
            stateVariables[objName] = 0.0

    for name, subcomponent in component.subnodes.items():
        stateVariables = collectStateVariables(rootName + subcomponent.name, subcomponent, stateVariables, initialValues)

    return stateVariables

def collectRegimes(root, component, regimes, activeRegimes = []):
    rootName = root
    if rootName != '':
        rootName += '.'
    available_regimes = []
    active_regime = None
    for obj in component.regimes:
        available_regimes.append(obj.name)
        objName = rootName + obj.name
        if objName in activeRegimes:
            active_regime = obj.name

    if len(available_regimes) > 0:
        if active_regime == None:
            active_regime = available_regimes[0]
        regimes[root] = [available_regimes, active_regime]

    for name, subcomponent in component.subnodes.items():
        regimes = collectRegimes(rootName + subcomponent.name, subcomponent, regimes, activeRegimes)

    return regimes

def collectAnalogPorts(root, component, analog_ports):
    rootName = root
    if rootName != '':
        rootName += '.'
    for obj in component.analog_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = rootName + obj.name
            analog_ports[objName] = ''

    for name, subcomponent in component.subnodes.items():
        analog_ports = collectAnalogPorts(rootName + subcomponent.name, subcomponent, analog_ports)

    return analog_ports

def collectEventPorts(root, component, event_ports):
    rootName = root
    if rootName != '':
        rootName += '.'
    for obj in component.event_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = rootName + obj.name
            event_ports[objName] = ''

    for name, subcomponent in component.subnodes.items():
        event_ports = collectEventPorts(rootName + subcomponent.name, subcomponent, event_ports)

    return event_ports

def collectResultsVariables(root, component, results_variables, checkedVariables = []):
    rootName = root
    if rootName != '':
        rootName += '.'

    for obj in component.aliases:
        objName = rootName + obj.lhs
        if objName in checkedVariables:
            results_variables[objName] = True
        else:
            results_variables[objName] = False

    for obj in component.state_variables:
        objName = rootName + obj.name
        if objName in checkedVariables:
            results_variables[objName] = True
        else:
            results_variables[objName] = False

    for obj in component.analog_ports:
        objName = rootName + obj.name
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            if objName in checkedVariables:
                results_variables[objName] = True
            else:
                results_variables[objName] = False

    for name, subcomponent in component.subnodes.items():
        results_variables = collectResultsVariables(rootName + subcomponent.name, subcomponent, results_variables, checkedVariables)

    return results_variables

def addItemsToTree(treeWidget, dictItems, rootName, editableItems = True):
    rootItem = QtGui.QTreeWidgetItem(treeWidget, [rootName, ''])

    for key, value in dictItems.items():
        names = key.split(".")
        item_path = names[:-1]
        item_name = names[-1]

        currentItem = rootItem

        # First find the parent QTreeWidgetItem
        for item in item_path:
            found = False
            if currentItem:
                for c in range(0, currentItem.childCount()):
                    child = currentItem.child(c)
                    cname = child.text(0)
                    if item == cname:
                        found = True
                        currentItem = currentItem.child(c)
                        break

            if found == False:
                currentItem = QtGui.QTreeWidgetItem(currentItem, [item, ''])

        # Now we have the parrent in the currentItem, so add the new item to it with the variable data
        varItem = QtGui.QTreeWidgetItem(currentItem, [item_name, str(value)])
        varData = QtCore.QVariant(key)
        varItem.setData(1, QtCore.Qt.UserRole, varData)
        if editableItems:
            varItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled)
        else:
            varItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    treeWidget.expandAll()
    treeWidget.resizeColumnToContents(0)
    treeWidget.resizeColumnToContents(1)

def addItemsToResultsVariablesTree(treeWidget, dictItems, rootName):
    rootItem = QtGui.QTreeWidgetItem(treeWidget, [rootName])

    for key, value in dictItems.items():
        names = key.split(".")
        item_path = names[:-1]
        item_name = names[-1]

        currentItem = rootItem

        # First find the parent QTreeWidgetItem
        for item in item_path:
            found = False
            if currentItem:
                for c in range(0, currentItem.childCount()):
                    child = currentItem.child(c)
                    cname = child.text(0)
                    if item == cname:
                        found = True
                        currentItem = currentItem.child(c)
                        break

            if found == False:
                currentItem = QtGui.QTreeWidgetItem(currentItem, [item])

        # Now we have the parrent in the currentItem, so add the new item to it with the variable data
        varItem = QtGui.QTreeWidgetItem(currentItem, [item_name])
        varData = QtCore.QVariant(key)
        varItem.setData(0, QtCore.Qt.UserRole, varData)
        varItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        if value:
            varItem.setCheckState(0, Qt.Checked)
        else:
            varItem.setCheckState(0, Qt.Unchecked)

    treeWidget.expandAll()
    
def addItemsToRegimesTree(treeWidget, dictItems, rootName):
    rootItem = QtGui.QTreeWidgetItem(treeWidget, [rootName, ''])

    for key, value in dictItems.items():
        names = key.split(".")
        item_path = names[:-1]
        item_name = names[-1]

        currentItem = rootItem

        # First find the parent QTreeWidgetItem
        for item in item_path:
            found = False
            if currentItem:
                for c in range(0, currentItem.childCount()):
                    child = currentItem.child(c)
                    cname = child.text(0)
                    if item == cname:
                        found = True
                        currentItem = currentItem.child(c)
                        break

            if found == False:
                currentItem = QtGui.QTreeWidgetItem(currentItem, [item, ''])

        # Now we have the parrent in the currentItem, so add the new item to it with the variable data
        varItem = QtGui.QTreeWidgetItem(currentItem, [item_name, value[1]])
        varData = QtCore.QVariant((key, value[0]))
        varItem.setData(1, QtCore.Qt.UserRole, varData)
        varItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    treeWidget.expandAll()
    treeWidget.resizeColumnToContents(0)
    treeWidget.resizeColumnToContents(1)

class nineml_tester(QtGui.QDialog):
    def __init__(self, ninemlComponent, parametersValues = {}, initialConditionsValues = {}, activeRegimes = [], resultsVariables = []):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ninemlSimulator()
        self.ui.setupUi(self)

        self.ninemlComponent   = ninemlComponent
        self.parser            = ExpressionParser()
        # Dictionaries 'key' : floating-point-value
        self.parameters        = {}
        self.state_variables   = {}
        # Dictionaries: 'key' : 'expression'
        self.analog_ports      = {}
        self.event_ports       = {}
        # Dictionary 'key' : [ [list-of-available-regimes], 'current-active-regime']
        self.active_regimes    = {}
        # Dictionaries 'key' : boolean-value
        self.results_variables = {}

        self.connect(self.ui.treeParameters,        QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotParameterItemChanged)
        self.connect(self.ui.treeInitialConditions, QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotInitialConditionItemChanged)
        self.connect(self.ui.treeRegimes,           QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotRegimesItemDoubleClicked)
        self.connect(self.ui.treeAnalogPorts,       QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotAnalogPortsItemDoubleClicked)
        self.connect(self.ui.treeEventPorts,        QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotEventPortsItemDoubleClicked)
        self.connect(self.ui.treeResultsVariables,  QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotResultsVariablesItemChanged)

        collectParameters      ('', self.ninemlComponent, self.parameters,      parametersValues)
        collectStateVariables  ('', self.ninemlComponent, self.state_variables, initialConditionsValues)
        collectRegimes         ('', self.ninemlComponent, self.active_regimes,  activeRegimes)
        collectAnalogPorts     ('', self.ninemlComponent, self.analog_ports)
        collectEventPorts      ('', self.ninemlComponent, self.event_ports)
        collectResultsVariables('', self.ninemlComponent, self.results_variables, resultsVariables)

        self.printResults()

        addItemsToTree                (self.ui.treeParameters,        self.parameters,        self.ninemlComponent.name)
        addItemsToTree                (self.ui.treeInitialConditions, self.state_variables,   self.ninemlComponent.name)
        addItemsToTree                (self.ui.treeAnalogPorts,       self.analog_ports,      self.ninemlComponent.name, False)
        addItemsToTree                (self.ui.treeEventPorts,        self.event_ports,       self.ninemlComponent.name, False)
        addItemsToRegimesTree         (self.ui.treeRegimes,           self.active_regimes,    self.ninemlComponent.name)
        addItemsToResultsVariablesTree(self.ui.treeResultsVariables,  self.results_variables, self.ninemlComponent.name)

    def printResults(self):
        printDictionary(self.parameters)
        printDictionary(self.state_variables)
        printDictionary(self.active_regimes)
        printDictionary(self.analog_ports)
        printDictionary(self.event_ports)
        printDictionary(self.results_variables)

    def slotParameterItemChanged(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            varValue = QtCore.QVariant(item.text(1))

            value, isOK = varValue.toDouble()
            if not isOK:
                QtGui.QMessageBox.warning(None, "NineML Simulator", "Invalid value entered for the item: " + item.text(0))
                item.setText(1, str(self.parameters[key]))
                return

            if self.parameters[key] != value:
                self.parameters[key] = value

    def slotInitialConditionItemChanged(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            varValue = QtCore.QVariant(item.text(1))

            value, isOK = varValue.toDouble()
            if not isOK:
                QtGui.QMessageBox.warning(None, "NineML Simulator", "Invalid value entered for the item: " + item.text(0))
                item.setText(1, str(self.state_variables[key]))
                return

            if self.state_variables[key] != value:
                self.state_variables[key] = value

    def slotResultsVariableItemChanged(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            varValue = QtCore.QVariant(item.text(1))

            value, isOK = varValue.toDouble()
            if not isOK:
                QtGui.QMessageBox.warning(None, "NineML Simulator", "Invalid value entered for the item: " + item.text(0))
                item.setText(1, str(self.parameters[key]))
                return

            if self.parameters[key] != value:
                self.parameters[key] = value
            if True:
                item.setCheckState(0,QtCore.Qt.Checked)
            else:
                item.setCheckState(0,QtCore.Qt.Unchecked)

    def slotEventPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Event Port Input", "Set the input event expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, new_expression)
                self.event_ports[key] = str(new_expression)

    def slotAnalogPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Analog Port Input", "Set the analog port input expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, str(new_expression))
                self.analog_ports[key] = str(new_expression)

    def slotRegimesItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key, available_regimes = data.toPyObject()
            active_regime, ok = QtGui.QInputDialog.getItem(self, "Available regimes", "Select the new active regime:", available_regimes, 0, False)
            if ok:
                item.setText(1, active_regime)
                self.active_regimes[key][1] = str(active_regime)

    def slotResultsVariablesItemChanged(self, item, column):
        if column == 0:
            data = item.data(0, QtCore.Qt.UserRole)
            key  = str(data.toString())
            if item.checkState(0) == Qt.Checked:
                self.results_variables[key] = True
            else:
                self.results_variables[key] = False

coba_iaf_base = TestableComponent('hierachical_iaf_1coba')
coba_iaf = coba_iaf_base()

destexhe_ampa_base = TestableComponent('destexhe_ampa')
destexhe_ampa = destexhe_ampa_base()

app = QtGui.QApplication(sys.argv)

parameters = {
    'CobaSyn.q' : 3.0,
    'CobaSyn.tau' : 5.0,
    'CobaSyn.vrev' : 0.0,
    'iaf.cm' : 1,
    'iaf.gl' : 50,
    'iaf.taurefrac' : 0.008,
    'iaf.vreset' : -60,
    'iaf.vrest' : -60,
    'iaf.vthresh' : -40
}
initial_conditions = {
    'CobaSyn.g' : 0.0,
    'iaf.V' : -60,
    'iaf.tspike' : -1E99
}
active_regimes = [
    'CobaSyn.cobadefaultregime',
    'iaf.subthresholdregime'
]
results_variables = [
    'CobaSyn.g',
    'iaf.tspike'
]

s = nineml_tester(coba_iaf, parameters, initial_conditions, active_regimes, results_variables)
s.exec_()
s.printResults()
#sys.exit(app.exec_())

