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
        regimes[root] = (available_regimes, active_regime)

    for name, subcomponent in component.subnodes.items():
        regimes = collectRegimes(rootName + subcomponent.name, subcomponent, regimes, activeRegimes)

    return regimes

def addItemsToTree(treeWidget, dictItems, rootName):
    #items = sorted(dictItems.iteritems())

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
        varItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled)

    treeWidget.expandAll()

class nineml_simulator(QtGui.QDialog):
    def __init__(self, ninemlComponent, parametersValues = {}, initialConditionsValues = {}, activeRegimes = []):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ninemlSimulator()
        self.ui.setupUi(self)

        self.ninemlComponent  = ninemlComponent
        self.parser           = ExpressionParser()
        self.parameters       = {}
        self.state_variables  = {}
        self.analog_ports     = {}
        self.event_ports      = {}
        self.active_regimes   = {}

        self.connect(self.ui.treeParameters,        QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),   self.slotParameterItemChanged)
        self.connect(self.ui.treeInitialConditions, QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),   self.slotInitialConditionItemChanged)

        collectParameters    ('', self.ninemlComponent, self.parameters,      parametersValues)
        collectStateVariables('', self.ninemlComponent, self.state_variables, initialConditionsValues)
        collectRegimes       ('', self.ninemlComponent, self.active_regimes,  activeRegimes)

        printDictionary(self.parameters)
        printDictionary(self.state_variables)
        printDictionary(self.active_regimes)

        addItemsToTree(self.ui.treeParameters,        self.parameters,      self.ninemlComponent.name)
        addItemsToTree(self.ui.treeInitialConditions, self.state_variables, self.ninemlComponent.name)

    def printResults(self):
        printDictionary(self.parameters)
        printDictionary(self.state_variables)
        printDictionary(self.active_regimes)

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

s = nineml_simulator(coba_iaf, parameters, initial_conditions, active_regimes)
s.exec_()
s.printResults()
#sys.exit(app.exec_())

