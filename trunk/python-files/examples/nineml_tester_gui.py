#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation
import os, sys, math
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_tester_ui import Ui_ninemlTester

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
        parameters = collectParameters(rootName + name, subcomponent, parameters, initialValues)

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
        stateVariables = collectStateVariables(rootName + name, subcomponent, stateVariables, initialValues)

    return stateVariables

# Regimes are different from other elements, for they appear in the following form:
#  - Component | regime
#     - Component | regime
#     - Component | regime
#        - Component | regime
# that is components dont have other children except their subnodes,
# and the regime is linked to the item holding the component name
def collectRegimes(root, component, regimes, activeRegimes = []):
    if root == '':
        root = component.name
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
        regimes = collectRegimes(rootName + name, subcomponent, regimes, activeRegimes)

    return regimes

# This function collects canonical names of all ports that are connected
def getConnectedAnalogPorts(root_model_name, component, connected_ports):
    rootName = root_model_name
    if rootName != '':
        rootName += '.'

    for port_connection in component.portconnections:
        connected_ports.append(rootName + '.'.join(port_connection[0].loctuple))
        connected_ports.append(rootName + '.'.join(port_connection[1].loctuple))

    for name, subcomponent in component.subnodes.items():
        connected_ports = getConnectedAnalogPorts(rootName + name, subcomponent, connected_ports)

    return connected_ports
    
def collectAnalogPorts(root_model_name, component, analog_ports, connected_ports, expressions):
    rootName = root_model_name
    if rootName != '':
        rootName += '.'
    for obj in component.analog_ports:
        # Add only if it is inlet or reduce port
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = rootName + obj.name
            # Add only if it is not in the list of connected ports
            if not objName in connected_ports:
                # If the initial expression is given set it; otherwise set the empty string
                if objName in expressions:
                    analog_ports[objName] = expressions[objName]
                else:
                    analog_ports[objName] = ''

    for name, subcomponent in component.subnodes.items():
        analog_ports = collectAnalogPorts(rootName + name, subcomponent, analog_ports, connected_ports, expressions)

    return analog_ports

def collectEventPorts(root, component, event_ports, expressions):
    rootName = root
    if rootName != '':
        rootName += '.'
    for obj in component.event_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = rootName + obj.name
            if objName in expressions:
                event_ports[objName] = expressions[objName]
            else:
                event_ports[objName] = ''

    for name, subcomponent in component.subnodes.items():
        event_ports = collectEventPorts(rootName + name, subcomponent, event_ports, expressions)

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
        results_variables = collectResultsVariables(rootName + name, subcomponent, results_variables, checkedVariables)

    return results_variables

def addItemsToTree(treeWidget, dictItems, rootName, editableItems = True):
    if len(dictItems) == 0:
        return
        
    rootItem = QtGui.QTreeWidgetItem(treeWidget, [rootName, ''])
    rootItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

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
        item_path = names # Achtung, achtung! Not: [:-1]

        currentItem = rootItem

        # First find the parent QTreeWidgetItem
        for item in item_path:
            found = False
            cname = currentItem.text(0)
            # First check whether it is equal to the current item
            # If it is not, check its children
            # This is useful if the root component has got regimes
            if item == cname:
                found = True
                currentItem = currentItem
            else:
                for c in range(0, currentItem.childCount()):
                    child = currentItem.child(c)
                    cname = child.text(0)
                    if item == cname:
                        found = True
                        currentItem = currentItem.child(c)
                        break

            if found == False:
                currentItem = QtGui.QTreeWidgetItem(currentItem, [item, ''])

        currentItem.setText(1, value[1])
        varData = QtCore.QVariant((key, value[0]))
        currentItem.setData(1, QtCore.Qt.UserRole, varData)
        currentItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)

    treeWidget.expandAll()
    treeWidget.resizeColumnToContents(0)
    treeWidget.resizeColumnToContents(1)

class nineml_tester:
    def __init__(self, ninemlComponent, **kwargs):
        _parameters               = kwargs.get('parameters',               {})
        _initial_conditions       = kwargs.get('initial_conditions',       {})
        _active_states            = kwargs.get('active_states',            [])
        _variables_to_report      = kwargs.get('variables_to_report',      [])
        _analog_ports_expressions = kwargs.get('analog_ports_expressions', {})
        _event_ports_expressions  = kwargs.get('event_ports_expressions',  {})

        self.ninemlComponent   = ninemlComponent
        # Dictionaries 'key' : floating-point-value
        self.parameters         = {}
        self.initial_conditions = {}
        # Dictionaries: 'key' : 'expression'
        self.analog_ports_expressions = {}
        self.event_ports_expressions  = {}
        # Dictionary 'key' : [ [list-of-available-states], 'current-active-state']
        self.active_states = {}
        # Dictionaries 'key' : boolean-value
        self.variables_to_report = {}

        collectParameters      ('', self.ninemlComponent, self.parameters,               _parameters)
        collectStateVariables  ('', self.ninemlComponent, self.initial_conditions,       _initial_conditions)
        collectRegimes         ('', self.ninemlComponent, self.active_states,            _active_states)
        collectEventPorts      ('', self.ninemlComponent, self.event_ports_expressions,  _event_ports_expressions)
        collectResultsVariables('', self.ninemlComponent, self.variables_to_report,      _variables_to_report)
        connected_ports = []
        connected_ports = getConnectedAnalogPorts('', self.ninemlComponent, connected_ports)
        collectAnalogPorts('', self.ninemlComponent, self.analog_ports_expressions, connected_ports, _analog_ports_expressions)

    @property
    def parametersValues(self):
        return self.parameters

    @property
    def initialConditions(self):
        return self.initial_conditions

    @property
    def analogPortsExpressions(self):
        return self.analog_ports_expressions

    @property
    def eventPortsExpressions(self):
        return self.event_ports_expressions

    @property
    def activeStates(self):
        results = []
        for key, value in self.active_states.items():
            results.append(key + '.' + value[1])
        return results

    @property
    def variablesToReport(self):
        results = []
        for key, value in self.variables_to_report.items():
            if value:
                results.append(key)
        return results

    def printResults(self):
        print 'parameters:'
        printDictionary(self.parameters)
        print 'initial_conditions:'
        printDictionary(self.initial_conditions)
        print 'active_states:'
        printDictionary(self.active_states)
        print 'analog_ports_expressions:'
        printDictionary(self.analog_ports_expressions)
        print 'event_ports_expressions:'
        printDictionary(self.event_ports_expressions)
        print 'variables_to_report:'
        printDictionary(self.variables_to_report)

class nineml_tester_htmlGUI(nineml_tester):
    def __init__(self, ninemlComponent, **kwargs):
        nineml_tester.__init__(self, ninemlComponent, **kwargs)

    def generateHTMLForm(self):
        form_template = """
        <form method="post">
            <h1>NineMl component: {0}</h1>
            {1}
            <br/>
            <input type="submit" value="Submit" />
        </form>
        """

        if len(self.parameters) > 0:
            content = '<form method="post"><h2>Parameters</h2>\n'
            content += '<ul>'
            for name, value in self.parameters.items():
                content += '<li><span>{0}</span> <input type="text" name="{0}" value="{1}"/> </li>'.format(name, value)
            content += '</ul>'
            content += '\n'

        if len(self.initial_conditions) > 0:
            content += '<h2>Initial Conditions</h2>\n'
            content += '<ul>'
            for name, value in self.initial_conditions.items():
                content += '<li><span>{0}</span> <input type="text" name="{0}" value="{1}"/> </li>'.format(name, value)
            content += '</ul>'
            content += '\n'

        if len(self.active_states) > 0:
            content += '<h2>Active Regimes</h2>\n'
            content += '<ul>'
            for name, value in self.active_states.items():
                content += '<li><span>{0}</span> <select name="{0}">'.format(name)
                for available_regime in value[0]:
                    content += '<option value="{0}">{0}</option>'.format(available_regime)
                content += '</select> </li>'
            content += '</ul>'
            content += '\n'

        if len(self.analog_ports_expressions) > 0:
            content += '<h2>Analog Ports Expressions</h2>\n'
            content += '<ul>'
            for name, value in self.analog_ports_expressions.items():
                content += '<li><span>{0}</span> <input type="text" name="{0}" value="{1}"/> </li>'.format(name, value)
            content += '</ul>'
            content += '\n'

        if len(self.event_ports_expressions) > 0:
            content += '<h2>Event Ports Expressions</h2>\n'
            content += '<ul>'
            for name, value in self.event_ports_expressions.items():
                content += '<li><span>{0}</span> <input type="text" name="{0}" value="{1}"/> </li>'.format(name, value)
            content += '</ul>'
            content += '\n'

        if len(self.variables_to_report) > 0:
            content += '<h2>Variables To Report</h2>\n'
            content += '<ul>'
            for name, value in self.variables_to_report.items():
                content += '<li><span>{0}</span> <input type="text" name="{0}" value="{1}"/> </li>'.format(name, value)
            content += '</ul>'
            content += '\n'

        return form_template.format(self.ninemlComponent.name, content)
        
class nineml_tester_qtGUI(nineml_tester, QtGui.QDialog):
    def __init__(self, ninemlComponent, **kwargs):
        nineml_tester.__init__(self, ninemlComponent, **kwargs)
        QtGui.QDialog.__init__(self)

        self.ui = Ui_ninemlTester()
        self.ui.setupUi(self)
        
        self.connect(self.ui.buttonOk,              QtCore.SIGNAL('clicked()'),                                self.slotOK)
        self.connect(self.ui.buttonCancel,          QtCore.SIGNAL('clicked()'),                                self.slotCancel)
        self.connect(self.ui.treeParameters,        QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotParameterItemChanged)
        self.connect(self.ui.treeInitialConditions, QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotInitialConditionItemChanged)
        self.connect(self.ui.treeRegimes,           QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotRegimesItemDoubleClicked)
        self.connect(self.ui.treeAnalogPorts,       QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotAnalogPortsItemDoubleClicked)
        self.connect(self.ui.treeEventPorts,        QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotEventPortsItemDoubleClicked)
        self.connect(self.ui.treeResultsVariables,  QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotResultsVariablesItemChanged)

        addItemsToTree                (self.ui.treeParameters,        self.parameters,               self.ninemlComponent.name)
        addItemsToTree                (self.ui.treeInitialConditions, self.initial_conditions,       self.ninemlComponent.name)
        addItemsToTree                (self.ui.treeAnalogPorts,       self.analog_ports_expressions, self.ninemlComponent.name, False)
        addItemsToTree                (self.ui.treeEventPorts,        self.event_ports_expressions,  self.ninemlComponent.name, False)
        addItemsToRegimesTree         (self.ui.treeRegimes,           self.active_states,            self.ninemlComponent.name)
        addItemsToResultsVariablesTree(self.ui.treeResultsVariables,  self.variables_to_report,      self.ninemlComponent.name)

    def slotOK(self):
        self.done(QtGui.QDialog.Accepted)
    
    def slotCancel(self):
        self.done(QtGui.QDialog.Rejected)

    def slotParameterItemChanged(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            varValue = QtCore.QVariant(item.text(1))

            value, isOK = varValue.toDouble()
            if not isOK:
                QtGui.QMessageBox.warning(None, "NineML", "Invalid value entered for the item: " + item.text(0))
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
                QtGui.QMessageBox.warning(None, "NineML", "Invalid value entered for the item: " + item.text(0))
                item.setText(1, str(self.initial_conditions[key]))
                return

            if self.initial_conditions[key] != value:
                self.initial_conditions[key] = value

    def slotEventPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Event Port Input", "Set the input event expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, new_expression)
                self.event_ports_expressions[key] = str(new_expression)

    def slotAnalogPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key  = str(data.toString())
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Analog Port Input", "Set the analog port input expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, str(new_expression))
                self.analog_ports_expressions[key] = str(new_expression)

    def slotRegimesItemDoubleClicked(self, item, column):
        if column == 1:
            data = item.data(column, QtCore.Qt.UserRole)
            key, available_regimes = data.toPyObject()
            active_state, ok = QtGui.QInputDialog.getItem(self, "Available regimes", "Select the new active regime:", available_regimes, 0, False)
            if ok:
                item.setText(1, active_state)
                self.active_states[key][1] = str(active_state)

    def slotResultsVariablesItemChanged(self, item, column):
        if column == 0:
            data = item.data(0, QtCore.Qt.UserRole)
            key  = str(data.toString())
            if item.checkState(0) == Qt.Checked:
                self.variables_to_report[key] = True
            else:
                self.variables_to_report[key] = False

