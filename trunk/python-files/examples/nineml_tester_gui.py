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
    for key, value in dictionary.iteritems():
        print key + ' : ' + repr(value)
    print '\n'

def printList(l):
    for value in l:
        print repr(value)
    print '\n'

class treeItem:
    typeNoValue = -1
    typeFloat   =  0
    typeInteger =  1
    typeString  =  2
    typeBoolean =  3
    typeList    =  4

    def __init__(self, parent, name, value, data, itemType = typeNoValue):
        self.parent   = parent
        self.children = []
        self.name     = name
        self.value    = value
        self.data     = data
        self.itemType = itemType
        if parent:
            parent.children.append(self)

    @property
    def canonicalName(self):
        if self.parent:
            return self.parent.canonicalName + '.' + self.name
        else:
            return self.name

    @property
    def level(self):
        if self.parent:
            return self.parent.level + 1
        else:
            return 0

    @property
    def hasChildren(self):
        if len(self.children) == 0:
            return False
        else:
            return True
            
    def getDictionary(self):
        dictItems = {}
        if self.itemType != treeItem.typeNoValue:
            dictItems[self.canonicalName] = self.value

        for child in self.children:
            dictItems = dict(dictItems.items() + child.getDictionary().items())

        return dictItems

    def getHTMLFormList(self):
        content = '<ul>'

        if self.itemType == treeItem.typeFloat:
            content += '<li><span>{0}</span><input type="text" name="{1}" value="{2}"/></li>'.format(self.name, self.canonicalName, self.value)
        elif self.itemType == treeItem.typeInteger:
            content += '<li><span>{0}</span><input type="text" name="{1}" value="{2}"/></li>'.format(self.name, self.canonicalName, self.value)
        elif self.itemType == treeItem.typeString:
            content += '<li><span>{0}</span><input type="text" name="{1}" value="{2}"/></li>'.format(self.name, self.canonicalName, self.value)
        elif self.itemType == treeItem.typeBoolean:
            if self.value:
                content += '<li><span>{0}</span><input type="checkbox" name="{1}" checked/></li>'.format(self.name, self.canonicalName)
            else:
                content += '<li><span>{0}</span><input type="checkbox" name="{1}"/></li>'.format(self.name, self.canonicalName)
        elif self.itemType == treeItem.typeList:
            content += '<li><span>{0}</span> <select name="{1}">'.format(self.name, self.canonicalName)
            for available_regime in self.data:
                content += '<option value="{0}">{0}</option>'.format(available_regime)
            content += '</select></li>'
        else:
            content += '<li><span>{0}</span></li>'.format(self.name)

        for child in self.children:
            content += child.getHTMLFormList()

        content += '</ul>'
        return content
        
    def __str__(self):
        indent = self.level * '    '
        res = '{0}- {1}: {2}\n'.format(indent, self.name, self.value)
        for child in sorted(self.children):
            res += str(child)
        return res

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

def getValueFromDictionary(canonicalName, dictValues, defaultValue, excludeRootName = False):
    if excludeRootName:
        names = canonicalName.split('.')
        if len(names) == 1:
            key = names[0]
        else:
            key = '.'.join(names[1:])
    else:
        key = canonicalName
    #print 'canonicalName = {0} -> key = {1}'.format(canonicalName, key)
    if dictValues.has_key(key):
        return dictValues[key]
    else:
        return defaultValue

def isValueInList(canonicalName, listValues, excludeRootName = False):
    if excludeRootName:
        names = canonicalName.split('.')
        if len(names) == 1:
            key = names[0]
        else:
            key = '.'.join(names[1:])
    else:
        key = canonicalName
    #print 'canonicalName = {0} -> key = {1}'.format(canonicalName, key)
    return (key in listValues)

def collectParameters(nodeItem, component, dictParameters, initialValues = {}):
    for obj in component.parameters:
        objName = nodeItem.canonicalName + '.' + obj.name
        value   = getValueFromDictionary(objName, initialValues, 0.0, True)
        dictParameters[objName] = value
        item = treeItem(nodeItem, obj.name, value, None, treeItem.typeFloat)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectParameters(subnodeItem, subcomponent, dictParameters, initialValues)

def collectStateVariables(nodeItem, component, dictStateVariables, initialValues = {}):
    for obj in component.state_variables:
        objName = nodeItem.canonicalName + '.' + obj.name
        value   = getValueFromDictionary(objName, initialValues, 0.0, True)
        dictStateVariables[objName] = value
        item = treeItem(nodeItem, obj.name, value, None, treeItem.typeFloat)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectStateVariables(subnodeItem, subcomponent, dictStateVariables, initialValues)

def collectRegimes(nodeItem, component, dictRegimes, activeRegimes = []):
    available_regimes = []
    active_regime     = None

    for obj in component.regimes:
        available_regimes.append(obj.name)
        objName = nodeItem.canonicalName + '.' + obj.name
        if isValueInList(objName, activeRegimes, True) == True:
            active_regime = obj.name

    if len(available_regimes) > 0:
        if active_regime == None:
            active_regime = available_regimes[0]

        dictRegimes[nodeItem.canonicalName] = active_regime

        nodeItem.itemType = treeItem.typeList
        nodeItem.value    = active_regime
        nodeItem.data     = available_regimes

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectRegimes(subnodeItem, subcomponent, dictRegimes, activeRegimes)

def collectAnalogPorts(nodeItem, component, dictAnalogPortsExpressions, connected_ports, expressions = {}):
    for obj in component.analog_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = nodeItem.canonicalName + '.' + obj.name
            if isValueInList(objName, connected_ports, False) == False:
                value   = getValueFromDictionary(objName, expressions, '', True)
                dictAnalogPortsExpressions[objName] = value
                item = treeItem(nodeItem, obj.name, value, None, treeItem.typeString)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectAnalogPorts(subnodeItem, subcomponent, dictAnalogPortsExpressions, connected_ports, expressions)

def collectEventPorts(nodeItem, component, dictEventPortsExpressions, expressions = {}):
    for obj in component.event_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = nodeItem.canonicalName + '.' + obj.name
            value   = getValueFromDictionary(objName, expressions, '', True)
            dictEventPortsExpressions[objName] = value
            item = treeItem(nodeItem, obj.name, value, None, treeItem.typeString)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectEventPorts(subnodeItem, subcomponent, dictEventPortsExpressions, expressions)

def collectVariablesToReport(nodeItem, component, dictVariablesToReport, variables_to_report = []):
    for obj in component.aliases:
        objName = nodeItem.canonicalName + '.' + obj.lhs
        if isValueInList(objName, variables_to_report, True) == True:
            checked = True
        else:
            checked = False
        dictVariablesToReport[objName] = checked
        item = treeItem(nodeItem, obj.lhs, checked, None, treeItem.typeBoolean)

    for obj in component.state_variables:
        objName = nodeItem.canonicalName + '.' + obj.name
        if isValueInList(objName, variables_to_report, True) == True:
            checked = True
        else:
            checked = False
        dictVariablesToReport[objName] = checked
        item = treeItem(nodeItem, obj.name, checked, None, treeItem.typeBoolean)

    for obj in component.analog_ports:
        objName = nodeItem.canonicalName + '.' + obj.name
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            if isValueInList(objName, variables_to_report, True) == True:
                checked = True
            else:
                checked = False
            dictVariablesToReport[objName] = checked
            item = treeItem(nodeItem, obj.name, checked, None, treeItem.typeBoolean)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectVariablesToReport(subnodeItem, subcomponent, dictVariablesToReport, variables_to_report)

def addItem(parent, item):
    widgetItem = QtGui.QTreeWidgetItem(parent, [item.name, ''])

    # Item's data is always the tree item object
    widgetItem.setData(1, QtCore.Qt.UserRole, QtCore.QVariant(item))

    # The common flags
    widgetItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    # Depending on the type set the text(0) or something else
    if item.itemType == treeItem.typeFloat or item.itemType == treeItem.typeInteger or item.itemType == treeItem.typeString:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsEditable)
        widgetItem.setText(1, str(item.value))
            
    elif item.itemType == treeItem.typeBoolean:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsUserCheckable)
        if item.value:
            widgetItem.setCheckState(0, Qt.Checked)
        else:
            widgetItem.setCheckState(0, Qt.Unchecked)

    elif item.itemType == treeItem.typeList:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsEditable)
        widgetItem.setText(1, str(item.value))

    return widgetItem
    
def addItemsToTree(parent, tree_item):
    new_parent = addItem(parent, tree_item)
    for child in tree_item.children:
        addItemsToTree(new_parent, child)

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

        self.treeParameters = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectParameters(self.treeParameters, self.ninemlComponent, self.parameters, _parameters)

        self.treeInitialConditions = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectStateVariables(self.treeInitialConditions, self.ninemlComponent, self.initial_conditions, _initial_conditions)

        self.treeActiveStates = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectRegimes(self.treeActiveStates, self.ninemlComponent, self.active_states, _active_states)

        self.treeEventPorts = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectEventPorts(self.treeEventPorts, self.ninemlComponent, self.event_ports_expressions, _event_ports_expressions)

        self.treeVariablesToReport = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectVariablesToReport(self.treeVariablesToReport, self.ninemlComponent, self.variables_to_report, _variables_to_report)

        connected_ports = []
        connected_ports = getConnectedAnalogPorts(self.ninemlComponent.name, self.ninemlComponent, connected_ports)
        self.treeAnalogPorts = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectAnalogPorts(self.treeAnalogPorts, self.ninemlComponent, self.analog_ports_expressions, connected_ports, _analog_ports_expressions)
    """
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
    """
    
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

    def printTrees(self):
        print 'tree parameters:'
        print str(self.treeParameters)
        print 'tree initial_conditions:'
        print str(self.treeInitialConditions)
        print 'tree active_states:'
        print str(self.treeActiveStates)
        print 'tree event_ports_expressions:'
        print str(self.treeEventPorts)
        print 'tree analog_ports_expressions:'
        print str(self.treeAnalogPorts)
        print 'tree variables_to_report:'
        print str(self.treeVariablesToReport)

    def printTreeDictionaries(self):
        print 'tree parameters dictionary:'
        printDictionary(self.treeParameters.getDictionary())
        print 'tree initial_conditions dictionary:'
        printDictionary(self.treeInitialConditions.getDictionary())
        print 'tree active_states dictionary:'
        printDictionary(self.treeActiveStates.getDictionary())
        print 'tree event_ports_expressions dictionary:'
        printDictionary(self.treeEventPorts.getDictionary())
        print 'tree analog_ports_expressions dictionary:'
        printDictionary(self.treeAnalogPorts.getDictionary())
        print 'tree variables_to_report dictionary:'
        printDictionary(self.treeVariablesToReport.getDictionary())
        
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

        content = ''
        if len(self.parameters) > 0:
            content += '<h2>Parameters</h2>\n'
            content += self.treeParameters.getHTMLFormList()
            content += '\n'

        if len(self.initial_conditions) > 0:
            content += '<h2>Initial Conditions</h2>\n'
            content += self.treeInitialConditions.getHTMLFormList() + '\n'

        if len(self.active_states) > 0:
            content += '<h2>Active Regimes</h2>\n'
            content += self.treeActiveStates.getHTMLFormList()

        if len(self.analog_ports_expressions) > 0:
            content += '<h2>Analog Ports Expressions</h2>\n'
            content += self.treeAnalogPorts.getHTMLFormList() + '\n'

        if len(self.event_ports_expressions) > 0:
            content += '<h2>Event Ports Expressions</h2>\n'
            content += self.treeEventPorts.getHTMLFormList() + '\n'

        if len(self.variables_to_report) > 0:
            content += '<h2>Variables To Report</h2>\n'
            content += self.treeVariablesToReport.getHTMLFormList() + '\n'

        return form_template.format(self.ninemlComponent.name, content)
        
class nineml_tester_qtGUI(nineml_tester, QtGui.QDialog):
    def __init__(self, ninemlComponent, **kwargs):
        nineml_tester.__init__(self, ninemlComponent, **kwargs)
        QtGui.QDialog.__init__(self)

        self.ui = Ui_ninemlTester()
        self.ui.setupUi(self)
        
        addItemsToTree(self.ui.treeParameters,        self.treeParameters)
        addItemsToTree(self.ui.treeInitialConditions, self.treeInitialConditions)
        addItemsToTree(self.ui.treeAnalogPorts,       self.treeAnalogPorts)
        addItemsToTree(self.ui.treeEventPorts,        self.treeEventPorts)
        addItemsToTree(self.ui.treeRegimes,           self.treeActiveStates)
        addItemsToTree(self.ui.treeResultsVariables,  self.treeVariablesToReport)

        self.connect(self.ui.buttonOk,              QtCore.SIGNAL('clicked()'),                                self.slotOK)
        self.connect(self.ui.buttonCancel,          QtCore.SIGNAL('clicked()'),                                self.slotCancel)
        self.connect(self.ui.treeParameters,        QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotTreeItemChanged)
        self.connect(self.ui.treeInitialConditions, QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotTreeItemChanged)
        self.connect(self.ui.treeRegimes,           QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotRegimesItemDoubleClicked)
        self.connect(self.ui.treeAnalogPorts,       QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotAnalogPortsItemDoubleClicked)
        self.connect(self.ui.treeEventPorts,        QtCore.SIGNAL("itemDoubleClicked(QTreeWidgetItem*, int)"), self.slotEventPortsItemDoubleClicked)
        self.connect(self.ui.treeResultsVariables,  QtCore.SIGNAL("itemChanged(QTreeWidgetItem*, int)"),       self.slotTreeItemChanged)

        self.ui.treeParameters.expandAll()
        self.ui.treeParameters.resizeColumnToContents(0)
        self.ui.treeParameters.resizeColumnToContents(1)
        self.ui.treeInitialConditions.expandAll()
        self.ui.treeInitialConditions.resizeColumnToContents(0)
        self.ui.treeInitialConditions.resizeColumnToContents(1)
        self.ui.treeAnalogPorts.expandAll()
        self.ui.treeAnalogPorts.resizeColumnToContents(0)
        self.ui.treeAnalogPorts.resizeColumnToContents(1)
        self.ui.treeEventPorts.expandAll()
        self.ui.treeEventPorts.resizeColumnToContents(0)
        self.ui.treeEventPorts.resizeColumnToContents(1)
        self.ui.treeRegimes.expandAll()
        self.ui.treeRegimes.resizeColumnToContents(0)
        self.ui.treeRegimes.resizeColumnToContents(1)
        self.ui.treeResultsVariables.expandAll()
        self.ui.treeResultsVariables.resizeColumnToContents(0)
        self.ui.treeResultsVariables.resizeColumnToContents(1)

    def slotOK(self):
        self.done(QtGui.QDialog.Accepted)
    
    def slotCancel(self):
        self.done(QtGui.QDialog.Rejected)

    def slotTreeItemChanged(self, item, column):
        if column == 1:
            data = item.data(1, QtCore.Qt.UserRole)
            if not data:
                return
            tree_item = data.toPyObject()

            if tree_item.itemType == treeItem.typeFloat:
                varValue  = QtCore.QVariant(item.text(1))
                newValue, isOK = varValue.toDouble()
                if not isOK:
                    msg = 'Invalid floating point value ({0}) entered for the item: {1}'.format(item.text(1), item.text(0))
                    QtGui.QMessageBox.warning(None, "NineML", msg)
                    item.setText(1, str(tree_item.value))
                    return
                tree_item.value = newValue
                
            elif tree_item.itemType == treeItem.typeInteger:
                varValue  = QtCore.QVariant(item.text(1))
                newValue, isOK = varValue.toInteger()
                if not isOK:
                    msg = 'Invalid integer value ({0}) entered for the item: {1}'.format(item.text(1), item.text(0))
                    QtGui.QMessageBox.warning(None, "NineML", msg)
                    item.setText(1, str(tree_item.value))
                    return
                tree_item.value = newValue

            elif tree_item.itemType == treeItem.typeString:
                tree_item.value  = str(item.text(1))

            elif tree_item.itemType == treeItem.typeList:
                tree_item.value  = str(item.text(1))

        # Only for boolean data (whith a check-box)
        elif column == 0:
            data = item.data(1, QtCore.Qt.UserRole)
            if not data:
                return
            tree_item = data.toPyObject()
            if tree_item.itemType == treeItem.typeBoolean:
                if item.checkState(0) == Qt.Checked:
                    tree_item.value = True
                else:
                    tree_item.value = False

    def slotEventPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data      = item.data(1, QtCore.Qt.UserRole)
            tree_item = data.toPyObject()
            if tree_item.value == None:
                return
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Event Port Input", "Set the input event expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, new_expression)
                tree_item.value = str(new_expression)

    def slotAnalogPortsItemDoubleClicked(self, item, column):
        if column == 1:
            data      = item.data(1, QtCore.Qt.UserRole)
            tree_item = data.toPyObject()
            if tree_item.value == None:
                return
            old_expression = item.text(1)
            new_expression, ok = QtGui.QInputDialog.getText(self, "Analog Port Input", "Set the analog port input expression:", QtGui.QLineEdit.Normal, old_expression)
            if ok:
                item.setText(1, new_expression)
                tree_item.value = str(new_expression)

    def slotRegimesItemDoubleClicked(self, item, column):
        if column == 1:
            data      = item.data(1, QtCore.Qt.UserRole)
            tree_item = data.toPyObject()
            if tree_item.value == None:
                return
            available_regimes = tree_item.data
            active_state, ok = QtGui.QInputDialog.getItem(self, "Available regimes", "Select the new active regime:", available_regimes, 0, False)
            if ok:
                item.setText(1, active_state)
                tree_item.value = str(active_state)

