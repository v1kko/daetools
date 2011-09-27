#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
import os, sys, math, collections
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from nineml_tester_ui import Ui_ninemlTester
from expression_parser import ExpressionParser
from units_parser import UnitsParser
from StringIO import StringIO
import dot2tex

def printDictionary(dictionary):
    for key, value in dictionary.iteritems():
        print('    {0} : {1}'.format(key, repr(value)))

def printList(l):
    for value in l:
        print('    {0}'.format(repr(value)))

def updateDictionary(dictOld, dictNew):
    for key, value in dictNew.items():
        if key in dictOld:
            if isinstance(dictOld[key], float):
                new_value = float(value)
            elif isinstance(dictOld[key], basestring):
                new_value = str(value)
            else:
                new_value = value
            if dictOld[key] != new_value:
                dictOld[key] = new_value

def updateTree(item, dictNew):
    key = item.canonicalName
    if key in dictNew:
        if item.itemType == treeItem.typeFloat:
            item.value = float(dictNew[key])

        elif item.itemType == treeItem.typeInteger:
            item.value = int(dictNew[key])

        elif item.itemType == treeItem.typeString:
            item.value = str(dictNew[key])

        elif item.itemType == treeItem.typeBoolean:
            item.value = bool(dictNew[key])

        elif item.itemType == treeItem.typeList:
            item.value = str(dictNew[key])

    for child in item.children:
        updateTree(child, dictNew)

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
    #print('canonicalName = {0} -> key = {1}'.format(canonicalName, key))
    if key in dictValues:
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
    #print('canonicalName = {0} -> key = {1}'.format(canonicalName, key))
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

def collectRegimes(nodeItem, component, dictRegimes, activeRegimes = {}):
    available_regimes = []
    active_regime     = None

    for obj in component.regimes:
        available_regimes.append(obj.name)
        objName = nodeItem.canonicalName + '.' + obj.name
        value   = getValueFromDictionary(nodeItem.canonicalName, activeRegimes, None, True)
        if value == obj.name:
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
                value   = str(getValueFromDictionary(objName, expressions, '', True))
                dictAnalogPortsExpressions[objName] = value
                item = treeItem(nodeItem, obj.name, value, None, treeItem.typeString)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectAnalogPorts(subnodeItem, subcomponent, dictAnalogPortsExpressions, connected_ports, expressions)

def collectEventPorts(nodeItem, component, dictEventPortsExpressions, expressions = {}):
    for obj in component.event_ports:
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            objName = nodeItem.canonicalName + '.' + obj.name
            value   = str(getValueFromDictionary(objName, expressions, '', True))
            dictEventPortsExpressions[objName] = value
            item = treeItem(nodeItem, obj.name, value, None, treeItem.typeString)

    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectEventPorts(subnodeItem, subcomponent, dictEventPortsExpressions, expressions)

def collectVariablesToReport(nodeItem, component, dictVariablesToReport, variables_to_report = {}):
    for obj in component.aliases:
        objName = nodeItem.canonicalName + '.' + obj.lhs
        checked = getValueFromDictionary(objName, variables_to_report, False, True)
        dictVariablesToReport[objName] = checked
        item = treeItem(nodeItem, obj.lhs, checked, None, treeItem.typeBoolean)

    for obj in component.state_variables:
        objName = nodeItem.canonicalName + '.' + obj.name
        checked = getValueFromDictionary(objName, variables_to_report, False, True)
        dictVariablesToReport[objName] = checked
        item = treeItem(nodeItem, obj.name, checked, None, treeItem.typeBoolean)

    # Get crashes with this included
    """
    for obj in component.analog_ports:
        objName = nodeItem.canonicalName + '.' + obj.name
        if (obj.mode == 'recv') or (obj.mode == 'reduce'):
            checked = getValueFromDictionary(objName, variables_to_report, False, True)
            dictVariablesToReport[objName] = checked
            item = treeItem(nodeItem, obj.name, checked, None, treeItem.typeBoolean)
    """
    
    for name, subcomponent in component.subnodes.items():
        subnodeItem = treeItem(nodeItem, name, None, None, treeItem.typeNoValue)
        collectVariablesToReport(subnodeItem, subcomponent, dictVariablesToReport, variables_to_report)

def addItem(treeWidget, parent, item):
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
    
def addItemsToTree(treeWidget, parent, tree_item):
    new_parent = addItem(treeWidget, parent, tree_item)
    for child in tree_item.children:
        addItemsToTree(treeWidget, new_parent, child)

class nineml_component_qtGUI(QtGui.QDialog):
    def __init__(self, inspector):
        QtGui.QDialog.__init__(self)
        self.ui = Ui_ninemlTester()
        self.ui.setupUi(self)

        validator = QtGui.QDoubleValidator(self)
        self.ui.timeHorizonSLineEdit.setValidator(validator)
        self.ui.reportingIntervalSLineEdit.setValidator(validator)

        self.inspector = inspector
        addItemsToTree(self.ui.treeParameters,          self.ui.treeParameters,        self.inspector.treeParameters)
        addItemsToTree(self.ui.treeInitialConditions,   self.ui.treeInitialConditions, self.inspector.treeInitialConditions)
        addItemsToTree(self.ui.treeAnalogPorts,         self.ui.treeAnalogPorts,       self.inspector.treeAnalogPorts)
        addItemsToTree(self.ui.treeEventPorts,          self.ui.treeEventPorts,        self.inspector.treeEventPorts)
        addItemsToTree(self.ui.treeRegimes,             self.ui.treeRegimes,           self.inspector.treeActiveStates)
        addItemsToTree(self.ui.treeResultsVariables,    self.ui.treeResultsVariables,  self.inspector.treeVariablesToReport)

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
        #self.ui.treeResultsVariables.resizeColumnToContents(1)

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

        # Only for boolean data (with a check-box)
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

def latex_table(header_flags, header_items, rows_items, caption = ''):
    table_template = """
\\begin{{table}}[placement=!h]
{3}
\\begin{{center}}
\\begin{{tabular}}{{ {0} }}
\\hline
{1}
\\hline
{2}
\\end{{tabular}}
\\end{{center}}
\\end{{table}}
    """
    #flags  = ' | ' + ' | '.join(header_flags) + ' | '
    flags  = ' '.join(header_flags)
    header = ' & '.join(header_items) + ' \\\\'
    rows = ''
    for item in rows_items:
        rows += ' & '.join(item) + ' \\\\ \n'
    if caption:
        title = '\\caption{{{0}}} \n'.format(caption)
    else:
        title = ''
    return table_template.format(flags, header, rows, title)

def latex_regime_table(header_flags, regime, odes, _on_conditions, _on_events, caption = ''):
    table_template = """
\\begin{{table}}[placement=!h]
{3}
\\begin{{center}}
\\begin{{tabular}}{{ {0} }}
\\hline
\\multicolumn{{2}}{{c}}{{ {1} }} \\\\
\\hline
{2}
\\end{{tabular}}
\\end{{center}}
\\end{{table}}
    """
    flags  = ' '.join(header_flags)
    rows = '\\multirow{{{0}}}{{*}}{{ODEs}} & '.format(len(odes))
    for item in odes:
        rows += item + ' \\\\ \n'

    rows += '\\multirow{{{0}}}{{*}}{{On condition}} & '.format(len(_on_conditions))
    for item in _on_conditions:
        rows += item + ' \\\\ \n'

    rows += '\\multirow{{{0}}}{{*}}{{On event}} & '.format(len(_on_events))
    for item in _on_events:
        rows += item + ' \\\\ \n'

    if caption:
        title = '\\caption{{{0}}} \n'.format(caption)
    else:
        title = ''
    return table_template.format(flags, regime, rows, title)

class nineml_component_inspector:
    categoryParameters              = '___PARAMETERS___'
    categoryInitialConditions       = '___INITIAL_CONDITIONS___'
    categoryActiveStates            = '___ACTIVE_STATES___'
    categoryAnalogPortsExpressions  = '___INLET_ANALOG_PORTS_EXPRESSIONS___'
    categoryEventPortsExpressions   = '___EVENT_PORTS_EXPRESSIONS___'
    categoryVariablesToReport       = '___VARIABLES_TO_REPORT___'

    begin_itemize = '\\begin{itemize}\n'
    item          = '\\item '
    end_itemize   = '\\end{itemize}\n\n'

    def __init__(self):
        # NineML component
        self.ninemlComponent = None

        self.timeHorizon       = 0.0
        self.reportingInterval = 0.0
        
        # Dictionaries 'key' : floating-point-value
        self.parameters                 = {}
        self.initial_conditions         = {}
        # Dictionaries: 'key' : 'expression'
        self.analog_ports_expressions   = {}
        self.event_ports_expressions    = {}
        # Dictionary 'key' : 'current-active-state'
        self.active_regimes             = {}
        # Dictionaries 'key' : boolean-value
        self.variables_to_report        = {}

        self.treeParameters         = None
        self.treeInitialConditions  = None
        self.treeActiveStates       = None
        self.treeEventPorts         = None
        self.treeVariablesToReport  = None
        self.treeAnalogPorts        = None

    def inspect(self, component, **kwargs):
        if isinstance(component, nineml.abstraction_layer.ComponentClass):
            self.ninemlComponent = component
        elif isinstance(component, basestring):
            self.ninemlComponent = nineml.abstraction_layer.parse(component)
        else:
            raise RuntimeError('the input NineML component must be either ComponentClass or a path to xml file')
        if not self.ninemlComponent:
            raise RuntimeError('Invalid input NineML component')

        _parameters               = kwargs.get('parameters',               {})
        _initial_conditions       = kwargs.get('initial_conditions',       {})
        _active_regimes           = kwargs.get('active_regimes',           {})
        _analog_ports_expressions = kwargs.get('analog_ports_expressions', {})
        _event_ports_expressions  = kwargs.get('event_ports_expressions',  {})
        _variables_to_report      = kwargs.get('variables_to_report',      {})

        if not isinstance(_parameters, dict):
            raise RuntimeError('parameters argument must be a dictionary')
        if not isinstance(_initial_conditions, dict):
            raise RuntimeError('initial_conditions argument must be a dictionary')
        if not isinstance(_active_regimes, dict):
            raise RuntimeError('active_regimes argument must be a dictionary')
        if not isinstance(_analog_ports_expressions, dict):
            raise RuntimeError('analog_ports_expressions argument must be a dictionary')
        if not isinstance(_event_ports_expressions, dict):
            raise RuntimeError('event_ports_expressions argument must be a dictionary')
        if not isinstance(_variables_to_report, dict):
            raise RuntimeError('variables_to_report argument must be a dictionary')

        if 'timeHorizon' in kwargs:
            self.timeHorizon = float(kwargs.get('timeHorizon'))
        if 'reportingInterval' in kwargs:
            self.reportingInterval = float(kwargs.get('reportingInterval'))

        self.treeParameters = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectParameters(self.treeParameters, self.ninemlComponent, self.parameters, _parameters)

        self.treeInitialConditions = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectStateVariables(self.treeInitialConditions, self.ninemlComponent, self.initial_conditions, _initial_conditions)

        self.treeActiveStates = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectRegimes(self.treeActiveStates, self.ninemlComponent, self.active_regimes, _active_regimes)

        self.treeEventPorts = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectEventPorts(self.treeEventPorts, self.ninemlComponent, self.event_ports_expressions, _event_ports_expressions)

        self.treeVariablesToReport = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectVariablesToReport(self.treeVariablesToReport, self.ninemlComponent, self.variables_to_report, _variables_to_report)

        connected_ports = []
        connected_ports = getConnectedAnalogPorts(self.ninemlComponent.name, self.ninemlComponent, connected_ports)
        self.treeAnalogPorts = treeItem(None, self.ninemlComponent.name, None, None, treeItem.typeNoValue)
        collectAnalogPorts(self.treeAnalogPorts, self.ninemlComponent, self.analog_ports_expressions, connected_ports, _analog_ports_expressions)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        res = []
        res.append('tree parameters:')
        res.append(str(self.treeParameters))
        res.append('tree initial_conditions:')
        res.append(str(self.treeInitialConditions))
        return '\n'.join(res)

    def printCollectedData(self):
        print('timeHorizon: ' + str(self.timeHorizon))
        print('reportingInterval: ' + str(self.reportingInterval))
        print('parameters:')
        printDictionary(self.parameters)
        print('initial_conditions:')
        printDictionary(self.initial_conditions)
        print('active_regimes:')
        printDictionary(self.active_regimes)
        print('analog_ports_expressions:')
        printDictionary(self.analog_ports_expressions)
        print('event_ports_expressions:')
        printDictionary(self.event_ports_expressions)
        print('variables_to_report:')
        printDictionary(self.variables_to_report)
    
    def printTrees(self):
        print('timeHorizon: ' + str(self.timeHorizon))
        print('reportingInterval: ' + str(self.reportingInterval))
        print('tree parameters:')
        print(str(self.treeParameters))
        print('tree initial_conditions:')
        print(str(self.treeInitialConditions))
        print('tree active_regimes:')
        print(str(self.treeActiveStates))
        print('tree event_ports_expressions:')
        print(str(self.treeEventPorts))
        print('tree analog_ports_expressions:')
        print(str(self.treeAnalogPorts))
        print('tree variables_to_report:')
        print(str(self.treeVariablesToReport))

    def printTreeDictionaries(self):
        print('tree parameters dictionary:')
        printDictionary(self.treeParameters.getDictionary())
        print('tree initial_conditions dictionary:')
        printDictionary(self.treeInitialConditions.getDictionary())
        print('tree active_regimes dictionary:')
        printDictionary(self.treeActiveStates.getDictionary())
        print('tree event_ports_expressions dictionary:')
        printDictionary(self.treeEventPorts.getDictionary())
        print('tree analog_ports_expressions dictionary:')
        printDictionary(self.treeAnalogPorts.getDictionary())
        print('tree variables_to_report dictionary:')
        printDictionary(self.treeVariablesToReport.getDictionary())

    def updateData(self, **kwargs):
        _parameters               = kwargs.get('parameters',               {})
        _initial_conditions       = kwargs.get('initial_conditions',       {})
        _active_regimes           = kwargs.get('active_regimes',           {})
        _analog_ports_expressions = kwargs.get('analog_ports_expressions', {})
        _event_ports_expressions  = kwargs.get('event_ports_expressions',  {})
        _variables_to_report      = kwargs.get('variables_to_report',      {})

        if not isinstance(_parameters, dict):
            raise RuntimeError('parameters argument must be a dictionary')
        if not isinstance(_initial_conditions, dict):
            raise RuntimeError('initial_conditions argument must be a dictionary')
        if not isinstance(_active_regimes, dict):
            raise RuntimeError('active_regimes argument must be a dictionary')
        if not isinstance(_analog_ports_expressions, dict):
            raise RuntimeError('analog_ports_expressions argument must be a dictionary')
        if not isinstance(_event_ports_expressions, dict):
            raise RuntimeError('event_ports_expressions argument must be a dictionary')
        if not isinstance(_variables_to_report, dict):
            raise RuntimeError('variables_to_report argument must be a dictionary')

        if 'timeHorizon' in kwargs:
            self.timeHorizon = float(kwargs.get('timeHorizon'))
        if 'reportingInterval' in kwargs:
            self.reportingInterval = float(kwargs.get('reportingInterval'))
        updateDictionary(self.parameters,               _parameters)
        updateDictionary(self.initial_conditions,       _initial_conditions)
        updateDictionary(self.active_regimes,           _active_regimes)
        updateDictionary(self.analog_ports_expressions, _analog_ports_expressions)
        updateDictionary(self.event_ports_expressions,  _event_ports_expressions)
        updateDictionary(self.variables_to_report,      _variables_to_report)

    def updateTrees(self):
        updateTree(self.treeParameters,         self.parameters)
        updateTree(self.treeInitialConditions,  self.initial_conditions)
        updateTree(self.treeActiveStates,       self.active_regimes)
        updateTree(self.treeAnalogPorts,        self.analog_ports_expressions)
        updateTree(self.treeEventPorts,         self.event_ports_expressions)
        updateTree(self.treeVariablesToReport,  self.variables_to_report)

    def getComponentXMLSourceCode(self, flatten = True):
        f = StringIO()
        nineml.al.writers.XMLWriter.write(self.ninemlComponent, f, flatten)
        xmlSource = f.getvalue()
        return xmlSource

    def getComponentALObject(self):
        return self.ninemlComponent

    def writeComponentToXMLFile(self, filename, flatten = True):
        if not self.ninemlComponent or not isinstance(self.ninemlComponent, nineml.abstraction_layer.ComponentClass):
            raise RuntimeError('Invalid input NineML component')
        nineml.al.writers.XMLWriter.write(self.ninemlComponent, filename, flatten)

    def generateHTMLForm(self):
        if not self.ninemlComponent or not isinstance(self.ninemlComponent, nineml.abstraction_layer.ComponentClass):
            raise RuntimeError('Invalid input NineML component')
        
        content = ''
        content += '<fieldset>'
        content += '<legend>General</legend>'
        content += '<label for="testName">Test name</label>'
        content += '<input type="text" name="testName" value="Dummy test"/><br/>'
        content += '<label for="testDescription">Test description</label>'
        content += '<textarea name="testDescription" rows="2" cols="50">Dummy test description</textarea><br/>'
        content += '</fieldset>\n'

        content += '<fieldset>'
        content += '<legend>Simulation</legend>'
        content += '<label for="timeHorizon">Time horizon</label>'
        content += '<input type="text" name="timeHorizon" value="{0}"/><br/>'.format(self.timeHorizon)
        content += '<label for="reportingInterval">Reporting interval</label>'
        content += '<input type="text" name="reportingInterval" value="{0}"/><br/>'.format(self.reportingInterval)
        content += '</fieldset>\n'

        if len(self.parameters) > 0:
            content += '<fieldset>'
            content += '<legend>Parameters</legend>\n'
            content += self._generateHTMLFormTree(self.treeParameters, nineml_component_inspector.categoryParameters)
            content += '</fieldset>\n'

        if len(self.initial_conditions) > 0:
            content += '<fieldset>'
            content += '<legend>Initial conditions</legend>\n'
            content += self._generateHTMLFormTree(self.treeInitialConditions, nineml_component_inspector.categoryInitialConditions) + '\n'
            content += '</fieldset>\n'

        if len(self.active_regimes) > 0:
            content += '<fieldset>'
            content += '<legend>Initially active regimes</legend>\n'
            content += self._generateHTMLFormTree(self.treeActiveStates, nineml_component_inspector.categoryActiveStates)
            content += '</fieldset>\n'

        if len(self.analog_ports_expressions) > 0:
            content += '<fieldset>'
            content += '<legend>Analog-ports inputs</legend>\n'
            content += self._generateHTMLFormTree(self.treeAnalogPorts, nineml_component_inspector.categoryAnalogPortsExpressions) + '\n'
            content += '</fieldset>\n'

        if len(self.event_ports_expressions) > 0:
            content += '<fieldset>'
            content += '<legend>Event-ports inputs</legend>\n'
            content += self._generateHTMLFormTree(self.treeEventPorts, nineml_component_inspector.categoryEventPortsExpressions) + '\n'
            content += '</fieldset>\n'

        if len(self.variables_to_report) > 0:
            content += '<fieldset>'
            content +='<legend>Variables to report</legend>\n'
            content += self._generateHTMLFormTree(self.treeVariablesToReport, nineml_component_inspector.categoryVariablesToReport) + '\n'
            content += '</fieldset>\n'

        return content

    def _generateHTMLFormTree(self, item, category = ''):
        if category == '':
            inputName = item.canonicalName
        else:
            inputName = category + '.' + item.canonicalName

        content = '<ul>'
        if item.itemType == treeItem.typeFloat:
            content += '<li><label for="{1}">{0}</label><input type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.value)

        elif item.itemType == treeItem.typeInteger:
            content += '<li><label for="{1}">{0}</label><input type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.value)

        elif item.itemType == treeItem.typeString:
            content += '<li><label for="{1}">{0}</label><input type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.value)

        elif item.itemType == treeItem.typeBoolean:
            if item.value:
                content += '<li><label for="{1}">{0}</label><input type="checkbox" name="{1}" checked/></li>'.format(item.name, inputName)
            else:
                content += '<li><label for="{1}">{0}</label><input type="checkbox" name="{1}"/></li>'.format(item.name, inputName)

        elif item.itemType == treeItem.typeList:
            if isinstance(item.data, collections.Iterable) and len(item.data) > 0:
                content += '<li><label for="{1}">{0}</label> <select name="{1}">'.format(item.name, inputName)
                for available_regime in item.data:
                    if available_regime == item.value:
                        content += '<option value="{0}" selected>{0}</option>'.format(available_regime)
                    else:
                        content += '<option value="{0}">{0}</option>'.format(available_regime)
                content += '</select></li>'
            else:
                content += '<li>{0}</li>'.format(item.name)

        else:
            content += '<li>{0}</li>'.format(item.name)

        for child in item.children:
            content += self._generateHTMLFormTree(child, category)

        content += '</ul>'
        return content

    def generateHTMLReport(self):
        if not self.ninemlComponent or not isinstance(self.ninemlComponent, nineml.abstraction_layer.ComponentClass):
            raise RuntimeError('Invalid input NineML component')

        form_template = """
        <h1>NineMl component: {0}</h1>
        {1}
        """
        content = ''
        if len(self.parameters) > 0:
            content += '<h2>Parameters</h2>\n'
            content += self._generateHTMLReportTree(self.treeParameters)
            content += '\n'

        if len(self.initial_conditions) > 0:
            content += '<h2>Initial Conditions</h2>\n'
            content += self._generateHTMLReportTree(self.treeInitialConditions) + '\n'

        if len(self.active_regimes) > 0:
            content += '<h2>Active Regimes</h2>\n'
            content += self._generateHTMLReportTree(self.treeActiveStates)

        if len(self.analog_ports_expressions) > 0:
            content += '<h2>Analog Ports Expressions</h2>\n'
            content += self._generateHTMLReportTree(self.treeAnalogPorts) + '\n'

        if len(self.event_ports_expressions) > 0:
            content += '<h2>Event Ports Expressions</h2>\n'
            content += self._generateHTMLReportTree(self.treeEventPorts) + '\n'

        if len(self.variables_to_report) > 0:
            content += '<h2>Variables To Report</h2>\n'
            content += self._generateHTMLReportTree(self.treeVariablesToReport) + '\n'

        return form_template.format(self.ninemlComponent.name, content)

    def _generateHTMLReportTree(self, item):
        content = '<ul>'
        if item.itemType == treeItem.typeFloat:
            content += '<li>{0} ({1})</li>'.format(item.name, item.value)

        elif item.itemType == treeItem.typeInteger:
            content += '<li>{0} ({1})</li>'.format(item.name, item.value)

        elif item.itemType == treeItem.typeString:
            content += '<li>{0} ({1})</li>'.format(item.name, item.value)

        elif item.itemType == treeItem.typeBoolean:
            content += '<li>{0} ({1})</li>'.format(item.name, item.value)

        elif item.itemType == treeItem.typeList:
            content += '<li>{0} ({1})</li>'.format(item.name, item.value)

        else:
            content += '<li>{0}</li>'.format(item.name)

        for child in item.children:
            content += self._generateHTMLReportTree(child)

        content += '</ul>'
        return content
        
    def showQtGUI(self):
        if not self.ninemlComponent or not isinstance(self.ninemlComponent, nineml.abstraction_layer.ComponentClass):
            raise RuntimeError('Invalid input NineML component')

        app = QtGui.QApplication(sys.argv)
        gui = nineml_component_qtGUI(self)
        gui.ui.timeHorizonSLineEdit.setText(str(self.timeHorizon))
        gui.ui.reportingIntervalSLineEdit.setText(str(self.reportingInterval))

        isOK = gui.exec_()
        if isOK == QtGui.QDialog.Accepted:
            self.updateData(timeHorizon              = float(str(gui.ui.timeHorizonSLineEdit.text())),
                            reportingInterval        = float(str(gui.ui.reportingIntervalSLineEdit.text())),
                            parameters               = self.treeParameters.getDictionary(),
                            initial_conditions       = self.treeInitialConditions.getDictionary(),
                            active_regimes           = self.treeActiveStates.getDictionary(),
                            analog_ports_expressions = self.treeAnalogPorts.getDictionary(),
                            event_ports_expressions  = self.treeEventPorts.getDictionary(),
                            variables_to_report      = self.treeVariablesToReport.getDictionary())
            results = {}
            results['timeHorizon']              = self.timeHorizon
            results['reportingInterval']        = self.reportingInterval
            results['parameters']               = self.parameters
            results['initial_conditions']       = self.initial_conditions
            results['active_regimes']           = self.active_regimes
            results['analog_ports_expressions'] = self.analog_ports_expressions
            results['event_ports_expressions']  = self.event_ports_expressions
            results['variables_to_report']      = self.variables_to_report
            return results
        else:
            return None

    def generateLatexReport(self, tests = []):
        """
        'tests' argument is an array of tuples: (varName, xPoints, yPoints, plotFileName)
        """
        if not self.ninemlComponent or not isinstance(self.ninemlComponent, nineml.abstraction_layer.ComponentClass):
            raise RuntimeError('Invalid input NineML component')

        content       = []
        tests_content = []
        parser        = ExpressionParser()

        # Collect all unique components from sub-nodes:
        unique_components = {}
        self._detectUniqueComponents(self.ninemlComponent, unique_components)

        # Add all detected components to the report
        for name, component in unique_components.items():
            self._addComponentToReport(content, component, name, parser)

        # Add all tests to the report
        for test in tests:
            self._addTestToReport(tests_content, test)

        return (''.join(content), ''.join(tests_content))

    def _detectUniqueComponents(self, component, unique_components):
        if not component.name in unique_components:
            unique_components[component.name] = component

        for name, subcomponent in component.subnodes.items():
            self._detectUniqueComponents(subcomponent, unique_components)

    def _addTestToReport(self, content, test):
        #testName        = test[0]
        #testDescription = test[1]
        #dictInputs      = test[2]
        #plots           = test[3]
        testName, testDescription, dictInputs, plots, log_output = test
        
        testInputs = '\\begin{verbatim}\n'
        testInputs += 'Time horizon = {0}\n'.format(dictInputs['timeHorizon'])
        testInputs += 'Reporting interval = {0}\n'.format(dictInputs['reportingInterval'])
        
        testInputs += 'Parameters:\n'
        for name, value in dictInputs['parameters'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += 'Initial conditions:\n'
        for name, value in dictInputs['initial_conditions'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += 'Analog ports expressions:\n'
        for name, value in dictInputs['analog_ports_expressions'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += 'Event ports expressions:\n'
        for name, value in dictInputs['event_ports_expressions'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += 'Initially active regimes:\n'
        for name, value in dictInputs['active_regimes'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += 'Variables to report:\n'
        for name, value in dictInputs['variables_to_report'].items():
            testInputs += '    {0} = {1}\n'.format(name, value)
        
        testInputs += '\\end{verbatim}\n'

        content.append('\\subsection*{{Test: {0}}}\n\n'.format(testName))
        content.append('Description: \n{0}\\newline\n'.format(testDescription))
        content.append('Input data: \n{0}\\newline\n'.format(testInputs))
        for plot in plots:
            varName      = plot[0]
            xPoints      = plot[1]
            yPoints      = plot[2]
            pngFileName  = plot[3]
            csvFileName  = plot[4]
            tex_plot = '\\begin{center}\n\\includegraphics{./' + pngFileName + '}\n\\end{center}\n'
            content.append(tex_plot)

        
    def _addComponentToReport(self, content, component, name, parser):
        comp_name = name.replace('_', '\\_')
        content.append('\\section{{NineML Component: {0}}}\n\n'.format(comp_name))

        # 1) Create parameters
        parameters = list(component.parameters)
        if len(parameters) > 0:
            content.append('\\subsection*{Parameters}\n\n')
            header_flags = ['l', 'c', 'l']
            header_items = ['Name', 'Units', 'Notes']
            rows_items = []
            for param in parameters:
                _name = param.name.replace('_', '\\_')
                rows_items.append([_name, ' - ', ' '])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')

        # 2) Create state-variables (diff. variables)
        state_variables = list(component.state_variables)
        if len(state_variables) > 0:
            content.append('\\subsection*{State-Variables}\n\n')
            header_flags = ['l', 'c', 'l']
            header_items = ['Name', 'Units', 'Notes']
            rows_items = []
            for var in state_variables:
                _name = var.name.replace('_', '\\_')
                rows_items.append([_name, ' - ', ' '])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')

        # 3) Create alias variables (algebraic)
        aliases = list(component.aliases)
        if len(aliases) > 0:
            content.append('\\subsection*{Aliases}\n\n')
            header_flags = ['l', 'l', 'c', 'l']
            header_items = ['Name', 'Expression', 'Units', 'Notes']
            rows_items = []
            for alias in aliases:
                _name = '${0}$'.format(alias.lhs)
                _rhs  = '${0}$'.format(parser.parse_to_latex(alias.rhs))
                rows_items.append([_name, _rhs, ' - ', ' '])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')

        # 4) Create analog-ports and reduce-ports
        analog_ports = list(component.analog_ports)
        if len(analog_ports) > 0:
            content.append('\\subsection*{Analog Ports}\n\n')
            header_flags = ['l', 'l', 'c', 'l']
            header_items = ['Name', 'Type', 'Units', 'Notes']
            rows_items = []
            for port in analog_ports:
                _name = port.name.replace('_', '\\_')
                _type = port.mode
                rows_items.append([_name, _type, ' - ', ' '])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')

        # 5) Create event-ports
        event_ports = list(component.event_ports)
        if len(event_ports) > 0:
            content.append('\\subsection*{Event ports}\n\n')
            header_flags = ['l', 'l', 'c', 'l']
            header_items = ['Name', 'Type', 'Units', 'Notes']
            rows_items = []
            for port in event_ports:
                _name = port.name.replace('_', '\\_')
                _type = port.mode
                rows_items.append([_name, _type, ' - ', ' '])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')
        '''
        content.append("""\\begin{center}
            \\begin{tabular}{ | l | l | l | p{5cm} |}
            \\hline
            Day & Min Temp & Max Temp & Summary \\\\ \\hline
            Monday & 11C & 22C & A clear day with lots of sunshine.
            However, the strong breeze will bring down the temperatures. \\\\ \\hline
            Tuesday & 9C & 19C & Cloudy with rain, \\newline across many northern regions. \\newline Clear spells
            across most of Scotland and Northern Ireland,
            but rain reaching the far northwest. \\\\ \\hline
            Wednesday & 10C & 21C & Rain will still linger for the morning.
            Conditions will improve by early afternoon and continue
            throughout the evening. \\\\
            \\hline
            \\end{tabular}
            \\end{center}""")
        '''
        
        # 6) Create sub-nodes
        if len(component.subnodes.items()) > 0:
            content.append('\\subsection*{Sub-nodes}\n\n')
            content.append(nineml_component_inspector.begin_itemize)
            for name, subcomponent in component.subnodes.items():
                _name = name.replace('_', '\\_')
                tex = nineml_component_inspector.item + _name + '\n'
                content.append(tex)
            content.append(nineml_component_inspector.end_itemize)
            content.append('\n')

        # 7) Create port connections
        portconnections = list(component.portconnections)
        if len(portconnections) > 0:
            content.append('\\subsection*{Port Connections}\n\n')
            header_flags = ['l', 'l']
            header_items = ['From', 'To']
            rows_items = []
            for port_connection in portconnections:
                portFrom = '.'.join(port_connection[0].loctuple)
                portTo   = '.'.join(port_connection[1].loctuple)
                _fromname = portFrom.replace('_', '\\_')
                _toname   = portTo.replace('_', '\\_')
                rows_items.append([_fromname, _toname])
            content.append(latex_table(header_flags, header_items, rows_items))
            content.append('\n')

        # 8) Create regimes
        regimes = list(component.regimes)
        """
        if len(regimes) > 0:
            content.append('\\subsection*{Regimes}\n\n')
            for regime in regimes:
                header_flags = ['l', 'l', 'l']
                header_items = ['ODEs', 'Transitions']
                rows_items = []

                _name = regime.name.replace('_', '\\_')
                _odes = []
                _on_events = []
                _on_conditions = []

                for time_deriv in regime.time_derivatives:
                    _odes.append('$\\frac{{d{0}}}{{dt}} = {1}$'.format(time_deriv.dependent_variable, parser.parse_to_latex(time_deriv.rhs)))

                for on_condition in regime.on_conditions:
                    _on_conditions.append('\\mbox{If } $' + parser.parse_to_latex(on_condition.trigger.rhs) + '$\mbox{:}')

                    if on_condition.target_regime.name != '':
                        _on_conditions.append('\\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(on_condition.target_regime.name))

                    for state_assignment in on_condition.state_assignments:
                        _on_conditions.append('\\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, parser.parse_to_latex(state_assignment.rhs)))

                    for event_output in on_condition.event_outputs:
                        _on_conditions.append('\\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name))

                # 8c) Create on_event actions
                for on_event in regime.on_events:
                    _on_events.append('\\mbox{On } $' + on_event.src_port_name + '$\mbox{:}')

                    if on_event.target_regime.name != '':
                        _on_events.append('\\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(on_event.target_regime.name))

                    for state_assignment in on_event.state_assignments:
                        _on_events.append('\\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, parser.parse_to_latex(state_assignment.rhs)))

                    for event_output in on_event.event_outputs:
                        _on_events.append('\\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name))

                content.append(latex_regime_table(header_flags, _name, _odes, _on_conditions, _on_events))
                content.append('\n')
        """
        if len(regimes) > 0:
            regimes_list     = []
            transitions_list = []

            content.append('\\subsection*{Regimes}\n\n')
            for ir, regime in enumerate(regimes):
                regimes_list.append(regime.name)

                tex = ''
                # 8a) Create time derivatives
                counter = 0
                for time_deriv in regime.time_derivatives:
                    if counter != 0:
                        tex += ' \\\\ '
                    tex += '\\frac{{d{0}}}{{dt}} = {1}'.format(time_deriv.dependent_variable, parser.parse_to_latex(time_deriv.rhs))
                    counter += 1

                # 8b) Create on_condition actions
                for on_condition in regime.on_conditions:
                    regimeFrom = regime.name
                    if on_condition.target_regime.name == '':
                        regimeTo = regimeFrom
                    else:
                        regimeTo = on_condition.target_regime.name
                    condition  = parser.parse_to_latex(on_condition.trigger.rhs)

                    tex += ' \\\\ \\mbox{If } ' + condition + '\mbox{:}'

                    if regimeTo != regimeFrom:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(regimeTo)

                    for state_assignment in on_condition.state_assignments:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, parser.parse_to_latex(state_assignment.rhs))

                    for event_output in on_condition.event_outputs:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name)

                    transition = '{0} -> {1} [label="{2}"];'.format(regimeFrom, regimeTo, condition)
                    transitions_list.append(transition)

                # 8c) Create on_event actions
                for on_event in regime.on_events:
                    regimeFrom = regime.name
                    if on_event.target_regime.name == '':
                        regimeTo = regimeFrom
                    else:
                        regimeTo = on_event.target_regime.name
                    source_port = on_event.src_port_name

                    tex += ' \\\\ \\mbox{On } ' + source_port + '\mbox{:}'

                    if regimeTo != regimeFrom:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{switch to }} {0}'.format(regimeTo)

                    for state_assignment in on_event.state_assignments:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{set }} {0} = {1}'.format(state_assignment.lhs, parser.parse_to_latex(state_assignment.rhs))

                    for event_output in on_event.event_outputs:
                        tex += ' \\\\ \\hspace*{{0.2in}} \\mbox{{emit }} {0}'.format(event_output.port_name)

                    transition = '{0} -> {1} [label="{2}"];'.format(regimeFrom, regimeTo, source_port)
                    transitions_list.append(transition)

                tex = '${0} = \\begin{{cases}} {1} \\end{{cases}}$\n'.format(regime.name, tex)
                tex += '\\newline \n'
                content.append(tex)

            dot_graph_template = '''
            digraph finite_state_machine {{
                rankdir=LR;
                node [shape=ellipse]; {0};
                {1}
            }}
            '''
            if len(regimes_list) > 1:
                dot_graph = dot_graph_template.format(' '.join(regimes_list), '\n'.join(transitions_list))
                graph     = dot2tex.dot2tex(dot_graph, autosize=True, texmode='math', format='tikz', crop=True, figonly=True)
                tex_graph = '\\begin{center}\n' + graph + '\\end{center}\n'
                content.append('\\newline \n')
                content.append(tex_graph)
                content.append('\n')
            
            content.append('\\newpage')
            content.append('\n')

if __name__ == "__main__":
    #nineml_component = '/home/ciroki/Data/daetools/trunk/python-files/examples/iaf.xml'
    nineml_component = TestableComponent('hierachical_iaf_1coba')()
    if not nineml_component:
        raise RuntimeError('Cannot load NineML component')

    timeHorizon = 10
    reportingInterval = 0.01
    parameters = {
        'cobaExcit.q' : 3.0,
        'cobaExcit.tau' : 5.0,
        'cobaExcit.vrev' : 0.0,
        'iaf.cm' : 1,
        'iaf.gl' : 50,
        'iaf.taurefrac' : 0.008,
        'iaf.vreset' : -60,
        'iaf.vrest' : -60,
        'iaf.vthresh' : -40
    }
    initial_conditions = {
        'cobaExcit.g' : 0.0,
        'iaf.V' : -60,
        'iaf.tspike' : -1E99
    }
    analog_ports_expressions = {}
    event_ports_expressions = {}
    active_regimes = {
        'cobaExcit' : 'cobadefaultregime',
        'iaf' : 'subthresholdregime'
    }
    variables_to_report = {
        'cobaExcit.I' : True,
        'iaf.V' : True
    }

    inspector = nineml_component_inspector()
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    isOK = inspector.showQtGUI()
    inspector.printCollectedData()

    variables_to_report = {}
    variables_to_report['iaf_1coba.iaf.tspike'] = True
    inspector.updateData(variables_to_report = variables_to_report)
    inspector.updateTrees()
    print('New data')
    inspector.printTreeDictionaries()
    
    #print(inspector.getComponentXMLSourceCode())
    #print(inspector.generateHTMLForm())
    #print(inspector.generateHTMLReport())
    #print(inspector.generateLatexReport())
    
    import pickle

    ins = pickle.dumps(inspector)

    inspector = pickle.loads(ins)
    inspector.inspect(nineml_component, timeHorizon              = timeHorizon,
                                        reportingInterval        = reportingInterval,
                                        parameters               = parameters,
                                        initial_conditions       = initial_conditions,
                                        active_regimes           = active_regimes,
                                        analog_ports_expressions = analog_ports_expressions,
                                        event_ports_expressions  = event_ports_expressions,
                                        variables_to_report      = variables_to_report)
    isOK = inspector.showQtGUI()
    inspector.printCollectedData()
