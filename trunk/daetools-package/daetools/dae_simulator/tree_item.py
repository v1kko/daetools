#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                              tree_item.py
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
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from editor_parameter_ui import Ui_EditorParameter
from editor_parameter_array_ui import Ui_EditorParameterArray
from editor_state_transition_ui import Ui_EditorStateTransition

class runtimeInformation(object):
    def __init__(self, model):
        self.model = model
        self.name  = ''
        self.description = ''
        self.TimeHorizon = ''
        self.ReportingInterval = ''

class treeItem(object):
    typeNone             = -1
    typeDomain           =  0
    typeParameter        =  1
    typeInitialCondition =  2
    typeDOF              =  3
    typeStateTransition  =  4
    typeOutputVariable   =  5

    def __init__(self, parent, dae_object, itemType = typeNone):
        self.parent         = parent
        self.children       = []
        self.daeObject      = dae_object
        self.itemType       = itemType
        self.editor         = None
        self.treeWidgetItem = None
        self._value         = None
        if parent:
            parent.children.append(self)

    def getValue(self):
        return self._value
        
    def getValueAsText(self):
        """
        Must be implemented in derived classes
        """
        raise RuntimeError('Not implemented')
        
    def setValue(self, value):
        """
        Must be implemented in derived classes
        """
        raise RuntimeError('Not implemented')
    
    @property
    def name(self):
        if self.daeObject:
            return self.daeObject.GetStrippedName()
        else:
            return 'UnsetName'
        
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
        res = '{0}- {1}: {2} [{3}]\n'.format(indent, self.name, self.value, self.editor)
        for child in sorted(self.children):
            res += str(child)
        return res

class editor_Parameter(QtGui.QFrame):
    def __init__(self, treeItem, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorParameter()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        self.ui.valueEdit.setText(value)
        self.ui.unitsEdit.setText(units)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        self.treeItem.setValue( (str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text())) )

class editor_ParameterArray(QtGui.QFrame):
    def __init__(self, treeItem, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorParameterArray()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        self.ui.valueEdit.setPlainText(value)
        self.ui.unitsEdit.setText(units)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        #print 'Update: %s with (%s %s)' % (self.treeItem.name, str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text()))
        self.treeItem.setValue( (str(self.ui.valueEdit.toPlainText()), str(self.ui.unitsEdit.text())) )

class treeItem_Parameter(treeItem):
    def __init__(self, parent, parameter, value, units):
        treeItem.__init__(self, parent, parameter, treeItem.typeParameter)

        self.setValue((value, units))
        if parameter.NumberOfPoints == 1:
            self._editor = editor_Parameter(self, str(value), str(units))
        else:
            self._editor = editor_ParameterArray(self, str(value), str(units))
            
        self._layout = QtGui.QHBoxLayout()
        self._layout.setObjectName("paramLayout")
        self._layout.addWidget(self._editor)
        self.editor = self._layout
       
    def setValue(self, value):
        """
        value is a tuple of two strings:
          [0] float or (multi-dimensional) list of floats
          [1] daetools unit object
        """
        if not isinstance(value, tuple):
            raise RuntimeError('Invalid value: %s for the parameter tree item: %s' % (str(value), self.name))
        
        if isinstance(value[0], basestring):
            try:
                val = eval(value[0])
            except Exception as e:
                errorMsg = 'Cannot get value for the parameter tree item: %s\nError: Invalid value specified' % (self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

            if not isinstance(val, (float, list)):
                errorMsg = 'Cannot get value for the parameter tree item: %s\nIt must be either float or a list of floats' % self.name
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
        
        elif isinstance(value[0], (float, list)):
            val = value[0]
            
        else:
            raise RuntimeError('Invalid value: %s for the parameter tree item: %s (must be a string, float or a list)' % (str(value[0]), self.name))
            
        
        if isinstance(value[1], basestring):
            try:
                units = eval(value[1], {}, pyUnits.all_si_and_derived_units)
            except SyntaxError as e:
                errorMsg = 'Cannot get units for the parameter tree item: %s\nSyntax error at position %d in %s' % (self.name, e.offset, e.text)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            except Exception as e:
                errorMsg = 'Cannot get units for the parameter tree item: %s\nInvalid units specified: %s' % (self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
        
        elif isinstance(value[1], pyUnits.unit):
            units = value[1]
            
        else:
            raise RuntimeError('Invalid units: %s for the parameter tree item: %s (must be a string or an unit object)' % (str(value[1]), self.name))
        
        self._value = (val, units)
        if self.treeWidgetItem:
            self.treeWidgetItem.setText(1, self.getValueAsText())
     
    def getValueAsText(self):
        return '%s %s' % (str(self._value[0]), str(self._value[1]))
        
    def show(self, parent):
        self._editor.setParent(parent)
        self._layout.setParent(parent)
        self._editor.show()
        parent.adjustSize()
        
    def hide(self):
        self._editor.hide()
        
class treeItem_OutputVariable(treeItem):
    def __init__(self, parent, variable, value):
        treeItem.__init__(self, parent, variable, treeItem.typeOutputVariable)
        
        self.setValue(value)
    
    def setValue(self, value):
        if isinstance(value, bool):
            self._value = value
        elif str(value) in ['true', 'True', 'yes', 'y', '1']:
            self._value = True
        else:
            self._value = False
    
    def getValueAsText(self):
        if self._value:
            return 'True'
        else:
            return 'False'
        

class editor_StateTransition(QtGui.QFrame):
    def __init__(self, treeItem, states, activeState):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorStateTransition()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        for i, state in enumerate(states):
            self.ui.activeStateComboBox.addItem(state)
            if activeState == state:
                self.ui.activeStateComboBox.setCurrentIndex(i)
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        index = self.ui.activeStateComboBox.currentIndex()
        self.treeItem.setValue( str(self.ui.activeStateComboBox.currentText()) )

class treeItem_StateTransition(treeItem):
    def __init__(self, parent, stn):
        treeItem.__init__(self, parent, stn, treeItem.typeStateTransition)

        self.setValue(stn.ActiveState)
        states = [state.Name for state in stn.States]
        self._editor = editor_StateTransition(self, states, stn.ActiveState)
            
        self._layout = QtGui.QHBoxLayout()
        self._layout.setObjectName("stnLayout")
        self._layout.addWidget(self._editor)
        self.editor = self._layout
       
    def setValue(self, value):
        self._value = value
        if self.treeWidgetItem:
            self.treeWidgetItem.setText(1, self.getValueAsText())
       
    def getValueAsText(self):
        return self._value
        
    def show(self, parent):
        self._editor.setParent(parent)
        self._layout.setParent(parent)
        self._editor.show()
        parent.adjustSize()
        
    def hide(self):
        self._editor.hide()
        
        
def addItem(treeWidget, parent, item):
    widgetItem = QtGui.QTreeWidgetItem(parent, [item.name, ''])
    
    # Set tree item widget item so it can  update it
    item.treeWidgetItem = widgetItem

    # Item's data is always the tree item object
    widgetItem.setData(1, QtCore.Qt.UserRole, QtCore.QVariant(item))

    # The common flags
    widgetItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    # Depending on the type set the text(0) or something else
    if item.itemType == treeItem.typeNone:
        pass
    
    elif item.itemType == treeItem.typeOutputVariable:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsUserCheckable)
        if item.getValue():
            widgetItem.setCheckState(0, Qt.Checked)
        else:
            widgetItem.setCheckState(0, Qt.Unchecked)

    else:
        widgetItem.setFlags(widgetItem.flags())
        widgetItem.setText(1, item.getValueAsText())

    return widgetItem
    
def addItemsToTree(treeWidget, parent, tree_item):
    """
    Recursively adds the whole tree of treeItems to QTreeWidget tree.
    """
    new_parent = addItem(treeWidget, parent, tree_item)
    for child in tree_item.children:
        addItemsToTree(treeWidget, new_parent, child)
