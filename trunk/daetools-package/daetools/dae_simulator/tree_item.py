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
from editor_quantity_ui import Ui_EditorQuantity
from editor_quantity_array_ui import Ui_EditorQuantityArray
from editor_state_transition_ui import Ui_EditorStateTransition
from editor_domain_array_ui import Ui_EditorArrayDomain
from editor_domain_distributed_ui import Ui_EditorDistributedDomain

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
    typeQuantity         =  1
    typeStateTransition  =  2
    typeOutputVariable   =  3

    def __init__(self, parent, name, itemType = typeNone):
        self.parent         = parent
        self.children       = []
        self.name           = name
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
        res = '{0}- {1}: {2} [{3}]\n'.format(indent, self.name, self.getValue(), self.editor)
        for child in sorted(self.children):
            res += str(child)
        return res

class editor_Quantity(QtGui.QFrame):
    def __init__(self, treeItem, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorQuantity()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        d_validator = QtGui.QDoubleValidator(self)
        self.ui.valueEdit.setValidator(d_validator)
        
        self.ui.valueEdit.setText(value)
        self.ui.unitsEdit.setText(units)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        self.treeItem.setValue( (str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text())) )

class editor_QuantityArray(QtGui.QFrame):
    def __init__(self, treeItem, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorQuantityArray()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        self.ui.valueEdit.setPlainText(value)
        self.ui.unitsEdit.setText(units)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        #print 'Update: %s with (%s %s)' % (self.treeItem.name, str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text()))
        self.treeItem.setValue( (str(self.ui.valueEdit.toPlainText()), str(self.ui.unitsEdit.text())) )

def flatten(lst):
    return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )
  
class treeItem_Quantity(treeItem):
    def __init__(self, parent, name, value, units):
        treeItem.__init__(self, parent, name, treeItem.typeQuantity)
        
        self.units = units
        self.setValue((value, units))
        
        if isinstance(value, float):
            self._editor = editor_Quantity(self, str(value), str(units))
        else:
            self._editor = editor_QuantityArray(self, str(value), str(units))
            
        self._layout = QtGui.QHBoxLayout()
        self._layout.setObjectName("paramLayout")
        self._layout.addWidget(self._editor)
        self.editor = self._layout
       
    def setValue(self, value):
        """
        value is a tuple of two strings:
          [0] float/int/long or (multi-dimensional) list of floats/ints/longs
          [1] daetools unit object
        """
        if not isinstance(value, tuple):
            raise RuntimeError('Invalid value: %s for the tree item: %s' % (str(value), self.name))
        
        if isinstance(value[0], basestring):
            try:
                val = eval(value[0])
            except Exception as e:
                errorMsg = 'Cannot set value for the tree item: %s\nError: Invalid value specified' % (self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

            if not isinstance(val, (float, int, long, list)):
                errorMsg = 'Cannot set value for the tree item: %s\nIt must be either float or a list of floats' % self.name
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            
            if isinstance(val, list):
                flat_val = flatten(val)
                for v in flat_val:
                    if not isinstance(v, (float, int, long)):
                        errorMsg = 'Not all items are floats in the list of values for the tree item: %s' % self.name
                        QtGui.QMessageBox.critical(None, "Error", errorMsg)
                        return
        
        elif isinstance(value[0], (float, int, long, list)):
            val = value[0]
            
        else:
            raise RuntimeError('Invalid value: %s for the tree item: %s (must be a string, float or a list)' % (str(value[0]), self.name))
            
        
        if isinstance(value[1], basestring):
            try:
                # First try to evaluate the expression
                unit_expr = value[1].strip()
                if unit_expr == '':
                    units = pyUnits.unit()
                else:
                    units = eval(unit_expr, {}, pyUnits.all_si_and_derived_units)
            except SyntaxError as e:
                errorMsg = 'Cannot set units for the tree item: %s\nSyntax error at position %d in %s' % (self.name, e.offset, e.text)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            except Exception as e:
                errorMsg = 'Cannot set units for the tree item: %s\nInvalid units specified: %s' % (self.name, value[1])
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            
            # If successful, check if the result is puUnits.unit object
            if not isinstance(units, pyUnits.unit):
                errorMsg = 'Cannot set units for the tree item: %s\nInvalid units specified: %s' % (self.name, value[1])
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
                
            # Finally, try to scale the units to the objects units. If succesful the units are consistent.
            try:
                quantity(1.0, self.units).scaleTo(units)
            except Exception as e:
                errorMsg = 'Cannot set units for the tree item: %s\nNew units %s not consistent with objects units %s' % (self.name, value[1], self.units)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

        elif isinstance(value[1], pyUnits.unit):
            units = value[1]
            
        else:
            raise RuntimeError('Invalid units: %s for the tree item: %s (must be a string or unit object)' % (str(value[1]), self.name))
        
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
    def __init__(self, parent, name, value):
        treeItem.__init__(self, parent, name, treeItem.typeOutputVariable)
        
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
    def __init__(self, parent, name, states, active_state):
        treeItem.__init__(self, parent, name, treeItem.typeStateTransition)

        self.setValue(active_state)
        self._editor = editor_StateTransition(self, states, active_state)
            
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
        

class editor_ArrayDomain(QtGui.QFrame):
    def __init__(self, treeItem, numberOfPoints):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorArrayDomain()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        i_validator = QtGui.QIntValidator(self)
        self.ui.numberOfPointsEdit.setValidator(i_validator)
        
        self.ui.numberOfPointsEdit.setText(str(numberOfPoints))
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        #self.treeItem.setValue( str(self.ui.numberOfPointsEdit.text()) )
        pass # Nothing is edited

class editor_DistibutedDomain(QtGui.QFrame):
    def __init__(self, treeItem, discrMethod, order, numberOfIntervals, lowerBound, upperBound, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorDistributedDomain()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        d_validator = QtGui.QDoubleValidator(self)
        self.ui.lowerBoundEdit.setValidator(d_validator)
        self.ui.upperBoundEdit.setValidator(d_validator)
        
        self.ui.discrMethodEdit.setText(str(discrMethod))
        self.ui.orderEdit.setText(str(order))
        self.ui.numberOfIntervalsEdit.setText(str(numberOfIntervals))
        self.ui.lowerBoundEdit.setText(str(lowerBound))
        self.ui.upperBoundEdit.setText(str(upperBound))
        self.ui.unitsEdit.setText(str(units))
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        self.treeItem.setValue( (None, None, None, str(self.ui.lowerBoundEdit.text()), str(self.ui.upperBoundEdit.text()), None) )

class treeItem_Domain(treeItem):
    def __init__(self, parent, name, **kwargs):
        treeItem.__init__(self, parent, name, treeItem.typeStateTransition)

        if 'numberOfPoints' in kwargs:
            self.isArray = True
            numberOfPoints = kwargs['numberOfPoints']
            self.setValue(numberOfPoints)
            self._editor = editor_ArrayDomain(self, numberOfPoints)
        else:
            self.isArray = False
            self.discrMethod        = kwargs['discrMethod']       # not edited
            self.order              = kwargs['order']             # not edited
            self.numberOfIntervals  = kwargs['numberOfIntervals'] # not edited
            self.units              = kwargs['units']             # not edited
            lowerBound              = kwargs['lowerBound']
            upperBound              = kwargs['upperBound']
            self.setValue((self.discrMethod, self.order, self.numberOfIntervals, lowerBound, upperBound, self.units))
            self._editor = editor_DistibutedDomain(self, self.discrMethod, self.order, self.numberOfIntervals, lowerBound, upperBound, self.units)
            
        self._layout = QtGui.QHBoxLayout()
        self._layout.setObjectName("domainLayout")
        self._layout.addWidget(self._editor)
        self.editor = self._layout
       
    def setValue(self, value):
        if self.isArray:
            try:
                self._value = int(value)
            except Exception as e:
                errorMsg = 'Cannot get value for the domain tree item: %s\nError: Invalid value specified' % (self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

        else:
            if not len(value) == 6:
                errorMsg = 'Invalid size of the value for the distributed domain tree item: %s\nIt must be 6 (%d sent)' % (self.name. len(value))
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            try:
                lowerBound =  float(value[3])
                upperBound =  float(value[4])
                self._value = (self.discrMethod, self.order, self.numberOfIntervals, lowerBound, upperBound, self.units)
            except Exception as e:
                errorMsg = 'Cannot get value for the domain tree item: %s\nInvalid lower/upper bounds (must be floats)' % self.name
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            
        if self.treeWidgetItem:
            self.treeWidgetItem.setText(1, self.getValueAsText())
       
    def getValueAsText(self):
        if self.isArray:
            return 'Array(%d)' % self._value
        else:
            return 'Distributed(%s, %d, %d, %f, %f, %s)' % (self.discrMethod, self.order, self.numberOfIntervals, 
                                                            self._value[3], self._value[4], str(self.units) if str(self.units) != '' else '-')
        
    def show(self, parent):
        self._editor.setParent(parent)
        self._layout.setParent(parent)
        self._editor.show()
        parent.adjustSize()
        
    def hide(self):
        self._editor.hide()
        
        
def addItem(treeWidget, parent, item):
    widgetItem = QtGui.QTreeWidgetItem(parent, [item.name, ''])
    
    # Set tree item widget item so it can update it
    item.treeWidgetItem = widgetItem

    # Item's data is always the tree item object
    widgetItem.setData(1, QtCore.Qt.UserRole, QtCore.QVariant(item))

    # The common flags
    widgetItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    # Depending on the type set the text(0) or something else
    if item.itemType == treeItem.typeNone:
        font  = widgetItem.font(0)
        #font.setBold(True)
        brush = QtGui.QBrush(Qt.blue)
        widgetItem.setFont(0, font)
        widgetItem.setForeground(0, brush)
    
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
