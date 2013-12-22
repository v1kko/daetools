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

import sys, tempfile, numpy, json
from daetools.pyDAE import *
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from .editor_quantity_ui import Ui_EditorQuantity
from .editor_quantity_array_ui import Ui_EditorQuantityArray
from .editor_state_transition_ui import Ui_EditorStateTransition
from .editor_domain_array_ui import Ui_EditorArrayDomain
from .editor_domain_distributed_ui import Ui_EditorDistributedDomain
_numbers_and_list_ = (float, int, list)
_numbers_          = (float, int)

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
    typeSTN              =  2
    typeOutputVariable   =  3
    typeState            =  4

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
            
    def toDictionary(self):
        dictItems = {}
        d = {}
        if self.itemType == treeItem.typeDomain:
            if self.type == eArray:
                d['Type']                 = str(self.type)
                d['NumberOfPoints']       = int(self._value)
            elif self.type == eUnstructuredGrid:
                d['Type']                 = str(self.type)
                d['NumberOfPoints']       = int(self._value)
            else:
                d['Type']                 = str(self.type)
                d['DiscretizationMethod'] = str(self.discrMethod)
                d['DiscretizationOrder']  = int(self.order)
                d['NumberOfIntervals']    = int(self.numberOfIntervals)
                d['Points']               = list(self._value)
                d['Units']                = self.units.toDict()
            dictItems[self.canonicalName] = d
        
        elif self.itemType == treeItem.typeQuantity:
            if isinstance(self._value[0], list):
                a = numpy.array(self._value[0], dtype = float)
                d['Shape'] = [s for s in a.shape] # Shape
                d['Value'] = self._value[0]       # List of values
            else:
                d['Value'] = self._value[0]       # Float                
            d['Units'] = self._value[1].toDict()  # units
            dictItems[self.canonicalName] = d
        
        elif self.itemType == treeItem.typeState:
            if self._value:
                # We have to split the canonical name into: stnCanonicalName & stateName
                names = self.canonicalName.split('.')
                stnCanonicalName = '.'.join(names[0:-1])
                stateName        = names[-1]                
                d['ActiveState'] = stateName
                dictItems[stnCanonicalName] = d
        
        #elif self.itemType == treeItem.typeSTN:
        #    d['ActiveState'] = str(self._value)
        #    dictItems[self.canonicalName] = d
        
        elif self.itemType == treeItem.typeOutputVariable:
            dictItems[self.canonicalName] = bool(self._value)
        
        for child in self.children:
            dictItems = dict(list(dictItems.items()) + list(child.toDictionary().items()))

        return dictItems
            
    def __str__(self):
        indent = self.level * '    '
        res = '{0}- {1}: {2} [{3}]\n'.format(indent, self.name, self.getValue(), self.editor)
        for child in sorted(self.children):
            res += str(child)
        return res

class editor_Quantity(QtGui.QFrame):
    def __init__(self, treeItem, description, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorQuantity()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        d_validator = QtGui.QDoubleValidator(self)
        self.ui.valueEdit.setValidator(d_validator)
        
        self.ui.descriptionEdit.setHtml(description)
        self.ui.valueEdit.setText(value)
        self.ui.unitsEdit.setText(units)
        
        self.ui.descriptionEdit.setAutoFillBackground(False)
        self.ui.descriptionEdit.setBackgroundRole(QtGui.QPalette.NoRole)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        self.treeItem.setValue( (str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text())) )

class editor_QuantityArray(QtGui.QFrame):
    def __init__(self, treeItem, description, value, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorQuantityArray()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        self.ui.descriptionEdit.setHtml(description)
        self.ui.valueEdit.setPlainText(value)
        self.ui.unitsEdit.setText(units)
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        #print 'Update: %s with (%s %s)' % (self.treeItem.name, str(self.ui.valueEdit.text()), str(self.ui.unitsEdit.text()))
        self.treeItem.setValue( (str(self.ui.valueEdit.toPlainText()), str(self.ui.unitsEdit.text())) )

def flatten(lst):
    return sum( ([x] if not isinstance(x, list) else flatten(x) for x in lst), [] )
  
class treeItem_Quantity(treeItem):
    def __init__(self, parent, name, description, value, units, checkIfItemsAreFloats):
        treeItem.__init__(self, parent, name, treeItem.typeQuantity)
        
        self.checkIfItemsAreFloats = checkIfItemsAreFloats
        self.units = units
        self.setValue((value, units))
        
        if isinstance(value, float):
            self._editor = editor_Quantity(self, description, str(value), str(units))
        else:
            self._editor = editor_QuantityArray(self, description, str(value), str(units))
            
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
            raise RuntimeError('Invalid value: %s for the tree item: %s' % (str(value), daeGetStrippedName(self.name)))
        
        if isinstance(value[0], str):
            try:
                val = eval(value[0])
            except Exception as e:
                errorMsg = 'Cannot set value for the tree item: %s\nError: Invalid value specified' % daeGetStrippedName(self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

            if not isinstance(val, _numbers_and_list_):
                errorMsg = 'Cannot set value for the tree item: %s\nIt must be either float or a list of floats' % daeGetStrippedName(self.name)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            
            if isinstance(val, list):
                if self.checkIfItemsAreFloats:
                    flat_val = flatten(val)
                    for v in flat_val:
                        if isinstance(v, _numbers_):
                            v = float(v)
                        else:
                            errorMsg = 'Not all items are floats in the list of values for the tree item: %s' % daeGetStrippedName(self.name)
                            QtGui.QMessageBox.critical(None, "Error", errorMsg)
                            return
            elif isinstance(val, _numbers_):
                val = float(val)
        
        elif isinstance(value[0], _numbers_and_list_):
            if isinstance(value[0], list):
                val = value[0]
                if self.checkIfItemsAreFloats:
                    flat_val = flatten(val)
                    for v in flat_val:
                        if isinstance(v, _numbers_):
                            v = float(v)
                        else:
                            errorMsg = 'Not all items are floats in the list of values for the tree item: %s' % daeGetStrippedName(self.name)
                            QtGui.QMessageBox.critical(None, "Error", errorMsg)
                            return
            elif isinstance(value[0], _numbers_):
                val = float(value[0])
            
        else:
            raise RuntimeError('Invalid value: %s for the tree item: %s (must be a string, float or a list)' % (str(value[0]), daeGetStrippedName(self.name)))
            
        
        if isinstance(value[1], str):
            try:
                # First try to evaluate the expression
                unit_expr = value[1].strip()
                if unit_expr == '':
                    units = pyUnits.unit()
                else:
                    units = eval(unit_expr, {}, pyUnits.all_si_and_derived_units)
            except SyntaxError as e:
                errorMsg = 'Cannot set units for the tree item: %s\nSyntax error at position %d in %s' % (daeGetStrippedName(self.name), e.offset, e.text)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            except Exception as e:
                errorMsg = 'Cannot set units for the tree item: %s\nInvalid units specified: %s' % (daeGetStrippedName(self.name), value[1])
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
            
            # If successful, check if the result is puUnits.unit object
            if not isinstance(units, pyUnits.unit):
                errorMsg = 'Cannot set units for the tree item: %s\nInvalid units specified: %s' % (daeGetStrippedName(self.name), value[1])
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
                
            # Finally, try to scale the units to the objects units. If succesful the units are consistent.
            try:
                quantity(1.0, self.units).scaleTo(units)
            except Exception as e:
                errorMsg = 'Cannot set units for the tree item: %s\nNew units %s not consistent with objects units %s' % (daeGetStrippedName(self.name), value[1], self.units)
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

        elif isinstance(value[1], pyUnits.unit):
            units = value[1]
            
        else:
            raise RuntimeError('Invalid units: %s for the tree item: %s (must be a string or unit object)' % (str(value[1]), daeGetStrippedName(self.name)))
        
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
    def __init__(self, treeItem, description, states, activeState):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorStateTransition()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        self.ui.descriptionEdit.setHtml(description)
        for i, state in enumerate(states):
            self.ui.activeStateComboBox.addItem(state)
            if activeState == state:
                self.ui.activeStateComboBox.setCurrentIndex(i)
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        index = self.ui.activeStateComboBox.currentIndex()
        self.treeItem.setValue( str(self.ui.activeStateComboBox.currentText()) )

class treeItem_STN(treeItem):
    def __init__(self, parent, name, description, states, active_state):
        treeItem.__init__(self, parent, name, treeItem.typeSTN)

        self.setValue(active_state)
        self._editor = editor_StateTransition(self, description, states, active_state)
            
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
        
class treeItem_State(treeItem):
    def __init__(self, parent, name, description, isActive):
        treeItem.__init__(self, parent, name, treeItem.typeState)

        self.setValue(isActive)
       
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
            
def uncheckAllChildren(item):
    # Unchecks all children
    for child in item.children:
        if child.itemType == treeItem.typeState:
            child.setValue(False)
        uncheckAllChildren(child)

def correctSelections(item):
    # Checks whether an item is unchecked but some of its children is checked and corrects it
    for child in item.children:
        if child.itemType == treeItem.typeState:
            # If it is not checked then uncheck the whole tree downwards
            if not child.getValue():
                uncheckAllChildren(child)
        
        correctSelections(child)

class editor_ArrayDomain(QtGui.QFrame):
    def __init__(self, treeItem, description, numberOfPoints):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorArrayDomain()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        i_validator = QtGui.QIntValidator(self)
        self.ui.numberOfPointsEdit.setValidator(i_validator)
        
        self.ui.numberOfPointsEdit.setText(str(numberOfPoints))
        self.ui.descriptionEdit.setHtml(description)
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        #self.treeItem.setValue( str(self.ui.numberOfPointsEdit.text()) )
        pass # Nothing is edited

class editor_DistributedDomain(QtGui.QFrame):
    def __init__(self, treeItem, description, discrMethod, order, numberOfIntervals, points, units):
        QtGui.QFrame.__init__(self)
        self.ui = Ui_EditorDistributedDomain()
        self.ui.setupUi(self)
        
        self.treeItem = treeItem
        
        self.ui.discrMethodEdit.setText(str(discrMethod))
        self.ui.orderEdit.setText(str(order))
        self.ui.numberOfIntervalsEdit.setText(str(numberOfIntervals))
        self.ui.pointsEdit.setPlainText(str(points))
        self.ui.unitsEdit.setText(str(units))
        self.ui.descriptionEdit.setHtml(description)
        
        self.connect(self.ui.updateButton, QtCore.SIGNAL('clicked()'), self.slotUpdate)
        
    def slotUpdate(self):
        self.treeItem.setValue(str(self.ui.pointsEdit.toPlainText()))

class treeItem_Domain(treeItem):
    def __init__(self, parent, type, name, description, **kwargs):
        treeItem.__init__(self, parent, name, treeItem.typeDomain)

        self.type = type
        if self.type == eArray:
            numberOfPoints = kwargs['numberOfPoints']
            self.setValue(numberOfPoints)
            self._editor = editor_ArrayDomain(self, description, numberOfPoints)
        
        elif self.type == eStructuredGrid:
            self.discrMethod        = kwargs['discrMethod']       # not edited
            self.order              = kwargs['order']             # not edited
            self.numberOfIntervals  = kwargs['numberOfIntervals'] # not edited
            self.units              = kwargs['units']             # not edited
            points                  = kwargs['points']
            self.setValue(points)
            self._editor = editor_DistributedDomain(self, description, self.discrMethod, self.order, self.numberOfIntervals, points, self.units)
        
        elif self.type == eUnstructuredGrid:
            numberOfPoints = kwargs['numberOfPoints']
            self.setValue(numberOfPoints)
            self._editor = editor_ArrayDomain(self, description, numberOfPoints)
            
        self._layout = QtGui.QHBoxLayout()
        self._layout.setObjectName("domainLayout")
        self._layout.addWidget(self._editor)
        self.editor = self._layout
       
    def setValue(self, value):
        if self.type == eArray:
            try:
                self._value = int(value)
            except Exception as e:
                errorMsg = 'Cannot set value for the domain tree item: %s\nError: Invalid value specified' % (daeGetStrippedName(self.name))
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return

        elif self.type == eUnstructuredGrid:
            try:
                self._value = int(value)
            except Exception as e:
                errorMsg = 'Cannot set value for the domain tree item: %s\nError: Invalid value specified' % (daeGetStrippedName(self.name))
                QtGui.QMessageBox.critical(None, "Error", errorMsg)
                return
                
        elif self.type == eStructuredGrid:
            if isinstance(value, str):
                try:
                    val = eval(value)
                except Exception as e:
                    errorMsg = 'Cannot set points for the tree item: %s\nError: Invalid value specified' % daeGetStrippedName(self.name)
                    QtGui.QMessageBox.critical(None, "Error", errorMsg)
                    return

                if not isinstance(val, list):
                    errorMsg = 'Cannot set points for the tree item: %s\nIt must be a list of floats' % daeGetStrippedName(self.name)
                    QtGui.QMessageBox.critical(None, "Error", errorMsg)
                    return
                
                flat_val = flatten(val)
                for v in flat_val:
                    if isinstance(v, _numbers_):
                        v = float(v)
                    else:
                        errorMsg = 'Not all items are floats in the list of points for the tree item: %s' % daeGetStrippedName(self.name)
                        QtGui.QMessageBox.critical(None, "Error", errorMsg)
                        return
                
                if len(self._value) != len(val):
                    errorMsg = 'Cannot set points for the tree item: %s\nInvalid number of points (%s) (required is %d)' % (daeGetStrippedName(self.name), len(val), len(self._value))
                    QtGui.QMessageBox.critical(None, "Error", errorMsg)
                    return
            
            elif isinstance(value, list):
                val = value
            
            else:
                raise RuntimeError('Invalid value: %s for the tree item: %s (must be a string or a list)' % (str(value), daeGetStrippedName(self.name)))
                
            self._value = val
            
        if self.treeWidgetItem:
            self.treeWidgetItem.setText(1, self.getValueAsText())
       
    def getValueAsText(self):
        if self.type == eArray:
            return 'Array(%d)' % self._value
        elif self.type == eUnstructuredGrid:
            return 'UnstructuredGrid(%d)' % self._value
        else:
            return 'Distributed(%s, %d, %d, %s, %s)' % (self.discrMethod, self.order, self.numberOfIntervals, 
                                                        self._value, str(self.units) if str(self.units) != '' else '-')
        
    def show(self, parent):
        self._editor.setParent(parent)
        self._layout.setParent(parent)
        self._editor.show()
        parent.adjustSize()
        
    def hide(self):
        self._editor.hide()
        
def areAllChildrenEmpty(item):
    for child in item.children:
        if not areAllChildrenEmpty(child):
            return False

    # If all children are empty and the type is typeNone then the whole branch from this point is empty
    if item.itemType == treeItem.typeNone:
        return True
    else:
        return False
    
def addItem(treeWidget, parent, item):
    widgetItem = QtGui.QTreeWidgetItem(parent, [daeGetStrippedName(item.name), ''])
    
    # Set tree item widget item so it can update it
    item.treeWidgetItem = widgetItem

    # Item's data is always the tree item object
    widgetItem.setData(0, QtCore.Qt.UserRole, item) #QtCore.QVariant(item))

    # The common flags
    widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    # Depending on the type set the text(0) or something else
    if item.itemType == treeItem.typeNone:
        font  = widgetItem.font(0)
        font.setBold(True)
        brush = QtGui.QBrush(Qt.blue)
        widgetItem.setFont(0, font)
        #widgetItem.setForeground(0, brush)
    
    elif item.itemType == treeItem.typeOutputVariable:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsUserCheckable)
        if item.getValue():
            widgetItem.setCheckState(0, Qt.Checked)
        else:
            widgetItem.setCheckState(0, Qt.Unchecked)
    
    elif item.itemType == treeItem.typeState:
        widgetItem.setFlags(widgetItem.flags() | Qt.ItemIsUserCheckable)
        if item.getValue():
            widgetItem.setCheckState(0, Qt.Checked)
        else:
            widgetItem.setCheckState(0, Qt.Unchecked)

    else:
        widgetItem.setFlags(widgetItem.flags())
        widgetItem.setText(1, item.getValueAsText())

    return widgetItem
    
def addItemsToTree(treeWidget, parent, item):
    """
    Recursively adds the whole tree of treeItems to QTreeWidget tree.
    """
    if areAllChildrenEmpty(item):
        return
        
    new_parent = addItem(treeWidget, parent, item)
    for child in item.children:
        addItemsToTree(treeWidget, new_parent, child)
