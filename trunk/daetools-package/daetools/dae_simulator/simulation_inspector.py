#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                         simulation_inspector.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2014
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
from .tree_item import *

def _collectParameters(nodeItem, model_or_port, dictParameters):
    """
    Recursively looks for parameters in the model/port and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    if not isinstance(model_or_port, (daePort, daeModel)):
        raise RuntimeError('Invalid type of object in _collectDOFs function')

    # Process parameters in the current port/model
    parameters = model_or_port.Parameters

    for obj in parameters:
        name = obj.Name
        if obj.NumberOfPoints == 1 and len(obj.Domains) == 0:
            value = float(obj.GetValue())
        else:
            value = obj.npyValues.tolist()
        units = obj.Units
        description = obj.Description
        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = True)
        dictParameters[obj.CanonicalName] = (obj, item)

    if isinstance(model_or_port, daeModel):
        model = model_or_port
        for port in model.Ports:
            portItem = treeItem(nodeItem, port.Name, treeItem.typeNone)
            _collectParameters(portItem, port, dictParameters)
        for component in model.Components:
            componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
            _collectParameters(componentItem, component, dictParameters)

def _collectDomains(nodeItem, model_or_port, dictDomains):
    """
    Recursively looks for domains in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    if not isinstance(model_or_port, (daePort, daeModel)):
        raise RuntimeError('Invalid type of object in _collectDOFs function')

    # Process domains in the current port/model
    domains = model_or_port.Domains

    for obj in domains:
        name  = obj.Name
        type  = obj.Type 
        units = obj.Units
        description = obj.Description
        if obj.Type == eArray:
            numberOfPoints = obj.NumberOfPoints
            item = treeItem_Domain(nodeItem, type, name, description, numberOfPoints=numberOfPoints, units=units)
        elif obj.Type == eStructuredGrid:
            discrMethod        = obj.DiscretizationMethod # not edited
            order              = obj.DiscretizationOrder  # not edited
            numberOfIntervals  = obj.NumberOfIntervals    # not edited
            points             = obj.Points
            item = treeItem_Domain(nodeItem, type, name, description, discrMethod = discrMethod, 
                                                   order = order, numberOfIntervals = numberOfIntervals, 
                                                   points = points, units = units)
        elif obj.Type == eUnstructuredGrid:
            discrMethod    = obj.DiscretizationMethod # not edited
            order          = obj.DiscretizationOrder  # not edited
            numberOfPoints = obj.NumberOfPoints       # not edited
            points         = obj.Coordinates
            item = treeItem_Domain(nodeItem, type, name, description, numberOfPoints=numberOfPoints, units=units)
        
        dictDomains[obj.CanonicalName] = (obj, item)

    if isinstance(model_or_port, daeModel):
        model = model_or_port
        for port in model.Ports:
            portItem = treeItem(nodeItem, port.Name, treeItem.typeNone)
            _collectDomains(portItem, port, dictDomains)
        for component in model.Components:
            componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
            _collectDomains(componentItem, component, dictDomains)

def _collectOutputVariables(nodeItem, model_or_port, dictOutputVariables):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    if not isinstance(model_or_port, (daePort, daeModel)):
        raise RuntimeError('Invalid type of object in _collectDOFs function')

    # Process variables/parameters in the current port/model
    variables = model_or_port.Variables
    parameters = model_or_port.Parameters

    for obj in variables:
        name  = obj.Name
        value = obj.ReportingOn
        item  = treeItem_OutputVariable(nodeItem, name, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)
    
    for obj in parameters:
        name  = obj.Name
        value = obj.ReportingOn
        item  = treeItem_OutputVariable(nodeItem, name, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)

    if isinstance(model_or_port, daeModel):
        model = model_or_port
        for port in model.Ports:
            portItem = treeItem(nodeItem, port.Name, treeItem.typeNone)
            _collectOutputVariables(portItem, port, dictOutputVariables)
        for component in model.Components:
            componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
            _collectOutputVariables(componentItem, component, dictOutputVariables)

def _checkIfAnyOfNestedSTNsIsdaeSTN(state):
    # Returns True if there is any daeSTN type of STN in any of nested STNs
    # or in any of states in those nestedSTNs etc...
    
    # If there are no  nested STNs return False
    if len(state.NestedSTNs) == 0:
        return False
    
    # Iterate over nested STNs
    for nestedSTN in state.NestedSTNs:
        # If a nested STN is daeSTN return true
        if nestedSTN.Type == eSTN:
            return True
        
        # Iterate over states in the current nested state and check
        # if nested STNs within it have daeSTN type of STNs 
        for nested_state in nestedSTN:
            if _checkIfAnyOfNestedSTNsIsdaeSTN(nested_state):
                return True
    
    return False
    
def _processSTN(nodeItem, stn, dictSTNs):
    """
    Recursively process states in the STN and all nestedSTNs that exist in those states
    and adds new treeItem objects to the parent item 'nodeItem'.
    """
    if stn.Type == eSTN:
        # Add an empty item for the STN
        stnName = stn.Name
        stnItem = treeItem(nodeItem, stnName, treeItem.typeNone)
        
        # Iterate over states and add a checkable item for each of them 
        listStates = []
        dictSTNs[stn.CanonicalName] = (stn, listStates)
        for state in stn.States:
            name        = state.Name
            description = state.Description
            isActive    = True if stn.ActiveState == state.Name else False
            item        = treeItem_State(stnItem, name, description, isActive)
            listStates.append((state, item))
            
            # Process nested STNs (if existing)
            for nestedSTN in state.NestedSTNs:
                _processSTN(item, nestedSTN, dictSTNs)
    
    elif stn.Type == eIF:
        # Here we process only Nested states (if existing)        
        # If there is a nested STN in any of states add the ifItem
        # which represents a parent to states with nested STNs
        ifItem = None
        for state in stn.States:
            if _checkIfAnyOfNestedSTNsIsdaeSTN(state):
                stnName = stn.Name
                ifItem = treeItem(nodeItem, stnName, treeItem.typeNone)
                break
        
        # If ifItem is None that means there are no nested STNs in any of states - immediately return
        if not ifItem:
            return
            
        # Now we have a parent for states with nested STNs
        # Iterate over states and add an item for each of them 
        for state in stn.States:
            if _checkIfAnyOfNestedSTNsIsdaeSTN(state):
                # If there are STNs of type daeSTN in the current state add a new empty item
                stateName = state.Name
                stateItem = treeItem(ifItem, stateName, treeItem.typeNone)
                
                # Process nested STNs (if existing)
                for nestedSTN in state.NestedSTNs:
                    _processSTN(stateItem, nestedSTN, dictSTNs)

def _collectStates(nodeItem, model, dictSTNs):
    """
    Recursively looks for STNs in the 'model' and all its child-models
    and process them accordingly.
    """
    for stn in model.STNs:
        _processSTN(nodeItem, stn, dictSTNs)
        
    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectStates(componentItem, component, dictSTNs)

def _collectInitialConditions(nodeItem, model_or_port, dictInitialConditions, IDs):
    """
    Recursively looks for variables in the model/port and all models child-models and
    adds a new treeItem object to the parent item 'nodeItem' for all those
    who are differential and need an initial conditions to be set.
    """
    if not isinstance(model_or_port, (daePort, daeModel)):
        raise RuntimeError('Invalid type of object in _collectDOFs function')

    # Process variables in the current port/model
    variables = model_or_port.Variables

    for var in variables:
        name              = var.Name
        domainsIndexesMap = var.GetDomainsIndexesMap(indexBase = 0)
        units             = var.VariableType.Units
        description       = var.Description
        
        # Check if there is diff. flag set for any point in the variable
        # If there is not - skip the variable
        if not cnDifferential in IDs[var.OverallIndex : var.OverallIndex + var.NumberOfPoints]:
            continue
        
        if var.NumberOfPoints == 1 and len(var.Domains) == 0:
            value = var.GetValue()
        else:
            values = numpy.array(var.npyValues, dtype=object)
            # Iterate over points and set None for the points which are *not* differential
            for var_index, domainIndexes in list(domainsIndexesMap.items()):
                #print var_index, domainIndexes, values[tuple(domainIndexes)]
                if IDs[var.OverallIndex + var_index] != cnDifferential:
                    values[tuple(domainIndexes)] = None
            value = values.tolist()
        
        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = False)
        dictInitialConditions[name] = (var, item)

    if isinstance(model_or_port, daeModel):
        model = model_or_port
        for port in model.Ports:
            portItem = treeItem(nodeItem, port.Name, treeItem.typeNone)
            _collectInitialConditions(portItem, port, dictInitialConditions, IDs)
        for component in model.Components:
            componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
            _collectInitialConditions(componentItem, component, dictInitialConditions, IDs)
    
def _collectDOFs(nodeItem, model_or_port, dictDOFs, IDs):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem' for all those
    who are differential and need an initial conditions to be set.
    ACHTUNG, ACHTUNG!!!!!
    DOFs should be collected from models and ports.
    """

    if not isinstance(model_or_port, (daePort, daeModel)):
        raise RuntimeError('Invalid type of object in _collectDOFs function')

    # Process variables in the current port/model
    variables = model_or_port.Variables
    
    for var in variables:
        name              = var.Name
        domainsIndexesMap = var.GetDomainsIndexesMap(indexBase = 0)
        units             = var.VariableType.Units
        description       = var.Description

        # Check if there is assigned flag set for any point in the variable
        # If there is not then skip the variable
        if not cnAssigned in IDs[var.OverallIndex : var.OverallIndex + var.NumberOfPoints]:
            continue

        if var.NumberOfPoints == 1 and len(var.Domains) == 0:
            value = var.GetValue()
        else:
            values = numpy.array(var.npyValues, dtype=object)
            # Iterate over points and set None for the points which are *not* differential
            for var_index, domainIndexes in list(domainsIndexesMap.items()):
                if IDs[var.OverallIndex + var_index] != cnAssigned:
                    values[tuple(domainIndexes)] = None
            value = values.tolist()

        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = False)
        dictDOFs[name] = (var, item)

    if isinstance(model_or_port, daeModel):
        model = model_or_port
        for port in model.Ports:
            portItem = treeItem(nodeItem, port.Name, treeItem.typeNone)
            _collectDOFs(portItem, port, dictDOFs, IDs)
        for component in model.Components:
            componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
            _collectDOFs(componentItem, component, dictDOFs, IDs)
        
class daeSimulationInspector(object):
    categoryParameters              = '___PARAMETERS___'
    categoryInitialConditions       = '___INITIAL_CONDITIONS___'
    categoryActiveStates            = '___ACTIVE_STATES___'
    categoryDOFs                    = '___DOFS___'
    categoryDomains                 = '___DOMAINS___'
    categoryVariablesToReport       = '___VARIABLES_TO_REPORT___'

    def __init__(self, simulation):
        self.simulation = simulation
        
        self.domains            = {}
        self.parameters         = {}
        self.initial_conditions = {}
        self.dofs               = {}
        self.stns               = {}
        self.output_variables   = {}
        
        self.treeDomains            = None
        self.treeParameters         = None
        self.treeInitialConditions  = None
        self.treeDOFs               = None
        self.treeStates             = None
        self.treeOutputVariables    = None
        
        IDs = self.simulation.VariableTypes

        self.treeDomains = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectDomains(self.treeDomains, self.simulation.m, self.domains)
        
        self.treeParameters = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectParameters(self.treeParameters, self.simulation.m, self.parameters)

        self.treeInitialConditions = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectInitialConditions(self.treeInitialConditions, self.simulation.m, self.initial_conditions, IDs)

        self.treeDOFs = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectDOFs(self.treeDOFs, self.simulation.m, self.dofs, IDs)

        self.treeStates = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectStates(self.treeStates, self.simulation.m, self.stns)
        # Checks whether there are some unchecked items that have some of its children checked and corrects it
        correctSelections(self.treeStates)

        self.treeOutputVariables = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectOutputVariables(self.treeOutputVariables, self.simulation.m, self.output_variables)
        
    def _generateHTMLForm(self):
        content = ''
        content += '<fieldset>'
        content += '<legend>General</legend>'
        content += '<label for="testName">Test name</label>'
        content += '<input class="ninemlString required" id="id_testName" type="text" name="testName" value=""/><br/>'
        content += '<label for="testDescription">Test description</label>'
        content += '<textarea class="ninemlMultilineString" id="id_testDescription" name="testDescription" rows="2" cols="2"></textarea><br/>'
        content += '</fieldset>\n'

        content += '<fieldset>'
        content += '<legend>Simulation</legend>'
        #content += '<label for="timeHorizon">Time horizon</label>'
        #content += '<input class="ninemlFloat required number" id="id_timeHorizon" type="text" name="timeHorizon" value="{0}"/><br/>'.format(self.timeHorizon)
        #content += '<label for="reportingInterval">Reporting interval</label>'
        #content += '<input class="ninemlFloat required number" id="id_reportingInterval" type="text" name="reportingInterval" value="{0}"/><br/>'.format(self.reportingInterval)
        content += '</fieldset>\n'

        if self.treeParameters:
            content += '<fieldset>'
            content += '<legend>Parameters</legend>\n'
            content += self._generateHTMLFormTree(self.treeParameters, daeSimulationInspector.categoryParameters)
            content += '</fieldset>\n'

        if self.treeDomains:
            content += '<fieldset>'
            content += '<legend>Domains</legend>\n'
            content += self._generateHTMLFormTree(self.treeDomains, daeSimulationInspector.categoryDomains)
            content += '</fieldset>\n'

        if self.treeInitialConditions:
            content += '<fieldset>'
            content += '<legend>Initial conditions</legend>\n'
            content += self._generateHTMLFormTree(self.treeInitialConditions, daeSimulationInspector.categoryInitialConditions) + '\n'
            content += '</fieldset>\n'

        if self.treeDOFs:
            content += '<fieldset>'
            content += '<legend>Degrees Of Freedom</legend>\n'
            content += self._generateHTMLFormTree(self.treeDOFs, daeSimulationInspector.categoryDOFs)
            content += '</fieldset>\n'

        if self.treeStates:
            content += '<fieldset>'
            content += '<legend>Initially active states</legend>\n'
            content += self._generateHTMLFormTree(self.treeStates, daeSimulationInspector.categoryActiveStates)
            content += '</fieldset>\n'

        if self.treeOutputVariables:
            content += '<fieldset>'
            content +='<legend>Output variables</legend>\n'
            content += self._generateHTMLFormTree(self.treeOutputVariables, daeSimulationInspector.categoryVariablesToReport) + '\n'
            content += '</fieldset>\n'

        return content

    def _generateHTMLFormTree(self, item, category = '', required = False):
        """
        Internal function to recursively generate a HTML form for the given treeItem root item.

        :param item: treeItem object
        :param category: string
        :param required: boolean

        :rtype: string
        :raises:
        """
        if category == '':
            inputName = item.canonicalName
        else:
            inputName = category + '.' + item.canonicalName

        content = '<ul>'
        if item.itemType == treeItem.typeQuantity:
            content += '<li><label for="{1}">{0}</label><input class="ninemlFloat required number" type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.getValueAsText())

        elif item.itemType == treeItem.typeDomain:
            content += '<li><label for="{1}">{0}</label><input class="ninemlString" type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.getValueAsText())

        elif item.itemType == treeItem.typeState:
            #    if isinstance(item.data, collections.Iterable) and len(item.data) > 0:
            #        content += '<li><label for="{1}">{0}</label> <select class="ninemlComboBox" name="{1}">'.format(item.name, inputName)
            #        for available_regime in item.data:
            #            if available_regime == item.value:
            #                content += '<option value="{0}" selected>{0}</option>'.format(available_regime)
            #            else:
            #                content += '<option value="{0}">{0}</option>'.format(available_regime)
            #        content += '</select></li>'
            #    else:
            #        content += '<li>{0}</li>'.format(item.name)
            if required:
                content += '<li><label for="{1}">{0}</label><input class="ninemlString required" type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.getValueAsText())
            else:
                content += '<li><label for="{1}">{0}</label><input class="ninemlString" type="text" name="{1}" value="{2}"/></li>'.format(item.name, inputName, item.getValueAsText())

        elif item.itemType == treeItem.typeOutputVariable:
            if item.getValue():
                content += '<li><label for="{1}">{0}</label><input class="ninemlCheckBox" type="checkbox" name="{1}" checked/></li>'.format(item.name, inputName)
            else:
                content += '<li><label for="{1}">{0}</label><input class="ninemlCheckBox" type="checkbox" name="{1}"/></li>'.format(item.name, inputName)
                
        else:
            content += '<li>{0}</li>'.format(item.name)

        for child in item.children:
            content += self._generateHTMLFormTree(child, category, required)

        content += '</ul>'
        return content
        