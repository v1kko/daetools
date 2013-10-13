#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
***********************************************************************************
                         simulation_inspector.py
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
from tree_item import *

def _collectParameters(nodeItem, model, dictParameters):
    """
    Recursively looks for parameters in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.Parameters:
        name = obj.Name
        if obj.NumberOfPoints == 1:
            value = float(obj.GetValue())
        else:
            value = obj.npyValues.tolist()
        units = obj.Units
        description = obj.Description
        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = True)
        dictParameters[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectParameters(componentItem, component, dictParameters)

def _collectDomains(nodeItem, model, dictDomains):
    """
    Recursively looks for domains in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.Domains:
        name  = obj.Name
        type  = obj.Type 
        units = obj.Units
        description = obj.Description
        if obj.Type == eArray:
            numberOfPoints = obj.NumberOfPoints
            item = treeItem_Domain(nodeItem, type, name, description, numberOfPoints=numberOfPoints, 
                                                                units=units)
        else:
            discrMethod        = obj.DiscretizationMethod # not edited
            order              = obj.DiscretizationOrder  # not edited
            numberOfIntervals  = obj.NumberOfIntervals    # not edited
            lowerBound         = obj.LowerBound
            upperBound         = obj.UpperBound
            item = treeItem_Domain(nodeItem, type, name, description, discrMethod = discrMethod, 
                                                   order = order, numberOfIntervals = numberOfIntervals, 
                                                   lowerBound = lowerBound, upperBound = upperBound, units = units)
        dictDomains[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectDomains(componentItem, component, dictDomains)

def _collectOutputVariables(nodeItem, model, dictOutputVariables):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.Variables:
        name  = obj.Name
        value = obj.ReportingOn
        item  = treeItem_OutputVariable(nodeItem, name, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)
    
    for obj in model.Parameters:
        name  = obj.Name
        value = obj.ReportingOn
        item  = treeItem_OutputVariable(nodeItem, name, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectOutputVariables(componentItem, component, dictOutputVariables)

def _collectSTNs(nodeItem, model, dictSTNs):
    """
    Recursively looks for STNs (excluding IFs) in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.STNs:
        if obj.Type == eSTN:
            name        = obj.Name
            description = obj.Description
            states      = [state.Name for state in obj.States]
            item = treeItem_StateTransition(nodeItem, name, description, states, obj.ActiveState)
            dictSTNs[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectSTNs(componentItem, component, dictSTNs)

def _collectInitialConditions(nodeItem, model, dictInitialConditions, IDs):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem' for all those
    who are differential and need an initial conditions to be set.
    """
    for var in model.Variables:
        name              = var.Name
        domainsIndexesMap = var.GetDomainsIndexesMap(indexBase = 0)
        units             = var.VariableType.Units
        description       = var.Description
        
        # Check if there is diff. flag set for any point in the variable
        # If there is not then skip the variable
        if not cnDifferential in IDs[var.OverallIndex : var.OverallIndex + var.NumberOfPoints]:
            continue
        
        if var.NumberOfPoints == 1:
            value = var.GetValue()
        else:
            values = numpy.array(var.npyValues, dtype=object)
            # Iterate over points and set None for the points which are *not* differential
            for var_index, domainIndexes in domainsIndexesMap.iteritems():
                #print var_index, domainIndexes, values[tuple(domainIndexes)]
                if IDs[var.OverallIndex + var_index] != cnDifferential:
                    values[tuple(domainIndexes)] = None
            value = values.tolist()
        
        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = False)
        dictInitialConditions[name] = (var, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectInitialConditions(componentItem, component, dictInitialConditions, IDs)
    
def _collectDOFs(nodeItem, model, dictDOFs, IDs):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem' for all those
    who are differential and need an initial conditions to be set.
    """
    for var in model.Variables:
        name              = var.Name
        domainsIndexesMap = var.GetDomainsIndexesMap(indexBase = 0)
        units             = var.VariableType.Units
        description       = var.Description
        
        # Check if there is assigned flag set for any point in the variable
        # If there is not then skip the variable
        if not cnAssigned in IDs[var.OverallIndex : var.OverallIndex + var.NumberOfPoints]:
            continue
        
        if var.NumberOfPoints == 1:
            value = var.GetValue()
        else:
            values = numpy.array(var.npyValues, dtype=object)
            # Iterate over points and set None for the points which are *not* differential
            for var_index, domainIndexes in domainsIndexesMap.iteritems():
                if IDs[var.OverallIndex + var_index] != cnAssigned:
                    values[tuple(domainIndexes)] = None
            value = values.tolist()

        item = treeItem_Quantity(nodeItem, name, description, value, units, checkIfItemsAreFloats = False)
        dictDOFs[name] = (var, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component.Name, treeItem.typeNone)
        _collectDOFs(componentItem, component, dictDOFs, IDs)
        
class daeSimulationInspector(object):
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
        self.treeSTNs               = None
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

        self.treeSTNs = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectSTNs(self.treeSTNs, self.simulation.m, self.stns)

        self.treeOutputVariables = treeItem(None, self.simulation.m.Name, treeItem.typeNone)
        _collectOutputVariables(self.treeOutputVariables, self.simulation.m, self.output_variables)
