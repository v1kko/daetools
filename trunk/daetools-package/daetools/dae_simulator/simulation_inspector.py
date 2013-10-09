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
        item = treeItem_Parameter(nodeItem, obj)
        dictParameters[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component, None, treeItem.typeNone)
        _collectParameters(componentItem, component, dictParameters)

def _collectDomains(nodeItem, model, dictDomains):
    """
    Recursively looks for domains in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.Domains:
        item = treeItem_Domain(nodeItem, obj)
        dictDomains[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component, None, treeItem.typeNone)
        _collectDomains(componentItem, component, dictDomains)

def _collectOutputVariables(nodeItem, model, dictOutputVariables):
    """
    Recursively looks for variables in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.Variables:
        value = obj.ReportingOn
        item = treeItem_OutputVariable(nodeItem, obj, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)
    
    for obj in model.Parameters:
        value = obj.ReportingOn
        item = treeItem_OutputVariable(nodeItem, obj, value)
        dictOutputVariables[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component, None, treeItem.typeNone)
        _collectOutputVariables(componentItem, component, dictOutputVariables)

def _collectStateTransitions(nodeItem, model, dictSTNs):
    """
    Recursively looks for parameters in the 'model' and all its child-models and
    adds a new treeItem object to the parent item 'nodeItem'.
    """
    for obj in model.STNs:
        item = treeItem_StateTransition(nodeItem, obj)
        dictSTNs[obj.CanonicalName] = (obj, item)

    for component in model.Components:
        componentItem = treeItem(nodeItem, component, None, treeItem.typeNone)
        _collectStateTransitions(componentItem, component, dictSTNs)

        
class daeSimulationInspector(object):
    def __init__(self, simulation):
        self.simulation = simulation
        
        self.domains            = {}
        self.parameters         = {}
        self.initial_conditions = {}
        self.dofs               = {}
        self.state_transitions  = {}
        self.output_variables   = {}
        
        self.treeDomains            = None
        self.treeParameters         = None
        self.treeInitialConditions  = None
        self.treeDOFs               = None
        self.treeStateTransitions   = None
        self.treeOutputVariables    = None
        
        self.treeDomains = treeItem(None, self.simulation.m, treeItem.typeNone)
        _collectDomains(self.treeDomains, self.simulation.m, self.domains)
        
        self.treeParameters = treeItem(None, self.simulation.m, treeItem.typeNone)
        _collectParameters(self.treeParameters, self.simulation.m, self.parameters)

        self.treeInitialConditions = treeItem(None, self.simulation.m, treeItem.typeNone)
        #_collectInitialConditions(self.treeInitialConditions, self.simulation.m, self.initial_conditions)

        self.treeDOFs = treeItem(None, self.simulation.m, treeItem.typeNone)
        #_collectDOFs(self.treeDOFs, self.simulation.m, self.dofs)

        self.treeStateTransitions = treeItem(None, self.simulation.m, treeItem.typeNone)
        _collectStateTransitions(self.treeStateTransitions, self.simulation.m, self.state_transitions)

        self.treeOutputVariables = treeItem(None, self.simulation.m, treeItem.typeNone)
        _collectOutputVariables(self.treeOutputVariables, self.simulation.m, self.output_variables)
