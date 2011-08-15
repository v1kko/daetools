#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nineml
from nineml.abstraction_layer.testing_utils import RecordValue, TestableComponent
from nineml.abstraction_layer import ComponentClass
from nineml.abstraction_layer.testing_utils import std_pynn_simulation

import os, sys
from time import localtime, strftime
from daetools.pyDAE.daeParser import daeExpressionParser
from daetools.pyDAE.daeGetParserDictionary import getParserDictionary
from daetools.pyDAE import *

def getParserDictionary(model):
    """
    Dictionary should contain the following type of items:
     - string : adouble (for parameters and variables)
     - string : callable-object (these has to be implemented: sin, cos, tan, exp, ln, log, sqrt)
    """
    dictNameValue = {}

    # adouble values for parameters
    for p in model.Parameters:
        dictNameValue[p.Name] = p()

    # adouble values for variables
    for v in model.Variables:
        dictNameValue[v.Name] = v()

    # adouble values for ports variable ('value')
    for port in model.Ports:
        if port.Type == eInletPort:
            for v in port.Variables:
                dictNameValue[port.Name] = v()

    # DAE Tools mathematical functions:
    dictNameValue['t']    = model.time()

    # DAE Tools mathematical functions:
    dictNameValue['sin']  = Sin
    dictNameValue['cos']  = Cos
    dictNameValue['tan']  = Tan
    dictNameValue['log']  = Log10
    dictNameValue['ln']   = Log
    dictNameValue['sqrt'] = Sqrt
    dictNameValue['exp']  = Exp

    return dictNameValue

def printComponent(c, name, indent_string = '  ', level = 0):
    indent = level*indent_string
    print indent + '+ COMPONENT: [{0}], class: [{1}]'.format(name, c.__class__.__name__)

    indent = indent + '  '

    print indent + 'parameters:'
    for o in c.parameters:
        print indent + indent_string, o

    print indent + 'state_variables:'
    for o in c.state_variables:
        print indent + indent_string, o

    print indent + 'aliases:'
    for o in c.aliases:
        print indent + indent_string, o

    print indent + 'regimes:'
    for o in c.regimes:
        print indent + indent_string, o

        print indent + indent_string + 'transitions:'
        for t in o.on_conditions:
            print indent + 2*indent_string + 'source_regime.name: ' + t.source_regime.name
            print indent + 2*indent_string + 'target_regime.name: ' + t.target_regime.name
            print indent + 2*indent_string + 'trigger: ' + t.trigger.rhs
            print indent + 2*indent_string + 'state_assignments:'
            for sa in t.state_assignments:
                print indent + 3*indent_string + sa.lhs + ' = ' + sa.rhs
            print indent + 2*indent_string + 'event_outputs:'
            for eo in t.event_outputs:
                print indent + 3*indent_string + str(eo)
                
        for t in o.on_events:
            print indent + 2*indent_string + 'src_port_name: ' + t.src_port_name
            print indent + 2*indent_string + 'source_regime_name: ' + t.source_regime.name
            print indent + 2*indent_string + 'target_regime_name: ' + t.target_regime.name
            print indent + 2*indent_string + 'state_assignments:'
            for sa in t.state_assignments:
                print indent + 3*indent_string + sa.lhs + ' = ' + sa.rhs
            print indent + 2*indent_string + 'event_outputs:'
            for eo in t.event_outputs:
                print indent + 3*indent_string + str(eo)


    print indent + 'analog_ports:'
    for o in c.analog_ports:
        print indent + indent_string, o

    print indent + 'event_ports:'
    for o in c.event_ports:
        print indent + indent_string, o

    print indent + 'portconnections:'
    for o in c.portconnections:
        print indent + indent_string + ' {0} -> {1}'.format(o[0].getstr('.'), o[1].getstr('.'))

    print indent + 'subnodes:'
    for name, subc in c.subnodes.items():
        printComponent(subc, name, indent_string, level+1)

def findObjectInModel(model, name, **kwargs):
    look_for_domains    = kwargs.get('look_for_domains',    False)
    look_for_parameters = kwargs.get('look_for_parameters', False)
    look_for_variables  = kwargs.get('look_for_variables',  False)
    look_for_ports      = kwargs.get('look_for_ports',      False)
    look_for_eventports = kwargs.get('look_for_eventports', False)
    look_for_models     = kwargs.get('look_for_models',     False)

    objects = []
    look_for_all = True

    # If any of those is set then set look_for_all to False
    if look_for_domains or look_for_parameters or look_for_variables or look_for_ports or look_for_eventports or look_for_models:
        look_for_all = False
    
    if look_for_all:
        objects = model.Domains + model.Parameters + model.Variables + model.Ports + model.Models
    else:
        if look_for_domains:
            objects = objects + model.Domains
        if look_for_parameters:
            objects = objects + model.Parameters
        if look_for_variables:
            objects = objects + model.Variables
        if look_for_ports:
            objects = objects + model.Ports
        if look_for_eventports:
            objects = objects + model.EventPorts
        if look_for_models:
            objects = objects + model.Models

    mapObjects = {}
    for o in objects:
        mapObjects[o.Name] = o
    
    if name in mapObjects.keys():
        return mapObjects[name]
    return None

def getObjectFromNamespaceAddress(rootModel, address, **kwargs):
    canonicalName = '.'.join(address.loctuple)
    return getObjectFromCanonicalName(rootModel, canonicalName)

def getObjectFromCanonicalName(rootModel, canonicalName, **kwargs):
    """
    rootModel: daModel object
    canonicalName: a 'path' to the object ('model1.model2.object')
    """
    relativeName = daeGetRelativeName(rootModel.CanonicalName, canonicalName)
    #print 'relativeName = {0} for root = {1} and canonicalName = {2}'.format(relativeName, rootModel.CanonicalName, canonicalName)
    listCanonicalName = relativeName.split('.')
    objectName = listCanonicalName[-1]
    objectPath = listCanonicalName[:-1]

    root = rootModel
    if len(objectPath) > 0:
        for name in objectPath:
            #print 'name: {0} in model {1}'.format(name, root.Name)
            root = findObjectInModel(root, name, look_for_models = True)
            if root == None:
                raise RuntimeError('Could not locate object {0} in {1}'.format(name, ".".join(objectPath)))

    # Now we have the model where port should be located (root)
    # Search for the 'name' in ALL objects (domains, params, vars, ports, models) in the 'root' model
    return findObjectInModel(root, objectName, **kwargs)
    
class ninemlAnalogPort(daePort):
    def __init__(self, Name, PortType, Model, Description = ""):
        daePort.__init__(self, Name, PortType, Model, Description)

        # NineML ports always contain only one variable, and that variable is referred to by the port name
        # Here we name this variable 'value'
        self.value = daeVariable("value", no_t, self, "")

class nineml_daetools_bridge(daeModel):
    def __init__(self, Name, ninemlComponent, Parent = None, Description = ""):
        daeModel.__init__(self, Name, Parent, Description)

        self.ninemlComponent = ninemlComponent

        # 1) Create parameters
        self.nineml_parameters = []
        for param in self.ninemlComponent.parameters:
            self.nineml_parameters.append( daeParameter(param.name, eReal, self, "") )

        # 2) Create state-variables (diff. variables)
        self.nineml_state_variables = []
        for var in self.ninemlComponent.state_variables:
            self.nineml_state_variables.append( daeVariable(var.name, no_t, self, "") )

        # 3) Create alias variables (algebraic)
        self.nineml_aliases = []
        for alias in self.ninemlComponent.aliases:
            self.nineml_aliases.append( daeVariable(alias.lhs, no_t, self, "") )

        # 4) Create analog-ports
        self.nineml_analog_ports = []
        for analog_port in self.ninemlComponent.analog_ports:
            port_type = eInletPort
            if analog_port.mode == 'send':
                port_type = eOutletPort
            elif analog_port.mode == 'recv':
                port_type = eInletPort
            elif analog_port.mode == 'reduce':
                port_type = eInletPort
            else:
                raise RuntimeError("")
            self.nineml_analog_ports.append( ninemlAnalogPort(analog_port.name, port_type, self, "") )

        # 5) Create event-ports
        self.nineml_event_ports = []
        for event_port in self.ninemlComponent.event_ports:
            port_type = eInletPort
            if event_port.mode == 'send':
                port_type = eOutletPort
            elif event_port.mode == 'recv':
                port_type = eInletPort
            elif event_port.mode == 'reduce':
                port_type = eInletPort
            else:
                raise RuntimeError("")
            self.nineml_event_ports.append( daeEventPort(event_port.name, port_type, self, "") )

        # 6) Create sub-nodes
        self.ninemlSubComponents = []
        for name, subcomponent in ninemlComponent.subnodes.items():
            self.ninemlSubComponents.append( nineml_daetools_bridge(name, subcomponent, self) )

        # 7) Connect event ports
        if self.Name == 'iaf_1coba':
            spikeoutput = getObjectFromCanonicalName(self, 'iaf_1coba.iaf.spikeoutput',      look_for_eventports = True)
            spikeinput  = getObjectFromCanonicalName(self, 'iaf_1coba.cobaExcit.spikeinput', look_for_eventports = True)
            #print 'spikeoutput =', spikeoutput
            #print 'spikeinput =', spikeinput
            self.ConnectEventPorts(spikeinput, spikeoutput)
            
    def DeclareEquations(self):
        # Create the epression parser and set its Identifier:Value dictionary
        parser = daeExpressionParser()
        parser.dictNamesValues = getParserDictionary(self)
        #for key, value in parser.dictNamesValues.items():
        #    print key + ' : ' + repr(value)

        # 1) Create aliases (algebraic equations)
        aliases = list(self.ninemlComponent.aliases)
        if len(aliases) > 0:
            for i, alias in enumerate(aliases):
                eq = self.CreateEquation(alias.lhs, "")
                eq.Residual = self.nineml_aliases[i]() - parser.parse_and_evaluate(alias.rhs)

        # 2) Create regimes
        regimes         = list(self.ninemlComponent.regimes)
        state_variables = list(self.ninemlComponent.state_variables)
        if len(regimes) > 0:
            # Create STN for model
            self.STN('regimes')

            for regime in regimes:
                # 2a) Create State for each regime
                self.STATE(regime.name)

                # Sometime a time_derivative equation is not given and in that case the derivative is equal to zero
                # I have to discover which variables do not have a corresponding ODE
                # I do that by creating a map {'state_var' : 'RHS'} which initially has
                # set rhs to '0'. RHS will be set later while iterating through ODEs
                map_statevars_timederivs = {}
                for state_var in state_variables:
                    map_statevars_timederivs[state_var.name] = 0

                time_derivatives = list(regime.time_derivatives)
                for time_deriv in time_derivatives:
                    map_statevars_timederivs[time_deriv.dependent_variable] = time_deriv.rhs
                #print map_statevars_timederivs

                # 2b) Create equations for all state variables/time derivatives
                for var_name, rhs in map_statevars_timederivs.items():
                    variable = self.findVariable(var_name)
                    if variable == None:
                        raise RuntimeError('Cannot find state variable {0}'.format(var_name))

                    # Create equation
                    eq = self.CreateEquation(var_name, "")

                    # If right-hand side expression is 0 do not parse it
                    if rhs == 0:
                        eq.Residual = variable.dt()
                    else:
                        eq.Residual = variable.dt() - parser.parse_and_evaluate(rhs)

                # 2c) Create on_condition actions (state transitions, etc)
                for on_condition in regime.on_conditions:
                    condition         = parser.parse_and_evaluate(on_condition.trigger.rhs)
                    switchTo          = on_condition.target_regime.name
                    triggerEvents     = []
                    setVariableValues = []

                    for state_assignment in on_condition.state_assignments:
                        variable   = getObjectFromCanonicalName(self, state_assignment.lhs, look_for_variables = True)
                        if variable == None:
                            raise RuntimeError('Cannot find variable {0}'.format(state_assignment.lhs))
                        expression = parser.parse_and_evaluate(state_assignment.rhs)
                        setVariableValues.append( (variable, expression) )

                    for event_output in on_condition.event_outputs:
                        event_port = getObjectFromCanonicalName(self, event_output.port_name, look_for_eventports = True)
                        if event_port == None:
                            raise RuntimeError('Cannot find event port {0}'.format(event_output.port_name))
                        triggerEvents.append( (event_port, 0) )

                    self.ON_CONDITION(condition, switchTo          = switchTo,
                                                 triggerEvents     = triggerEvents,
                                                 setVariableValues = setVariableValues )

                    #self.SWITCH_TO(on_condition.target_regime.name, parser.parse_and_evaluate(on_condition.trigger.rhs))
                
                # 2d) Create on_event actions
                for on_event in regime.on_events:
                    print self.Name, on_event.src_port_name
                    source_event_port = getObjectFromCanonicalName(self, on_event.src_port_name, look_for_eventports = True)
                    if source_event_port == None:
                        raise RuntimeError('Cannot find event port {0}'.format(on_event.src_port_name))
                    switchToStates    = []
                    triggerEvents     = []
                    setVariableValues = []

                    for state_assignment in on_event.state_assignments:
                        variable   = getObjectFromCanonicalName(self, state_assignment.lhs, look_for_variables = True)
                        if variable == None:
                            raise RuntimeError('Cannot find variable {0}'.format(state_assignment.lhs))
                        expression = parser.parse_and_evaluate(state_assignment.rhs)
                        setVariableValues.append( (variable, expression) )

                    for event_output in on_event.event_outputs:
                        event_port = getObjectFromCanonicalName(self, event_output.port_name, look_for_eventports = True)
                        if event_port == None:
                            raise RuntimeError('Cannot find event port {0}'.format(event_output.port_name))
                        triggerEvents.append( (event_port, 0) )

                    self.ON_EVENT(source_event_port, switchToStates    = switchToStates,
                                                     triggerEvents     = triggerEvents,
                                                     setVariableValues = setVariableValues )
                                                 
            self.END_STN()

        # 3) Create equations for outlet analog-ports: port.value() - variable() = 0
        for analog_port in self.nineml_analog_ports:
            if analog_port.Type == eOutletPort:
                eq = self.CreateEquation(analog_port.Name + '_portequation', "")
                var_to = findObjectInModel(self, analog_port.Name, look_for_variables = True)
                if var_to == None:
                    raise RuntimeError('Cannot find state variable {0}'.format(analog_port.Name))
                eq.Residual = analog_port.value() - var_to()

        # 4) Create port connections
        for port_connection in self.ninemlComponent.portconnections:
            #print 'try to connect {0} to {1}'.format(port_connection[0].getstr('.'), port_connection[1].getstr('.'))
            portFrom = getObjectFromNamespaceAddress(self, port_connection[0])
            portTo   = getObjectFromNamespaceAddress(self, port_connection[1])

            print '  {0} -> {1}\n'.format(portFrom.CanonicalName, portTo.CanonicalName)
            self.ConnectPorts(portFrom, portTo)

    def findVariable(self, name):
        for var in self.nineml_state_variables:
            if var.Name == name:
                return var
        return None

    #def parseExpression(self, expression, **kwargs):
    #    """
    #    Parses the expression, evaluates it using 'dictNameValue' and returns the result (adouble object)
    #    """
    #    print_result = kwargs.get('print_result', False)
    #
    #    result = parser.parse(expression)
    #    if print_result:
    #        print 'Expression: {0}\nParse result: {1}'.format(expression, str(result))
    #    return parser.evaluate()

    """
    def setParameters(self):
        for param in self.nineml_parameters:
            param.SetValue(0.0)
        for model in self.ninemlSubComponents:
            model.setParameters()

    def assignVariables(self):
        #for var in self.nineml_state_variables:
        #    var.AssignValue(0.0)
        #for var in self.nineml_aliases:
        #    var.AssignValue(0.0)
        for model in self.ninemlSubComponents:
            model.assignVariables()

    def setInitialConditions(self):
        for var in self.nineml_state_variables:
            var.SetInitialCondition(0.0)
        for model in self.ninemlSubComponents:
            model.setInitialConditions()
    """
