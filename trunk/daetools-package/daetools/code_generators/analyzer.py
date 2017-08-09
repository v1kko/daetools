"""
***********************************************************************************
                            analyzer.py
                DAE Tools: pyDAE module, www.daetools.com
                Copyright (C) Dragan Nikolic
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
************************************************************************************
"""
import sys, numpy, traceback
from daetools.pyDAE import *

class daeCodeGeneratorAnalyzer(object):
    def __init__(self):
        self._simulation        = None
        self.ports              = []
        self.dictPorts          = {}
        self.models             = []
        self.dictModels         = {}
        self.runtimeInformation = {}

    def analyzeSimulation(self, simulation):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        self._simulation        = simulation
        self.ports              = []
        self.dictPorts          = {}
        self.models             = []
        self.dictModels         = {}
        self.runtimeInformation = {
                                    'Domains'         : [],
                                    'Parameters'      : [],
                                    'Variables'       : [],
                                    'Equations'       : [],
                                    'STNs'            : [],
                                    'PortConnections' : []
                                  }

        self._collectObjects(self._simulation.m)
        #print self.ports
        #print self.models
        
        self.models.reverse()

        for port_class, dict_port in self.ports:
            data = self.analyzePort(dict_port['port'])
            # Update dict_port
            dict_port['data'] = data

        for model_class, dict_model in self.models:
            data = self.analyzeModel(dict_model['model'])
            # Update dict_model
            dict_model['data'] = data

        self.runtimeInformation['ModelType']              = self._simulation.m.ModelType
        self.runtimeInformation['IsModelDynamic']         = self._simulation.m.IsModelDynamic
        self.runtimeInformation['TotalNumberOfVariables'] = self._simulation.TotalNumberOfVariables
        self.runtimeInformation['NumberOfEquations']      = self._simulation.NumberOfEquations
        self.runtimeInformation['IDs']                    = self._simulation.VariableTypes
        self.runtimeInformation['Values']                 = self._simulation.Values
        self.runtimeInformation['TimeDerivatives']        = self._simulation.TimeDerivatives
        self.runtimeInformation['IndexMappings']          = self._simulation.IndexMappings
        self.runtimeInformation['RelativeTolerance']      = self._simulation.RelativeTolerance
        self.runtimeInformation['AbsoluteTolerances']     = self._simulation.AbsoluteTolerances
        self.runtimeInformation['TimeHorizon']            = self._simulation.TimeHorizon
        self.runtimeInformation['ReportingInterval']      = self._simulation.ReportingInterval
        self.runtimeInformation['QuasiSteadyState']       = True if self._simulation.InitialConditionMode == eQuasiSteadyState else False

        self._collectRuntimeInformationFromModel(self._simulation.m)
       
    def analyzePort(self, port):
        result = { 'Class'         : '',
                   'Name'          : '',
                   'Parameters'    : [],
                   'Domains'       : [],
                   'Variables'     : [],
                   'VariableTypes' : {}
                 }
                 
        result['Class'] = port.__class__.__name__
        result['Name']  = port.Name
        
        #if len(port.Domains) > 0:
        #    raise RuntimeError('Ports cannot contain domains')
        #if len(port.Parameters) > 0:
        #    raise RuntimeError('Ports cannot contain parameters')

        # Domains
        for domain in port.Domains:
            data = {}
            data['Name']                 = domain.Name
            data['Type']                 = str(domain.Type)
            data['Units']                = domain.Units
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            #data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            #data['DiscretizationOrder']  = domain.DiscretizationOrder
            data['Description']          = domain.Description

            result['Domains'].append(data)

        # Parameters
        for parameter in port.Parameters:
            data = {}
            data['Name']                 = parameter.Name
            data['Domains']              = [daeGetRelativeName(port, domain) for domain in parameter.Domains]
            data['Units']                = parameter.Units
            data['Description']          = parameter.Description

            result['Parameters'].append(data)

        for variable in port.Variables:
            #if len(variable.Domains) > 0:
            #    raise RuntimeError('Ports cannot contain distributed variables')

            data = {}
            data['Name']              = variable.Name
            data['Domains']           = [daeGetRelativeName(port, domain) for domain in variable.Domains]
            data['Type']              = variable.VariableType.Name
            data['Description']       = variable.Description
            if variable.NumberOfPoints == 1:
                IDs = [variable.npyIDs]
            else:
                IDs = variable.npyIDs
            if cnDifferential in IDs:
                data['RuntimeHint'] = 'differential'
            elif cnAssigned in IDs:
                data['RuntimeHint'] = 'assigned'
            else:
                data['RuntimeHint'] = 'algebraic'
            if not variable.VariableType.Name in result['VariableTypes']:
                vt_data = {}
                vt_data['Name']              = variable.VariableType.Name
                vt_data['Units']             = variable.VariableType.Units
                vt_data['LowerBound']        = variable.VariableType.LowerBound
                vt_data['UpperBound']        = variable.VariableType.UpperBound
                vt_data['InitialGuess']      = variable.VariableType.InitialGuess
                vt_data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
                if not variable.VariableType.Name in result['VariableTypes']:
                    result['VariableTypes'][variable.VariableType.Name] = vt_data

            result['Variables'].append(data)

        return result
    
    def analyzeModel(self, model):
        result = { 'Class'                : '',
                   'Name'                 : '',
                   'CanonicalName'        : '',
                   'VariableTypes'        : {},
                   'Parameters'           : [],
                   'Domains'              : [],
                   'Variables'            : [],
                   'Ports'                : [],
                   'EventPorts'           : [],
                   'Components'           : [],
                   'Equations'            : [],
                   'PortConnections'      : [],
                   'EventPortConnections' : [],
                   'STNs'                 : [],
                   'OnConditionActions'   : [],
                   'OnEventActions'       : [],
                   'PortArrays'           : [], # ??
                   'ComponentArrays'      : []  # ??
                 }
 
        result['Class']         = model.__class__.__name__
        result['Name']          = model.Name
        result['CanonicalName'] = model.CanonicalName
        
        # Domains
        for domain in model.Domains:
            data = {}
            data['Name']                 = domain.Name
            data['Type']                 = str(domain.Type)
            data['Units']                = domain.Units
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            #data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            #data['DiscretizationOrder']  = domain.DiscretizationOrder
            data['Description']          = domain.Description

            result['Domains'].append(data)

        # Parameters
        for parameter in model.Parameters:
            data = {}
            data['Name']                 = parameter.Name
            data['Domains']              = [daeGetRelativeName(model, domain) for domain in parameter.Domains]
            data['Units']                = parameter.Units
            data['Description']          = parameter.Description

            result['Parameters'].append(data)
        
        # Variables
        for variable in model.Variables:
            data = {}
            data['Name']              = variable.Name
            data['Domains']           = [daeGetRelativeName(model, domain) for domain in variable.Domains]
            data['Type']              = variable.VariableType.Name
            data['Description']       = variable.Description
            if variable.NumberOfPoints == 1:
                IDs = [variable.npyIDs]
            else:
                IDs = variable.npyIDs
            if cnDifferential in IDs:
                data['RuntimeHint'] = 'differential'
            elif cnAssigned in IDs:
                data['RuntimeHint'] = 'assigned'
            else:
                data['RuntimeHint'] = 'algebraic'
            if not variable.VariableType.Name in result['VariableTypes']:
                vt_data = {}
                vt_data['Name']              = variable.VariableType.Name
                vt_data['Units']             = variable.VariableType.Units
                vt_data['LowerBound']        = variable.VariableType.LowerBound
                vt_data['UpperBound']        = variable.VariableType.UpperBound
                vt_data['InitialGuess']      = variable.VariableType.InitialGuess
                vt_data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
                if not variable.VariableType.Name in result['VariableTypes']:
                    result['VariableTypes'][variable.VariableType.Name] = vt_data

            result['Variables'].append(data)

        # PortConnections
        for port_connection in model.PortConnections:
            data = {}
            data['PortFrom']  = daeGetRelativeName(model, port_connection.PortFrom)
            data['PortTo']    = daeGetRelativeName(model, port_connection.PortTo)
            data['Equations'] = self._processEquations(port_connection.Equations, model)

            result['PortConnections'].append(data)

        # EventPortConnections
        for event_port_connection in model.EventPortConnections:
            data = {}
            data['PortFrom']  = daeGetRelativeName(model, event_port_connection.PortFrom)
            data['PortTo']    = daeGetRelativeName(model, event_port_connection.PortTo)

            result['EventPortConnections'].append(data)
        
        # Ports
        for port in model.Ports:
            data = {}
            data['Name']              = port.Name
            data['Class']             = port.__class__.__name__
            data['Type']              = str(port.Type)
            data['Description']       = port.Description

            result['Ports'].append(data)
        
        # EventPorts
        for event_port in model.EventPorts:
            data = {}
            data['Name']              = event_port.Name
            data['Class']             = event_port.__class__.__name__
            data['Type']              = str(event_port.Type)
            data['Description']       = event_port.Description

            result['EventPorts'].append(data)

        # Equations
        result['Equations'] = self._processEquations(model.Equations, model)

        # StateTransitionNetworks
        result['STNs'] = self._processSTNs(model.STNs, model)
        
        # OnConditionActions
        result['OnConditionActions'] = self._processOnConditionActions(model.OnConditionActions, model)
        
        # OnEventActions
        result['OnEventActions'] = self._processOnEventActions(model.OnEventActions, model)

        # Components
        for component in model.Components:
            data = {}
            data['Name']              = component.Name
            data['Class']             = component.__class__.__name__
            data['Description']       = component.Description

            result['Components'].append(data)

        # PortArrays
        if len(model.PortArrays) > 0:
            raise RuntimeError('PortArrays are not supported by the code analyzer')

        # ComponentArrays
        if len(model.ComponentArrays) > 0:
            raise RuntimeError('ComponentArrays are not supported by the code analyzer')

        return result

    def _collectObjects(self, model):
        if not model.__class__.__name__ in self.dictModels:
            self.dictModels[model.__class__.__name__] = None
            self.models.append( (model.__class__.__name__, {'model' : model, 'data' : None}) )

        for port in model.Ports:
            if not port.__class__.__name__ in self.dictPorts:
                self.dictPorts[port.__class__.__name__] = None
                self.ports.append( (port.__class__.__name__, {'port' : port, 'data' : None}) )

        for model in model.Components:
            self._collectObjects(model)

    def _processEquations(self, equations, model):
        eqns = []
        for equation in equations:
            data = {}
            data['Name']                           = equation.Name
            data['Scaling']                        = equation.Scaling
            data['Residual']                       = equation.Residual
            data['Description']                    = equation.Description
            data['EquationType']                   = equation.EquationType
            data['DistributedEquationDomainInfos'] = []
            for dedi in equation.DistributedEquationDomainInfos:
                dedi_data = {}
                # Get a name relative to the equation's parent model 
                dedi_data['Domain']       = daeGetRelativeName(equation.Model, dedi.Domain)
                dedi_data['DomainBounds'] = str(dedi.DomainBounds)
                dedi_data['DomainPoints'] = dedi.DomainPoints
                
                data['DistributedEquationDomainInfos'].append(dedi_data)

            data['EquationExecutionInfos']         = []
            for eeinfo in equation.EquationExecutionInfos:
                eedata = {}
                eedata['ResidualRuntimeNode'] = eeinfo.Node
                eedata['VariableIndexes']     = eeinfo.VariableIndexes
                eedata['EquationType']        = str(eeinfo.EquationType)

                data['EquationExecutionInfos'].append(eedata)

            eqns.append(data)

        return eqns
                    
    def _processSTNs(self, STNs, model):
        stns  = []        
        for stn in STNs:
            data = {}
            if isinstance(stn, daeIF):
                data['Class'] = 'daeIF'
            else:
                data['Class'] = 'daeSTN'
            data['Name']          = stn.Name
            data['CanonicalName'] = stn.CanonicalName
            data['Description']   = stn.Description
            data['ActiveState']   = stn.ActiveState
            stateMap = {}
            for i, state in enumerate(stn.States):
                stateMap[state.Name] = i
            data['StateMap'] = stateMap
            data['States']   = []
            for i, state in enumerate(stn.States):
                state_data = {}
                state_data['Name']               = state.Name
                state_data['Equations']          = self._processEquations(state.Equations, model)
                state_data['NestedSTNs']         = self._processSTNs(state.NestedSTNs, model)
                state_data['OnConditionActions'] = self._processOnConditionActions(state.OnConditionActions, model)
                state_data['OnEventActions']     = self._processOnEventActions(state.OnEventActions, model)

                data['States'].append(state_data)

            stns.append(data)

        return stns

    def _processOnConditionActions(self, on_condition_actions, model):
        sOnConditionActions = []
        for on_condition_action in on_condition_actions:
            oca_data = {}
            oca_data['ConditionSetupNode']   = on_condition_action.Condition.SetupNode
            oca_data['ConditionRuntimeNode'] = on_condition_action.Condition.RuntimeNode
            oca_data['EventTolerance']       = on_condition_action.Condition.EventTolerance
            oca_data['Expressions']          = on_condition_action.Condition.Expressions
            oca_data['Actions']              = self._processActions(on_condition_action.Actions, model)
            if len(on_condition_action.UserDefinedActions) > 0:
                raise RuntimeError('UserDefinedActions cannot be exported')

            sOnConditionActions.append(oca_data)

        return sOnConditionActions
    
    def _processOnEventActions(self, on_event_actions, model):
        sOnEventActions = []
        for on_event_action in on_event_actions:
            oea_data = {}
            oea_data['EventPort']            = daeGetRelativeName(model, on_event_action.EventPort)
            oea_data['Actions']              = self._processActions(on_event_action.Actions, model)
            if len(on_event_action.UserDefinedActions) > 0:
                raise RuntimeError('UserDefinedActions cannot be exported')
            
            sOnEventActions.append(oea_data)

        return sOnEventActions
    
    def _processActions(self, actions, model):
        sActions = []
        for action in actions:
            data = {}
            data['Type'] = str(action.Type)
            if action.Type == eChangeState:
                data['STN']              = action.STN.Name
                data['STNCanonicalName'] = action.STN.CanonicalName
                data['StateTo']          = action.StateTo.Name

            elif action.Type == eSendEvent:
                data['SendEventPort']    = daeGetRelativeName(model, action.SendEventPort)
                data['VariableWrapper']  = action.VariableWrapper

            elif action.Type == eReAssignOrReInitializeVariable:
                data['VariableWrapper']       = action.VariableWrapper
                data['SetupNode']             = action.SetupNode
                data['RuntimeNode']           = action.RuntimeNode
                data['DomainIndexes']         = action.VariableWrapper.DomainIndexes
                data['VariableCanonicalName'] = action.VariableWrapper.Variable.CanonicalName
                data['OverallIndex']          = action.VariableWrapper.OverallIndex
                data['ID']                    = action.VariableWrapper.VariableType

            elif action.Type == eUserDefinedAction:
                raise RuntimeError('User defined actions are not supported')
       
            sActions.append(data)

        return sActions

    def _collectRuntimeInformationFromModel(self, model):
        for domain in model.Domains:
            data = {}
            data['CanonicalName']        = domain.CanonicalName
            data['Description']          = domain.Description
            data['Type']                 = str(domain.Type)
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            #data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            #data['DiscretizationOrder']  = domain.DiscretizationOrder
            if domain.Type == eUnstructuredGrid:
                data['Points']           = numpy.array([i for i in range(domain.NumberOfPoints)])
            else:
                data['Points']           = domain.Points
            data['Units']                = domain.Units

            self.runtimeInformation['Domains'].append(data)

        for parameter in model.Parameters:
            data = {}
            data['CanonicalName']        = parameter.CanonicalName
            data['Description']          = parameter.Description
            data['Domains']              = [domain.NumberOfPoints for domain in parameter.Domains]
            data['DomainNames']          = [domain.CanonicalName for domain in parameter.Domains]
            data['NumberOfPoints']       = parameter.NumberOfPoints
            data['Values']               = parameter.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = parameter.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = parameter.ReportingOn
            data['Units']                = parameter.Units

            self.runtimeInformation['Parameters'].append(data)

        for variable in model.Variables:
            data = {}
            data['CanonicalName']        = variable.CanonicalName
            data['Description']          = variable.Description
            data['Domains']              = [domain.NumberOfPoints for domain in variable.Domains]
            data['DomainNames']          = [domain.CanonicalName for domain in variable.Domains]
            data['NumberOfPoints']       = variable.NumberOfPoints
            data['OverallIndex']         = variable.OverallIndex
            data['Values']               = variable.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['IDs']                  = variable.npyIDs                   # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = variable.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = variable.ReportingOn
            data['Units']                = variable.VariableType.Units
            data['VariableType']         = variable.VariableType

            self.runtimeInformation['Variables'].append(data)

        for port in model.Ports:
            self._collectRuntimeInformationFromPort(port)

        for port_connection in model.PortConnections:
            data = {}
            data['PortFrom']        = port_connection.PortFrom.CanonicalName
            data['PortTo']          = port_connection.PortTo.CanonicalName
            data['Equations']       = self._processEquations(port_connection.Equations, model)

            self.runtimeInformation['PortConnections'].append(data)

        # equations
        self.runtimeInformation['Equations'].extend( self._processEquations(model.Equations, model) )

        # StateTransitionNetworks
        self.runtimeInformation['STNs'].extend( self._processSTNs(model.STNs, model) )

        for component in model.Components:
            self._collectRuntimeInformationFromModel(component)

    def _collectRuntimeInformationFromPort(self, port):
        for domain in port.Domains:
            data = {}
            data['CanonicalName']        = domain.CanonicalName
            data['Description']          = domain.Description
            data['Type']                 = str(domain.Type)
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            #data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            #data['DiscretizationOrder']  = domain.DiscretizationOrder
            if domain.Type == eUnstructuredGrid:
                data['Points']           = numpy.array([i for i in range(domain.NumberOfPoints)])
            else:
                data['Points']           = domain.npyPoints
            data['Units']                = domain.Units

            self.runtimeInformation['Domains'].append(data)

        for parameter in port.Parameters:
            data = {}
            data['CanonicalName']        = parameter.CanonicalName
            data['Description']          = parameter.Description
            data['Domains']              = [domain.NumberOfPoints for domain in parameter.Domains]
            data['DomainNames']          = [domain.CanonicalName for domain in parameter.Domains]
            data['NumberOfPoints']       = parameter.NumberOfPoints
            data['Values']               = parameter.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = parameter.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = parameter.ReportingOn
            data['Units']                = parameter.Units

            self.runtimeInformation['Parameters'].append(data)

        for variable in port.Variables:
            data = {}
            data['CanonicalName']        = variable.CanonicalName
            data['Description']          = variable.Description
            data['Domains']              = [domain.NumberOfPoints for domain in variable.Domains]
            data['DomainNames']          = [domain.CanonicalName for domain in variable.Domains]
            data['NumberOfPoints']       = variable.NumberOfPoints
            data['OverallIndex']         = variable.OverallIndex
            data['Values']               = variable.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['IDs']                  = variable.npyIDs                   # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = variable.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = variable.ReportingOn
            data['Units']                = variable.VariableType.Units
            data['VariableType']         = variable.VariableType

            self.runtimeInformation['Variables'].append(data)
