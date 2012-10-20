import sys, numpy, traceback
from daetools.pyDAE import *

class daeCodeGeneratorAnalyzer(object):
    def __init__(self):
        self._simulation        = None
        self.ports              = {}
        self.models             = {}
        self.runtimeInformation = {}

    def analyzeSimulation(self, simulation):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        self._simulation        = simulation
        self.ports              = {}
        self.models             = {}
        self.runtimeInformation = {
                                    'Domains'     : [],
                                    'Parameters'  : [],
                                    'Variables'   : [],
                                    'STNs'        : []
                                  }

        self._collectObjects(self._simulation.m)

        for port_class, dict_port in self.ports.items():
            data = self.analyzePort(dict_port['port'])

            self.ports[port_class]['data'] = data

        for model_class, dict_model in self.models.items():
            data = self.analyzeModel(dict_model['model'])

            self.models[model_class]['data'] = data

        self._collectRuntimeInformationFromModel(self._simulation.m)
       
    def analyzePort(self, port):
        result = { 'Class'         : '',
                   'Parameters'    : [],
                   'Domains'       : [],
                   'Variables'     : [],
                   'VariableTypes' : {}
                 }
                 
        result['Class'] = port.__class__.__name__
        
        if len(port.Domains) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        if len(port.Parameters) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        # Domains
        for domain in port.Domains:
            data = {}
            data['Name']                 = domain.Name
            data['Type']                 = str(domain.Type)
            data['Units']                = domain.Units
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            data['DiscretizationOrder']  = domain.DiscretizationOrder
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
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')

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
                   'CanonicalName'        : '',
                   'VariableTypes'        : {},
                   'Parameters'           : [],
                   'Domains'              : [],
                   'Variables'            : [],
                   'Ports'                : [],
                   'Components'           : [],
                   'Equations'            : [],
                   'PortConnections'      : [],
                   'EventPortConnections' : [],
                   'STNs'                 : []
                 }
 
        result['Class']         = model.__class__.__name__
        result['CanonicalName'] = model.CanonicalName
        
        # Domains
        for domain in model.Domains:
            data = {}
            data['Name']                 = domain.Name
            data['Type']                 = str(domain.Type)
            data['Units']                = domain.Units
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            data['DiscretizationOrder']  = domain.DiscretizationOrder
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
            data['PortFrom'] = daeGetRelativeName(model, port_connection.PortFrom)
            data['PortTo']   = daeGetRelativeName(model, port_connection.PortTo)

            result['PortConnections'].append(data)

        # Ports
        for port in model.Ports:
            data = {}
            data['Name']              = port.Name
            data['Class']             = port.__class__.__name__
            data['Type']              = str(port.Type)
            data['Description']       = port.Description

            result['Ports'].append(data)

        # equations
        result['Equations'] = self._processEquations(model.Equations, model)

        # StateTransitionNetworks
        result['STNs'] = self._processSTNs(model.STNs, model)

        # Components
        for component in model.Components:
            data = {}
            data['Name']              = component.Name
            data['Class']             = component.__class__.__name__
            data['Description']       = component.Description

            result['Components'].append(data)

        return result

    def _collectObjects(self, model):
        self.models[model.__class__.__name__] = {'model' : model, 'data' : None}

        for port in model.Ports:
            if not port.__class__.__name__ in self.ports:
                self.ports[port.__class__.__name__] = {'port' : port, 'data' : None}

        for model in model.Components:
            if not model.__class__.__name__ in self.models:
                self.models[model.__class__.__name__] = {'model' : model, 'data' : None}

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
            data['DistributedEquationDomainInfos'] = []
            for dedi in equation.DistributedEquationDomainInfos:
                dedi_data = {}
                dedi_data['Domain']       = dedi.Domain.Name
                dedi_data['DomainBounds'] = str(dedi.DomainBounds)
                dedi_data['DomainPoints'] = dedi.DomainPoints
                
                data['DistributedEquationDomainInfos'].append(dedi_data)

            data['EquationExecutionInfos']         = []
            for eeinfo in equation.EquationExecutionInfos:
                eedata = {}
                eedata['ResidualRuntimeNode'] = eeinfo.Node

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
            data['Name']        = stn.Name
            data['ActiveState'] = stn.ActiveState
            data['States']      = []
            for i, state in enumerate(stn.States):
                state_data = {}
                state_data['Name']              = state.Name
                state_data['Equations']         = self._processEquations(state.Equations, model)
                state_data['NestedSTNs']        = self._processSTNs(state.NestedSTNs, model)
                state_data['StateTransitions']  = []
                for state_transition in state.StateTransitions:
                    st_data = {}
                    st_data['ConditionSetupNode']   = state_transition.Condition.SetupNode
                    st_data['ConditionRuntimeNode'] = state_transition.Condition.RuntimeNode
                    st_data['Actions']              = self._processActions(state_transition.Actions, model)

                    state_data['StateTransitions'].append(st_data)

                data['States'].append(state_data)

            stns.append(data)

        return stns

    def _processActions(self, actions, model):
        sActions = []
        for action in actions:
            data = {}
            data['Type']            = str(action.Type)
            data['STN']             = action.STN.Name
            data['StateTo']         = action.StateTo.Name
            data['SendEventPort']   = daeGetRelativeName(model, action.SendEventPort)
            data['VariableWrapper'] = action.VariableWrapper
            data['SetupNode']       = action.SetupNode
            data['RuntimeNode']     = action.RuntimeNode
       
            sActions.append(data)

        return sActions

    def _collectRuntimeInformationFromModel(self, model):
        for domain in model.Domains:
            data = {}
            data['CanonicalName']        = domain.CanonicalName
            data['Type']                 = str(domain.Type)
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            data['DiscretizationOrder']  = domain.DiscretizationOrder
            data['Points']               = domain.npyPoints

            self.runtimeInformation['Domains'].append(data)

        for parameter in model.Parameters:
            data = {}
            data['CanonicalName']        = parameter.CanonicalName
            data['Domains']              = [domain.NumberOfPoints for domain in parameter.Domains]
            data['NumberOfPoints']       = parameter.NumberOfPoints
            data['Values']               = parameter.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = parameter.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = parameter.ReportingOn

            self.runtimeInformation['Parameters'].append(data)

        for variable in model.Variables:
            data = {}
            data['CanonicalName']        = variable.CanonicalName
            data['Domains']              = [domain.NumberOfPoints for domain in variable.Domains]
            data['NumberOfPoints']       = variable.NumberOfPoints
            data['Values']               = variable.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['IDs']                  = variable.npyIDs                   # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = variable.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = variable.ReportingOn

            self.runtimeInformation['Variables'].append(data)

        for port in model.Ports:
            self._collectRuntimeInformationFromPort(port)

        for stn in model.STNs:
            if isinstance(stn, daeSTN):
                data = {}
                data['CanonicalName']   = stn.CanonicalName
                data['ActiveState']     = stn.ActiveState
                stateMap = {}
                for i, state in enumerate(stn.States):
                    stateMap[state.Name] = i
                data['StateMap']        = stateMap

                self.runtimeInformation['STNs'].append(data)

        for component in model.Components:
            self._collectRuntimeInformationFromModel(component)

    def _collectRuntimeInformationFromPort(self, port):
        for domain in port.Domains:
            data = {}
            data['CanonicalName']        = domain.CanonicalName
            data['Type']                 = str(domain.Type)
            data['NumberOfIntervals']    = domain.NumberOfIntervals
            data['NumberOfPoints']       = domain.NumberOfPoints
            data['DiscretizationMethod'] = str(domain.DiscretizationMethod)
            data['DiscretizationOrder']  = domain.DiscretizationOrder
            data['Points']               = domain.npyPoints

            self.runtimeInformation['Domains'].append(data)

        for parameter in port.Parameters:
            data = {}
            data['CanonicalName']        = parameter.CanonicalName
            data['Domains']              = [domain.NumberOfPoints for domain in parameter.Domains]
            data['NumberOfPoints']       = parameter.NumberOfPoints
            data['Values']               = parameter.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = parameter.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = parameter.ReportingOn

            self.runtimeInformation['Parameters'].append(data)

        for variable in port.Variables:
            data = {}
            data['CanonicalName']        = variable.CanonicalName
            data['Domains']              = [domain.NumberOfPoints for domain in variable.Domains]
            data['NumberOfPoints']       = variable.NumberOfPoints
            data['Values']               = variable.npyValues                # nd_array[d1][d2]...[dn] or float if no domains
            data['IDs']                  = variable.npyIDs                   # nd_array[d1][d2]...[dn] or float if no domains
            data['DomainsIndexesMap']    = variable.GetDomainsIndexesMap(0)  # {index : [domains indexes]}
            data['ReportingOn']          = variable.ReportingOn

            self.runtimeInformation['Variables'].append(data)
