import sys, numpy, traceback
from daetools.pyDAE import *

class daeExportAnalyzer(object):
    def __init__(self, expressionFormatter):
        if not expressionFormatter:
            raise RuntimeError('Invalid expression formatter object')

        self.expressionFormatter     = expressionFormatter
        self.simulation              = None
        self.portsToAnalyze          = {}
        self.modelsToAnalyze         = {}

    def analyzeSimulation(self, simulation):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        self.simulation              = simulation
        self.portsToAnalyze          = {}
        self.modelsToAnalyze         = {}

        self._collectObjects(self.simulation.m)

        for port_class, dict_port in self.portsToAnalyze.items():
            data = self.analyzePort(dict_port['port'])
            self.portsToAnalyze[port_class]['data'] = data

        for model_class, dict_model in self.modelsToAnalyze.items():
            data = self.analyzeModel(dict_model['model'])
            self.modelsToAnalyze[model_class]['data'] = data

    def _collectObjects(self, model):
        self.modelsToAnalyze[model.__class__.__name__] = {'model' : model, 'data' : None}

        for port in model.Ports:
            if not port.__class__.__name__ in self.portsToAnalyze:
                self.portsToAnalyze[port.__class__.__name__] = {'port' : port, 'data' : None}

        for model in model.Components:
            if not model.__class__.__name__ in self.modelsToAnalyze:
                self.modelsToAnalyze[model.__class__.__name__] = {'model' : model, 'data' : None}

        for model in model.Components:
            self._collectObjects(model)

        #print 'Models collected:', self.modelsToAnalyze.keys()
        #print 'Ports collected:', self.portsToAnalyze.keys()
        
    def analyzePort(self, port):
        result = { 'Class'         : None,
                   'Parameters'    : [],
                   'Domains'       : [],
                   'Variables'     : [],
                   'VariableTypes' : []
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
            data['LowerBound']           = domain.LowerBound
            data['UpperBound']           = domain.UpperBound
            data['Description']          = domain.Description
            data['Points']               = domain.npyPoints

            result['Domains'].append(data)

        # Parameters
        for parameter in port.Parameters:
            data = {}
            data['Name']                 = parameter.Name
            data['Domains']              = [daeGetRelativeName(port, domain) for domain in parameter.Domains]
            data['Units']                = parameter.Units
            data['Description']          = parameter.Description
            data['Values']               = parameter.npyValues
            data['ReportingOn']          = parameter.ReportingOn

            result['Parameters'].append(data)

        for variable in port.Variables:
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')

            data = {}
            data['Name']              = variable.Name
            data['Domains']           = [daeGetRelativeName(port, domain) for domain in variable.Domains]
            data['Type']              = variable.VariableType.Name
            #data['Units']             = variable.VariableType.Units
            #data['LowerBound']        = variable.VariableType.LowerBound
            #data['UpperBound']        = variable.VariableType.UpperBound
            #data['InitialGuess']      = variable.VariableType.InitialGuess
            #data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
            data['Description']       = variable.Description
            data['Values']            = variable.npyValues
            data['IDs']               = variable.npyIDs
            data['DomainsIndexesMap'] = variable.GetDomainsIndexesMap(0)
            data['ReportingOn']       = variable.ReportingOn
            if not variable.VariableType.Name in result['VariableTypes']:
                vt_data = {}
                vt_data['Name']              = variable.VariableType.Name
                vt_data['Units']             = variable.VariableType.Units
                vt_data['LowerBound']        = variable.VariableType.LowerBound
                vt_data['UpperBound']        = variable.VariableType.UpperBound
                vt_data['InitialGuess']      = variable.VariableType.InitialGuess
                vt_data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
                result['VariableTypes'].append(vt_data)

            result['Variables'].append(data)

        return result
    
    def analyzeModel(self, model):
        result = { 'Class'                : None,
                   'VariableTypes'        : [],
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
        sParameters      = []
        sVariables       = []
        sPorts           = []
        sComponents      = []
        sPortConnections = []
        sEquations       = []
        sSTNs            = []

        self.expressionFormatter.model = model

        result['Class'] = model.__class__.__name__
        
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
            data['LowerBound']           = domain.LowerBound
            data['UpperBound']           = domain.UpperBound
            data['Description']          = domain.Description
            data['Points']               = domain.npyPoints

            result['Domains'].append(data)

        # Parameters
        for parameter in model.Parameters:
            data = {}
            data['Name']                 = parameter.Name
            data['Domains']              = [daeGetRelativeName(model, domain) for domain in parameter.Domains]
            data['Units']                = parameter.Units
            data['Description']          = parameter.Description
            data['Values']               = parameter.npyValues
            data['ReportingOn']          = parameter.ReportingOn

            result['Parameters'].append(data)
        
        # Variables
        for variable in model.Variables:
            data = {}
            data['Name']              = variable.Name
            data['Domains']           = [daeGetRelativeName(model, domain) for domain in variable.Domains]
            data['Type']              = variable.VariableType.Name
            #data['Units']             = variable.VariableType.Units
            #data['LowerBound']        = variable.VariableType.LowerBound
            #data['UpperBound']        = variable.VariableType.UpperBound
            #data['InitialGuess']      = variable.VariableType.InitialGuess
            #data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
            data['Description']       = variable.Description
            data['Values']            = variable.npyValues
            data['IDs']               = variable.npyIDs
            data['DomainsIndexesMap'] = variable.GetDomainsIndexesMap(0)
            data['ReportingOn']       = variable.ReportingOn
            if not variable.VariableType.Name in result['VariableTypes']:
                vt_data = {}
                vt_data['Name']              = variable.VariableType.Name
                vt_data['Units']             = variable.VariableType.Units
                vt_data['LowerBound']        = variable.VariableType.LowerBound
                vt_data['UpperBound']        = variable.VariableType.UpperBound
                vt_data['InitialGuess']      = variable.VariableType.InitialGuess
                vt_data['AbsoluteTolerance'] = variable.VariableType.AbsoluteTolerance
                result['VariableTypes'].append(vt_data)

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

    def _processEquations(self, equations, model):
        eqns = []
        for equation in equations:
            data = {}
            data['Name']                           = equation.Name
            data['Scaling']                        = equation.Scaling
            data['Residual']                       = equation.Residual
            data['Description']                    = equation.Description
            data['EquationExecutionInfos']         = []
            data['DistributedEquationDomainInfos'] = []
            for dedi in equation.DistributedEquationDomainInfos:
                dedi_data = {}
                dedi_data['Domain']       = dedi.Domain.Name
                dedi_data['DomainBounds'] = str(dedi.DomainBounds)
                dedi_data['DomainPoints'] = dedi.DomainPoints
                
                data['DistributedEquationDomainInfos'].append(dedi_data)
                
            for eeinfo in equation.EquationExecutionInfos:
                data['EquationExecutionInfos'].append( self.expressionFormatter.formatRuntimeNode(eeinfo.Node) ) #############

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
                    st_data['ConditionRuntimeNode'] = self.expressionFormatter.formatRuntimeConditionNode(state_transition.Condition.RuntimeNode) ############
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

