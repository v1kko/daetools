import sys, numpy, traceback
from daetools.pyDAE import *

portTemplate = """connector %(port)s
%(variables)s
end %(port)s;

"""

modelTemplate = """class %(model)s
/* Import libs */
  import Modelica.Math.*;
%(parameters)s
%(variables)s
%(ports)s
%(components)s

equation
%(port_connections)s
%(equations)s
%(stns)s
end %(model)s;

"""

wrapperTemplate = """class %(model)s_simulation
  annotation(experiment(StartTime = %(start_time)s, StopTime = %(end_time)s, Tolerance = %(tolerance)s));
  
%(model_instance)s

%(initial_equation)s
end %(model)s_simulation;

"""

def printNumpyArrayForModelica(nd_arr):
    return '{' + ','.join([str(val) for val in nd_arr]) + '}'
    
def nameFormat(name):
    return name.replace('&', '').replace(';', '')

def unitFormat(unit):
    res = []
    for u, exp in unit.unitDictionary.items():
        if exp >= 0:
            if exp == 1:
                res.append('{0}'.format(u))
            elif int(exp) == exp:
                res.append('{0}{1}'.format(u, int(exp)))
            else:
                res.append('{0}{1}'.format(u, exp))

    for u, exp in unit.unitDictionary.items():
        if exp < 0:
            if int(exp) == exp:
                res.append('{0}{1}'.format(u, int(exp)))
            else:
                res.append('{0}{1}'.format(u, exp))

    return '.'.join(res)

class exportNodeContext(object):
    def __init__(self, model):
        self.model = model
        
class daeModelicaExport(object):
    def __init__(self, simulation = None):
        self.wrapperInstanceName     = 'wrapper'
        self.defaultIndent           = '  '

        self.ports                   = {}
        self.models                  = {}
        self.parametersValues        = {}
        self.assignedVariablesValues = {}
        self.initialConditions       = {}
        self.initiallyActiveStates   = {}
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None
        self.variableTypes           = []
        self.initialVariableValues   = []

    def exportModel(self, model, filename):
        try:
            if not model:
                return

            numpy.set_string_function(printNumpyArrayForModelica, repr=False)

            self.ports                   = {}
            self.models                  = {}
            self.parametersValues        = {}
            self.assignedVariablesValues = {}
            self.initialConditions       = {}
            self.initiallyActiveStates   = {}
            self.warnings                = []
            self.simulation              = None
            self.topLevelModel           = model
            self.variableTypes           = []
            self.initialVariableValues   = []

            indent   = 1
            s_indent = indent * self.defaultIndent

            result = self._processModel(model, indent)

            f = open(filename, "w")
            f.write(result)
            f.close()

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print e
            traceback.print_tb(exc_traceback)
        finally:
            # Restore numpy printing to defaults
            numpy.set_string_function(None, repr=False)
            
    def exportPort(self, port, filename):
        try:
            if not port:
                return

            numpy.set_string_function(printNumpyArrayForModelica, repr=False)

            self.ports                   = {}
            self.models                  = {}
            self.parametersValues        = {}
            self.assignedVariablesValues = {}
            self.initialConditions       = {}
            self.initiallyActiveStates   = {}
            self.warnings                = []
            self.simulation              = None
            self.topLevelModel           = port.Model
            self.variableTypes           = []
            self.initialVariableValues   = []

            indent   = 1
            s_indent = indent * self.defaultIndent

            result = self._processPort(port, indent)

            f = open(filename, "w")
            f.write(result)
            f.close()

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print e
            traceback.print_tb(exc_traceback)
        finally:
            # Restore numpy printing to defaults
            numpy.set_string_function(None, repr=False)

    def exportSimulation(self, simulation, filename):
        try:
            if not simulation:
                return

            numpy.set_string_function(printNumpyArrayForModelica, repr=False)

            self.ports                   = {}
            self.models                  = {}
            self.parametersValues        = {}
            self.assignedVariablesValues = {}
            self.initialConditions       = {}
            self.initiallyActiveStates   = {}
            self.warnings                = []
            self.simulation              = simulation
            self.topLevelModel           = simulation.m
            self.variableTypes           = simulation.VariableTypes
            self.initialVariableValues   = simulation.InitialValues
            self._collectObjects(self.simulation.m)

            indent   = 1
            s_indent = indent * self.defaultIndent

            result = ''
            for port_class, port in self.ports.items():
                result += self._processPort(port, indent)
            for model_class, model in self.models.items():
                result += self._processModel(model, indent)

            self._collectRuntimeInformationFromModel(self.topLevelModel)

            f = open(filename, "w")
            f.write(result)

            model_instance = s_indent + '{0} {1}('.format(self.topLevelModel.__class__.__name__, self.wrapperInstanceName)
            indent = ' ' * len(model_instance)

            params        = ['{0} = {1}'     .format(key, value) for key, value in self.parametersValues.items()]
            assigned_vars = ['{0}{1} = {2};' .format(s_indent, key, value) for key, value in self.assignedVariablesValues.items()]
            init_conds    = ['{0}{1} := {2};'.format(s_indent, key, value) for key, value in self.initialConditions.items()]
            init_states   = ['{0}{1} := {2};'.format(s_indent, key, value) for key, value in self.initiallyActiveStates.items()]

            params.sort(key=str.lower)
            assigned_vars.sort(key=str.lower)
            init_conds.sort(key=str.lower)
            init_states.sort(key=str.lower)

            join_format = ',\n%s' % indent
            arguments = join_format.join(params)

            initial_equations = ''
            if len(assigned_vars) > 0:
                initial_equations  = 'equation\n/* Assigned variables (DOFs) */\n' + '\n'.join(sorted(assigned_vars))

            if len(init_states) > 0 or len(init_conds) > 0:
                initial_equations += '\n\ninitial algorithm\n'

            if len(init_states) > 0:
                initial_equations += '\n/* Initially active states */\n' + '\n'.join(sorted(init_states))

            if len(init_conds) > 0:
                initial_equations += '\n/* Initial conditions */\n' + '\n'.join(sorted(init_conds))

            model_instance = model_instance + arguments + ');'
            dictModel = {
                        'model'            : self.topLevelModel.__class__.__name__,
                        'model_instance'   : model_instance,
                        'initial_equation' : initial_equations,
                        'start_time'       : 0.0,
                        'end_time'         : self.simulation.TimeHorizon,
                        'tolerance'        : daeGetConfig().GetFloat('daetools.IDAS.relativeTolerance', 1e-5)
                        }
            wrapper = wrapperTemplate % dictModel
            f.write(wrapper)
            f.close()

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print e
            traceback.print_tb(exc_traceback)
        finally:
            # Restore numpy printing to defaults
            numpy.set_string_function(None, repr=False)

    def _collectObjects(self, model):
        self.models[model.__class__.__name__] = model

        for port in model.Ports:
            if not port.__class__.__name__ in self.ports:
                self.ports[port.__class__.__name__] = port

        for model in model.Components:
            if not model.__class__.__name__ in self.models:
                self.models[model.__class__.__name__] = model

        for model in model.Components:
            self._collectObjects(model)

        print 'Models collected:', self.models.keys()
        print 'Ports collected:', self.ports.keys()
        
    def _processPort(self, port, indent):
        sVariables  = []
        s_indent = indent * self.defaultIndent

        if len(port.Domains) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        if len(port.Parameters) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        for variable in port.Variables:
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')
                
            sVariables.append(s_indent + 'Real {0}(unit="{1}") "{2}";'.format(nameFormat(variable.Name),
                                                                              unitFormat(variable.VariableType.Units),
                                                                              variable.Description) )

        dictPort = {
                     'port'      : nameFormat(port.__class__.__name__),
                     'variables' : '\n'.join(sVariables)
                   }
        result = portTemplate % dictPort
        return result
    
    def _processModel(self, model, indent):
        sParameters      = []
        sVariables       = []
        sPorts           = []
        sComponents      = []
        sPortConnections = []
        sEquations       = []
        sSTNs            = []
        s_indent = indent * self.defaultIndent

        # Domains
        for domain in model.Domains:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, domain))
            sParameters.append(s_indent + 'parameter Integer {0}_np "Number of points in domain {0}";'.format(nameFormat(domain.Name)))
            sParameters.append(s_indent + 'parameter Real[{0}_np] {1}(unit="{2}") "{3}";\n'.format(nameFormat(domain.Name),
                                                                                                   nameFormat(domain.Name),
                                                                                                   unitFormat(domain.Units),
                                                                                                   domain.Description) )
                                                                                                        
        # Parameters
        for parameter in model.Parameters:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, parameter))
            domains = ''
            if len(parameter.Domains) > 0:
                domains = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in parameter.Domains))

            sParameters.append(s_indent + 'parameter Real{0} {1}(unit="{2}") "{3}";'.format(domains,
                                                                                            nameFormat(parameter.Name),
                                                                                            unitFormat(parameter.Units),
                                                                                            parameter.Description) )

        # Variables
        for variable in model.Variables:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            startValue    = ''
            if len(variable.Domains) > 0:
                domains    = '[{0}]'.format(','.join(nameFormat(d.Name)+'_np' for d in variable.Domains))
                varType = 'distributed'
                startValue = ''
                try:
                    for vt in self.variableTypes[variable.OverallIndex : variable.OverallIndex+variable.NumberOfPoints]:
                        if vt == cnDifferential:
                            varType    = 'distributed differential'
                            startValue = 'start = 0.0, '
                            break
                except:
                    pass

            else:
                domains = ''
                varType = 'unknown'
                startValue = ''
                try:
                    if self.variableTypes[variable.OverallIndex] == cnDifferential:
                        varType    = 'differential'
                        startValue = 'start = 0.0, '
                    elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                        varType = 'assigned'
                    else:
                        varType = 'algebraic'
                except:
                    pass
                
            sVariables.append(s_indent + '/* type: {0} */'.format(varType))
            sVariables.append(s_indent + 'Real{0} {1}({2}unit="{3}") "{4}";'.format(domains,
                                                                                    nameFormat(variable.Name),
                                                                                    startValue,
                                                                                    unitFormat(variable.VariableType.Units),
                                                                                    variable.Description) )

        context = exportNodeContext(model)

        # equations
        sEquations.extend(self._processEquations(model.Equations, indent, context))

        # PortConnections
        for port_connection in model.PortConnections:
            nameFrom = nameFormat(daeGetRelativeName(model, port_connection.PortFrom))
            nameTo   = nameFormat(daeGetRelativeName(model, port_connection.PortTo))
            sPortConnections.append(s_indent + 'connect({0}, {1});'.format(nameFrom, nameTo))

        # Ports
        for port in model.Ports:
            if port.Type == eInletPort:
                portType = 'inlet'
            elif port.Type == eOutletPort:
                portType = 'outlet'
            else:
                portType = 'unknown'
            sPorts.append(s_indent + '/* type: {0} */'.format(portType))
            sPorts.append(s_indent + '{0} {1} "{2}";'.format(port.__class__.__name__,
                                                             nameFormat(port.Name),
                                                             port.Description))

        # StateTransitionNetworks
        sSTNs.extend(self._processSTNs(model.STNs, indent, context, sVariables))

        # Components
        for component in model.Components:
            sComponents.append(s_indent + '{0} {1} "{2}";'.format(component.__class__.__name__,
                                                                  nameFormat(component.Name),
                                                                  component.Description))

        # Put all together                                                                    
        _ports            = ('\n/* Ports */      \n' if len(sPorts)      else '') + '\n'.join(sPorts)
        _parameters       = ('\n/* Parameters */ \n' if len(sParameters) else '') + '\n'.join(sParameters)
        _variables        = ('\n/* Variables */  \n' if len(sVariables)  else '') + '\n'.join(sVariables)
        _components       = ('\n/* Components */ \n' if len(sComponents) else '') + '\n'.join(sComponents)
        _port_connections =                                                         '\n'.join(sPortConnections)
        _equations        =                                                         '\n'.join(sEquations)
        _stns             =                                                         '\n'.join(sSTNs)
        dictModel = {
                      'model'            : model.__class__.__name__,
                      'ports'            : _ports,
                      'parameters'       : _parameters,
                      'variables'        : _variables,
                      'components'       : _components,
                      'port_connections' : _port_connections,
                      'equations'        : _equations,
                      'stns'             : _stns
                    }
        result = modelTemplate % dictModel
        return result

    def _collectRuntimeInformationFromModel(self, model):
        for domain in model.Domains:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, domain))
            self.parametersValues[relativeName+'_np'] = str(domain.NumberOfPoints)
            self.parametersValues[relativeName]       = '{{{0}}}'.format(','.join(str(p) for p in domain.Points))

        for parameter in model.Parameters:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, parameter))
            if len(parameter.Domains) > 0:
                p_vals = parameter.GetNumPyArray()
                values = str(p_vals)
            else:
                values = str(parameter.GetValue())
            self.parametersValues[relativeName] = values

        for variable in model.Variables:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            if len(variable.Domains) > 0:
                # This will return dictionary {overallIndex : [domain_indexes]}
                # Domain indexes are 0-based and must be converted to 1-base
                dictIndexes = variable.OverallVSDomainsIndexesMap
                for overall_index, list_of_domain_indexes in dictIndexes.iteritems():
                    vt = self.variableTypes[overall_index]
                    # ACHTUNG, ACHTUNG!!
                    # Modelica uses 1-based indexing
                    name = '{0}.{1}[{2}]'.format(self.wrapperInstanceName, relativeName, ','.join([str(di+1) for di in list_of_domain_indexes]))
                    if vt == cnDifferential:
                        self.initialConditions[name] = value
                    elif vt == cnAssigned:
                        self.assignedVariablesValues[name] = value
                    
            else:
                name  = '{0}.{1}'.format(self.wrapperInstanceName, relativeName)
                value = variable.GetValue()
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    self.initialConditions[name] = value
                elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                    self.assignedVariablesValues[name] = value

        for port in model.Ports:
            self._collectRuntimeInformationFromPort(port)

        for stn in model.STNs:
            if isinstance(stn, daeSTN):
                name = '{0}.{1}'.format(self.wrapperInstanceName, nameFormat(daeGetRelativeName(self.topLevelModel, stn)))
                stateMap = {}
                for i, state in enumerate(stn.States):
                    stateMap[state.Name] = i
                self.initiallyActiveStates[name] = stateMap[stn.ActiveState]

        for component in model.Components:
            self._collectRuntimeInformationFromModel(component)

    def _collectRuntimeInformationFromPort(self, port):
        for variable in port.Variables:
            relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, variable))
            if len(variable.Domains) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')
            else:
                name  = '{0}.{1}'.format(self.wrapperInstanceName, relativeName)
                value = variable.GetValue()
                if self.variableTypes[variable.OverallIndex] == cnDifferential:
                    self.initialConditions[name] = value
                elif self.variableTypes[variable.OverallIndex] == cnAssigned:
                    self.assignedVariablesValues[name] = value

    def _processEquations(self, Equations, indent, context):
        sEquations = []
        s_indent = indent * self.defaultIndent
        for equation in Equations:
            sEquations.append(s_indent + '/* {0} */'.format(equation.Description))
            for eeinfo in equation.EquationExecutionInfos:
                node = eeinfo.Node
                sEquations.append(s_indent + '{0} = 0;'.format(self._processNode(node, context)))
        return sEquations
                    
    def _processSTNs(self, STNs, indent, context, sVariables):
        sSTNs  = []
        sWhens = []
        s_indent  = indent * self.defaultIndent
        
        for stn in STNs:
            if isinstance(stn, daeIF):
                for i, state in enumerate(stn.States):
                    if i == 0:
                        state_transition = state.StateTransitions[0]
                        condition = self._processConditionNode(state_transition.Condition.RuntimeNode, context)
                        sSTNs.append(s_indent + 'if {0} then'.format(condition))
                        sSTNs.extend(self._processEquations(state.Equations, indent+1, context))

                    elif i != len(stn.States)-1:
                        state_transition = state.StateTransitions[0]
                        condition = self._processConditionNode(state_transition.Condition.RuntimeNode, context)
                        sSTNs.append(s_indent + 'else if {0} then'.format(condition))
                        sSTNs.extend(self._processEquations(state.Equations, indent+1, context))

                    else:
                        sSTNs.append(s_indent + 'else')
                        sSTNs.extend(self._processEquations(state.Equations, indent+1, context))

                    sSTNs.extend(self._processSTNs(state.NestedSTNs, indent+1, context, sVariables))

                sSTNs.append(s_indent + 'end if;')

            elif isinstance(stn, daeSTN):
                relativeName = nameFormat(daeGetRelativeName(self.topLevelModel, stn))
                stnVariableName = relativeName
                activeState = stn.ActiveState
                stateMap = {}
                for i, state in enumerate(stn.States):
                    stateMap[state.Name] = i
                sortedStateMap = sorted(stateMap.iteritems(), key=lambda x:x[1])
                
                sVariables.append(s_indent + '/* State transition network */')
                sVariables.append(s_indent + 'Integer {0};'.format(stnVariableName))

                for i, state in enumerate(stn.States):
                    if i == 0:
                        sSTNs.append(s_indent + '/* STN {0}: {1}*/'.format(stnVariableName, ', '.join([st + ' = ' + str(i) for st, i in sortedStateMap])))
                        sSTNs.append(s_indent + 'if {0} == {1} then'.format(stnVariableName, str(stateMap[state.Name])))
                    elif i != len(stn.States)-1:
                        sSTNs.append(s_indent + 'elseif {0} == {1} then'.format(stnVariableName, str(stateMap[state.Name])))
                    else:
                        sSTNs.append(s_indent + 'else')

                    sSTNs.extend(self._processEquations(state.Equations, indent+1, context))

                    for state_transition in state.StateTransitions:
                        condition = self._processConditionNode(state_transition.Condition.RuntimeNode, context)
                        if len(sWhens) == 0:
                            sWhens.append(s_indent + '/* State transitions of {0} */'.format(stnVariableName))
                            sWhens.append(s_indent + 'when ({0} == {1}) and ({2}) then'.format(stnVariableName, str(stateMap[state.Name]), condition))
                        else:
                            sWhens.append(s_indent + 'elsewhen ({0} == {1}) and ({2}) then'.format(stnVariableName, str(stateMap[state.Name]), condition))
                        sWhens.extend(self._processActions(state_transition.Actions, indent+1, context, stateMap))

                    if len(state.NestedSTNs) > 0:
                        raise RuntimeError('Nested state transition networks (daeSTN) canot be exported to modelica')

                if len(sWhens) > 0:
                    sWhens.append(s_indent + 'end when;')

                sSTNs.append(s_indent + 'end if;')
                sSTNs.extend(sWhens)

        return sSTNs

    def _processActions(self, Actions, indent, context, stateMap):
        sActions = []
        s_indent  = indent * self.defaultIndent

        for action in Actions:
            if action.Type == eChangeState:
                sActions.append(s_indent + '{0} = {1};'.format(action.STN.Name, str(stateMap[action.StateTo.Name])))

            elif action.Type == eSendEvent:
                pass

            elif action.Type == eReAssignOrReInitializeVariable:
                pass

            elif action.Type == eUserDefinedAction:
                pass

            else:
                pass
       
        return sActions
    
    def _processConditionNode(self, node, context):
        res = ''
        if isinstance(node, condUnaryNode):
            n = '(' + self._processConditionNode(node.Node, context) + ')'
            if node.LogicalOperator == eNot:
                res = 'not {0}'.format(n)
            else:
                raise RuntimeError('Not supported unary logical operator')

        elif isinstance(node, condBinaryNode):
            left  = '(' + self._processConditionNode(node.LNode, context) + ')'
            right = '(' + self._processConditionNode(node.RNode, context) + ')'

            if node.LogicalOperator == eAnd:
                res = '{0} and {1}'.format(left, right)
            elif node.LogicalOperator == eOr:
                res = '{0} or {1}'.format(left, right)
            else:
                raise RuntimeError('Not supported binary logical operator')

        elif isinstance(node, condExpressionNode):
            left  = '(' + self._processNode(node.LNode, context) + ')'
            right = '(' + self._processNode(node.RNode, context) + ')'

            if node.ConditionType == eNotEQ: # !=
                res = '{0} != {1}'.format(left, right)
            elif node.ConditionType == eEQ: # ==
                res = '{0} == {1}'.format(left, right)
            elif node.ConditionType == eGT: # >
                res = '{0} > {1}'.format(left, right)
            elif node.ConditionType == eGTEQ: # >=
                res = '{0} >= {1}'.format(left, right)
            elif node.ConditionType == eLT: # <
                res = '{0} < {1}'.format(left, right)
            elif node.ConditionType == eLTEQ: # <=
                res = '{0} <= {1}'.format(left, right)
            else:
                raise RuntimeError('Not supported condition type')
        else:
            raise RuntimeError('Not supported condition node')

        return res

    def _processNode(self, node, context):
        res = ''
        if isinstance(node, adConstantNode):
            res = '{0}'.format(node.Quantity.value) #, node.Quantity.units)

        elif isinstance(node, adTimeNode):
            res = 'time'

        elif isinstance(node, adUnaryNode):
            n = '(' + self._processNode(node.Node, context) + ')'

            if node.Function == eSign:
                res = '-{0}'.format(n)
            elif node.Function == eSqrt:
                res = 'sqrt({0})'.format(n)
            elif node.Function == eExp:
                res = 'exp({0})'.format(n)
            elif node.Function == eLog:
                res = 'log10({0})'.format(n)
            elif node.Function == eLn:
                res = 'log({0})'.format(n)
            elif node.Function == eAbs:
                res = 'abs({0})'.format(n)
            elif node.Function == eSin:
                res = 'sin({0})'.format(n)
            elif node.Function == eCos:
                res = 'cos({0})'.format(n)
            elif node.Function == eTan:
                res = 'tan({0})'.format(n)
            elif node.Function == eArcSin:
                res = 'asin({0})'.format(n)
            elif node.Function == eArcCos:
                res = 'acos({0})'.format(n)
            elif node.Function == eArcTan:
                res = 'atan({0})'.format(n)
            elif node.Function == eCeil:
                res = 'ceil({0})'.format(n)
            elif node.Function == eFloor:
                res = 'floor({0})'.format(n)
            else:
                raise RuntimeError('Not supported unary function')

        elif isinstance(node, adBinaryNode):
            left  = '(' + self._processNode(node.LNode, context) + ')'
            right = '(' + self._processNode(node.RNode, context) + ')'

            if node.Function == ePlus:
                res = '{0} + {1}'.format(left, right)
            elif node.Function == eMinus:
                res = '{0} - {1}'.format(left, right)
            elif node.Function == eMulti:
                res = '{0} * {1}'.format(left, right)
            elif node.Function == eDivide:
                res = '{0} / {1}'.format(left, right)
            elif node.Function == ePower:
                res = '{0} ^ {1}'.format(left, right)
            elif node.Function == eMin:
                res = 'min({0}, {1})'.format(left, right)
            elif node.Function == eMax:
                res = 'max({0}, {1})'.format(left, right)
            else:
                raise RuntimeError('Not supported binary function')

        elif isinstance(node, adScalarExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adVectorExternalFunctionNode):
            raise RuntimeError('External functions are not supported')

        elif isinstance(node, adDomainIndexNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            name = nameFormat(daeGetRelativeName(context.model, node.Domain))
            res = '{0}[{1}]'.format(name, node.Index+1)

        elif isinstance(node, adRuntimeParameterNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Parameter))
            res = '{0}{1}'.format(name, domainindexes)

        elif isinstance(node, adRuntimeVariableNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Variable))
            res = '{0}{1}'.format(name, domainindexes)

        elif isinstance(node, adRuntimeTimeDerivativeNode):
            # ACHTUNG, ACHTUNG!!
            # Modelica uses 1-based indexing
            domainindexes = ''
            if len(node.DomainIndexes) > 0:
                domainindexes = '[{0}]'.format(','.join([str(di + 1) for di in node.DomainIndexes]))

            name = nameFormat(daeGetRelativeName(context.model, node.Variable))
            res = 'der({0}{1})'.format(name, domainindexes)

        else:
            raise RuntimeError('Not supported node')

        return res
