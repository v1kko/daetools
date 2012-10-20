import sys, numpy, math, traceback
from daetools.pyDAE import *
from formatter import daeExpressionFormatter
from analyzer import daeCodeGeneratorAnalyzer


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


class daeModelicaExpressionFormatter(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)

        # Index base in arrays
        self.indexBase = 1

        # Use relative names
        self.useRelativeNames = True

        # One- and multi-dimensional parameters/variables and domain points
        self.indexLeftBracket  = '['
        self.indexRightBracket = ']'
        self.indexFormat = 'python_index_style'

        # String format for the time derivative: der(variable)
        self.derivative = 'der({variable}{indexes})'

        # Logical operators
        self.AND   = 'and'
        self.OR    = 'or'
        self.NOT   = 'not'

        self.EQ    = '=='
        self.NEQ   = '!='
        self.LT    = '<'
        self.LTEQ  = '<='
        self.GT    = '>'
        self.GTEQ  = '>='

        # Mathematical operators
        self.PLUS   = '+'
        self.MINUS  = '-'
        self.MULTI  = '*'
        self.DIVIDE = '/'
        self.POWER  = '^'

        # Mathematical functions
        self.SIN    = 'sin'
        self.COS    = 'cos'
        self.TAN    = 'tan'
        self.ASIN   = 'asin'
        self.ACOS   = 'acos'
        self.ATAN   = 'atan'
        self.EXP    = 'exp'
        self.SQRT   = 'sqrt'
        self.LOG    = 'log'
        self.LOG10  = 'log10'
        self.FLOOR  = 'floor'
        self.CEIL   = 'ceil'
        self.ABS    = 'abs'
        self.MIN    = 'min'
        self.MAX    = 'max'

        # Current time in simulation
        self.TIME   = 'time'

    def formatNumpyArray(self, arr):
        if isinstance(arr, numpy.ndarray):
            return '{' + ','.join([self.formatNumpyArray(val) for val in arr]) + '}'
        else:
            return str(arr)

    def formatQuantity(self, quantity):
        # Formats constants/quantities in equations that have a value and units
        return str(quantity.value)

    def formatUnits(self, units):
        # Format: m.kg2/s-2 meaning m * kg**2 / s**2
        positive = []
        negative = []
        for u, exp in units.unitDictionary.items():
            if exp >= 0:
                if exp == 1:
                    positive.append('{0}'.format(u))
                elif int(exp) == exp:
                    positive.append('{0}{1}'.format(u, int(exp)))
                else:
                    positive.append('{0}{1}'.format(u, exp))

        for u, exp in units.unitDictionary.items():
            if exp < 0:
                if exp == -1:
                    negative.append('{0}'.format(u))
                elif int(exp) == exp:
                    negative.append('{0}{1}'.format(u, int(math.fabs(exp))))
                else:
                    negative.append('{0}{1}'.format(u, math.fabs(exp)))

        sPositive = ' '.join(positive)
        if len(negative) == 0:
            sNegative = ''
        elif len(negative) == 1:
            sNegative = '/' + '.'.join(negative)
        else:
            sNegative = '/(' + '.'.join(negative) + ')'

        return sPositive + sNegative
        
class daeModelicaExport(object):
    def __init__(self, simulation = None):
        self.wrapperInstanceName     = ''
        self.defaultIndent           = '  '
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None
        
        self.exprFormatter = daeModelicaExpressionFormatter()
        self.analyzer      = daeCodeGeneratorAnalyzer()

    def exportModel(self, model, filename = None):
        if not model:
            raise RuntimeError('Invalid model object')

        self.parametersValues        = {}
        self.assignedVariablesValues = {}
        self.initialConditions       = {}
        self.initiallyActiveStates   = {}
        self.warnings                = []
        self.simulation              = None
        self.topLevelModel           = model

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeModel(model)
        result = self._processModel(model, indent)

        if filename:
            f = open(filename, "w")
            f.write(result)
            f.close()

        return result

    def exportPort(self, port, filename = None):
        if not port:
            raise RuntimeError('Invalid port object')

        self.parametersValues        = {}
        self.assignedVariablesValues = {}
        self.initialConditions       = {}
        self.initiallyActiveStates   = {}
        self.warnings                = []
        self.simulation              = None
        self.topLevelModel           = port.Model

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzePort(port)
        result = self._processPort(port, indent)

        if filename:
            f = open(filename, "w")
            f.write(result)
            f.close()

        return result

    def exportSimulation(self, simulation, filename = None):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        self.parametersValues        = {}
        self.assignedVariablesValues = {}
        self.initialConditions       = {}
        self.initiallyActiveStates   = {}
        self.warnings                = []
        self.simulation              = simulation
        self.topLevelModel           = simulation.m
        self.wrapperInstanceName     = self.exprFormatter.formatIdentifier(simulation.m.Name)

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)

        resultModel = ''
        for port_class, info in self.analyzer.ports.items():
            resultModel += self._processPort(info['data'], indent)

        for model_class, info in self.analyzer.models.items():
            resultModel += self._processModel(info['data'], indent)

        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

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
            initial_equations  = 'equation\n/* Assigned variables (DOFs) */\n' + '\n'.join(assigned_vars)

        if len(init_states) > 0 or len(init_conds) > 0:
            initial_equations += '\n\ninitial algorithm\n'

        if len(init_states) > 0:
            initial_equations += '\n/* Initially active states */\n' + '\n'.join(init_states)

        if len(init_conds) > 0:
            initial_equations += '\n/* Initial conditions */\n' + '\n'.join(init_conds)

        model_instance = model_instance + arguments + ');'
        dictModel = {
                    'model'            : self.topLevelModel.__class__.__name__,
                    'model_instance'   : model_instance,
                    'initial_equation' : initial_equations,
                    'start_time'       : 0.0,
                    'end_time'         : self.simulation.TimeHorizon,
                    'tolerance'        : daeGetConfig().GetFloat('daetools.IDAS.relativeTolerance', 1e-5)
                    }
        resultWrapper = wrapperTemplate % dictModel

        if filename:
            f = open(filename, "w")
            f.write(resultModel)
            f.write(resultWrapper)
            f.close()

        return resultModel, resultWrapper

    def _processPort(self, data, indent):
        sVariables  = []
        s_indent = indent * self.defaultIndent

        if len(data['Domains']) > 0:
            raise RuntimeError('Modelica ports cannot contain domains')

        if len(data['Parameters']) > 0:
            raise RuntimeError('Modelica ports cannot contain parameters')

        # Variables
        for variable in data['Variables']:
            if len(variable['Domains']) > 0:
                raise RuntimeError('Modelica ports cannot contain distributed variables')

            name        = self.exprFormatter.formatIdentifier(variable['Name'])
            units       = self.exprFormatter.formatUnits(data['VariableTypes'][variable['Type']]['Units'] if variable['Type'] in data['VariableTypes'] else '')
            domains     = ''
            attributes  = '(unit="{units}")'.format(units = units)
            description = variable['Description']

            varTemplate = s_indent + 'Real{domains} {name}{attributes} "{description}";'
            sVariables.append(varTemplate.format(name = name,
                                                 domains = domains,
                                                 attributes = attributes,
                                                 description = description))

        dictPort = {
                     'port'      : data['Class'],
                     'variables' : '\n'.join(sVariables)
                   }
        result = portTemplate % dictPort
        return result
    
    def _processModel(self, data, indent):
        sParameters      = []
        sVariables       = []
        sPorts           = []
        sComponents      = []
        sPortConnections = []
        sEquations       = []
        sSTNs            = []

        s_indent = indent * self.defaultIndent

        # Achtung, Achtung!!
        self.exprFormatter.modelCanonicalName = data['CanonicalName']
        
        # Domains
        for domain in data['Domains']:
            name        = self.exprFormatter.formatIdentifier(domain['Name'])
            units       = self.exprFormatter.formatUnits(domain['Units'])
            noPoints    = domain['NumberOfPoints']
            domains     = '[{name}_np]'.format(name = name)
            attributes  = '(unit="{units}")'.format(units = units)
            description = domain['Description']
            
            domTemplate   = s_indent + 'parameter Integer {name}_np "Number of points in domain {name}";'
            paramTemplate = s_indent + 'parameter Real{domains} {name}{attributes} "{description}";\n'
            
            sParameters.append(domTemplate.format(name = name,
                                                  noPoints = noPoints))
            sParameters.append(paramTemplate.format(name = name,
                                                    domains = domains,
                                                    attributes = attributes,
                                                    description = description))
                                                                                                        
        # Parameters
        for parameter in data['Parameters']:
            name        = self.exprFormatter.formatIdentifier(parameter['Name'])
            units       = self.exprFormatter.formatUnits(parameter['Units'])
            domains = ''
            if len(parameter['Domains']) > 0:
                domains = '[{0}]'.format(','.join(self.exprFormatter.formatIdentifier(d)+'_np' for d in parameter['Domains']))
            attributes  = '(unit="{units}")'.format(units = units)
            description = parameter['Description']

            paramTemplate = s_indent + 'parameter Real{domains} {name}{attributes} "{description}";'
            sParameters.append(paramTemplate.format(name = name,
                                                    domains = domains,
                                                    attributes = attributes,
                                                    description = description))

        # Variables
        for variable in data['Variables']:
            name        = self.exprFormatter.formatIdentifier(variable['Name'])
            units       = self.exprFormatter.formatUnits(data['VariableTypes'][variable['Type']]['Units'] if variable['Type'] in data['VariableTypes'] else '')
            if len(variable['Domains']) > 0:
                domains = '[{0}]'.format(','.join(self.exprFormatter.formatIdentifier(d)+'_np' for d in variable['Domains']))
                if variable['RuntimeHint'] == 'differential':
                    attributes  = '(start=0.0, unit="{units}")'.format(units = units)
                else:
                    attributes  = '(unit="{units}")'.format(units = units)
            else:
                domains = ''
                if variable['RuntimeHint'] == 'differential':
                    attributes  = '(start=0.0, unit="{units}")'.format(units = units)
                else:
                    attributes  = '(unit="{units}")'.format(units = units)
            description = variable['Description']

            varTemplate = s_indent + 'Real{domains} {name}{attributes} "{description}";'
            sVariables.append(s_indent + '/* type hint: {0} variable */'.format(variable['RuntimeHint']))
            sVariables.append(varTemplate.format(name = name,
                                                 domains = domains,
                                                 attributes = attributes,
                                                 description = description))
        
        # Equations
        sEquations.extend(self._processEquations(data['Equations'], indent))

        # PortConnections
        for port_connection in data['PortConnections']:
            portconnTemplate = s_indent + 'connect({portFrom}, {portTo});'
            sPortConnections.append(portconnTemplate.format(portFrom = port_connection['PortFrom'],
                                                            portTo = port_connection['PortTo']))

        # Ports
        for port in data['Ports']:
            class_      = port['Class']
            name        = port['Name']
            if port['Type'] == 'eInletPort':
                portType = 'inlet'
            elif port['Type'] == 'eOutletPort':
                portType = 'outlet'
            else:
                portType = 'unknown'
            description = port['Description']

            portTemplate = s_indent + '{class_} {name} "{description}";'
            sPorts.append(s_indent + '/* type: {0} port */'.format(portType))
            sPorts.append(portTemplate.format(class_ = class_,
                                              name = name,
                                              description = description))

        # StateTransitionNetworks
        sSTNs.extend(self._processSTNs(data['STNs'], indent, sVariables))

        # Components
        for component in data['Components']:
            class_      = component['Class']
            name        = component['Name']
            description = component['Description']

            compTemplate = s_indent + '{class_} {name} "{description}";'
            sComponents.append(compTemplate.format(class_ = class_,
                                                   name = name,
                                                   description = description))
                
        # Put all together                                                                    
        _ports            = ('\n/* Ports */      \n' if len(sPorts)      else '') + '\n'.join(sPorts)
        _parameters       = ('\n/* Parameters */ \n' if len(sParameters) else '') + '\n'.join(sParameters)
        _variables        = ('\n/* Variables */  \n' if len(sVariables)  else '') + '\n'.join(sVariables)
        _components       = ('\n/* Components */ \n' if len(sComponents) else '') + '\n'.join(sComponents)
        _port_connections =                                                         '\n'.join(sPortConnections)
        _equations        =                                                         '\n'.join(sEquations)
        _stns             =                                                         '\n'.join(sSTNs)
        dictModel = {
                      'model'            : data['Class'],
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

    def _processEquations(self, Equations, indent):
        sEquations = []
        s_indent = indent * self.defaultIndent
        eqTemplate = s_indent + '{residual} = 0;'
        for equation in Equations:
            description = equation['Description']
            sEquations.append(s_indent + '/* {0} */'.format(description))
            for eeinfo in equation['EquationExecutionInfos']:
                residual = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                sEquations.append(eqTemplate.format(residual = residual))
        return sEquations
                    
    def _processSTNs(self, STNs, indent, sVariables):
        sSTNs  = []
        sWhens = []
        s_indent  = indent * self.defaultIndent
        
        for stn in STNs:
            nStates = len(stn['States'])
            if stn['Class'] == 'daeIF':
                for i, state in enumerate(stn['States']):
                    if i == 0:
                        state_transition = state['StateTransitions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
                        sSTNs.append(s_indent + 'if {0} then'.format(condition))
                        sSTNs.extend(self._processEquations(state['Equations'], indent+1))

                    elif (i > 0) and (i < nStates - 1):
                        state_transition = state['StateTransitions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
                        sSTNs.append(s_indent + 'else if {0} then'.format(condition))
                        sSTNs.extend(self._processEquations(state['Equations'], indent+1))

                    else:
                        sSTNs.append(s_indent + 'else')
                        sSTNs.extend(self._processEquations(state['Equations'], indent+1))

                    sSTNs.extend(self._processSTNs(state['NestedSTNs'], indent+1, sVariables))

                sSTNs.append(s_indent + 'end if;')

            elif stn['Class'] == 'daeSTN':
                stnVariableName = stn['Name'] + '_stn'
                activeState     = stn['ActiveState']
                #stateMap = {}
                #for i, state in enumerate(stn['States']):
                #    stateMap[state['Name']] = i
                #sortedStateMap = sorted(stateMap.iteritems(), key=lambda x:x[1])
                
                sVariables.append(s_indent + '/* State transition network */')
                sVariables.append(s_indent + 'String {0};'.format(stnVariableName))

                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    if i == 0:
                        sSTNs.append(s_indent + '/* STN {0}: {1}*/'.format(stnVariableName, ', '.join([st['Name'] for st in stn['States']])))
                        sSTNs.append(s_indent + 'if {0} == "{1}" then'.format(stnVariableName, state['Name']))
                    elif (i > 0) and (i < nStates - 1):
                        sSTNs.append(s_indent + 'elseif {0} == "{1}" then'.format(stnVariableName, state['Name']))
                    else:
                        sSTNs.append(s_indent + 'else')

                    sSTNs.extend(self._processEquations(state['Equations'], indent+1))

                    for state_transition in state['StateTransitions']:
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
                        if len(sWhens) == 0:
                            sWhens.append(s_indent + '/* State transitions of {0} */'.format(stnVariableName))
                            sWhens.append(s_indent + 'when ({0} == "{1}") and ({2}) then'.format(stnVariableName, state['Name'], condition))
                        else:
                            sWhens.append(s_indent + 'elsewhen ({0} == "{1}") and ({2}) then'.format(stnVariableName, state['Name'], condition))
                        sWhens.extend(self._processActions(state_transition['Actions'], indent+1))

                    if len(state['NestedSTNs']) > 0:
                        raise RuntimeError('Nested state transition networks (daeSTN) canot be exported to modelica')

                if len(sWhens) > 0:
                    sWhens.append(s_indent + 'end when;')

                sSTNs.append(s_indent + 'end if;')
                sSTNs.extend(sWhens)

        return sSTNs

    def _processActions(self, Actions, indent):
        sActions = []
        s_indent  = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                stnVariableName = action['STN'] + '_stn'
                stateTo         = action['StateTo']
                sActions.append(s_indent + '{0} = "{1}";'.format(stnVariableName, stateTo))

            elif action['Type'] == 'eSendEvent':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            elif action['Type'] == 'eReAssignOrReInitializeVariable':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            elif action['Type'] == 'eUserDefinedAction':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            else:
                pass
       
        return sActions

    def _generateRuntimeInformation(self, runtimeInformation):
        for domain in runtimeInformation['Domains']:
            relativeName = daeGetRelativeName(self.wrapperInstanceName, domain['CanonicalName'])
            relativeName = self.exprFormatter.formatIdentifier(relativeName)
            self.parametersValues[relativeName+'_np'] = str(domain['NumberOfPoints'])
            self.parametersValues[relativeName]       = self.exprFormatter.formatNumpyArray(domain['Points']) # Numpy array
            
        for parameter in runtimeInformation['Parameters']:
            relativeName = daeGetRelativeName(self.wrapperInstanceName, parameter['CanonicalName'])
            relativeName = self.exprFormatter.formatIdentifier(relativeName)
            if parameter['NumberOfPoints'] == 1:
                self.parametersValues[relativeName] = str(parameter['Values'])
            else:
                self.parametersValues[relativeName] = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array

        for variable in runtimeInformation['Variables']:
            canonicalName = self.exprFormatter.formatIdentifier(variable['CanonicalName'])
            n = variable['NumberOfPoints']
            if n == 1:
                ID    = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value = float(variable['Values'])   # numpy float
                name  = canonicalName

                if ID == cnDifferential:
                    self.initialConditions[name] = value
                elif ID == cnAssigned:
                    self.assignedVariablesValues[name] = value

            else:
                for i in range(0, n):
                    domIndexes = tuple(variable['DomainsIndexesMap'][i])  # list of integers
                    ID         = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value      = float(variable['Values'][domIndexes])    # numpy float
                    # ACHTUNG, ACHTUNG!!
                    # Modelica uses 1-based indexing
                    name = canonicalName + '[' + ','.join([str(index + self.exprFormatter.indexBase) for index in domIndexes]) + ']'

                    if ID == cnDifferential:
                        self.initialConditions[name] = value
                    elif ID == cnAssigned:
                        self.assignedVariablesValues[name] = value

        for stn in runtimeInformation['STNs']:
            stnVariableName = stn['CanonicalName'] + '_stn'
            self.initiallyActiveStates[stnVariableName] = '"' + stn['ActiveState'] + '"'
