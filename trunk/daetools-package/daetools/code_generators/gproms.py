import os, shutil, sys, numpy, math, traceback
from daetools.pyDAE import *
from .formatter import daeExpressionFormatter
from .analyzer import daeCodeGeneratorAnalyzer
from .code_generator import daeCodeGenerator

modelTemplate = """\
{ daetools-gPROMS code-generator warnings:
%(warnings)s
}

%(variable_types_defs)s

MODEL %(model_name)s
PARAMETER
  %(domains_defs)s
  %(parameters_defs)s

VARIABLE
  %(variables_defs)s

SELECTOR
  %(stns_defs)s

EQUATION
  %(equations_defs)s
END

PROCESS %(model_name)s
UNIT
  %(instance_name)s as %(model_name)s

SET
  %(domains_inits)s
  %(parameters_inits)s

ASSIGN
  %(assigned_inits)s

%(stns_inits)s

INITIAL
  %(initial_conditions)s

SOLUTIONPARAMETERS
  ReportingInterval := %(reporting_interval)f;

SCHEDULE
  CONTINUE FOR %(time_horizon)f;
END
"""

class daeExpressionFormatter_gPROMS(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)
        
        # Index base for arrays
        self.indexBase = 1

        self.useFlattenedNamesForAssignedVariables = False
        self.IDs      = {}
        self.indexMap = {}

        # Use relative names and flatten identifiers
        self.useRelativeNames   = True
        self.flattenIdentifiers = True

        self.domain                   = '{domain}({index})'

        self.parameter                = '{parameter}{indexes}'
        self.parameterIndexStart      = '('
        self.parameterIndexEnd        = ')'
        self.parameterIndexDelimiter  = ','

        self.variable                 = '{variable}{indexes}'
        self.variableIndexStart       = '('
        self.variableIndexEnd         = ')'
        self.variableIndexDelimiter   = ','

        self.assignedVariable         = '{variable}'

        self.feMatrixItem             = '{value}'
        self.feVectorItem             = '{value}'

        # String format for the time derivative, ie. der(variable[1,2]) in Modelica
        # daetools use: variable.dt(1,2), gPROMS $variable(1,2) ...
        self.derivative               = '${variable}{indexes}'
        self.derivativeIndexStart     = '('
        self.derivativeIndexEnd       = ')'
        self.derivativeIndexDelimiter = ','

        # Constants
        self.constant = '{value}'

        # External functions
        self.scalarExternalFunction = '{name}()'
        self.vectorExternalFunction = '{name}()'

        # Logical operators
        self.AND   = '{leftValue} and {rightValue}'
        self.OR    = '{leftValue} or {rightValue}'
        self.NOT   = 'not {value}'

        self.EQ    = '{leftValue} == {rightValue}'
        self.NEQ   = '{leftValue} != {rightValue}'
        self.LT    = '{leftValue} < {rightValue}'
        self.LTEQ  = '{leftValue} <= {rightValue}'
        self.GT    = '{leftValue} > {rightValue}'
        self.GTEQ  = '{leftValue} >= {rightValue}'

        # Mathematical operators
        self.SIGN   = '-{value}'

        self.PLUS   = '{leftValue} + {rightValue}'
        self.MINUS  = '{leftValue} - {rightValue}'
        self.MULTI  = '{leftValue} * {rightValue}'
        self.DIVIDE = '{leftValue} / {rightValue}'
        self.POWER  = '{leftValue} ^ {rightValue}'

        # Mathematical functions
        self.SIN    = 'SIN({value})'
        self.COS    = 'COS({value})'
        self.TAN    = 'TAN({value})'
        self.ASIN   = 'ASIN({value})'
        self.ACOS   = 'ACOS({value})'
        self.ATAN   = 'ATAN({value})'
        self.EXP    = 'EXP({value})'
        self.SQRT   = 'SQRT({value})'
        self.LOG    = 'LOG({value})'
        self.LOG10  = 'LOG10({value})'
        self.FLOOR  = 'FLOOR({value})'
        self.CEIL   = 'CEIL({value})'
        self.ABS    = 'ABS({value})'

        self.MIN    = 'MIN({leftValue}, {rightValue})'
        self.MAX    = 'MAX({leftValue}, {rightValue})'

        # Current time in simulation
        self.TIME   = '__TIME__'

    def formatNumpyArray(self, arr):
        if isinstance(arr, (numpy.ndarray, list)):
            return '[' + ', '.join([self.formatNumpyArray(val) for val in arr]) + ']'
        else:
            return str(arr)
     
class daeCodeGenerator_gPROMS(daeCodeGenerator):
    def __init__(self):
        self.exprFormatter = daeExpressionFormatter_gPROMS()
        self.analyzer      = daeCodeGeneratorAnalyzer()

        self._reset()

    def _reset(self):
        self.rootModelName = ''
        self.defaultIndent = '  '
        self.warnings      = []
        self.topLevelModel = None
        self.simulation    = None

        self.variable_types = []
        self.domains        = []
        self.parameters     = []
        self.variables      = []
        self.stns           = []
        self.equations      = []

        self.domains_inits      = []
        self.parameters_inits   = []
        self.assigned_inits     = []
        self.stns_inits         = []
        self.initial_conditions = []
        
    def generateSimulation(self, simulation, directory):
        # Reset all arrays
        self._reset()

        if not simulation:
            raise RuntimeError('Invalid simulation object')

        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        # Achtung, Achtung!!
        # rootModelName and exprFormatter.modelCanonicalName should not be stripped
        # of illegal characters, since they are used to get relative names
        self.rootModelName                    = simulation.m.Name
        self.exprFormatter.modelCanonicalName = simulation.m.Name

        self.simulation    = simulation
        self.topLevelModel = simulation.m

        model_name    = simulation.m.GetStrippedName()
        instance_name = simulation.m.GetStrippedName()

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)
        self.exprFormatter.IDs      = self.analyzer.runtimeInformation['IDs']
        self.exprFormatter.indexMap = self.analyzer.runtimeInformation['IndexMappings']

        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

        separator = '\n' + self.defaultIndent
        warnings  = '\n'.join(self.warnings)

        # Model
        variable_types_defs = '\n'.join(self.variable_types)
        domains_defs        = separator.join(self.domains)
        parameters_defs     = separator.join(self.parameters)
        variables_defs      = separator.join(self.variables)
        stns_defs           = separator.join(self.stns)
        equations_defs      = separator.join(self.equations)

        # Simulation
        domains_inits      = separator.join(self.domains_inits)
        parameters_inits   = separator.join(self.parameters_inits)
        assigned_inits     = separator.join(self.assigned_inits)
        stns_inits         = ''
        if self.stns_inits:
            stns_inits = 'SELECTOR\n  '
            stns_inits += separator.join(self.stns_inits)
        initial_conditions = separator.join(self.initial_conditions)
        reporting_interval = self.simulation.ReportingInterval
        time_horizon       = self.simulation.TimeHorizon
        relative_tolerance = self.simulation.DAESolver.RelativeTolerance
            
        dictInfo = {
                     'instance_name' :      instance_name,
                     'model_name' :         model_name,
                     
                     'variable_types_defs': variable_types_defs,
                     'domains_defs' :       domains_defs,
                     'parameters_defs' :    parameters_defs,
                     'variables_defs' :     variables_defs,
                     'stns_defs' :          stns_defs,
                     'equations_defs' :     equations_defs,
                        
                     'domains_inits' :      domains_inits,
                     'parameters_inits' :   parameters_inits,
                     'assigned_inits' :     assigned_inits,
                     'stns_inits' :         stns_inits,
                     'initial_conditions' : initial_conditions,
                     'reporting_interval' : reporting_interval,
                     'time_horizon' :       time_horizon,
                     'relative_tolerance' : relative_tolerance,
                     
                     'warnings' :           warnings
                   }

        model_contents = modelTemplate % dictInfo;

        f = open(os.path.join(directory, '%s.gPROMS' % model_name), "w")
        f.write(model_contents)
        f.close()

        if len(self.warnings) > 0:
            print('CODE GENERATOR WARNINGS:')
            print(warnings)

    def _processEquations(self, Equations, indent):
        s_indent  = indent     * self.defaultIndent
        s_indent2 = (indent+1) * self.defaultIndent
        
        for equation in Equations:
            description = equation['Description']
            if description:
                self.equations.append(s_indent + '# {0}:'.format(description))
            
            for eeinfo in equation['EquationExecutionInfos']:
                res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                # self.equations.append(s_indent + '# Type: {0}'.format(eeinfo['EquationType']))
                self.equations.append(s_indent + '{0} = 0;'.format(res))

    def _processSTNs(self, STNs, indent):
        s_indent = indent * self.defaultIndent
        s_indent1 = (indent + 1) * self.defaultIndent
        s_indent2 = (indent + 2) * self.defaultIndent
        rootModel = self.exprFormatter.formatIdentifier(self.rootModelName)

        for stn in STNs:
            if stn['Class'] == 'daeIF':
                relativeName    = daeGetRelativeName(self.rootModelName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName)
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    # Not all states have state_transitions ('else' state has no state transitions)
                    on_condition_action = None
                    if i == 0:
                        # There is only one OnConditionAction in IF
                        on_condition_action = state['OnConditionActions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                        temp = s_indent + 'IF ({0}) THEN'.format(condition)
                        self.equations.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        # There is only one OnConditionAction in ELSE_IFs
                        on_condition_action = state['OnConditionActions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                        temp = s_indent + 'ELSE IF ({0}) THEN'.format(condition)
                        self.equations.append(temp)

                    else:
                        temp = s_indent + 'ELSE '
                        self.equations.append(temp)

                    # 1a. Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+1)
                    
                    # 1b. Put equations into the residuals list
                    self._processEquations(state['Equations'], indent+1)

                end = s_indent
                for i in range(nStates-1):
                    end += 'END '
                self.equations.append(end)

            elif stn['Class'] == 'daeSTN':
                relativeName    = daeGetRelativeName(self.rootModelName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName)
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                if description:
                    stnTemplate = '# {description}:'
                    self.stns.append(stnTemplate.format(description = description))

                stnTemplate = '{name} AS ({states})'
                self.stns.append(stnTemplate.format(name = stnVariableName,
                                                    states = states))
                stnTemplate = '{rootModel}.{name} := {rootModel}.{activeState};'
                self.stns_inits.append(stnTemplate.format(name = stnVariableName,
                                                          rootModel = rootModel,
                                                          activeState = activeState))

                temp = s_indent + 'CASE {0} OF'.format(stnVariableName)
                self.equations.append(temp)
                
                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    temp = s_indent1 + 'WHEN {state}:'.format(state = state['Name'])
                    self.equations.append(temp)

                    # 1a. Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+2)
                    
                    # 1b. Put equations into the residuals list
                    self._processEquations(state['Equations'], indent+2)

                    # 3. actions
                    for i, on_condition_action in enumerate(state['OnConditionActions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])
                        self._processActions(on_condition_action['Actions'], condition, indent+2)
                        
                temp = s_indent + 'END'
                self.equations.append(temp)

    def _processActions(self, Actions, condition, indent):
        s_indent = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                stateTo = action['StateTo']
                self.equations.append(s_indent + 'SWITCH TO {0} IF {1};'.format(stateTo, condition))

            elif action['Type'] == 'eSendEvent':
                self.warnings.append('gPROMS code cannot be generated for SendEvent actions on event port [%s]' % action['SendEventPort'])

            elif action['Type'] == 'eReAssignOrReInitializeVariable':
                self.warnings.append('gPROMS code cannot be generated for actions that re-assign/initialize variable [%s]' % action['VariableCanonicalName'])
                
            elif action['Type'] == 'eUserDefinedAction':
                self.warnings.append('gPROMS code cannot be generated for UserDefined actions')

            else:
                raise RuntimeError('Unknown action type')

    def _generateRuntimeInformation(self, runtimeInformation):
        Ntotal             = runtimeInformation['TotalNumberOfVariables']
        Neq                = runtimeInformation['NumberOfEquations']
        IDs                = runtimeInformation['IDs']
        initValues         = runtimeInformation['Values']
        initDerivatives    = runtimeInformation['TimeDerivatives']
        indexMappings      = runtimeInformation['IndexMappings']
        absoluteTolerances = runtimeInformation['AbsoluteTolerances']

        blockIDs             = Neq * [-1]
        blockInitValues      = Neq * [-1]
        absTolerances        = Neq * [1E-5]
        blockInitDerivatives = Neq * [0.0]
        for oi, bi in list(indexMappings.items()):
            if IDs[oi] == cnAlgebraic:
               blockIDs[bi] = cnAlgebraic
            elif IDs[oi] == cnDifferential:
               blockIDs[bi] = cnDifferential

            if IDs[oi] != cnAssigned:
                blockInitValues[bi]      = initValues[oi]
                blockInitDerivatives[bi] = initDerivatives[oi]
                absTolerances[bi]        = absoluteTolerances[oi]

        rootModel = self.exprFormatter.formatIdentifier(self.rootModelName)
        
        for domain in runtimeInformation['Domains']:
            relativeName   = daeGetRelativeName(self.rootModelName, domain['CanonicalName'])
            formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(formattedName)
            description    = domain['Description']
            numberOfPoints = domain['NumberOfPoints']
            units          = self.exprFormatter.formatUnits(domain['Units'])
            domains        = '[' + str(domain['NumberOfPoints']) + ']'
            points         = self.exprFormatter.formatNumpyArray(domain['Points']) # Numpy array

            domTemplate   = '{name}_np as integer'
            self.domains.append(domTemplate.format(name = name))
            domTemplate   = '{name} as array({name}_np) of real'
            self.domains.append(domTemplate.format(name = name))

            domTemplate   = '{rootModel}.{name}_np := {numberOfPoints};'
            self.domains_inits.append(domTemplate.format(name = name,
                                                         rootModel = rootModel,
                                                         numberOfPoints = numberOfPoints))
            domTemplate   = '{rootModel}.{name} := {points}; # {units}'
            self.domains_inits.append(domTemplate.format(name = name,
                                                         rootModel = rootModel,
                                                         units = units,
                                                         points = points))
            
        for parameter in runtimeInformation['Parameters']:
            relativeName    = daeGetRelativeName(self.rootModelName, parameter['CanonicalName'])
            formattedName   = self.exprFormatter.formatIdentifier(relativeName)
            name            = self.exprFormatter.flattenIdentifier(formattedName)
            description     = parameter['Description']
            numberOfPoints  = int(parameter['NumberOfPoints'])
            units           = self.exprFormatter.formatUnits(parameter['Units'])
            values          = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array
            numberOfDomains = len(parameter['Domains'])
            
            paramTemplate = '# {description} [{units}]:'
            self.parameters.append(paramTemplate.format(units = units,
                                                        description = description))
            if numberOfDomains > 0:
                relativeDomains = [daeGetRelativeName(self.rootModelName, dn)     for dn in variable['DomainNames']]
                formatDomains   = [self.exprFormatter.formatIdentifier(dn)        for dn in relativeDomains]
                flatDomains     = [self.exprFormatter.flattenIdentifier(dn)+'_np' for dn in formatDomains]
                domains = '{0}'.format(', '.join(flatDomains))

                paramTemplate = '{name} as array({domains}) of real'
                self.parameters.append(paramTemplate.format(name = name,
                                                            domains = domains))

                paramTemplate = '{rootModel}.{name} := [{values}]; # {units}'
                self.parameters_inits.append(paramTemplate.format(name = name,
                                                                  units = units,
                                                                  rootModel = rootModel,
                                                                  values = values))
            else:
                paramTemplate = '{name} as real'
                self.parameters.append(paramTemplate.format(name = name))

                paramTemplate = '{rootModel}.{name} := {value}; # {units}'
                self.parameters_inits.append(paramTemplate.format(name = name,
                                                                  units = units,
                                                                  rootModel = rootModel,
                                                                  value = values))

        variableTypesUsed = {}
        for variable in runtimeInformation['Variables']:
            relativeName   = daeGetRelativeName(self.rootModelName, variable['CanonicalName'])
            formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(formattedName)
            description    = variable['Description']
            numberOfPoints = variable['NumberOfPoints']
            units          = self.exprFormatter.formatUnits(variable['Units'])
            varTypeName    = variable['VariableType'].Name
            varType        = variable['VariableType']
            numberOfDomains = len(variable['Domains'])

            vartypeTemplate = '  {name} = {default} : {min} : {max} UNIT = "{units}"'
            if not varTypeName in variableTypesUsed:
                variableTypesUsed[varTypeName] = None
                self.variable_types.append('DECLARE TYPE')
                self.variable_types.append(vartypeTemplate.format(name = varTypeName,
                                                                  min = varType.LowerBound,
                                                                  max = varType.UpperBound,
                                                                  default = varType.InitialGuess,
                                                                  units = units,
                                                                  absTol = varType.AbsoluteTolerance))
                self.variable_types.append('END')
                
            varTemplate = '#  {description} [{units}]:'
            self.variables.append(varTemplate.format(units = units,
                                                     description = description))
            
            if numberOfDomains > 0:
                relativeDomains = [daeGetRelativeName(self.rootModelName, dn)     for dn in variable['DomainNames']]
                formatDomains   = [self.exprFormatter.formatIdentifier(dn)        for dn in relativeDomains]
                flatDomains     = [self.exprFormatter.flattenIdentifier(dn)+'_np' for dn in formatDomains]
                domains = '{0}'.format(', '.join(flatDomains))

                varTemplate = '{name} as array({domains}) of {type}'
                self.variables.append(varTemplate.format(name = name,
                                                         domains = domains,
                                                         type = varTypeName,
                                                         units = units,
                                                         relativeName = relativeName,
                                                         description = description))

                domainsIndexesMap = variable['DomainsIndexesMap']
                for i in range(0, numberOfPoints):
                    domIndexes   = tuple(domainsIndexesMap[i])              # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    #fullName    = name + '(' + ','.join(str(di + self.exprFormatter.indexBase) for di in domIndexes) + ')'
                    fullName     = self.exprFormatter.formatVariable(variable['CanonicalName'], domIndexes, overallIndex)

                    if ID == cnDifferential:
                        temp = '{rootModel}.{name} = {value}; # {units}'.format(name = fullName,
                                                                                 value = value,
                                                                                 rootModel = rootModel,
                                                                                 units = units)
                        self.initial_conditions.append(temp)

                    elif ID == cnAssigned:
                        temp = '{rootModel}.{name} := {value}; # {units}'.format(name = fullName,
                                                                                 value = value,
                                                                                 rootModel = rootModel,
                                                                                 units = units)
                        self.assigned_inits.append(temp)

            else:
                ID    = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value = float(variable['Values'])   # numpy float

                varTemplate = '{name} as {type}'
                self.variables.append(varTemplate.format(name = name,
                                                         type = varTypeName,
                                                         units = units,
                                                         relativeName = relativeName,
                                                         description = description))

                if ID == cnDifferential:
                    varTemplate = '{rootModel}.{name} = {value}; # {units}'
                    self.initial_conditions.append(varTemplate.format(name = name,
                                                                      units = units,
                                                                      rootModel = rootModel,
                                                                      value = value))
                elif ID == cnAssigned:
                    varTemplate = '{rootModel}.{name} := {value}; # {units}'
                    self.assigned_inits.append(varTemplate.format(name = name,
                                                                  units = units,
                                                                  rootModel = rootModel,
                                                                  value = value))

        indent = 1
        s_indent = indent * self.defaultIndent

        # First, generate equations for port connections
        for port_connection in runtimeInformation['PortConnections']:
            self.equations.append(s_indent + '# Port connection: {0} -> {1}'.format(port_connection['PortFrom'], port_connection['PortTo']))
            self._processEquations(port_connection['Equations'], indent)

        # Then, generate ordinary equations
        self._processEquations(runtimeInformation['Equations'], indent)

        # Finally, generate equations for IFs/STNs
        self._processSTNs(runtimeInformation['STNs'], indent)

        # gPROMS has no current TIME reserved word... add variable __TIME__
        equations_s = ''.join(self.equations)
        if equations_s.find(self.exprFormatter.TIME):
            self.variable_types.append('DECLARE TYPE')
            self.variable_types.append('  gproms_time_t = 0 : 0 : +1e30 UNIT = "s"')
            self.variable_types.append('END')
            self.variables.append('%s as gproms_time_t' % self.exprFormatter.TIME)
            self.equations.append('$%s = 1;' % self.exprFormatter.TIME)
            self.initial_conditions.append('{rootModel}.{time} = 0; # s'.format(rootModel = rootModel,
                                                                                time = self.exprFormatter.TIME))
