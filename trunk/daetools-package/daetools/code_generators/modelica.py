import os, shutil, sys, numpy, math, traceback
from daetools.pyDAE import *
from .formatter import daeExpressionFormatter
from .analyzer import daeCodeGeneratorAnalyzer
from .code_generator import daeCodeGenerator

modelTemplate = """\
class %(model_name)s

/* Description:
%(doc_string)s
%(warnings)s
*/

/* Import libs */
  import Modelica.Math.*;
  
  %(parameters_defs)s
  
  %(variables_defs)s

equation
  %(equations_defs)s

  /* DOFs */
  %(assigned_inits)s

  /* OnConditionActions from STNs */
  %(whens)s

initial equation
  %(initial_conditions)s

  annotation(experiment(StartTime = %(start_time)s, StopTime = %(end_time)s, Tolerance = %(tolerance)s));

end %(model_name)s;
"""

class daeExpressionFormatter_Modelica(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)

        # Index base in arrays
        self.indexBase = 1

        self.useFlattenedNamesForAssignedVariables = False
        self.IDs      = {}
        self.indexMap = {}

        # Use relative names and flatten identifiers
        self.useRelativeNames   = True
        self.flattenIdentifiers = True

        self.domain                   = '{domain}[{index}]'

        self.parameter                = '{parameter}{indexes}'
        self.parameterIndexStart      = '['
        self.parameterIndexEnd        = ']'
        self.parameterIndexDelimiter  = ','

        self.variable                 = '{variable}{indexes}'
        self.variableIndexStart       = '['
        self.variableIndexEnd         = ']'
        self.variableIndexDelimiter   = ','

        self.assignedVariable         = '{variable}'

        self.feMatrixItem             = '{value}'
        self.feVectorItem             = '{value}'

        # String format for the time derivative, ie. der(variable[1,2]) in Modelica
        # daetools use: variable.dt(1,2), gPROMS $variable(1,2) ...
        self.derivative               = 'der({variable}{indexes})'
        self.derivativeIndexStart     = '['
        self.derivativeIndexEnd       = ']'
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
        self.NEQ   = '{leftValue} <> {rightValue}'
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
        self.SIN    = 'sin({value})'
        self.COS    = 'cos({value})'
        self.TAN    = 'tan({value})'
        self.ASIN   = 'asin({value})'
        self.ACOS   = 'acos({value})'
        self.ATAN   = 'atan({value})'
        self.EXP    = 'exp({value})'
        self.SQRT   = 'sqrt({value})'
        self.LOG    = 'log({value})'
        self.LOG10  = 'log10({value})'
        self.FLOOR  = 'floor({value})'
        self.CEIL   = 'ceil({value})'
        self.ABS    = 'abs({value})'

        self.MIN    = 'min({leftValue}, {rightValue})'
        self.MAX    = 'max({leftValue}, {rightValue})'

        # Current time in simulation
        self.TIME   = 'time'

    def formatNumpyArray(self, arr):
        if isinstance(arr, (numpy.ndarray, list)):
            return '{' + ', '.join([self.formatNumpyArray(val) for val in arr]) + '}'
        else:
            return str(arr)

    def formatQuantity(self, quantity):
        # Formats constants/quantities in equations that have a value and units
        return str(quantity.value)

    def formatUnits(self, units):
        # Format: m.kg2/s-2 meaning m * kg**2 / s**2
        positive = []
        negative = []
        for u, exp in list(units.toDict().items()):
            if exp >= 0:
                if exp == 1:
                    positive.append('{0}'.format(u))
                elif int(exp) == exp:
                    positive.append('{0}{1}'.format(u, int(exp)))
                else:
                    positive.append('{0}{1}'.format(u, exp))

        for u, exp in list(units.toDict().items()):
            if exp < 0:
                if exp == -1:
                    negative.append('{0}'.format(u))
                elif int(exp) == exp:
                    negative.append('{0}{1}'.format(u, int(math.fabs(exp))))
                else:
                    negative.append('{0}{1}'.format(u, math.fabs(exp)))

        if len(positive) == 0:
            sPositive = 'rad'
        else:
            sPositive = '.'.join(positive)

        if len(negative) == 0:
            sNegative = ''
        elif len(negative) == 1:
            sNegative = '/' + '.'.join(negative)
        else:
            sNegative = '/(' + '.'.join(negative) + ')'

        return sPositive + sNegative

class daeCodeGenerator_Modelica(daeCodeGenerator):
    def __init__(self):
        self.exprFormatter = daeExpressionFormatter_Modelica()
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
        self.equations      = []
        self.whens          = []

        self.domains_inits      = []
        self.parameters_inits   = []
        self.assigned_inits     = []
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
        doc_string    = sys.modules[simulation.__module__].__doc__

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)
        self.exprFormatter.IDs      = self.analyzer.runtimeInformation['IDs']
        self.exprFormatter.indexMap = self.analyzer.runtimeInformation['IndexMappings']

        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

        separator = '\n' + self.defaultIndent
        warnings = ''
        if len(self.warnings) > 0:
            warnings  = 'daetools-Modelica code-generator warnings:\n'
            warnings += '\n'.join(self.warnings)

        # Model
        variable_types_defs = separator.join(self.variable_types)
        domains_defs        = separator.join(self.domains)
        parameters_defs     = separator.join(self.parameters)
        variables_defs      = separator.join(self.variables)
        equations_defs      = separator.join(self.equations)

        # Simulation
        domains_inits      = separator.join(self.domains_inits)
        parameters_inits   = separator.join(self.parameters_inits)
        assigned_inits     = separator.join(self.assigned_inits)
        whens              = separator.join(self.whens)
        initial_conditions = separator.join(self.initial_conditions)
        reporting_interval = self.simulation.ReportingInterval
        time_horizon       = self.simulation.TimeHorizon
        relative_tolerance = self.simulation.DAESolver.RelativeTolerance

        dictInfo = {
                     'instance_name' :      instance_name,
                     'model_name' :         model_name,
                     'doc_string' :         doc_string,
                     
                     'variable_types_defs': variable_types_defs,
                     'domains_defs' :       domains_defs,
                     'parameters_defs' :    parameters_defs,
                     'variables_defs' :     variables_defs,
                     'equations_defs' :     equations_defs,
                        
                     'domains_inits' :      domains_inits,
                     'parameters_inits' :   parameters_inits,
                     'assigned_inits' :     assigned_inits,
                     'whens' :              whens,
                     'initial_conditions' : initial_conditions,
                     'start_time' :         0.0,
                     'end_time' :           time_horizon,
                     'tolerance' :          relative_tolerance,

                     'warnings' :           warnings
                   }

        model_contents = modelTemplate % dictInfo;

        f = open(os.path.join(directory, '%s.mo' % model_name), "w")
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
                self.equations.append('/* {0}: */'.format(description))
            
            for eeinfo in equation['EquationExecutionInfos']:
                res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                self.equations.append('{0} = 0;'.format(res))

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

                        temp = s_indent + 'if ({0}) then'.format(condition)
                        self.equations.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        # There is only one OnConditionAction in ELSE_IFs
                        on_condition_action = state['OnConditionActions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                        temp = s_indent + 'elseif ({0}) then'.format(condition)
                        self.equations.append(temp)

                    else:
                        temp = s_indent + 'else '
                        self.equations.append(temp)

                    # 1a. Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+1)
                    
                    # 1b. Put equations into the residuals list
                    self._processEquations(state['Equations'], indent+1)

                end = s_indent + 'end if;'
                self.equations.append(end)

            elif stn['Class'] == 'daeSTN':
                relativeName    = daeGetRelativeName(self.rootModelName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName)
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                stnTemplate = 'String {name} "{description}"; /* States: [{states}] */'
                self.variables.append(stnTemplate.format(name = stnVariableName,
                                                         description = description,
                                                         activeState = activeState,
                                                         states = states))
                stnTemplate = '{name} = "{activeState}";'
                self.initial_conditions.append(stnTemplate.format(name = stnVariableName,
                                                                  activeState = activeState))
                counter = 0
                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    # Not all states have state_transitions ('else' state has no state transitions)
                    on_condition_action = None
                    if i == 0:
                        temp = s_indent + 'if ({name} == "{state}") then'.format(name = stnVariableName,
                                                                                 state = state["Name"])
                        self.equations.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        temp = s_indent + 'elseif ({name} == "{state}") then'.format(name = stnVariableName,
                                                                                     state = state["Name"])
                        self.equations.append(temp)

                    else:
                        temp = s_indent + 'else '
                        self.equations.append(temp)

                    if len(state['OnConditionActions']) > 0:
                        temp = s_indent + '/* OnConditionActions for {stn}.{state} */'.format(stn = stnVariableName,
                                                                                              state = state['Name'])
                        self.whens.append(temp)

                        for on_condition_action in state['OnConditionActions']:
                            condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                            if counter == 0:
                                temp = s_indent + 'when ({0} == "{1}") and ({2}) then'.format(stnVariableName, state['Name'], condition)
                            else:
                                temp = s_indent + 'elsewhen ({0} == "{1}") and ({2}) then'.format(stnVariableName, state['Name'], condition)
                            self.whens.append(temp)

                            # Generate on condition actions
                            self._processActions(on_condition_action['Actions'], indent+1)
                            counter += 1

                    # Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+1)

                    # Generate equations
                    self._processEquations(state['Equations'], indent+1)

                if counter > 0:
                    end = s_indent + 'end when;'
                    self.whens.append(end)

                end = s_indent + 'end if;'
                self.equations.append(end)

    def _processActions(self, Actions, indent):
        s_indent  = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                stnVariableName = action['STN']
                stateTo         = action['StateTo']
                self.whens.append(s_indent + '{0} = "{1}";'.format(stnVariableName, stateTo))

            elif action['Type'] == 'eSendEvent':
                self.warnings.append('Modelica code cannot be generated for SendEvent actions - the model will not work as expected!!')

            elif action['Type'] == 'eReAssignOrReInitializeVariable':
                relativeName = daeGetRelativeName(self.wrapperInstanceName, action['VariableWrapper'].Variable.CanonicalName)
                relativeName = self.exprFormatter.formatIdentifier(relativeName)
                domainIndexes = action['VariableWrapper'].DomainIndexes
                node          = action['RuntimeNode']
                strDomainIndexes = ''
                if len(domainIndexes) > 0:
                    strDomainIndexes = '[' + ','.join() + ']'
                variableName = relativeName + strDomainIndexes
                value = self.exprFormatter.formatRuntimeNode(node)
                self.whens.append(s_indent + 'reinit({0}, {1});'.format(variableName, value))

            elif action['Type'] == 'eUserDefinedAction':
                self.warnings.append('Modelica code cannot be generated for UserDefined actions - the model will not work as expected!!')

            else:
                pass

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

            domTemplate   = 'parameter Integer {name}_np = {numberOfPoints} "Number of points in domain {name}";'
            self.parameters.append(domTemplate.format(name = name,
                                                      description = description,
                                                      numberOfPoints = numberOfPoints))
                                                      
            domTemplate   = 'parameter Real {name}[{name}_np](each unit = "{units}") = {points} "{description}";'
            self.parameters.append(domTemplate.format(name = name,
                                                      description = description,
                                                      units = units,
                                                      points = points,
                                                      numberOfPoints = numberOfPoints))
            
        for parameter in runtimeInformation['Parameters']:
            relativeName    = daeGetRelativeName(self.rootModelName, parameter['CanonicalName'])
            formattedName   = self.exprFormatter.formatIdentifier(relativeName)
            name            = self.exprFormatter.flattenIdentifier(formattedName)
            description     = parameter['Description']
            numberOfPoints  = int(parameter['NumberOfPoints'])
            units           = self.exprFormatter.formatUnits(parameter['Units'])
            values          = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array
            numberOfDomains = len(parameter['Domains'])
            
            if numberOfDomains > 0:
                relativeDomains = [daeGetRelativeName(self.rootModelName, dn)   for dn in parameter['DomainNames']]
                formatDomains = [self.exprFormatter.formatIdentifier(dn)        for dn in relativeDomains]
                flatDomains   = [self.exprFormatter.flattenIdentifier(dn)+'_np' for dn in formatDomains]
                domains       = '{0}'.format(', '.join(flatDomains))
                attributes    = '(each unit = "%s")' % units

                paramTemplate = 'parameter Real {name}[{domains}]{attributes} = {values} "{description}";'
                self.parameters.append(paramTemplate.format(name = name,
                                                            attributes = attributes,
                                                            description = description,
                                                            values = values,
                                                            domains = domains))

            else:
                attributes    = '(unit = "%s")' % units
                paramTemplate = 'parameter Real {name}{attributes} = {value} "{description}";'
                self.parameters.append(paramTemplate.format(name = name,
                                                            attributes = attributes,
                                                            value = values,
                                                            description = description))

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

            vartypeTemplate = '{name}({min}, {max}, {default}, {absTol});'
            if not varTypeName in variableTypesUsed:
                variableTypesUsed[varTypeName] = None
                self.variable_types.append(vartypeTemplate.format(name = varTypeName,
                                                                  min = varType.LowerBound,
                                                                  max = varType.UpperBound,
                                                                  default = varType.InitialGuess,
                                                                  absTol = varType.AbsoluteTolerance))
                
            if numberOfDomains > 0:
                relativeDomains = [daeGetRelativeName(self.rootModelName, dn)     for dn in variable['DomainNames']]
                formatDomains   = [self.exprFormatter.formatIdentifier(dn)        for dn in relativeDomains]
                flatDomains     = [self.exprFormatter.flattenIdentifier(dn)+'_np' for dn in formatDomains]
                domains         = '{0}'.format(', '.join(flatDomains))
                attributes      = '(each unit = "%s")' % units

                varTemplate = 'Real {name}[{domains}]{attributes} "{description}";'
                self.variables.append(varTemplate.format(name = name,
                                                         attributes = attributes,
                                                         description = description,
                                                         domains = domains))

                domainsIndexesMap = variable['DomainsIndexesMap']
                for i in range(0, numberOfPoints):
                    domIndexes   = tuple(domainsIndexesMap[i])              # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    #fullName     = name + '[' + ','.join(str(di+self.exprFormatter.indexBase) for di in domIndexes) + ']'
                    fullName     = self.exprFormatter.formatVariable(variable['CanonicalName'], domIndexes, overallIndex)

                    if ID == cnDifferential:
                        temp = '{name} = {value} /* {units} */;'.format(name = fullName,
                                                                        value = value,
                                                                        units = units)
                        self.initial_conditions.append(temp)

                    elif ID == cnAssigned:
                        temp = '{name} = {value} /* {units} */;'.format(name = fullName,
                                                                        value = value,
                                                                        units = units)
                        self.assigned_inits.append(temp)
                                                         
            else:
                ID    = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value = float(variable['Values'])   # numpy float

                if ID == cnDifferential:
                    attributes = '(start = 0.0, unit = "%s")' % units
                    varTemplate = 'Real {name}{attributes} "{description}";'

                    initTemplate = '{name} = {value} /* {units} */;'
                    self.initial_conditions.append(initTemplate.format(name = name,
                                                                       units = units,
                                                                       value = value))
                elif ID == cnAssigned:
                    attributes = '(unit = "%s")' % units
                    varTemplate = 'Real {name}{attributes} "{description}";'
                    
                    temp = '{name} = {value} /* {units} */;'.format(name = name,
                                                                    value = value,
                                                                    units = units)
                    self.assigned_inits.append(temp)

                else:
                    attributes = '(unit = "%s")' % units
                    varTemplate = 'Real {name}{attributes} "{description}";'

                self.variables.append(varTemplate.format(name = name,
                                                         attributes = attributes,
                                                         value = value,
                                                         rootModel = rootModel,
                                                         description = description))

        indent = 1
        s_indent = indent * self.defaultIndent

        # First, generate equations for port connections
        for port_connection in runtimeInformation['PortConnections']:
            self.equations.append(s_indent + '/* Port connection: {0} -> {1} */'.format(port_connection['PortFrom'], port_connection['PortTo']))
            self._processEquations(port_connection['Equations'], indent)

        # Then, generate ordinary equations
        self._processEquations(runtimeInformation['Equations'], indent)

        # Finally, generate equations for IFs/STNs
        self._processSTNs(runtimeInformation['STNs'], indent)

   
