import sys, numpy, math, traceback
from daetools.pyDAE import *
from formatter import daeExpressionFormatter
from analyzer import daeCodeGeneratorAnalyzer

mainTemplate = """
#include <string.h>
#include <stdio.h>

#define real_t   double
#define _v_(i)   adouble(_model_.m_valueProxies[i]->getValue(_values_),        (i == _current_index_for_jacobian_evaluation_) ? 1.0 : 0.0)
#define _dt_(i)  adouble(_model_.m_valueProxies[i]->getdt(_time_derivatives_), (i == _current_index_for_jacobian_evaluation_) ? _inverse_time_step_ : 0.0)
#define _time_   adouble(_current_time_, 0.0)

%(model)s
%(parameters)s
%(activeStates)s
%(assignedVariables)s
%(initialConditions)s

void residuals(real_t _current_time_,
               real_t* _values_,
               real_t* _time_derivatives_,
               real_t* _residuals_,
               cModel* _pmodel_)
{
    int _ec_ = 0;
    
%(residuals)s
}

void jacobian(long int _number_of_equations_,
              real_t _current_time_,
              real_t _inverse_time_step_,
              real_t* _values_,
              real_t* _time_derivatives_,
              real_t* _residuals_,
              daeMatrix<real_t>& _jacobian_,
              cModel* _pmodel_)
{
    int _current_index_for_jacobian_evaluation_ = 0;
    
%(jacobian)s
}

int number_of_roots()
{
    int _noRoots_ = 0;
    
%(numberOfRoots)s

    return _noRoots_;
}

void roots(real_t _current_time_,
           real_t* _values_,
           real_t* _time_derivatives_,
           real_t* _roots_,
           cModel* _pmodel_)
{
    int _rc_ = 0;
    
%(roots)s
}

bool check_for_discontinuities(real_t _current_time_,
                               real_t* _values_,
                               real_t* _time_derivatives_,
                               cModel* _pmodel_)
{
    bool reinitializationNeeded = false;
    
%(checkForDiscontinuities)s

    return reinitializationNeeded;
}


void execute_actions(real_t _current_time_,
                     real_t* _values_,
                     real_t* _time_derivatives_,
                     cModel* _pmodel_)
{
%(executeActions)s
}

"""


class daeCXXExpressionFormatter(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)

        # Index base in arrays
        self.indexBase = 0

        self.useFlattenedNamesForAssignedVariables = True
        self.IDs = {} # will be set later after analyzeSimulation()

        # Use relative names
        self.useRelativeNames   = True
        self.flattenIdentifiers = True

        self.domain                   = '{domain}[{index}]'

        self.parameter                = '{parameter}{indexes}'
        self.parameterIndexStart      = '['
        self.parameterIndexEnd        = ']'
        self.parameterIndexDelimiter  = ']['

        self.variable                 = '_v_({overallIndex})'
        self.variableIndexStart       = ''
        self.variableIndexEnd         = ''
        self.variableIndexDelimiter   = ''

        self.derivative               = '_dt_({overallIndex})'
        self.derivativeIndexStart     = ''
        self.derivativeIndexEnd       = ''
        self.derivativeIndexDelimiter = ''

        # Logical operators
        self.AND   = '&&'
        self.OR    = '||'
        self.NOT   = '!'

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
        self.POWER  = '???'

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
        self.TIME   = '_time_'

    def formatNumpyArray(self, arr):
        if isinstance(arr, (numpy.ndarray, list)):
            return '{' + ', '.join([self.formatNumpyArray(val) for val in arr]) + '}'
        else:
            return str(arr)

    def formatQuantity(self, quantity):
        # Formats constants/quantities in equations that have a value and units
        return str(quantity.value)
     
class daeCXXExport(object):
    def __init__(self, simulation = None):
        self.wrapperInstanceName     = ''
        self.defaultIndent           = '    '
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None

        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.initiallyActiveStates           = []
        self.residuals               = []
        self.checkForDiscontinuities = []
        self.executeActions          = []
        self.numberOfRoots           = []
        self.rootFunctions           = []
        
        self.exprFormatter = daeCXXExpressionFormatter()
        self.analyzer      = daeCodeGeneratorAnalyzer()

    def exportSimulation(self, simulation, filename = None):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.initiallyActiveStates           = []
        self.residuals               = []
        self.warnings                = []
        self.simulation              = simulation
        self.topLevelModel           = simulation.m
        self.wrapperInstanceName     = simulation.m.Name
        self.exprFormatter.modelCanonicalName = simulation.m.Name

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)
        self.exprFormatter.IDs = self.analyzer.runtimeInformation['IDs']
        
        #print self.analyzer.models
        #print self.analyzer.ports

        #import pprint
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(self.analyzer.runtimeInformation)

        """
        resultModel = ''
        for port_class, info in self.analyzer.ports:
            resultModel += self._processPort(info['data'], indent)

        for model_class, info in self.analyzer.models:
            resultModel += self._processModel(info['data'], indent)
        """
        
        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

        modelDef       = '\n/* General info */\n' + '\n'.join(self.modelDef) + '\n'
        paramsDef      = '\n/* Domains and parameters */\n' + '\n'.join(self.parametersDefs) + '\n'
        stnDef         = '\n/* STNs */\n' + '\n'.join(self.initiallyActiveStates) + '\n'
        assignedVars   = '\n/* Assigned variables */\n' + '\n'.join(self.assignedVariables) + '\n'
        initConds      = '\n/* Initial conditions */\n' + '\n'.join(self.initialConditions) + '\n'
        eqnsRes        = '\n'.join(self.residuals) + '\n'
        jacobRes       = ''
        rootsDef       = '\n'.join(self.rootFunctions)
        checkDiscont   = '\n'.join(self.checkForDiscontinuities)
        execActionsDef = '\n'.join(self.executeActions)
        noRootsDef     = '\n'.join(self.numberOfRoots)

        dictInfo = {
                        'model' : modelDef,
                        'parameters' : paramsDef,
                        'activeStates' : stnDef,
                        'assignedVariables' : assignedVars,
                        'initialConditions' : initConds,
                        'residuals' : eqnsRes,
                        'jacobian' : jacobRes,
                        'roots' : rootsDef,
                        'numberOfRoots' : noRootsDef,
                        'checkForDiscontinuities' : checkDiscont,
                        'executeActions' : execActionsDef
                   }

        results = mainTemplate % dictInfo;

        if filename:
            f = open(filename, "w")
            f.write(results)
            #f.write(modelDef)
            #f.write(paramsDef)
            #f.write(varsDef)
            #f.write(eqnsRes)
            #f.write(checkDiscont)
            #f.write(rootsDef)
            #f.write(execActionsDef)
            #f.write(noRootsDef)
            f.close()

        #return results

    def _processEquations(self, Equations, indent):
        s_indent  = indent * self.defaultIndent
        for equation in Equations:
            for eeinfo in equation['EquationExecutionInfos']:
                self.residuals.append(s_indent + '_residuals_[_ec_++] = ' + self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode']) + ';')
                    
    def _processSTNs(self, STNs, indent):
        s_indent = indent * self.defaultIndent
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
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                varTemplate = 'char* {name} = "{activeState}"; /* States: {states}; {description} */ \n'
                self.initiallyActiveStates.append(varTemplate.format(name = stnVariableName,
                                                             states = states,
                                                             activeState = activeState,
                                                             description = description))

                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    if i == 0:
                        temp = s_indent + '/* STN {0} */'.format(stnVariableName)
                        self.residuals.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        temp = s_indent + 'if(strcmp({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        temp = s_indent + 'else if(strcmp({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    else:
                        temp = s_indent + 'else {'
                        self.residuals.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    # 1. Put equations into the residuals list
                    self._processEquations(state['Equations'], indent+1)
                    self.residuals.append(s_indent + '}')

                    nStateTransitions = len(state['StateTransitions'])
                    s_indent2 = (indent + 1) * self.defaultIndent

                    # 2. checkForDiscontinuities
                    for i, state_transition in enumerate(state['StateTransitions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
                        if i == 0:
                            self.checkForDiscontinuities.append(s_indent2 + 'if({0}) {{ reinitializationNeeded = true; }}'.format(condition))

                        elif (i > 0) and (i < nStates - 1):
                            self.checkForDiscontinuities.append(s_indent2 + 'else if({0}) {{ reinitializationNeeded = true; }}'.format(condition))

                        else:
                            self.checkForDiscontinuities.append(s_indent2 + 'else { reinitializationNeeded = true; }')

                    self.checkForDiscontinuities.append(s_indent + '}')
                    
                    # 3. executeActions
                    for i, state_transition in enumerate(state['StateTransitions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
                        if i == 0:
                            self.executeActions.append(s_indent2 + 'if({0}) {{'.format(condition))

                        elif (i > 0) and (i < nStates - 1):
                            self.executeActions.append(s_indent2 + 'else if({0}) {{'.format(condition))

                        else:
                            self.executeActions.append(s_indent2 + 'else {')

                        self._processActions(state_transition['Actions'], indent+2)
                        self.executeActions.append(s_indent2 + '}')

                    self.executeActions.append(s_indent + '}')

                    # 4. numberOfRoots
                    for i, state_transition in enumerate(state['StateTransitions']):
                        nExpr = len(state_transition['Expressions'])
                        self.numberOfRoots.append(s_indent2 + '_noRoots_ += {0};'.format(nExpr))

                    self.numberOfRoots.append(s_indent + '}')

                    # 5. rootFunctions
                    for i, state_transition in enumerate(state['StateTransitions']):
                        for expression in state_transition['Expressions']:
                            self.rootFunctions.append(s_indent2 + '_roots_[_rc_++] = {0}'.format(self.exprFormatter.formatRuntimeNode(expression)))

                    self.rootFunctions.append(s_indent + '}')

                    if len(state['NestedSTNs']) > 0:
                        raise RuntimeError('Nested state transition networks (daeSTN) canot be exported to c')

    def _processActions(self, Actions, indent):
        s_indent = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, action['STNCanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                stateTo         = action['StateTo']
                self.executeActions.append(s_indent + '{0} = "{1}";'.format(stnVariableName, stateTo))

            elif action['Type'] == 'eSendEvent':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            elif action['Type'] == 'eReAssignOrReInitializeVariable':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            elif action['Type'] == 'eUserDefinedAction':
                raise RuntimeError('Unsupported action: {0}'.format(action['Type']))

            else:
                pass

    def _generateRuntimeInformation(self, runtimeInformation):
        indexMappings = runtimeInformation['IndexMappings']
        Ntotal = runtimeInformation['TotalNumberOfVariables']
        Neq    = runtimeInformation['NumberOfEquations']
        N      = len(indexMappings)

        Nvars = 'const int _Ntotal_vars_ = {0};'.format(Ntotal)
        self.modelDef.append(Nvars)

        Nvars = 'const int _Nvars_ = {0};'.format(N)
        self.modelDef.append(Nvars)

        Neqns = 'const int _Neqns_ = {0};'.format(Neq)
        self.modelDef.append(Neqns)

        IDs = 'int _IDs_[_Ntotal_vars_] = {0};'.format(self.exprFormatter.formatNumpyArray(runtimeInformation['IDs']))
        self.modelDef.append(IDs)

        # ACHTUNG, ACHTUNG!! IndexMappings does not contain assigned variables!!
        indexMapping = []
        for overallIndex in range(0, Ntotal):
            if overallIndex in indexMappings: # overallIndex is in the map
                indexMapping.append(indexMappings[overallIndex])
            else:
                indexMapping.append(overallIndex)

        indexMap = 'int _indexMap_[_Ntotal_vars_] = {0};'.format(self.exprFormatter.formatNumpyArray(indexMapping))
        self.modelDef.append(indexMap)

        initValues = 'real_t _initValues_[_Ntotal_vars_] = {0};'.format(self.exprFormatter.formatNumpyArray(runtimeInformation['InitialValues']))
        self.modelDef.append(initValues)

        #model = 'cModel _model_(_Ntotal_vars_, _Neqns_, _IDs_, _initValues_);'
        #self.modelDef.append(model)

        for domain in runtimeInformation['Domains']:
            relativeName = daeGetRelativeName(self.wrapperInstanceName, domain['CanonicalName'])
            relativeName   = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(relativeName)
            description    = domain['Description']
            numberOfPoints = domain['NumberOfPoints']
            domains        = '[' + str(domain['NumberOfPoints']) + ']'
            points         = self.exprFormatter.formatNumpyArray(domain['Points']) # Numpy array

            domTemplate   = 'const int {name}_np = {numberOfPoints}; /* Number of points in domain {name} */'
            paramTemplate = 'real_t {name}{domains} = {points}; /* {description} */ \n'

            self.parametersDefs.append(domTemplate.format(name = name,
                                                          numberOfPoints = numberOfPoints))
            self.parametersDefs.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            points = points,
                                                            description = description))
            
        for parameter in runtimeInformation['Parameters']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, parameter['CanonicalName'])
            relativeName   = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(relativeName)
            description    = parameter['Description']
            values         = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array
            domains = ''
            if len(parameter['Domains']) > 0:
                domains = '[{0}]'.format(']['.join(str(np) for np in parameter['Domains']))

            paramTemplate = 'real_t {name}{domains} = {values}; /* {description} */ \n'

            self.parametersDefs.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            values = values,
                                                            description = description))

        for variable in runtimeInformation['Variables']:
            relativeName  = daeGetRelativeName(self.wrapperInstanceName, variable['CanonicalName'])
            formattedName = self.exprFormatter.formatIdentifier(relativeName)
            name          = self.exprFormatter.flattenIdentifier(formattedName)

            n = variable['NumberOfPoints']
            if n == 1:
                ID           = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value        = float(variable['Values'])   # numpy float
                overallIndex = variable['OverallIndex']
                fullName = relativeName

                if ID == cnDifferential:
                    name_ = '_initValues_[{0}]'.format(overallIndex)
                    temp = '{name} = {value}; /* {fullName} */'.format(name = name_, value = value, fullName = fullName)
                    self.initialConditions.append(temp)

                elif ID == cnAssigned:
                    temp = 'real_t {name} = {value}; /* {fullName} */'.format(name = name, value = value, fullName = fullName)
                    self.assignedVariables.append(temp)

            else:
                for i in range(0, n):
                    domIndexes   = tuple(variable['DomainsIndexesMap'][i])  # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    fullName     = relativeName + '(' + ','.join(str(di) for di in domIndexes) + ')'

                    if ID == cnDifferential:
                        name_ = '_initValues_[{0}]'.format(overallIndex)
                        temp = '{name} = {value}; /* {fullName} */'.format(name = name_, value = value, fullName = fullName)
                        self.initialConditions.append(temp)

                    elif ID == cnAssigned:
                        temp = 'real_t {name} = {value}; /* {fullName} */'.format(name = name, value = value, fullName = fullName)
                        self.assignedVariables.append(temp)

        """
        for variable in runtimeInformation['Variables']:
            relativeName    = daeGetRelativeName(self.wrapperInstanceName, variable['CanonicalName'])
            relativeName    = self.exprFormatter.formatIdentifier(relativeName)
            name            = self.exprFormatter.flattenIdentifier(relativeName)
            numberOfDomains = len(variable['Domains'])
            domains         = '{' + ', '.join([str(d) for d in variable['Domains']]) + '}'
            overallIndex    = variable['OverallIndex']
            description     = variable['Description']

            if numberOfDomains > 0:
                self.variablesDefs.append('int {name}_domains[{numberOfDomains}] = {domains}'.format(name = name,
                                                                                                    numberOfDomains = numberOfDomains,
                                                                                                    domains = domains))
                varTemplate = 'cVariable {name}(&_model_, {overallIndex}, {name}_domains, {numberOfDomains}); // {description}\n'
            else:
                varTemplate = 'cVariable {name}(&_model_, {overallIndex}); // {description}\n'
                
            self.variablesDefs.append(varTemplate.format(name = name,
                                                         overallIndex = overallIndex,                                                         
                                                         numberOfDomains = numberOfDomains,
                                                         description = description))
        """
        
        indent = 1
        self._processEquations(runtimeInformation['Equations'], indent)
        self._processSTNs(runtimeInformation['STNs'], indent)