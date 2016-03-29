import os, shutil, sys, numpy, math, traceback
from daetools.pyDAE import *
from .formatter import daeExpressionFormatter
from .analyzer import daeCodeGeneratorAnalyzer
from .code_generator import daeCodeGenerator

"""
Compile with:
  gcc -O3 -Wall -std=c99 -pedantic -o daetools_simulation auxiliary.c adouble.c main.c -lsundials_idas -lsundials_nvecserial -lblas -llapack
"""

class daeExpressionFormatter_c99(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)
        self.indexBase                              = 0
        self.useFlattenedNamesForAssignedVariables  = True
        self.IDs                                    = {}
        self.indexMap                               = {}

        # Use relative names
        self.useRelativeNames         = True
        self.flattenIdentifiers       = True

        self.domain                   = '_adouble_(_m_->{domain}[{index}], 0)'

        self.parameter                = '_adouble_(_m_->{parameter}{indexes}, 0)'
        self.parameterIndexStart      = '['
        self.parameterIndexEnd        = ']'
        self.parameterIndexDelimiter  = ']['

        self.variable                 = '_v_({blockIndex})'
        self.variableIndexStart       = ''
        self.variableIndexEnd         = ''
        self.variableIndexDelimiter   = ''

        self.assignedVariable         = '_adouble_(_m_->{variable}, 0)'
        
        self.feMatrixItem             = '_adouble_({value}, 0)'
        self.feVectorItem             = '_adouble_({value}, 0)'

        self.derivative               = '_dv_({blockIndex})'
        self.derivativeIndexStart     = ''
        self.derivativeIndexEnd       = ''
        self.derivativeIndexDelimiter = ''

        # Constants
        self.constant = '_adouble_({value}, 0)'
        
        # External functions
        self.scalarExternalFunction = 'modCalculateScalarExtFunction("{name}", _m_, _current_time_, _values_, _time_derivatives_)'
        self.vectorExternalFunction = '_adouble_(0.0, 0.0)'

        # Logical operators
        self.AND   = '({leftValue} && {rightValue})'
        self.OR    = '({leftValue} || {rightValue})'
        self.NOT   = '(! {value})'

        self.EQ    = '_eq_({leftValue}, {rightValue})'
        self.NEQ   = '_neq_({leftValue}, {rightValue})'
        self.LT    = '_lt_({leftValue}, {rightValue})'
        self.LTEQ  = '_lteq_({leftValue}, {rightValue})'
        self.GT    = '_gt_({leftValue}, {rightValue})'
        self.GTEQ  = '_gteq_({leftValue}, {rightValue})'

        # Mathematical operators
        self.SIGN   = '_sign_({value})'

        self.PLUS   = '_plus_({leftValue}, {rightValue})'
        self.MINUS  = '_minus_({leftValue}, {rightValue})'
        self.MULTI  = '_multi_({leftValue}, {rightValue})'
        self.DIVIDE = '_divide_({leftValue}, {rightValue})'
        self.POWER  = '_pow_({leftValue}, {rightValue})'

        # Mathematical functions
        self.SIN    = '_sin_({value})'
        self.COS    = '_cos_({value})'
        self.TAN    = '_tan_({value})'
        self.ASIN   = '_asin_({value})'
        self.ACOS   = '_acos_({value})'
        self.ATAN   = '_atan_({value})'
        self.EXP    = '_exp_({value})'
        self.SQRT   = '_sqrt_({value})'
        self.LOG    = '_log_({value})'
        self.LOG10  = '_log10_({value})'
        self.FLOOR  = '_floor_({value})'
        self.CEIL   = '_ceil_({value})'
        self.ABS    = '_abs_({value})'
        self.SINH   = '_sinh_({value})'
        self.COSH   = '_cosh_({value})'
        self.TANH   = '_tanh_({value})'
        self.ASINH  = '_asinh_({value})'
        self.ACOSH  = '_acosh_({value})'
        self.ATANH  = '_atanh_({value})'
        self.ERF    = '_erf_({value})'

        self.MIN     = '_min_({leftValue}, {rightValue})'
        self.MAX     = '_max_({leftValue}, {rightValue})'
        self.ARCTAN2 = '_atan2_({leftValue}, {rightValue})'

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
     
class daeCodeGenerator_c99(daeCodeGenerator):
    def __init__(self):
        self.wrapperInstanceName     = ''
        self.defaultIndent           = '    '
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None
        self.equationGenerationMode  = ''
        
        self.assignedVariablesDefs   = []

        self.assignedVariablesInits  = []
        self.initialConditions       = []
        self.stnDefs                 = []
        self.initiallyActiveStates   = []
        self.runtimeInformation_h    = []
        self.runtimeInformation_c    = []
        self.runtimeInformation_init = []
        self.parametersDefs          = []
        self.parametersInits         = []
        self.intValuesReferences     = []
        self.floatValuesReferences   = []
        self.stringValuesReferences  = []
        self.residuals               = []
        self.jacobians               = []
        self.checkForDiscontinuities = []
        self.executeActions          = []
        self.numberOfRoots           = []
        self.rootFunctions           = []
        self.variableNames           = []

        self.fmiInterface            = []

        self.exprFormatter = daeExpressionFormatter_c99()
        self.analyzer      = daeCodeGeneratorAnalyzer()

    def generateSimulation(self, simulation, directory):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        if not os.path.isdir(directory):
            os.makedirs(directory)

        self.assignedVariablesDefs   = []
        self.assignedVariablesInits  = []
        self.initialConditions       = []
        self.stnDefs                 = []
        self.initiallyActiveStates   = []
        self.runtimeInformation_h    = []
        self.runtimeInformation_c    = []
        self.runtimeInformation_init = []
        self.parametersDefs          = []
        self.parametersInits         = []
        self.intValuesReferences     = []
        self.floatValuesReferences   = []
        self.stringValuesReferences  = []
        self.residuals               = []
        self.jacobians               = []
        self.checkForDiscontinuities = []
        self.executeActions          = []
        self.numberOfRoots           = []
        self.rootFunctions           = []
        self.variableNames           = []
        self.warnings                = []
        self.simulation              = simulation
        self.topLevelModel           = simulation.m
        
        # Achtung, Achtung!!
        # wrapperInstanceName and exprFormatter.modelCanonicalName should not be stripped 
        # of illegal characters, since they are used to get relative names
        self.wrapperInstanceName              = simulation.m.Name
        self.exprFormatter.modelCanonicalName = simulation.m.Name

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)
        self.exprFormatter.IDs      = self.analyzer.runtimeInformation['IDs']
        self.exprFormatter.indexMap = self.analyzer.runtimeInformation['IndexMappings']

        #import pprint
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(self.analyzer.runtimeInformation)

        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

        rtInformation_h   = '\n    '.join(self.runtimeInformation_h)
        rtInformation_c   = '\n'.join(self.runtimeInformation_c)
        rtInformation_init= '\n    '.join(self.runtimeInformation_init)
        paramsDef         = '\n    '.join(self.parametersDefs)
        paramsInits       = '\n    '.join(self.parametersInits)
        stnDef            = '\n    '.join(self.stnDefs)
        stnActiveStates   = '\n    '.join(self.initiallyActiveStates)
        assignedVarsDefs  = '\n    '.join(self.assignedVariablesDefs)
        assignedVarsInits = '\n    '.join(self.assignedVariablesInits)
        initConds         = '\n    '.join(self.initialConditions)
        eqnsRes           = '\n'.join(self.residuals)
        jacobRes          = '\n'.join(self.jacobians)
        rootsDef          = '\n'.join(self.rootFunctions)
        checkDiscont      = '\n'.join(self.checkForDiscontinuities)
        execActionsDef    = '\n'.join(self.executeActions)
        noRootsDef        = '\n'.join(self.numberOfRoots)
        warnings          = '\n'.join(self.warnings)

        # intValuesReferences
        if len(self.intValuesReferences) > 0:
            intValuesReferences_Def = 'int* intValuesReferences[{0}];'.format(len(self.intValuesReferences))
        else:
            intValuesReferences_Def = 'int* intValuesReferences[1]; /* Array is empty but we need its declaration */'

        valRefInit = []
        for i, (ref_type, ref_name, ref_flat_name, block_index) in enumerate(self.intValuesReferences):
            if ref_type == 'NumberOfPointsInDomain':
                init = '_m_->intValuesReferences[{0}] = &_m_->{1};'.format(i, ref_flat_name)
            else:
                raise RuntimeError('Invalid integer variable reference type')
            valRefInit.append(init)            
        intValuesReferences_Init = '\n    '.join(valRefInit)
            
        # floatValuesReferences
        if len(self.floatValuesReferences) > 0:
            floatValuesReferences_Def = 'real_t* floatValuesReferences[{0}];'.format(len(self.floatValuesReferences))
        else:
            floatValuesReferences_Def = 'real_t* floatValuesReferences[1]; /* Array is empty but we need its declaration */'

        # FMI: initialize value references tuple: (ref_type, name)
        valRefInit = []
        for i, (ref_type, ref_name, ref_flat_name, block_index) in enumerate(self.floatValuesReferences):
            if ref_type == 'Assigned':
                init = '_m_->floatValuesReferences[{0}] = &_m_->{1};'.format(i, ref_flat_name)

            elif ref_type == 'Algebraic' or ref_type == 'Differential':
                init = '_m_->floatValuesReferences[{0}] = &_m_->values[{1}];'.format(i, block_index)

            elif ref_type == 'Differential':
                init = '_m_->floatValuesReferences[{0}] = &_m_->values[{1}];'.format(i, block_index)

            elif ref_type == 'Parameter':
                init = '_m_->floatValuesReferences[{0}] = &_m_->{1};'.format(i, ref_flat_name)

            #elif ref_type == 'NumberOfPointsInDomain':
            #    init = '_m_->floatValuesReferences[{0}] = &_m_->{1};'.format(i, ref_flat_name)

            elif ref_type == 'DomainPoints':
                init = '_m_->floatValuesReferences[{0}] = &_m_->{1};'.format(i, ref_flat_name)

            else:
                raise RuntimeError('Invalid variable reference type (%s): %s' % (ref_type, ref_name))
            
            valRefInit.append(init)            
        floatValuesReferences_Init = '\n    '.join(valRefInit)
        
        if len(self.stringValuesReferences) > 0:
            stringValuesReferences_Def = 'char* stringValuesReferences[{0}];'.format(len(self.stringValuesReferences))
        else:
            stringValuesReferences_Def = 'char* stringValuesReferences[1]; /* Array is empty but we need its declaration */'
        
        valRefInit = []
        stringValuesReferences_Init = '\n    '.join(valRefInit)
            
        dictInfo = {    'runtimeInformation_h' : rtInformation_h,
                        'runtimeInformation_c' : rtInformation_c,
                        'runtimeInformation_init' : rtInformation_init,
                        'parameters' : paramsDef,
                        'parametersInits' : paramsInits,
                        'intValuesReferences_Init' : intValuesReferences_Init,
                        'intValuesReferences_Def' : intValuesReferences_Def,
                        'floatValuesReferences_Init' : floatValuesReferences_Init,
                        'floatValuesReferences_Def' : floatValuesReferences_Def,
                        'stringValuesReferences_Init' : stringValuesReferences_Init,
                        'stringValuesReferences_Def' : stringValuesReferences_Def,
                        'stns' : stnDef,
                        'stnActiveStates' : stnActiveStates,
                        'assignedVariablesDefs' : assignedVarsDefs,
                        'assignedVariablesInits' : assignedVarsInits,
                        'initialConditions' : initConds,
                        'residuals' : eqnsRes,
                        'jacobian' : jacobRes,
                        'roots' : rootsDef,
                        'numberOfRoots' : noRootsDef,
                        'checkForDiscontinuities' : checkDiscont,
                        'executeActions' : execActionsDef,
                        'warnings' : warnings
                   }

        c99_dir = os.path.join(os.path.dirname(__file__), 'c99')
        
        f = open(os.path.join(c99_dir, 'model.h'), "r")
        daetools_model_h_templ = f.read()
        f.close()
        
        f = open(os.path.join(c99_dir, 'model.c'), "r")
        daetools_model_c_templ = f.read()
        f.close()
        
        daetools_model_h_contents = daetools_model_h_templ % dictInfo;
        daetools_model_c_contents = daetools_model_c_templ % dictInfo;

        path, dirName = os.path.split(directory)

        shutil.copy2(os.path.join(c99_dir, 'main.c'),          os.path.join(directory, 'main.c'))
        shutil.copy2(os.path.join(c99_dir, 'adouble.h'),       os.path.join(directory, 'adouble.h'))
        shutil.copy2(os.path.join(c99_dir, 'adouble.c'),       os.path.join(directory, 'adouble.c'))
        shutil.copy2(os.path.join(c99_dir, 'typedefs.h'),      os.path.join(directory, 'typedefs.h'))
        shutil.copy2(os.path.join(c99_dir, 'daesolver.h'),     os.path.join(directory, 'daesolver.h'))
        shutil.copy2(os.path.join(c99_dir, 'daesolver.c'),     os.path.join(directory, 'daesolver.c'))
        shutil.copy2(os.path.join(c99_dir, 'simulation.h'),    os.path.join(directory, 'simulation.h'))
        shutil.copy2(os.path.join(c99_dir, 'simulation.c'),    os.path.join(directory, 'simulation.c'))
        shutil.copy2(os.path.join(c99_dir, 'auxiliary.h'),     os.path.join(directory, 'auxiliary.h'))
        shutil.copy2(os.path.join(c99_dir, 'auxiliary.c'),     os.path.join(directory, 'auxiliary.c'))
        shutil.copy2(os.path.join(c99_dir, 'Makefile-gcc'),    os.path.join(directory, 'Makefile'))
        shutil.copy2(os.path.join(c99_dir, 'vc++2008.vcproj'), os.path.join(directory, '%s.vcproj' % dirName))
        shutil.copy2(os.path.join(c99_dir, 'qt_project.pro'),  os.path.join(directory, '%s.pro' % dirName))

        f = open(os.path.join(directory, 'model.h'), "w")
        f.write(daetools_model_h_contents)
        f.close()
        
        f = open(os.path.join(directory, 'model.c'), "w")
        f.write(daetools_model_c_contents)
        f.close()

        if len(self.warnings) > 0:
            print('CODE GENERATOR WARNINGS:')
            print(warnings)

    def _processEquations(self, Equations, indent):
        s_indent  = indent     * self.defaultIndent
        s_indent2 = (indent+1) * self.defaultIndent
        
        if self.equationGenerationMode == 'residuals':
            for equation in Equations:
                for eeinfo in equation['EquationExecutionInfos']:
                    res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                    self.residuals.append(s_indent + '/* Equation type: ' + eeinfo['EquationType'] + ' */')
                    self.residuals.append(s_indent + '_temp_ = {0};'.format(res))
                    self.residuals.append(s_indent + '_residuals_[_ec_++] = _getValue_(&_temp_);')

        elif self.equationGenerationMode == 'jacobian':
            for equation in Equations:
                for eeinfo in equation['EquationExecutionInfos']:
                    overall_indexes = eeinfo['VariableIndexes']
                    n = len(overall_indexes)
                    ID = len(self.jacobians)
                    block_indexes = []
                    for oi in overall_indexes:
                        if oi in self.exprFormatter.indexMap:
                            bi = self.exprFormatter.indexBase + self.exprFormatter.indexMap[oi]
                        else:
                            bi = -1
                        block_indexes.append(bi)
                    str_indexes = self.exprFormatter.formatNumpyArray(block_indexes)

                    self.jacobians.append(s_indent + 'int _block_indexes_{0}[{1}] = {2};'.format(ID, n, str_indexes))
                    self.jacobians.append(s_indent + 'for(_i_ = 0; _i_ < {0}; _i_++) {{'.format(n))
                    self.jacobians.append(s_indent2 + '_block_index_ = _block_indexes_{0}[_i_];'.format(ID))
                    self.jacobians.append(s_indent2 + '_current_index_for_jacobian_evaluation_ = _block_index_;')
                    
                    res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                    self.jacobians.append(s_indent2 + '_temp_ = {0};'.format(res))
                    self.jacobians.append(s_indent2 + '_jacobianItem_ = _getDerivative_(&_temp_);')
                    self.jacobians.append(s_indent2 + '_set_matrix_item_(_jacobian_matrix_, _ec_, _block_index_, _jacobianItem_);')

                    self.jacobians.append(s_indent + '}')
                    self.jacobians.append(s_indent + '_ec_++;')

    def _processSTNs(self, STNs, indent):
        s_indent = indent * self.defaultIndent
        for stn in STNs:
            if stn['Class'] == 'daeIF':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_ifstn'
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                varTemplate = 'char* {name}; /* States: [{states}] ({description}) */'
                self.stnDefs.append(varTemplate.format(name = stnVariableName,
                                                       states = states,
                                                       description = description))
                varTemplate = '_m_->{name} = "{activeState}";'
                self.initiallyActiveStates.append(varTemplate.format(name = stnVariableName,
                                                                     activeState = activeState))

                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    # Not all states have state_transitions ('else' state has no state transitions)
                    on_condition_action = None
                    if i == 0:
                        temp = s_indent + '/* IF {0} */'.format(stnVariableName)
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        # There is only one OnConditionAction in IF
                        on_condition_action = state['OnConditionActions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                        temp = s_indent + 'if(_compare_strings_(_m_->{0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        # We need to detect the number of roots and root functions for the current state of affairs,
                        # that is active states after execute actions (or at the very beggining of a simulation).
                        # Therefore, put those here and not below in the: if(condition) block.
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        temp = s_indent + 'if({0}) {{'.format(condition)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        # There is only one OnConditionAction in ELSE_IFs
                        on_condition_action = state['OnConditionActions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])

                        temp = s_indent + 'else if(_compare_strings_(_m_->{0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        # We need to detect the number of roots and root functions for the current state of affairs,
                        # that is active states after execute actions (or at the very beggining of a simulation).
                        # Therefore, put those here and not below in the: if(condition) block.
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        temp = s_indent + 'else if({0}) {{'.format(condition)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)

                    else:
                        temp = s_indent + 'else {'
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        # Here it does not matter
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    # 1a. Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+1)
                    
                    # 1b. Put equations into the residuals list
                    self.equationGenerationMode  = 'residuals'
                    self._processEquations(state['Equations'], indent+1)
                    self.residuals.append(s_indent + '}')

                    # 1c. Put equations into the jacobians list
                    self.equationGenerationMode  = 'jacobian'
                    self._processEquations(state['Equations'], indent+1)
                    self.jacobians.append(s_indent + '}')

                    s_indent2 = (indent + 1) * self.defaultIndent
                    s_indent3 = (indent + 2) * self.defaultIndent

                    # 2. checkForDiscontinuities
                    self.checkForDiscontinuities.append(s_indent2 + 'if(! _compare_strings_(_m_->{0}, "{1}")) {{'.format(stnVariableName, state['Name']))
                    self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                    self.checkForDiscontinuities.append(s_indent2 + '}')
                    self.checkForDiscontinuities.append(s_indent + '}')

                    # 3. executeActions
                    self.executeActions.append(s_indent2 + 'printf("The state [{0}] in the IF [{1}] is active now.\\n");'.format(state['Name'], relativeName))
                    self.executeActions.append(s_indent2 + '_m_->{0} = "{1}";'.format(stnVariableName, state['Name']))
                    self.executeActions.append(s_indent + '}')

                    # 4. numberOfRoots
                    if on_condition_action: # For 'else' state has no state transitions
                        nExpr = len(on_condition_action['Expressions'])
                        self.numberOfRoots.append(s_indent + '_noRoots_ += {0};'.format(nExpr))
                    self.numberOfRoots.append(s_indent + '}')

                    # 5. rootFunctions
                    if on_condition_action: # For 'else' state has no state transitions
                        for expression in on_condition_action['Expressions']:
                            self.rootFunctions.append(s_indent + '_temp_ = {0};'.format(self.exprFormatter.formatRuntimeNode(expression)))
                            self.rootFunctions.append(s_indent + '_roots_[_rc_++] = _getValue_(&_temp_);')
                    self.rootFunctions.append(s_indent + '}')

            elif stn['Class'] == 'daeSTN':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                varTemplate = 'char* {name}; /* States: [{states}] ({description}) */'
                self.stnDefs.append(varTemplate.format(name = stnVariableName,
                                                       states = states,
                                                       description = description))
                varTemplate = '_m_->{name} = "{activeState}";'
                self.initiallyActiveStates.append(varTemplate.format(name = stnVariableName,
                                                                     activeState = activeState))

                nStates = len(stn['States'])
                for i, state in enumerate(stn['States']):
                    if i == 0:
                        temp = s_indent + '/* STN {0} */'.format(stnVariableName)
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        temp = s_indent + 'if(_compare_strings_(_m_->{0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        temp = s_indent + 'else if(_compare_strings_(_m_->{0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    else:
                        temp = s_indent + 'else {'
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    # 1a. Generate NestedSTNs
                    self._processSTNs(state['NestedSTNs'], indent+1)
                    
                    # 1b. Put equations into the residuals list
                    self.equationGenerationMode  = 'residuals'
                    self._processEquations(state['Equations'], indent+1)
                    self.residuals.append(s_indent + '}')

                    # 1c. Put equations into the jacobians list
                    self.equationGenerationMode  = 'jacobian'
                    self._processEquations(state['Equations'], indent+1)
                    self.jacobians.append(s_indent + '}')

                    s_indent2 = (indent + 1) * self.defaultIndent
                    s_indent3 = (indent + 2) * self.defaultIndent

                    # 2. checkForDiscontinuities
                    for i, on_condition_action in enumerate(state['OnConditionActions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])
                        if i == 0:
                            self.checkForDiscontinuities.append(s_indent2 + 'if({0}) {{'.format(condition))
                            self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                            self.checkForDiscontinuities.append(s_indent2 + '}')

                        elif (i > 0) and (i < nStates - 1):
                            self.checkForDiscontinuities.append(s_indent2 + 'else if({0}) {{'.format(condition))
                            self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                            self.checkForDiscontinuities.append(s_indent2 + '}')

                        else:
                            self.checkForDiscontinuities.append(s_indent2 + 'else {')
                            self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                            self.checkForDiscontinuities.append(s_indent2 + '}')

                    self.checkForDiscontinuities.append(s_indent + '}')
                    
                    # 3. executeActions
                    for i, on_condition_action in enumerate(state['OnConditionActions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(on_condition_action['ConditionRuntimeNode'])
                        if i == 0:
                            self.executeActions.append(s_indent2 + 'if({0}) {{'.format(condition))

                        elif (i > 0) and (i < nStates - 1):
                            self.executeActions.append(s_indent2 + 'else if({0}) {{'.format(condition))

                        else:
                            self.executeActions.append(s_indent2 + 'else {')

                        self._processActions(on_condition_action['Actions'], indent+2)
                        self.executeActions.append(s_indent2 + '}')

                    self.executeActions.append(s_indent + '}')

                    # 4. numberOfRoots
                    for i, on_condition_action in enumerate(state['OnConditionActions']):
                        nExpr = len(on_condition_action['Expressions'])
                        self.numberOfRoots.append(s_indent2 + '_noRoots_ += {0};'.format(nExpr))

                    self.numberOfRoots.append(s_indent + '}')

                    # 5. rootFunctions
                    for i, on_condition_action in enumerate(state['OnConditionActions']):
                        for expression in on_condition_action['Expressions']:
                            self.rootFunctions.append(s_indent2 + '_temp_ = {0};'.format(self.exprFormatter.formatRuntimeNode(expression)))
                            self.rootFunctions.append(s_indent2 + '_roots_[_rc_++] = _getValue_(&_temp_);')

                    self.rootFunctions.append(s_indent + '}')

    def _processActions(self, Actions, indent):
        s_indent = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, action['STNCanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                stateTo         = action['StateTo']
                self.executeActions.append(s_indent + '_log_message_("The state: [{0}] in the STN [{1}] is active now.\\n");'.format(stateTo, relativeName))
                self.executeActions.append(s_indent + '_m_->{0} = "{1}";'.format(stnVariableName, stateTo))

            elif action['Type'] == 'eSendEvent':
                self.warnings.append('C code cannot be generated for SendEvent actions on [%s] event port' % action['SendEventPort'])

            elif action['Type'] == 'eReAssignOrReInitializeVariable':
                relativeName  = daeGetRelativeName(self.wrapperInstanceName, action['VariableCanonicalName'])
                relativeName  = self.exprFormatter.formatIdentifier(relativeName)
                domainIndexes = action['DomainIndexes']
                overallIndex  = action['OverallIndex']
                ID            = action['ID']
                node          = action['RuntimeNode']
                strDomainIndexes = ''
                if len(domainIndexes) > 0:
                    strDomainIndexes = '(' + ','.join() + ')'
                variableName = relativeName + strDomainIndexes
                value = self.exprFormatter.formatRuntimeNode(node)

                self.executeActions.append(s_indent + '_log_message_("The variable [{0}] new value is [{1}] now.\\n");'.format(variableName, value))
                self.executeActions.append(s_indent + '_temp_ = {0};'.format(value))

                if ID == cnDifferential:
                    # Write a new value directly into the _values_ array
                    # All actions that need this variable will use a new value.
                    # Is that what we want??? Should be...
                    self.executeActions.append(s_indent + '_values_[{0}] = _getValue_(&_temp_);'.format(overallIndex))

                elif ID == cnAssigned:
                    self.executeActions.append(s_indent + '{0} = _getValue_(&_temp_);'.format(variableName))

                else:
                    raise RuntimeError('Cannot reset a value of the state variable: {0}'.format(relativeName))

                self.executeActions.append(s_indent + '_copy_values_to_solver_ = true;')

            elif action['Type'] == 'eUserDefinedAction':
                self.warnings.append('C code cannot be generated for UserDefined actions')

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

        self.variableNames = Neq * ['']
        
        self.runtimeInformation_h.append('int         IDs               [%d];' % Neq)
        self.runtimeInformation_h.append('real_t      initValues        [%d];' % Neq)
        self.runtimeInformation_h.append('real_t      initDerivatives   [%d];' % Neq)
        self.runtimeInformation_h.append('real_t      absoluteTolerances[%d];' % Neq)
        self.runtimeInformation_h.append('const char* variableNames     [%d];' % Neq)
        
        #self.runtimeInformation_c.append('#define __total_number_of_variables__ {0}'.format(Ntotal))
        #self.runtimeInformation_c.append('#define __number_of_equations__       {0}'.format(Neq))
        
        self.runtimeInformation_init.append('_m_->Ntotal_vars       = %d;' % Ntotal)
        self.runtimeInformation_init.append('_m_->Nequations        = %d;' % Neq)
        self.runtimeInformation_init.append('_m_->startTime         = %f;'   % 0.0)
        self.runtimeInformation_init.append('_m_->timeHorizon       = %f;'   % runtimeInformation['TimeHorizon'])
        self.runtimeInformation_init.append('_m_->reportingInterval = %f;'   % runtimeInformation['ReportingInterval'])
        self.runtimeInformation_init.append('_m_->relativeTolerance = %f;'   % runtimeInformation['RelativeTolerance'])        
        self.runtimeInformation_init.append('_m_->quasySteadyState  = %s;\n' % ('true' if runtimeInformation['QuasySteadyState'] else 'false'))

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

        self.runtimeInformation_c.append('const int    _IDs_                [%d] = %s;' % (Neq, self.exprFormatter.formatNumpyArray(blockIDs)))
        self.runtimeInformation_c.append('const real_t _init_values_        [%d] = %s;' % (Neq, self.exprFormatter.formatNumpyArray(blockInitValues)))
        self.runtimeInformation_c.append('const real_t _init_derivatives_   [%d] = %s;' % (Neq, self.exprFormatter.formatNumpyArray(blockInitDerivatives)))
        self.runtimeInformation_c.append('const real_t _absolute_tolerances_[%d] = %s;' % (Neq, self.exprFormatter.formatNumpyArray(absTolerances)))

        self.runtimeInformation_init.append('memcpy(_m_->IDs,                &_IDs_,                 %d * sizeof(int));'    % Neq)
        self.runtimeInformation_init.append('memcpy(_m_->initValues,         &_init_values_,         %d * sizeof(real_t));' % Neq)
        self.runtimeInformation_init.append('memcpy(_m_->initDerivatives,    &_init_derivatives_,    %d * sizeof(real_t));' % Neq)
        self.runtimeInformation_init.append('memcpy(_m_->absoluteTolerances, &_absolute_tolerances_, %d * sizeof(real_t));' % Neq)

        for domain in runtimeInformation['Domains']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, domain['CanonicalName'])
            formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(formattedName)
            description    = domain['Description']
            numberOfPoints = domain['NumberOfPoints']
            units          = ('-' if (domain['Units'] == unit()) else str(domain['Units']))
            domains        = '[' + str(domain['NumberOfPoints']) + ']'
            points         = self.exprFormatter.formatNumpyArray(domain['Points']) # Numpy array

            domTemplate   = 'int {name}_np; /* Number of points in domain: {relativeName} */'
            paramTemplate = 'real_t {name}{domains}; /* Domain: {relativeName} ({units}, {description}) */'
            self.parametersDefs.append(domTemplate.format(name = name, relativeName = relativeName))
            self.parametersDefs.append(paramTemplate.format(name = name,
                                                            units = units,
                                                            domains = domains,
                                                            relativeName = relativeName,
                                                            description = description))

            domTemplate   = 'const int {name}_np = {numberOfPoints};'
            paramTemplate = 'real_t {name}{domains} = {points}; /* {units} */'
            self.parametersInits.append(domTemplate.format(name = name,
                                                           numberOfPoints = numberOfPoints))
            self.parametersInits.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            points = points,
                                                            units = units,
                                                            description = description))

            domTemplate   = '_m_->{name}_np = {name}_np;'
            paramTemplate = 'memcpy(&_m_->{name}, &{name}, {numberOfPoints} * sizeof(real_t));\n'
            self.parametersInits.append(domTemplate.format(name = name))
            self.parametersInits.append(paramTemplate.format(name = name,
                                                             numberOfPoints = numberOfPoints))

            struct_name = formattedName + '_np'
            flat_name   = name + '_np'
            self.intValuesReferences.append( ('NumberOfPointsInDomain', struct_name, flat_name, None) )
            for i in range(len(domain['Points'])):
                struct_name = '{0}[{1}]'.format(formattedName, i)
                flat_name   = '{0}[{1}]'.format(name, i)
                self.floatValuesReferences.append( ('DomainPoints', struct_name, flat_name, None) )
            
        for parameter in runtimeInformation['Parameters']:
            relativeName    = daeGetRelativeName(self.wrapperInstanceName, parameter['CanonicalName'])
            formattedName   = self.exprFormatter.formatIdentifier(relativeName)
            name            = self.exprFormatter.flattenIdentifier(formattedName)
            description     = parameter['Description']
            numberOfPoints  = int(parameter['NumberOfPoints'])
            units           = ('-' if (parameter['Units'] == unit()) else str(parameter['Units']))
            values          = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array
            numberOfDomains = len(parameter['Domains'])
            domains         = ''
            if numberOfDomains > 0:
                domains = '[{0}]'.format(']['.join(str(np) for np in parameter['Domains']))

                paramTemplate = 'real_t {name}{domains}; /* Parameter: {relativeName} ({units}, {description}) */'
                self.parametersDefs.append(paramTemplate.format(name = name,
                                                                units = units,
                                                                domains = domains,
                                                                relativeName = relativeName,
                                                                description = description))

                paramTemplate = 'real_t {name}{domains} = {values} /* {units} */;'
                self.parametersInits.append(paramTemplate.format(name = name,
                                                                 units = units,
                                                                 domains = domains,
                                                                 values = values))

                paramTemplate = 'memcpy(&_m_->{name}, &{name}, {numberOfPoints} * sizeof(real_t));\n'
                self.parametersInits.append(paramTemplate.format(name = name,
                                                                numberOfPoints = numberOfPoints))
            else:
                paramTemplate = 'real_t {name}; /* Parameter: {relativeName} ({units}, {description}) */'
                self.parametersDefs.append(paramTemplate.format(name = name,
                                                                units = units,
                                                                relativeName = relativeName,
                                                                description = description))

                paramTemplate = '_m_->{name} = {value} /* {units} */;'
                self.parametersInits.append(paramTemplate.format(name = name,
                                                                 units = units,
                                                                 value = values))
                
            if numberOfDomains == 0:
                struct_name = formattedName
                flat_name   = self.exprFormatter.flattenIdentifier(struct_name)
                self.floatValuesReferences.append( ('Parameter', struct_name, flat_name, None) )
            else:
                domainsIndexesMap = parameter['DomainsIndexesMap']
                for i in range(0, numberOfPoints):
                    domIndexes = tuple(domainsIndexesMap[i])  # list of integers
                    struct_name = '{0}[{1}]'.format(formattedName,  ']['.join(str(index) for index in domIndexes))
                    flat_name   = '{0}[{1}]'.format(name,           ']['.join(str(index) for index in domIndexes))
                    self.floatValuesReferences.append( ('Parameter', struct_name, flat_name, None) )

                """
                it = numpy.nditer(parameter['Values'], flags=['c_index', 'multi_index'])
                while not it.finished:
                    #print name + "%s = %d" % (it.multi_index, it[0])
                    p_name = '{0}[{1}]'.format(name,  ']['.join(str(index) for index in it.multi_index))
                    self.floatValuesReferences.append( ('Parameter', p_name, value_ref) )
                    print p_name
                    value_ref += 1
                    it.iternext()
                """
                
        for variable in runtimeInformation['Variables']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, variable['CanonicalName'])
            formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(formattedName)
            numberOfPoints = variable['NumberOfPoints']
            units          = ('-' if (variable['Units'] == unit()) else str(variable['Units']))

            if numberOfPoints == 1:
                ID           = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value        = float(variable['Values'])   # numpy float
                overallIndex = variable['OverallIndex']
                fullName     = relativeName

                blockIndex = None
                if ID != cnAssigned:
                    blockIndex = indexMappings[overallIndex] + self.exprFormatter.indexBase
                    self.variableNames[blockIndex] = fullName

                if ID == cnDifferential:
                    name_ = 'values[{0}]'.format(blockIndex)
                    temp = '{name} = {value} /* {units} */; /* {fullName} */'.format(name = name_, value = value, units = units, fullName = fullName)
                    self.initialConditions.append(temp)
                elif ID == cnAssigned:
                    name_ = name
                    temp = 'real_t {name}; /* {fullName}, {units} */'.format(name = name, units = units, fullName = fullName)
                    self.assignedVariablesDefs.append(temp)
                    temp = '_m_->{name} = {value} /* {units} */;'.format(name = name, value = value, units = units)
                    self.assignedVariablesInits.append(temp)
                    
                if ID == cnAssigned:
                    ref_type = 'Assigned'
                elif ID == cnDifferential:
                    ref_type = 'Differential'
                else:
                    ref_type = 'Algebraic'

                struct_name = name # Name as it appears in daeModel_t (Assigned vars. only)
                flat_name   = self.exprFormatter.flattenIdentifier(struct_name)
                self.floatValuesReferences.append( (ref_type, struct_name, flat_name, blockIndex) )

            else:
                domainsIndexesMap = variable['DomainsIndexesMap']
                for i in range(0, numberOfPoints):
                    domIndexes   = tuple(domainsIndexesMap[i])              # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    fullName     = relativeName + '(' + ','.join(str(di) for di in domIndexes) + ')'

                    blockIndex = None
                    if ID != cnAssigned:
                        blockIndex = indexMappings[overallIndex] + self.exprFormatter.indexBase
                        self.variableNames[blockIndex] = fullName

                    if ID == cnDifferential:
                        name_ = 'values[{0}]'.format(blockIndex)
                        temp = '{name} = {value} /* {units}*/; /* {fullName} */'.format(name = name_, value = value, units = units, fullName = fullName)
                        self.initialConditions.append(temp)
                    elif ID == cnAssigned:
                        name_ = name + '_' + '_'.join(str(di) for di in domIndexes)
                        temp = 'real_t {name}; /* {fullName}, {units} */'.format(name = name_, units = units, fullName = fullName)
                        self.assignedVariablesDefs.append(temp)

                        temp = '_m_->{name} = {value} /* {units} */;'.format(name = name_, value = value, units = units)
                        self.assignedVariablesInits.append(temp)

                    if ID == cnAssigned:
                        ref_type = 'Assigned'
                        struct_name = name_ # Name as it appears in daeModel_t (Assigned vars. only)
                    elif ID == cnDifferential:
                        ref_type = 'Differential'
                        struct_name = name
                    else:
                        ref_type = 'Algebraic'
                        struct_name = name

                    flat_name   = self.exprFormatter.flattenIdentifier(struct_name)
                    self.floatValuesReferences.append( (ref_type, struct_name, flat_name, blockIndex) )

        varNames = ['"' + name_ + '"' for name_ in self.variableNames]
        
        self.runtimeInformation_c.append('const char*  _variable_names_     [%d] = %s;' % (Neq, self.exprFormatter.formatNumpyArray(varNames)))
        self.runtimeInformation_init.append('for(int i = 0; i < %d; i++)' % Neq)
        self.runtimeInformation_init.append('    _m_->variableNames[i] = _variable_names_[i];')

        #import pprint
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(self.floatValuesReferences)

        indent = 1
        s_indent = indent * self.defaultIndent

        # First generate residuals for equations and port connections
        self.equationGenerationMode  = 'residuals'
        for port_connection in runtimeInformation['PortConnections']:
            self.residuals.append(s_indent + '/* Port connection: {0} -> {1} */'.format(port_connection['PortFrom'], port_connection['PortTo']))
            self._processEquations(port_connection['Equations'], indent)

        self.residuals.append(s_indent + '/* Equations */')
        self._processEquations(runtimeInformation['Equations'], indent)

        # Then generate jacobians for equations and port connections
        self.equationGenerationMode  = 'jacobian'
        for port_connection in runtimeInformation['PortConnections']:
            self.jacobians.append(s_indent + '/* Port connection: {0} -> {1} */'.format(port_connection['PortFrom'], port_connection['PortTo']))
            self._processEquations(port_connection['Equations'], indent)

        self.jacobians.append(s_indent + '/* Equations */')
        self._processEquations(runtimeInformation['Equations'], indent)

        # Finally generate together residuals and jacobians for STNs
        # _processSTNs will take care of self.equationGenerationMode regime
        self._processSTNs(runtimeInformation['STNs'], indent)

   
