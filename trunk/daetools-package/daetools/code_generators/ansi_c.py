import os, shutil, sys, numpy, math, traceback
from daetools.pyDAE import *
from formatter import daeExpressionFormatter
from analyzer import daeCodeGeneratorAnalyzer


"""
Compile with:
  g++ -O3 -Wall -o daetools_simulation adouble.cpp main.cpp
"""

mainTemplate = """\
/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAETOOLS_MODEL_H
#define DAETOOLS_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

/* CODE GENERATOR WARNINGS!!!
%(warnings)s
*/

#include "auxiliary.h"
#include "adouble.h"

#define _v_(i)   _adouble_(_values_[i],           (i == _current_index_for_jacobian_evaluation_) ? 1.0 : 0.0)
#define _dt_(i)  _adouble_(_time_derivatives_[i], (i == _current_index_for_jacobian_evaluation_) ? _inverse_time_step_ : 0.0)
#define _time_   _adouble_(_current_time_, 0.0)

typedef struct
{
%(valuesReferences)s

/* Domains and parameters */
%(parameters)s

/* StateTransitionNetworks */
%(activeStates)s

/* Assigned variables */
%(assignedVariables)s
}daetools_model_t;

void initialize_model(daetools_model_t* _m_);
void set_initial_conditions(real_t* values);
int residuals(daetools_model_t* _m_,
              real_t _current_time_,
              real_t* _values_,
              real_t* _time_derivatives_,
              real_t* _residuals_);
int jacobian(daetools_model_t* _m_,
             long int _number_of_equations_,
             real_t _current_time_,
             real_t _inverse_time_step_,
             real_t* _values_,
             real_t* _time_derivatives_,
             real_t* _residuals_,
             matrix_t _jacobian_matrix_);
int number_of_roots(daetools_model_t* _m_);
int roots(daetools_model_t* _m_,
          real_t _current_time_,
          real_t* _values_,
          real_t* _time_derivatives_,
          real_t* _roots_);
bool check_for_discontinuities(daetools_model_t* _m_,
                               real_t _current_time_,
                               real_t* _values_,
                               real_t* _time_derivatives_);
bool execute_actions(daetools_model_t* _m_,
                     real_t _current_time_,
                     real_t* _values_,
                     real_t* _time_derivatives_);

/* General info */          
%(model)s

void initialize_model(daetools_model_t* _m_)
{
  %(parametersInits)s
}

void set_initial_conditions(real_t* values)
{
/* Initial conditions */
%(initialConditions)s
}

int residuals(daetools_model_t* _m_,
              real_t _current_time_,
              real_t* _values_,
              real_t* _time_derivatives_,
              real_t* _residuals_)
{
    adouble _temp_;
    real_t _inverse_time_step_;
    int i, _ec_, _current_index_for_jacobian_evaluation_;
    
    _ec_                                    = 0;
    _current_index_for_jacobian_evaluation_ = -1;
    _inverse_time_step_                     = 0.0;

%(residuals)s

    return 0;
}

int jacobian(daetools_model_t* _m_,
             long int _number_of_equations_,
             real_t _current_time_,
             real_t _inverse_time_step_,
             real_t* _values_,
             real_t* _time_derivatives_,
             real_t* _residuals_,
             matrix_t _jacobian_matrix_)
{
    adouble _temp_;
    real_t _jacobianItem_;
    int _i_, _ec_, _block_index_, _current_index_for_jacobian_evaluation_;

    _ec_                                    = 0;
    _current_index_for_jacobian_evaluation_ = -1;
    
%(jacobian)s

    return 0;
}

int number_of_roots(daetools_model_t* _m_)
{
    int _noRoots_;

    _noRoots_ = 0;
    
%(numberOfRoots)s

    return _noRoots_;
}

int roots(daetools_model_t* _m_,
          real_t _current_time_,
          real_t* _values_,
          real_t* _time_derivatives_,
          real_t* _roots_)
{
    adouble _temp_;
    real_t _inverse_time_step_;
    int _rc_, _current_index_for_jacobian_evaluation_;
    
    _rc_                                    = 0;
    _inverse_time_step_                     = 0.0;
    _current_index_for_jacobian_evaluation_ = -1;
    
%(roots)s

    return 0;
}

bool check_for_discontinuities(daetools_model_t* _m_,
                               real_t _current_time_,
                               real_t* _values_,
                               real_t* _time_derivatives_)
{
    adouble _temp_;
    bool foundDiscontinuity;
    real_t _inverse_time_step_;
    int _current_index_for_jacobian_evaluation_;

    _inverse_time_step_                     = 0.0;
    _current_index_for_jacobian_evaluation_ = -1;
    foundDiscontinuity                      = false;
    
%(checkForDiscontinuities)s

    return foundDiscontinuity;
}

bool execute_actions(daetools_model_t* _m_,
                     real_t _current_time_,
                     real_t* _values_,
                     real_t* _time_derivatives_)
{
    adouble _temp_;
    real_t _inverse_time_step_;
    bool _copy_values_to_solver_;
    int _current_index_for_jacobian_evaluation_;

    _inverse_time_step_                     = 0.0;
    _current_index_for_jacobian_evaluation_ = -1;
    _copy_values_to_solver_                 = false;

%(executeActions)s

    return _copy_values_to_solver_;
}

#ifdef __cplusplus
}
#endif

#endif
"""

class daeANSICExpressionFormatter(daeExpressionFormatter):
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

        self.derivative               = '_dt_({blockIndex})'
        self.derivativeIndexStart     = ''
        self.derivativeIndexEnd       = ''
        self.derivativeIndexDelimiter = ''

        # Constants
        self.constant = '_adouble_({value}, 0)'

        # Logical operators
        self.AND   = '_and_({leftValue}, {rightValue})'
        self.OR    = '_or_({leftValue}, {rightValue})'
        self.NOT   = '_not_({value})'

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

        self.MIN    = '_min_({leftValue}, {rightValue})'
        self.MAX    = '_max_({leftValue}, {rightValue})'

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
     
class daeCodeGenerator_ANSI_C(object):
    def __init__(self, simulation = None):
        self.wrapperInstanceName     = ''
        self.defaultIndent           = '    '
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None
        self.equationGenerationMode  = ''
        
        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.parametersInits         = []
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

        self.exprFormatter = daeANSICExpressionFormatter()
        self.analyzer      = daeCodeGeneratorAnalyzer()

    def generateSimulation(self, simulation, **kwargs):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        directory = kwargs.get('projectDirectory', None)

        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.parametersInits         = []
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
        self.wrapperInstanceName     = simulation.m.Name
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

        modelDef       = '\n'.join(self.modelDef)
        paramsDef      = '\n'.join(self.parametersDefs)
        paramsInits    = '\n  '.join(self.parametersInits)
        stnDef         = '\n'.join(self.initiallyActiveStates)
        assignedVars   = '\n'.join(self.assignedVariables)
        initConds      = '\n'.join(self.initialConditions)
        eqnsRes        = '\n'.join(self.residuals)
        jacobRes       = '\n'.join(self.jacobians)
        rootsDef       = '\n'.join(self.rootFunctions)
        checkDiscont   = '\n'.join(self.checkForDiscontinuities)
        execActionsDef = '\n'.join(self.executeActions)
        noRootsDef     = '\n'.join(self.numberOfRoots)
        warnings       = '\n'.join(self.warnings)

        valuesReferences = ''

        dictInfo = {
                        'model' : modelDef,
                        'parameters' : paramsDef,
                        'parametersInits' : paramsInits,
                        'valuesReferences' : valuesReferences,
                        'activeStates' : stnDef,
                        'assignedVariables' : assignedVars,
                        'initialConditions' : initConds,
                        'residuals' : eqnsRes,
                        'jacobian' : jacobRes,
                        'roots' : rootsDef,
                        'numberOfRoots' : noRootsDef,
                        'checkForDiscontinuities' : checkDiscont,
                        'executeActions' : execActionsDef,
                        'warnings' : warnings
                   }

        results = mainTemplate % dictInfo;

        ansic_dir = os.path.join(os.path.dirname(__file__), 'ansic')

        # If the argument 'directory' is given create the folder and the project
        if directory:
            path, dirName = os.path.split(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

            daetools_model_h = os.path.join(directory, 'daetools_model.h')
            shutil.copy2(os.path.join(ansic_dir, 'main-dense.c'),   os.path.join(directory, 'main.c'))
            shutil.copy2(os.path.join(ansic_dir, 'adouble.h'),      os.path.join(directory, 'adouble.h'))
            shutil.copy2(os.path.join(ansic_dir, 'adouble.c'),      os.path.join(directory, 'adouble.c'))
            shutil.copy2(os.path.join(ansic_dir, 'typedefs.h'),     os.path.join(directory, 'typedefs.h'))
            shutil.copy2(os.path.join(ansic_dir, 'auxiliary.h'),    os.path.join(directory, 'auxiliary.h'))
            shutil.copy2(os.path.join(ansic_dir, 'auxiliary.c'),    os.path.join(directory, 'auxiliary.c'))
            shutil.copy2(os.path.join(ansic_dir, 'qt_project.pro'), os.path.join(directory, '{0}.pro'.format(dirName)))

            f = open(daetools_model_h, "w")
            f.write(results)
            f.close()

        if len(self.warnings) > 0:
            print 'CODE GENERATOR WARNINGS:'
            print warnings
        
        return results

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
            nStates = len(stn['States'])
            if stn['Class'] == 'daeIF':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_ifstn'
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
                    # Not all states have state_transitions ('else' state has no state transitions)
                    state_transition = None
                    if i == 0:
                        temp = s_indent + '/* IF {0} */'.format(stnVariableName)
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        state_transition = state['StateTransitions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])

                        temp = s_indent + 'if(_compare_strings_({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)

                        temp = s_indent + 'if({0}) {{'.format(condition)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        state_transition = state['StateTransitions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])

                        temp = s_indent + 'else if(_compare_strings_({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)

                        temp = s_indent + 'else if({0}) {{'.format(condition)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)

                    else:
                        temp = s_indent + 'else {'
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)

                    # 1a. Put equations into the residuals list
                    self.equationGenerationMode  = 'residuals'
                    self._processEquations(state['Equations'], indent+1)
                    self.residuals.append(s_indent + '}')

                    # 1b. Put equations into the jacobians list
                    self.equationGenerationMode  = 'jacobian'
                    self._processEquations(state['Equations'], indent+1)
                    self.jacobians.append(s_indent + '}')

                    nStateTransitions = len(state['StateTransitions'])
                    s_indent2 = (indent + 1) * self.defaultIndent
                    s_indent3 = (indent + 2) * self.defaultIndent

                    # 2. checkForDiscontinuities
                    self.checkForDiscontinuities.append(s_indent2 + 'if(! _compare_strings_({0}, "{1}")) {{'.format(stnVariableName, state['Name']))
                    self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                    self.checkForDiscontinuities.append(s_indent2 + '}')
                    self.checkForDiscontinuities.append(s_indent + '}')

                    # 3. executeActions
                    self.executeActions.append(s_indent2 + 'printf("The state {0} from {1} is active now.\\n");'.format(state['Name'], stnVariableName))
                    self.executeActions.append(s_indent2 + '{0} = "{1}";'.format(stnVariableName, state['Name']))
                    self.executeActions.append(s_indent + '}')

                    # 4. numberOfRoots
                    if state_transition: # For 'else' state has no state transitions
                        nExpr = len(state_transition['Expressions'])
                        self.numberOfRoots.append(s_indent + '_noRoots_ += {0};'.format(nExpr))

                    # 5. rootFunctions
                    if state_transition: # For 'else' state has no state transitions
                        for expression in state_transition['Expressions']:
                            self.rootFunctions.append(s_indent + '_temp_ = {0};'.format(self.exprFormatter.formatRuntimeNode(expression)))
                            self.rootFunctions.append(s_indent + '_roots_[_rc_++] = _getValue_(&_temp_);')

                    if len(state['NestedSTNs']) > 0:
                        raise RuntimeError('C code cannot be generated for nested state transition networks')

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
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                        temp = s_indent + 'if(_compare_strings_({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        temp = s_indent + 'else if(_compare_strings_({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
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

                    # 1. Put equations into the residuals list
                    self.equationGenerationMode  = 'residuals'
                    self._processEquations(state['Equations'], indent+1)
                    self.residuals.append(s_indent + '}')

                    # 1b. Put equations into the jacobians list
                    self.equationGenerationMode  = 'jacobian'
                    self._processEquations(state['Equations'], indent+1)
                    self.jacobians.append(s_indent + '}')

                    nStateTransitions = len(state['StateTransitions'])
                    s_indent2 = (indent + 1) * self.defaultIndent
                    s_indent3 = (indent + 2) * self.defaultIndent

                    # 2. checkForDiscontinuities
                    for i, state_transition in enumerate(state['StateTransitions']):
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])
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
                            self.rootFunctions.append(s_indent2 + '_temp_ = {0};'.format(self.exprFormatter.formatRuntimeNode(expression)))
                            self.rootFunctions.append(s_indent2 + '_roots_[_rc_++] = _getValue_(&_temp_);')

                    self.rootFunctions.append(s_indent + '}')

                    if len(state['NestedSTNs']) > 0:
                        raise RuntimeError('C code cannot be generated for nested state transition networks')

    def _processActions(self, Actions, indent):
        s_indent = indent * self.defaultIndent

        for action in Actions:
            if action['Type'] == 'eChangeState':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, action['STNCanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                stateTo         = action['StateTo']
                self.executeActions.append(s_indent + '_log_message_("The state [{0}] in STN [{1}] is active now.\\n");'.format(stateTo, stnVariableName))
                self.executeActions.append(s_indent + '{0} = "{1}";'.format(stnVariableName, stateTo))

            elif action['Type'] == 'eSendEvent':
                self.warnings.append('C code cannot be generated for SendEvent actions - the model will not work as expected!!')

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
                self.warnings.append('C code cannot be generated for UserDefined actions - the model will not work as expected!!')

            else:
                raise RuntimeError('Unknown action type')

    def _generateRuntimeInformation(self, runtimeInformation):
        Ntotal             = runtimeInformation['TotalNumberOfVariables']
        Neq                = runtimeInformation['NumberOfEquations']
        IDs                = runtimeInformation['IDs']
        initValues         = runtimeInformation['InitialValues']
        initDerivatives    = runtimeInformation['InitialDerivatives']
        indexMappings      = runtimeInformation['IndexMappings']
        absoluteTolerances = runtimeInformation['AbsoluteTolerances']

        self.variableNames = Neq * ['']
        
        Nvars = '#define _Ntotal_vars_ {0}'.format(Ntotal)
        self.modelDef.append(Nvars)

        Neqns = '#define _Neqns_ {0}'.format(Neq)
        self.modelDef.append(Neqns)

        startTime = 'const real_t _start_time_ = {0};'.format(0.0)
        self.modelDef.append(startTime)

        endTime = 'const real_t _end_time_ = {0};'.format(runtimeInformation['TimeHorizon'])
        self.modelDef.append(endTime)

        reportingInterval = 'const real_t _reporting_interval_ = {0};'.format(runtimeInformation['ReportingInterval'])
        self.modelDef.append(reportingInterval)

        relTolerance = 'const real_t _relative_tolerance_ = {0};'.format(runtimeInformation['RelativeTolerance'])
        self.modelDef.append(relTolerance)

        blockIDs             = Neq * [-1]
        blockInitValues      = Neq * [-1]
        absTolerances        = Neq * [1E-5]
        blockInitDerivatives = Neq * [0.0]
        for oi, bi in indexMappings.items():
            if IDs[oi] == 0:
               blockIDs[bi] = 0 
            elif IDs[oi] == 1:
               blockIDs[bi] = 1

            if IDs[oi] != 2:
                blockInitValues[bi]      = initValues[oi]
                blockInitDerivatives[bi] = initDerivatives[oi]
                absTolerances[bi]        = absoluteTolerances[oi]

        strIDs = 'const int _IDs_[_Neqns_] = {0};'.format(self.exprFormatter.formatNumpyArray(blockIDs))
        self.modelDef.append(strIDs)
            
        strInitValues = 'const real_t _initValues_[_Neqns_] = {0};'.format(self.exprFormatter.formatNumpyArray(blockInitValues))
        self.modelDef.append(strInitValues)

        strInitDerivs = 'const real_t _initDerivatives_[_Neqns_] = {0};'.format(self.exprFormatter.formatNumpyArray(blockInitDerivatives))
        self.modelDef.append(strInitDerivs)

        strAbsTol = 'const real_t _absolute_tolerances_[_Neqns_] = {0};'.format(self.exprFormatter.formatNumpyArray(absTolerances))
        self.modelDef.append(strAbsTol)

        # Needed for FMI code generator
        value_ref = 0

        for domain in runtimeInformation['Domains']:
            relativeName = daeGetRelativeName(self.wrapperInstanceName, domain['CanonicalName'])
            relativeName   = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(relativeName)
            description    = domain['Description']
            numberOfPoints = domain['NumberOfPoints']
            domains        = '[' + str(domain['NumberOfPoints']) + ']'
            points         = self.exprFormatter.formatNumpyArray(domain['Points']) # Numpy array

            domTemplate   = 'int {name}_np; /* Number of points in domain {name} */'
            paramTemplate = 'real_t {name}{domains}; /* {description} */ \n'
            self.parametersDefs.append(domTemplate.format(name = name))
            self.parametersDefs.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            description = description))

            domTemplate   = 'const int {name}_np = {numberOfPoints};'
            paramTemplate = 'real_t {name}{domains} = {points};'
            self.parametersInits.append(domTemplate.format(name = name,
                                                           numberOfPoints = numberOfPoints))
            self.parametersInits.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            points = points,
                                                            description = description))

            domTemplate   = '_m_->{name}_np = {name}_np;'
            paramTemplate = 'memcpy(&_m_->{name}, &{name}, {numberOfPoints} * sizeof(real_t));\n'
            self.parametersInits.append(domTemplate.format(name = name))
            self.parametersInits.append(paramTemplate.format(name = name,
                                                             numberOfPoints = numberOfPoints))
           
            #self.fmiInterface.append( ('domainPoint', name, value_ref) )
            #value_ref += 1
            
        for parameter in runtimeInformation['Parameters']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, parameter['CanonicalName'])
            relativeName   = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(relativeName)
            description    = parameter['Description']
            numberOfPoints = int(parameter['NumberOfPoints'])
            values         = self.exprFormatter.formatNumpyArray(parameter['Values']) # Numpy array
            domains = ''
            if len(parameter['Domains']) > 0:
                domains = '[{0}]'.format(']['.join(str(np) for np in parameter['Domains']))

            paramTemplate = 'real_t {name}{domains}; /* {description} */ \n'
            self.parametersDefs.append(paramTemplate.format(name = name,
                                                            domains = domains,
                                                            description = description))

            paramTemplate = 'real_t {name}{domains} = {values};'
            self.parametersInits.append(paramTemplate.format(name = name,
                                                             domains = domains,
                                                             values = values))

            paramTemplate = 'memcpy(&_m_->{name}, &{name}, {numberOfPoints} * sizeof(real_t));\n'
            self.parametersInits.append(paramTemplate.format(name = name,
                                                             numberOfPoints = numberOfPoints))

                                                             
            if numberOfPoints == 1:
                self.floatValuesReferences.append( ('parameter', name, value_ref) )
                value_ref += 1

            else:
                for i in range(0, numberOfPoints):
                    _name = '{0}[{1}]'.format(name, i)
                    self.floatValuesReferences.append( ('parameter', _name, value_ref) )
                    value_ref += 1                

        print self.floatValuesReferences
        
        for variable in runtimeInformation['Variables']:
            relativeName   = daeGetRelativeName(self.wrapperInstanceName, variable['CanonicalName'])
            formattedName  = self.exprFormatter.formatIdentifier(relativeName)
            name           = self.exprFormatter.flattenIdentifier(formattedName)
            numberOfPoints = variable['NumberOfPoints']

            if numberOfPoints == 1:
                ID           = int(variable['IDs'])        # cnDifferential, cnAssigned or cnAlgebraic
                value        = float(variable['Values'])   # numpy float
                overallIndex = variable['OverallIndex']
                fullName     = relativeName

                if ID == cnDifferential:
                    blockIndex   = indexMappings[overallIndex] + self.exprFormatter.indexBase
                    name_ = 'values[{0}]'.format(blockIndex)
                    temp = '{name} = {value}; /* {fullName} */'.format(name = name_, value = value, fullName = fullName)
                    self.initialConditions.append(temp)

                elif ID == cnAssigned:
                    temp = 'real_t {name} = {value}; /* {fullName} */'.format(name = name, value = value, fullName = fullName)
                    self.assignedVariables.append(temp)

                if ID != cnAssigned:
                    blockIndex = indexMappings[overallIndex] + self.exprFormatter.indexBase
                    self.variableNames[blockIndex] = fullName

            else:
                for i in range(0, numberOfPoints):
                    domIndexes   = tuple(variable['DomainsIndexesMap'][i])  # list of integers
                    ID           = int(variable['IDs'][domIndexes])         # cnDifferential, cnAssigned or cnAlgebraic
                    value        = float(variable['Values'][domIndexes])    # numpy float
                    overallIndex = variable['OverallIndex'] + i
                    fullName     = relativeName + '(' + ','.join(str(di) for di in domIndexes) + ')'

                    if ID == cnDifferential:
                        blockIndex   = indexMappings[overallIndex] + self.exprFormatter.indexBase
                        name_ = 'values[{0}]'.format(blockIndex)
                        temp = '{name} = {value}; /* {fullName} */'.format(name = name_, value = value, fullName = fullName)
                        self.initialConditions.append(temp)

                    elif ID == cnAssigned:
                        temp = 'real_t {name} = {value}; /* {fullName} */'.format(name = name, value = value, fullName = fullName)
                        self.assignedVariables.append(temp)

                    if ID != cnAssigned:
                        blockIndex = indexMappings[overallIndex] + self.exprFormatter.indexBase
                        self.variableNames[blockIndex] = fullName

        varNames = ['"' + name_ + '"' for name_ in self.variableNames]
        strVariableNames = 'const char* _variable_names_[_Neqns_] = {0};'.format(self.exprFormatter.formatNumpyArray(varNames))
        self.modelDef.append(strVariableNames)

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

   
