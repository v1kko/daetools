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

#include "adouble.h"
#include "matrix.h"

#define _v_(i)   adouble_(_values_[ _indexMap_[i] ],           (i == _current_index_for_jacobian_evaluation_) ? 1.0 : 0.0)
#define _dt_(i)  adouble_(_time_derivatives_[ _indexMap_[i] ], (i == _current_index_for_jacobian_evaluation_) ? _inverse_time_step_ : 0.0)
#define _time_   adouble_(_current_time_, 0.0)

void initial_values();
int residuals(real_t _current_time_,
              real_t* _values_,
              real_t* _time_derivatives_,
              real_t* _residuals_);
int jacobian(long int _number_of_equations_,
             real_t _current_time_,
             real_t _inverse_time_step_,
             real_t* _values_,
             real_t* _time_derivatives_,
             real_t* _residuals_,
             DAE_MATRIX _jacobian_matrix_);
int number_of_roots(real_t _current_time_,
                    real_t* _values_,
                    real_t* _time_derivatives_);
int roots(real_t _current_time_,
          real_t* _values_,
          real_t* _time_derivatives_,
          real_t* _roots_);
bool check_for_discontinuities(real_t _current_time_,
                               real_t* _values_,
                               real_t* _time_derivatives_);
bool execute_actions(real_t _current_time_,
                     real_t* _values_,
                     real_t* _time_derivatives_);

/* General info */          
%(model)s

/* Domains and parameters */
%(parameters)s

/* StateTransitionNetworks */
%(activeStates)s

/* Assigned variables */
%(assignedVariables)s

void initial_values()
{
/* Initial conditions */
%(initialConditions)s
}

int residuals(real_t _current_time_,
              real_t* _values_,
              real_t* _time_derivatives_,
              real_t* _residuals_)
{
    adouble _temp_;
    int i;
    int _ec_ = 0;
    int _current_index_for_jacobian_evaluation_ = -1;
    real_t _inverse_time_step_ = 0;
    
%(residuals)s

    printf("Residal time: %%10.4e \\n", _current_time_);
    printf("_values_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _values_[i]);
    printf("\\n");
    printf("_time_derivatives_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _time_derivatives_[i]);
    printf("\\n");
    printf("_residuals_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _residuals_[i]);
    printf("\\n\\n");

    return 0;
}

int jacobian(long int _number_of_equations_,
             real_t _current_time_,
             real_t _inverse_time_step_,
             real_t* _values_,
             real_t* _time_derivatives_,
             real_t* _residuals_,
             DAE_MATRIX _jacobian_matrix_)
{
    adouble _temp_;
    int _ec_ = 0;
    int i, _block_index_;
    real_t _jacobianItem_;
    int _current_index_for_jacobian_evaluation_ = -1;
    
%(jacobian)s

    printf("Jacobian time: %%10.4e \\n", _current_time_);
    printf("InvDT: %%10.4e \\n", _inverse_time_step_);
    printf("_values_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _values_[i]);
    printf("\\n");
    printf("_time_derivatives_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _time_derivatives_[i]);
    printf("\\n");
    printf("_residuals_: \\n");
    for(i = 0; i < _Neqns_; i++)
        printf("%%12.4e", _residuals_[i]);
    printf("\\n\\n");

    daeDenseMatrix<real_t> j;
    j.InitMatrix(_Neqns_, _Neqns_, _jacobian_matrix_, eColumnWise);
    j.Print();


    return 0;
}

int number_of_roots(real_t _current_time_,
                    real_t* _values_,
                    real_t* _time_derivatives_)
{
    int _noRoots_ = 0;
    
%(numberOfRoots)s

    return _noRoots_;
}

int roots(real_t _current_time_,
          real_t* _values_,
          real_t* _time_derivatives_,
          real_t* _roots_)
{
    adouble _temp_;
    int _rc_ = 0;
    real_t _inverse_time_step_ = 0;
    int _current_index_for_jacobian_evaluation_ = -1;
    
%(roots)s

    return 0;
}

bool check_for_discontinuities(real_t _current_time_,
                               real_t* _values_,
                               real_t* _time_derivatives_)
{
    adouble _temp_;
    real_t _inverse_time_step_ = 0;
    bool foundDiscontinuity = false;
    int _current_index_for_jacobian_evaluation_ = -1;
    
%(checkForDiscontinuities)s

    return foundDiscontinuity;
}

bool execute_actions(real_t _current_time_,
                     real_t* _values_,
                     real_t* _time_derivatives_)
{
    adouble _temp_;
    real_t _inverse_time_step_ = 0;
    bool reinitializationNeeded = false;
    int _current_index_for_jacobian_evaluation_ = -1;

%(executeActions)s

    return reinitializationNeeded;
}

#endif
"""

class daeANSICExpressionFormatter(daeExpressionFormatter):
    def __init__(self):
        daeExpressionFormatter.__init__(self)

        # Details will be set in set_ANSI_C()/set_CXX() functions

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
        self.language                = 'c++'
        self.warnings                = []
        self.topLevelModel           = None
        self.simulation              = None
        self.equationGenerationMode  = ''
        
        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.residuals               = []
        self.jacobians               = []
        self.checkForDiscontinuities = []
        self.executeActions          = []
        self.numberOfRoots           = []
        self.rootFunctions           = []

        self.exprFormatter = daeANSICExpressionFormatter()
        self.analyzer      = daeCodeGeneratorAnalyzer()

    def generateSimulation(self, simulation, **kwargs):
        if not simulation:
            raise RuntimeError('Invalid simulation object')

        directory     = kwargs.get('projectDirectory', None)
        self.language = kwargs.get('language',        'c++')
        
        if self.language == 'c':
            self.set_ANSI_C()
        elif self.language == 'c++':
            self.set_CXX()
        else:
            raise RuntimeError('Invalid language')

        self.assignedVariables       = []
        self.initialConditions       = []
        self.initiallyActiveStates   = []
        self.modelDef                = []
        self.parametersDefs          = []
        self.residuals               = []
        self.jacobians               = []
        self.warnings                = []
        self.simulation              = simulation
        self.topLevelModel           = simulation.m
        self.wrapperInstanceName     = simulation.m.Name
        self.exprFormatter.modelCanonicalName = simulation.m.Name

        indent   = 1
        s_indent = indent * self.defaultIndent

        self.analyzer.analyzeSimulation(simulation)
        self.exprFormatter.IDs = self.analyzer.runtimeInformation['IDs']

        #import pprint
        #pp = pprint.PrettyPrinter(indent=2)
        #pp.pprint(self.analyzer.runtimeInformation)

        self._generateRuntimeInformation(self.analyzer.runtimeInformation)

        modelDef       = '\n'.join(self.modelDef)
        paramsDef      = '\n'.join(self.parametersDefs)
        stnDef         = '\n'.join(self.initiallyActiveStates)
        assignedVars   = '\n'.join(self.assignedVariables)
        initConds      = '\n'.join(self.initialConditions)
        eqnsRes        = '\n'.join(self.residuals)
        jacobRes       = '\n'.join(self.jacobians)
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

        code_generators_dir = os.path.join(os.path.dirname(__file__))

        # If the argument 'directory' is given create the folder and the project
        if directory:
            path, dirName = os.path.split(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

            daetools_model_h = os.path.join(directory, 'daetools_model.h')
            if self.language == 'c':
                shutil.copy2(os.path.join(code_generators_dir, 'main_c'),       os.path.join(directory, 'main.c'))
                shutil.copy2(os.path.join(code_generators_dir, 'adouble_h_c'),  os.path.join(directory, 'adouble.h'))
                shutil.copy2(os.path.join(code_generators_dir, 'adouble_c'),    os.path.join(directory, 'adouble.c'))
                shutil.copy2(os.path.join(code_generators_dir, 'matrix_h_c'),   os.path.join(directory, 'matrix.h'))
                shutil.copy2(os.path.join(code_generators_dir, 'qt_project_c'), os.path.join(directory, '{0}.pro'.format(dirName)))
            else:
                shutil.copy2(os.path.join(code_generators_dir, 'main_cxx'),       os.path.join(directory, 'main.cpp'))
                shutil.copy2(os.path.join(code_generators_dir, 'adouble_h_cxx'),  os.path.join(directory, 'adouble.h'))
                shutil.copy2(os.path.join(code_generators_dir, 'adouble_cxx'),    os.path.join(directory, 'adouble.cpp'))
                shutil.copy2(os.path.join(code_generators_dir, 'matrix_h_cxx'),   os.path.join(directory, 'matrix.h'))
                shutil.copy2(os.path.join(code_generators_dir, 'qt_project_cxx'), os.path.join(directory, '{0}.pro'.format(dirName)))

            f = open(daetools_model_h, "w")
            f.write(results)
            f.close()

        return results

    def set_CXX(self):
        # Use relative names
        self.exprFormatter.useRelativeNames         = True
        self.exprFormatter.flattenIdentifiers       = True

        self.exprFormatter.domain                   = 'adouble_({domain}[{index}])'

        self.exprFormatter.parameter                = 'adouble_({parameter}{indexes})'
        self.exprFormatter.parameterIndexStart      = '['
        self.exprFormatter.parameterIndexEnd        = ']'
        self.exprFormatter.parameterIndexDelimiter  = ']['

        self.exprFormatter.variable                 = '_v_({overallIndex})'
        self.exprFormatter.variableIndexStart       = ''
        self.exprFormatter.variableIndexEnd         = ''
        self.exprFormatter.variableIndexDelimiter   = ''

        self.exprFormatter.derivative               = '_dt_({overallIndex})'
        self.exprFormatter.derivativeIndexStart     = ''
        self.exprFormatter.derivativeIndexEnd       = ''
        self.exprFormatter.derivativeIndexDelimiter = ''

        # Constants
        self.exprFormatter.constant = 'adouble_({value}, 0)'

        # Logical operators
        self.exprFormatter.AND   = '{leftValue} && {rightValue}'
        self.exprFormatter.OR    = '{leftValue} || {rightValue}'
        self.exprFormatter.NOT   = '! {value}'

        self.exprFormatter.EQ    = '{leftValue} == {rightValue}'
        self.exprFormatter.NEQ   = '{leftValue} != {rightValue}'
        self.exprFormatter.LT    = '{leftValue} < {rightValue}'
        self.exprFormatter.LTEQ  = '{leftValue} <= {rightValue}'
        self.exprFormatter.GT    = '{leftValue} > {rightValue}'
        self.exprFormatter.GTEQ  = '{leftValue} >= {rightValue}'

        # Mathematical operators
        self.exprFormatter.SIGN   = '-{value}'

        self.exprFormatter.PLUS   = '{leftValue} + {rightValue}'
        self.exprFormatter.MINUS  = '{leftValue} - {rightValue}'
        self.exprFormatter.MULTI  = '{leftValue} * {rightValue}'
        self.exprFormatter.DIVIDE = '{leftValue} / {rightValue}'
        self.exprFormatter.POWER  = 'pow({leftValue}, {rightValue})'

        # Mathematical functions
        self.exprFormatter.SIN    = 'sin({value})'
        self.exprFormatter.COS    = 'cos({value})'
        self.exprFormatter.TAN    = 'tan({value})'
        self.exprFormatter.ASIN   = 'asin({value})'
        self.exprFormatter.ACOS   = 'acos({value})'
        self.exprFormatter.ATAN   = 'atan({value})'
        self.exprFormatter.EXP    = 'exp({value})'
        self.exprFormatter.SQRT   = 'sqrt({value})'
        self.exprFormatter.LOG    = 'log({value})'
        self.exprFormatter.LOG10  = 'log10({value})'
        self.exprFormatter.FLOOR  = 'floor({value})'
        self.exprFormatter.CEIL   = 'ceil({value})'
        self.exprFormatter.ABS    = 'abs({value})'

        self.exprFormatter.MIN    = 'min({leftValue}, {rightValue})'
        self.exprFormatter.MAX    = 'max({leftValue}, {rightValue})'

        # Current time in simulation
        self.exprFormatter.TIME   = '_time_'

    def set_ANSI_C(self):
        # Use relative names
        self.exprFormatter.useRelativeNames   = True
        self.exprFormatter.flattenIdentifiers = True

        self.exprFormatter.domain                   = 'adouble_({domain}[{index}], 0)'

        self.exprFormatter.parameter                = 'adouble_({parameter}{indexes}, 0)'
        self.exprFormatter.parameterIndexStart      = '['
        self.exprFormatter.parameterIndexEnd        = ']'
        self.exprFormatter.parameterIndexDelimiter  = ']['

        self.exprFormatter.variable                 = '_v_({overallIndex})'
        self.exprFormatter.variableIndexStart       = ''
        self.exprFormatter.variableIndexEnd         = ''
        self.exprFormatter.variableIndexDelimiter   = ''

        self.exprFormatter.derivative               = '_dt_({overallIndex})'
        self.exprFormatter.derivativeIndexStart     = ''
        self.exprFormatter.derivativeIndexEnd       = ''
        self.exprFormatter.derivativeIndexDelimiter = ''

        # Constants
        self.exprFormatter.constant = 'adouble_({value}, 0)'

        # Logical operators
        self.exprFormatter.AND   = 'AND({leftValue}, {rightValue})'
        self.exprFormatter.OR    = 'OR({leftValue}, {rightValue})'
        self.exprFormatter.NOT   = 'NOT({value})'

        self.exprFormatter.EQ    = 'EQ({leftValue}, {rightValue})'
        self.exprFormatter.NEQ   = 'NEQ({leftValue}, {rightValue})'
        self.exprFormatter.LT    = 'LT({leftValue}, {rightValue})'
        self.exprFormatter.LTEQ  = 'LTEQ({leftValue}, {rightValue})'
        self.exprFormatter.GT    = 'GT({leftValue}, {rightValue})'
        self.exprFormatter.GTEQ  = 'GTEQ({leftValue}, {rightValue})'

        # Mathematical operators
        self.exprFormatter.SIGN   = 'SIGN({value})'

        self.exprFormatter.PLUS   = 'PLUS({leftValue}, {rightValue})'
        self.exprFormatter.MINUS  = 'MINUS({leftValue}, {rightValue})'
        self.exprFormatter.MULTI  = 'MULTI({leftValue}, {rightValue})'
        self.exprFormatter.DIVIDE = 'DIVIDE({leftValue}, {rightValue})'
        self.exprFormatter.POWER  = 'pow_({leftValue}, {rightValue})'

        # Mathematical functions
        self.exprFormatter.SIN    = 'sin_({value})'
        self.exprFormatter.COS    = 'cos_({value})'
        self.exprFormatter.TAN    = 'tan_({value})'
        self.exprFormatter.ASIN   = 'asin_({value})'
        self.exprFormatter.ACOS   = 'acos_({value})'
        self.exprFormatter.ATAN   = 'atan_({value})'
        self.exprFormatter.EXP    = 'exp_({value})'
        self.exprFormatter.SQRT   = 'sqrt_({value})'
        self.exprFormatter.LOG    = 'log_({value})'
        self.exprFormatter.LOG10  = 'log10_({value})'
        self.exprFormatter.FLOOR  = 'floor_({value})'
        self.exprFormatter.CEIL   = 'ceil_({value})'
        self.exprFormatter.ABS    = 'abs_({value})'

        self.exprFormatter.MIN    = 'min_({leftValue}, {rightValue})'
        self.exprFormatter.MAX    = 'max_({leftValue}, {rightValue})'

        # Current time in simulation
        self.exprFormatter.TIME   = '_time_'

    def _processEquations(self, Equations, indent):
        s_indent  = indent     * self.defaultIndent
        s_indent2 = (indent+1) * self.defaultIndent
        
        if self.equationGenerationMode == 'residuals':
            for equation in Equations:
                for eeinfo in equation['EquationExecutionInfos']:
                    res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                    self.residuals.append(s_indent + '_temp_ = {0};'.format(res))
                    self.residuals.append(s_indent + '_residuals_[_ec_++] = getValue(&_temp_);')

        elif self.equationGenerationMode == 'jacobian':
            for equation in Equations:
                for eeinfo in equation['EquationExecutionInfos']:
                    overall_indexes = eeinfo['VariableIndexes']
                    n = len(overall_indexes)
                    ID = len(self.jacobians)
                    str_indexes = self.exprFormatter.formatNumpyArray(overall_indexes)
                    self.jacobians.append(s_indent + 'int _overall_indexes_{0}[{1}] = {2};'.format(ID, n, str_indexes))
                    self.jacobians.append(s_indent + 'for(i = 0; i < {0}; i++) {{'.format(n))
                    self.jacobians.append(s_indent2 + '_block_index_ = _indexMap_[ _overall_indexes_{0}[i] ];'.format(ID))
                    self.jacobians.append(s_indent2 + '_current_index_for_jacobian_evaluation_ = _overall_indexes_{0}[i];'.format(ID))
                    
                    res = self.exprFormatter.formatRuntimeNode(eeinfo['ResidualRuntimeNode'])
                    self.jacobians.append(s_indent2 + '_temp_ = {0};'.format(res))
                    self.jacobians.append(s_indent2 + '_jacobianItem_ = getDerivative(&_temp_);')
                    #                                            SetItem(Eq.index, blockIndex, value)
                    self.jacobians.append(s_indent2 + 'setMatrixItem(_jacobian_matrix_, _ec_, _block_index_, _jacobianItem_);')

                    self.jacobians.append(s_indent + '}')
                    self.jacobians.append(s_indent + '_ec_++;')

    def _processSTNs(self, STNs, indent):
        s_indent = indent * self.defaultIndent
        if self.language == 'c':
            string_def = 'char*'
        elif self.language == 'c++':
            string_def = 'std::string'
        
        for stn in STNs:
            nStates = len(stn['States'])
            if stn['Class'] == 'daeIF':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_ifstn'
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                varTemplate = '{string_def} {name} = "{activeState}"; /* States: {states}; {description} */ \n'
                self.initiallyActiveStates.append(varTemplate.format(string_def = string_def,
                                                                     name = stnVariableName,
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

                        temp = s_indent + 'if({0}) {{'.format(condition)
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        state_transition = state['StateTransitions'][0]
                        condition = self.exprFormatter.formatRuntimeConditionNode(state_transition['ConditionRuntimeNode'])

                        temp = s_indent + 'else if({0}) {{'.format(condition)
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
                    self.checkForDiscontinuities.append(s_indent2 + 'if(compareStrings({0}, "{1}")) {{'.format(stnVariableName, state['Name']))
                    self.checkForDiscontinuities.append(s_indent3 + 'foundDiscontinuity = true;')
                    self.checkForDiscontinuities.append(s_indent2 + '}')
                    self.checkForDiscontinuities.append(s_indent + '}')

                    # 3. executeActions
                    self.executeActions.append(s_indent2 + '{0} = "{1}";'.format(stnVariableName, state['Name']))
                    self.executeActions.append(s_indent + '}')

                    # 4. numberOfRoots
                    if state_transition: # For 'else' state has no state transitions
                        nExpr = len(state_transition['Expressions'])
                        self.numberOfRoots.append(s_indent2 + '_noRoots_ += {0};'.format(nExpr))

                    self.numberOfRoots.append(s_indent + '}')

                    # 5. rootFunctions
                    if state_transition: # For 'else' state has no state transitions
                        for expression in state_transition['Expressions']:
                            self.rootFunctions.append(s_indent2 + '_temp_ = {0};'.format(self.exprFormatter.formatRuntimeNode(expression)))
                            self.rootFunctions.append(s_indent2 + '_roots_[_rc_++] = getValue(&_temp_);')

                    self.rootFunctions.append(s_indent + '}')

                    if len(state['NestedSTNs']) > 0:
                        raise RuntimeError('Nested state transition networks (daeIF) canot be exported to c')

            elif stn['Class'] == 'daeSTN':
                relativeName    = daeGetRelativeName(self.wrapperInstanceName, stn['CanonicalName'])
                relativeName    = self.exprFormatter.formatIdentifier(relativeName)
                stnVariableName = self.exprFormatter.flattenIdentifier(relativeName) + '_stn'
                description     = stn['Description']
                states          = ', '.join(st['Name'] for st in stn['States'])
                activeState     = stn['ActiveState']

                varTemplate = '{string_def} {name} = "{activeState}"; /* States: {states}; {description} */ \n'
                self.initiallyActiveStates.append(varTemplate.format(string_def = string_def,
                                                                     name = stnVariableName,
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

                        temp = s_indent + 'if(compareStrings({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
                        self.residuals.append(temp)
                        self.jacobians.append(temp)
                        self.checkForDiscontinuities.append(temp)
                        self.executeActions.append(temp)
                        self.numberOfRoots.append(temp)
                        self.rootFunctions.append(temp)

                    elif (i > 0) and (i < nStates - 1):
                        temp = s_indent + 'else if(compareStrings({0}, "{1}")) {{'.format(stnVariableName, state['Name'])
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
                            self.rootFunctions.append(s_indent2 + '_roots_[_rc_++] = getValue(&_temp_);')

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

        Nvars = '#define _Ntotal_vars_ {0}'.format(Ntotal)
        self.modelDef.append(Nvars)

        Nvars = '#define _Nvars_ {0}'.format(N)
        self.modelDef.append(Nvars)

        Neqns = '#define _Neqns_ {0}'.format(Neq)
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
            self.residuals.append(s_indent + '/* Port connection: {0} -> {1} */'.format(port_connection['PortFrom'], port_connection['PortTo']))
            self._processEquations(port_connection['Equations'], indent)

        self.residuals.append(s_indent + '/* Equations */')
        self._processEquations(runtimeInformation['Equations'], indent)

        # Finally generate together residuals and jacobians for STNs
        # _processSTNs will take care of self.equationGenerationMode regime
        self._processSTNs(runtimeInformation['STNs'], indent)

   
