/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2013
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "model.h"

#define _v_(i)   _adouble_(_values_[i],           (i == _current_index_for_jacobian_evaluation_) ? 1.0 : 0.0)
#define _dv_(i)  _adouble_(_time_derivatives_[i], (i == _current_index_for_jacobian_evaluation_) ? _inverse_time_step_ : 0.0)
#define _time_   _adouble_(_current_time_, 0.0)

/* General info */          
%(runtimeInformation_c)s

void modInitialize(daeModel_t* _m_)
{
    %(runtimeInformation_init)s

    //memset(_m_->initValues, 0, 125 * sizeof(real_t));
    //memset(_m_->initDerivatives, 0, 125 * sizeof(real_t));
    modSetInitialConditions(_m_, _m_->initValues);
    
    %(parametersInits)s
    %(assignedVariablesInits)s
    %(stnActiveStates)s    
}

void modInitializeValuesReferences(daeModel_t* _m_, real_t* values, real_t* timeDerivatives)
{
    _m_->values          = values;
    _m_->timeDerivatives = timeDerivatives;

    /* Values references is an array of pointers that point to the values
     * of domains/parameters/DOFs/variables in the daeModel_t structure.
     * Can be used to set/get values from the model (ie. for FMI) having only
     * index of some value. */
    
    /* Integers */
    %(intValuesReferences_Init)s

    /* Floats */
    %(floatValuesReferences_Init)s
    
    /* Strings */
    %(stringValuesReferences_Init)s
}

void modSetInitialConditions(daeModel_t* _m_, real_t* values)
{
    if(_m_->quasySteadyState)
        return;

    %(initialConditions)s
}

int modResiduals(daeModel_t* _m_,
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

int modJacobian(daeModel_t* _m_,
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

int modNumberOfRoots(daeModel_t* _m_)
{
    int _noRoots_;

    _noRoots_ = 0;
    
%(numberOfRoots)s

    return _noRoots_;
}

int modRoots(daeModel_t* _m_,
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

bool modCheckForDiscontinuities(daeModel_t* _m_,
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

daeeDiscontinuityType modExecuteActions(daeModel_t* _m_,
                                        real_t _current_time_,
                                        real_t* _values_,
                                        real_t* _time_derivatives_)
{
    adouble _temp_;
    real_t _inverse_time_step_;
    daeeDiscontinuityType _discontinuity_type_;
    int _current_index_for_jacobian_evaluation_;

    _inverse_time_step_                     = 0.0;
    _current_index_for_jacobian_evaluation_ = -1;
    _discontinuity_type_                    = eModelDiscontinuityWithDataChange;

%(executeActions)s

    return _discontinuity_type_;
}

adouble modCalculateScalarExtFunction(char* fun_name, 
                                      daeModel_t* _m_,
                                      real_t _current_time_,
                                      real_t* _values_,
                                      real_t* _time_derivatives_)
{
    _log_message_("calculate_scalar_ext_function: ");    
    _log_message_(fun_name);    
    _log_message_("\\n");    
    
    if(_compare_strings_(fun_name, ""))
    {
    }
    return _adouble_(0.0, 0.0);
}

void modGetValue_float(daeModel_t* _m_, int index, real_t* value)
{
    *value = *_m_->floatValuesReferences[index];
}

void modSetValue_float(daeModel_t* _m_, int index, real_t value)
{
    *_m_->floatValuesReferences[index] = value;
}

void modGetValue_string(daeModel_t* _m_, int index, char* value)
{
    /* *value = _m_->stringValuesReferences[index]; */
}

void modSetValue_string(daeModel_t* _m_, int index, const char* value)
{
    /* _m_->stringValuesReferences[index] = value; */
}

void modGetValue_int(daeModel_t* _m_, int index, int* value)
{
    *value = *_m_->intValuesReferences[index];
}

void modSetValue_int(daeModel_t* _m_, int index, int value)
{
    *_m_->intValuesReferences[index] = value;
}
