/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
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
#include "adouble.h"
#include "mpi_sync.h"
#include "runtime_information.h"
#include <boost/container/flat_map.hpp>
using boost::container::flat_map;

// These are internal storage containers for MPI communication
std::vector<real_t> foreignValues;
std::vector<real_t> foreignTimeDerivatives;

// These two point to the IDAS vectors yy/yp(for owned indexes) and to foreignValues/TimeDerivatives (for foreign indexes)
flat_map<int, real_t*> _mapValues_;
flat_map<int, real_t*> _mapTimeDerivatives_;

// These macros now use the above boost::flat_maps (before the used std::vectors).
// The variables in equations still use the block indexes but they are now mapped using boost::flat_maps;
//   that's how we can still use block indexes.
#define _v_(i)   adouble(*_mapValues_[i],          (i == _current_index_for_jacobian_evaluation_) ? 1.0 : 0.0)
#define _dv_(i)  adouble(*_mapTimeDerivatives_[i], (i == _current_index_for_jacobian_evaluation_) ? _inverse_time_step_ : 0.0)
#define _time_   adouble(_current_time_, 0.0)

void modInitialize(daeModel_t* _m_)
{
    // Get synchronisation info for the current node
    runtimeInformationData rtnd = mapRuntimeInformationData.at(_m_->mpi_rank);

    %(runtimeInformation_init)s

/*  Achtung, Achtung!!
    This part did not work!!!
    _m_->IDs                = rtnd.ids.data(); // std:vector<TYPE> is c++11 feature
    _m_->initValues         = rtnd.init_values.data();
    _m_->initDerivatives    = rtnd.init_derivatives.data();
    _m_->absoluteTolerances = rtnd.absolute_tolerances.data();
*/
    _m_->IDs                = new int   [_m_->Nequations_local];
    _m_->initValues         = new real_t[_m_->Nequations_local];
    _m_->initDerivatives    = new real_t[_m_->Nequations_local];
    _m_->absoluteTolerances = new real_t[_m_->Nequations_local];
    for(int i = 0; i < _m_->Nequations_local; i++)
    {
        _m_->IDs[i]                = rtnd.ids[i]; // std:vector<TYPE>::data() is c++11 feature
        _m_->initValues[i]         = rtnd.init_values[i];
        _m_->initDerivatives[i]    = rtnd.init_derivatives[i];
        _m_->absoluteTolerances[i] = rtnd.absolute_tolerances[i];
    }
    // _m_->variableNames should be allocated (using the local node Nequation or the (i_start:i_end) range)
    //for(int i = 0; i < _m_->Nequations; i++)
    //    _m_->variableNames[i] = rtnd.variable_names[i].c_str();

    /* This function is redundant - all initial conditions are already set in std::map<int,runtimeInformationData>,
       provided that the model is generated after the call to simulation.Initialize() (which is always the case). */
    modSetInitialConditions(_m_, _m_->initValues);

    %(parametersInits)s
    %(assignedVariablesInits)s
    %(stnActiveStates)s
}

void modFinalize(daeModel_t* _m_)
{
}

void modInitializeValuesReferences(daeModel_t* _m_, real_t* values, real_t* timeDerivatives)
{
    mpiIndexesData indData = mapIndexesData.at(_m_->mpi_rank);

// These are not needed anymore
    delete[] _m_->IDs;
    delete[] _m_->initValues;
    delete[] _m_->initDerivatives;
    delete[] _m_->absoluteTolerances;
    _m_->IDs = NULL;
    _m_->initValues = NULL;
    _m_->initDerivatives = NULL;
    _m_->absoluteTolerances = NULL;

    // Reserve the size for internal vectors
    int fi_size = indData.foreign_indexes.size();
    foreignValues.resize(fi_size, 0.0);
    foreignTimeDerivatives.resize(fi_size, 0.0);

    int tot_size = (indData.i_end - indData.i_start) + indData.foreign_indexes.size();
    _mapValues_.reserve(tot_size);
    _mapTimeDerivatives_.reserve(tot_size);

    // Insert the pointers for the owned indexes (they point to the items in IDAS vectors)
    for(int block_index = indData.i_start, node_index = 0; block_index < indData.i_end; block_index++, node_index++)
    {
        _mapValues_[block_index]          = &values[node_index];
        _mapTimeDerivatives_[block_index] = &timeDerivatives[node_index];
    }

    // Insert the pointers for the foreign indexes (they point to the items in internal vectors)
    int foreign_block_index;
    for(int i = 0; i < indData.foreign_indexes.size(); i++)
    {
        foreign_block_index = indData.foreign_indexes[i];

        _mapValues_[foreign_block_index]          = &foreignValues[i];
        _mapTimeDerivatives_[foreign_block_index] = &foreignTimeDerivatives[i];
    }

    // Initialize pointer maps
    for(mpiSyncMap::iterator it = indData.send_to.begin(); it != indData.send_to.end(); it++)
    {
        // it->first is int (mpi_rank)
        // it->second is vector<int>
        int mpi_rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values(i_size, 0.0),     derivs(i_size, 0.0);
        std::vector<real_t*> pvalues(i_size, nullptr), pderivs(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues[i] = _mapValues_         [ indexes[i] ];
            pderivs[i] = _mapTimeDerivatives_[ indexes[i] ];
        }

        mapValuesData.send_to[mpi_rank]   = make_pair(values,  derivs);
        mapPointersData.send_to[mpi_rank] = make_pair(pvalues, pderivs);
    }

    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
    {
        // it->first is int (mpi_rank)
        // it->second is vector<int>
        int mpi_rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values(i_size, 0.0),     derivs(i_size, 0.0);
        std::vector<real_t*> pvalues(i_size, nullptr), pderivs(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues[i] = _mapValues_         [ indexes[i] ];
            pderivs[i] = _mapTimeDerivatives_[ indexes[i] ];
        }

        mapValuesData.receive_from[mpi_rank]   = make_pair(values,  derivs);
        mapPointersData.receive_from[mpi_rank] = make_pair(pvalues, pderivs);
    }

    CheckSynchronisationIndexes(_m_, _m_->mpi_world, _m_->mpi_rank);

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
    /* This function is redundant - all initial conditions are already set in std::map<int,runtimeInformationData> */
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

/*
    if(_m_->mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("res-node-") + std::to_string(_m_->mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out|std::ofstream::app);

        flat_map<int, real_t*>::const_iterator iter;
        ofs << "_mapValues_ " << _mapValues_.size() << std::endl;
        for(iter = _mapValues_.begin(); iter != _mapValues_.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "_mapTimeDerivatives_ " << _mapTimeDerivatives_.size() << std::endl;
        for(iter = _mapTimeDerivatives_.begin(); iter != _mapTimeDerivatives_.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "_residuals_ " << _m_->Nequations_local << std::endl;
        for(int i = 0; i < _m_->Nequations_local; i++)
            ofs << "["<< i << ":" << _residuals_[i] << "]";
        ofs << std::endl;
    }
*/
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
    int _i_, _ec_, _block_index_, _block_index_local_, _current_index_for_jacobian_evaluation_;

    mpiIndexesData _ind_Data_ = mapIndexesData.at(_m_->mpi_rank);
    int i_start_ = _ind_Data_.i_start;
    int i_end_   = _ind_Data_.i_end;
    std::map<int,int>& _map_bi_to_bi_local_ = _ind_Data_.bi_to_bi_local;

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

real_t modCalculateScalarExtFunction(char* fun_name,
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
    return 0; //_adouble_(0.0, 0.0);
}

void modGetValue_float(daeModel_t* _m_, int index, real_t* value)
{
    /* *value = *_m_->floatValuesReferences[index]; */
}

void modSetValue_float(daeModel_t* _m_, int index, real_t value)
{
    /* *_m_->floatValuesReferences[index] = value; */
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
    /* *value = *_m_->intValuesReferences[index]; */
}

void modSetValue_int(daeModel_t* _m_, int index, int value)
{
    /* *_m_->intValuesReferences[index] = value; */
}
