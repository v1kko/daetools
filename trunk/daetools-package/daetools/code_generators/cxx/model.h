/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2016
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

#include "auxiliary.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

/* C++ CODE GENERATOR WARNINGS!!!
%(warnings)s
*/

typedef struct
{
    /* MPI-related data */
    void*   mpi_world;
    int     mpi_rank;
    void*   mpi_comm;
    long    Nequations_local; /* Number of equations/state variables in the local node (MPI) */

    long    Ntotal_vars;      /* Total number of variables (including Degrees of Freedom) */
    long    Nequations;       /* Number of equations/state variables */
    real_t  startTime;
    real_t  timeHorizon;
    real_t  reportingInterval;
    real_t  relativeTolerance;
    bool    quasySteadyState;

    int*         IDs;
    real_t*      initValues;
    real_t*      initDerivatives;
    real_t*      absoluteTolerances;
    const char** variableNames;

    %(intValuesReferences_Def)s
    %(floatValuesReferences_Def)s
    %(stringValuesReferences_Def)s

    /* Domains and parameters */
    %(parameters)s

    /* Assigned variables (Degrees of Freedom) */
    %(assignedVariablesDefs)s

    /* State Transition Networks */
    %(stns)s

} daeModel_t;

void modInitialize(daeModel_t* _m_);
void modFinalize(daeModel_t* _m_);
void modInitializeValuesReferences(daeModel_t* _m_, real_t* values, real_t* timeDerivatives);
void modSetInitialConditions(daeModel_t* _m_, real_t* values);
void modGetValue_float(daeModel_t* _m_, int index, real_t* value);
void modSetValue_float(daeModel_t* _m_, int index, real_t value);
void modGetValue_string(daeModel_t* _m_, int index, char* value);
void modSetValue_string(daeModel_t* _m_, int index, const char* value);
void modGetValue_int(daeModel_t* _m_, int index, int* value);
void modSetValue_int(daeModel_t* _m_, int index, int value);
void modSynchroniseData(daeModel_t* _m_);

int modResiduals(daeModel_t* _m_,
                 real_t _current_time_,
                 real_t* _values_,
                 real_t* _time_derivatives_,
                 real_t* _residuals_);
int modJacobian(daeModel_t* _m_,
                long int _number_of_equations_,
                real_t _current_time_,
                real_t _inverse_time_step_,
                real_t* _values_,
                real_t* _time_derivatives_,
                real_t* _residuals_,
                matrix_t _jacobian_matrix_);
int modNumberOfRoots(daeModel_t* _m_);
int modRoots(daeModel_t* _m_,
             real_t _current_time_,
             real_t* _values_,
             real_t* _time_derivatives_,
             real_t* _roots_);
bool modCheckForDiscontinuities(daeModel_t* _m_,
                                real_t _current_time_,
                                real_t* _values_,
                                real_t* _time_derivatives_);
daeeDiscontinuityType modExecuteActions(daeModel_t* _m_,
                                        real_t _current_time_,
                                        real_t* _values_,
                                        real_t* _time_derivatives_);
real_t modCalculateScalarExtFunction(char* fun_name,
                                     daeModel_t* _m_,
                                     real_t _current_time_,
                                     real_t* _values_,
                                     real_t* _time_derivatives_);

// #ifdef __cplusplus
// }
// #endif

#endif
