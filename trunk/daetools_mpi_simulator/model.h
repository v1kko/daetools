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
#ifndef DAE_MODEL_H
#define DAE_MODEL_H

#include "auxiliary.h"

typedef struct daeModel_
{
    /* MPI-related data */
    void*        mpi_world;
    int          mpi_rank;
    void*        mpi_comm;

    long         Nequations_local; /* Number of equations/state variables in the local node (MPI) */
    long         Ntotal_vars;      /* Total number of variables (including Degrees of Freedom) */
    long         Nequations;       /* Number of equations/state variables */
    long         Ndofs;
    real_t       startTime;
    real_t       timeHorizon;
    real_t       reportingInterval;
    real_t       relativeTolerance;
    bool         quasiSteadyState;

    int*         ids;
    real_t*      initValues;
    real_t*      initDerivatives;
    real_t*      absoluteTolerances;
    const char** variableNames;
    std::string  inputDirectory;
} daeModel_t;

void modInitialize(daeModel_t* model, const std::string& inputDirectory);
void modFinalize(daeModel_t* model);
void modInitializeValuesReferences(daeModel_t* model);

void mpiCheckSynchronisationIndexes(daeModel_t* model, void* mpi_world, int mpi_rank);
void mpiSynchroniseData(daeModel_t* model, real_t* values, real_t* time_derivatives);

int modResiduals(daeModel_t* model,
                 real_t current_time,
                 real_t* values,
                 real_t* time_derivatives,
                 real_t* residuals);
int modJacobian(daeModel_t* model,
                long int number_of_equations,
                real_t current_time,
                real_t inverse_time_step,
                real_t* values,
                real_t* time_derivatives,
                real_t* residuals,
                void* matrix);
int modNumberOfRoots(daeModel_t* model);
int modRoots(daeModel_t* model,
             real_t  current_time,
             real_t* values,
             real_t* time_derivatives,
             real_t* roots);
bool modCheckForDiscontinuities(daeModel_t* model,
                                real_t  current_time,
                                real_t* values,
                                real_t* time_derivatives);
daeeDiscontinuityType modExecuteActions(daeModel_t* model,
                                        real_t  current_time,
                                        real_t* values,
                                        real_t* time_derivatives);

#endif
