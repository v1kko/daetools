/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_MODEL_H
#define CS_MODEL_H

#include <string>
#include <vector>
#include <map>
#include "cs_machine.h"
#include "cs_evaluator.h"

namespace cs
{
/* Partition data (used for inter-process communication in parallel simulations). */
typedef std::map< int32_t, std::vector<int32_t> > csPartitionIndexMap;
struct csPartitionData_t
{
    std::vector<int32_t>      foreignIndexes;
    std::map<int32_t,int32_t> biToBiLocal;
    csPartitionIndexMap       sendToIndexes;
    csPartitionIndexMap       receiveFromIndexes;
};

class csModel_t
{
public:
    virtual ~csModel_t(){}

    /* Model structure (variabless-related data). */
    uint32_t Nequations_local; /* Number of equations/state variables in this processing element */
    uint32_t Nequations;       /* Number of equations/state variables */
    uint32_t Ndofs;
    bool     quasiSteadyState; /* This belongs to the simulation interface!! */

    std::vector<real_t>      dofs;
    std::vector<real_t>      variableValues;
    std::vector<real_t>      variableDerivatives;
    std::vector<real_t>      absoluteTolerances;
    std::vector<int32_t>     ids;
    std::vector<std::string> variableNames;

    /* Model equations (ComputeStack-related data). */
    std::vector<csComputeStackItem_t>   computeStacks;
    std::vector<csJacobianMatrixItem_t> jacobianMatrixItems;
    std::vector<uint32_t>               activeEquationSetIndexes;

    /* Compute Stack Evaluator. */
    csComputeStackEvaluator_t* csEvaluator;

    /* Partition data. */
    csPartitionData_t partitionData;

    /* MPI-related data. */
    int    pe_rank;
    void*  mpi_world;
    void*  mpi_comm;
};

class csMatrixAccess_t
{
public:
    virtual ~csMatrixAccess_t(){}

    virtual void SetItem(size_t row, size_t col, real_t value) = 0;
};

typedef enum
{
    eDCTUnknown = 0,
    eGlobalDiscontinuity,
    eModelDiscontinuity,
    eModelDiscontinuityWithDataChange,
    eNoDiscontinuity
} csDiscontinuityType;


class csDAEModel_t : public csModel_t
{
public:
    virtual ~csDAEModel_t(){}

    virtual void Load(const std::string& inputDirectory, csComputeStackEvaluator_t* csEvaluator) = 0;

    virtual void Free() = 0;

    virtual void GetDAESystemStructure(int& N,
                                       int& NNZ,
                                       std::vector<int>& IA,
                                       std::vector<int>& JA) = 0;

    virtual void EvaluateResiduals(real_t time, real_t* residuals) = 0;

    virtual void EvaluateJacobian(real_t time, real_t inverseTimeStep, csMatrixAccess_t* ma) = 0;

    virtual void SetAndSynchroniseData(real_t  time,
                                       real_t* values,
                                       real_t* time_derivatives) = 0;
    virtual int NumberOfRoots() = 0;
    virtual void Roots(real_t time, real_t* values, real_t* time_derivatives, real_t* roots) = 0;
    virtual bool CheckForDiscontinuities(real_t time, real_t* values, real_t* time_derivatives) = 0;
    virtual csDiscontinuityType ExecuteActions(real_t time, real_t* values, real_t* time_derivatives) = 0;
};

}

#endif
