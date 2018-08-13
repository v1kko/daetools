/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_MODEL_H
#define CS_MODEL_H

#include <string>
#include <vector>
#include <map>
#include "cs_machine.h"
#include "cs_evaluator.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_Models_EXPORTS
#define OPENCS_MODELS_API __declspec(dllexport)
#else
#define OPENCS_MODELS_API __declspec(dllimport)
#endif
#else
#define OPENCS_MODELS_API
#endif

namespace cs
{
enum csModelType
{
    eSteadyState = 0,
    eDAE,
    eExplicitODE,
    eImplicitODE
};

enum csModelInputFileType
{
    eInputFile_ModelStructure = 0,
    eInputFile_ModelEquations,
    eInputFile_SparsityPattern,
    eInputFile_PartitionData
};

/* Partition data (used for inter-process communication in parallel simulations). */
typedef std::map< int32_t, std::vector<int32_t> > csPartitionIndexMap;
struct OPENCS_MODELS_API csPartitionData_t
{
    std::vector<int32_t>      foreignIndexes;
    std::map<int32_t,int32_t> biToBiLocal;
    csPartitionIndexMap       sendToIndexes;
    csPartitionIndexMap       receiveFromIndexes;
};

/* Model structure. */
struct OPENCS_MODELS_API csModelStructure_t
{
    uint32_t                 Nequations;       /* Number of variables in this Processing Element */
    uint32_t                 Nequations_total; /* Number of variables */
    uint32_t                 Ndofs;
    bool                     isODESystem;
    std::vector<real_t>      dofValues;
    std::vector<real_t>      variableValues;
    std::vector<real_t>      variableDerivatives;
    std::vector<std::string> variableNames;
    std::vector<int32_t>     variableTypes;
    std::vector<real_t>      absoluteTolerances;
};

/* Model equations (ComputeStack-related data). */
struct OPENCS_MODELS_API csModelEquations_t
{
    std::vector<csComputeStackItem_t>   computeStacks;
    std::vector<uint32_t>               activeEquationSetIndexes;
};

/* The sparsity pattern (in compressed row storage format). */
struct OPENCS_MODELS_API csSparsityPattern_t
{
    uint32_t                             Nnz;
    uint32_t                             Nequations;
    std::vector<uint32_t>                rowIndexes;
    std::vector<csIncidenceMatrixItem_t> incidenceMatrixItems;
};

class OPENCS_MODELS_API csModel_t
{
public:
    csModel_t();
    virtual ~csModel_t();

    void LoadModel(const std::string& inputDirectory);
    void SaveModel(const std::string& outputDirectory);

    /* Model structure. */
    csModelStructure_t structure;

    /* Model equations. */
    csModelEquations_t equations;

    /* Sparsity pattern. */
    csSparsityPattern_t sparsityPattern;

    /* Partition data. */
    csPartitionData_t partitionData;

    /* Compute Stack Evaluator. */
    csComputeStackEvaluator_t* csEvaluator;

    /* MPI-related data. */
    int    pe_rank;

    /* Names of CS input files. */
    constexpr static const char* inputFileNameTemplate     = "%s-%05d.csdata";
    constexpr static const char* modelEquationsFileName    = "model_equations";
    constexpr static const char* modelStructureFileName    = "model_structure";
    constexpr static const char* partitionDataFileName     = "partition_data";
    constexpr static const char* sparsityPatternFileName   = "sparsity_pattern";
    constexpr static const char* simulationOptionsFileName = "simulation_options.json";
};

/* A generic interface to a sparse matrix storage. */
class csMatrixAccess_t
{
public:
    virtual ~csMatrixAccess_t(){}

    virtual void SetItem(size_t row, size_t col, real_t value) = 0;
};

/* Discontinuity types. */
typedef enum
{
    eDCTUnknown = 0,
    eGlobalDiscontinuity,
    eModelDiscontinuity,
    eModelDiscontinuityWithDataChange,
    eNoDiscontinuity
} csDiscontinuityType;

/* Common ODE/DAE Model class interface. */
class OPENCS_MODELS_API csDifferentialEquationModel_t : public csModel_t
{
public:
    virtual ~csDifferentialEquationModel_t(){}

    virtual void Load(const std::string& inputDirectory, csComputeStackEvaluator_t* csEvaluator) = 0;
    virtual void Load(const csModel_t* csModel, csComputeStackEvaluator_t* csEvaluator) = 0;

    virtual void Free() = 0;

    virtual void GetSparsityPattern(int& N,
                                    int& NNZ,
                                    std::vector<int>& IA,
                                    std::vector<int>& JA) = 0;

    virtual void EvaluateEquations(real_t time, real_t* equations) = 0;

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
