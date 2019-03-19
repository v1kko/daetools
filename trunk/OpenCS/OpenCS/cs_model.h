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

#include <stdint.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <memory>
#include "cs_machine.h"
#include "cs_evaluator.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_MODELS_EXPORTS
#define OPENCS_MODELS_API __declspec(dllexport)
#else
#define OPENCS_MODELS_API __declspec(dllimport)
#endif
#else
#define OPENCS_MODELS_API
#endif

#define csThrowException(MSG) \
    throw std::runtime_error(std::string("Exception in ") + std::string(__FUNCTION__) + " (" + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "):\n" + std::string(MSG) + "\n");

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
    std::vector<std::string> dofNames;
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

    void LoadModel(int rank, const std::string& inputDirectory);
    void SaveModel(const std::string& outputDirectory);
    void Free();

    /* Model structure. */
    csModelStructure_t structure;

    /* Model equations. */
    csModelEquations_t equations;

    /* Sparsity pattern. */
    csSparsityPattern_t sparsityPattern;

    /* Partition data. */
    csPartitionData_t partitionData;

    /* Compute Stack Evaluator. */
    csComputeStackEvaluatorPtr csEvaluator;

    /* MPI-related data. */
    int pe_rank;

    /* Names of CS input files. */
    constexpr static const char* inputFileNameTemplate     = "%s-%05d.csdata";
    constexpr static const char* modelEquationsFileName    = "model_equations";
    constexpr static const char* modelStructureFileName    = "model_structure";
    constexpr static const char* partitionDataFileName     = "partition_data";
    constexpr static const char* sparsityPatternFileName   = "sparsity_pattern";
    constexpr static const char* simulationOptionsFileName = "simulation_options.json";
};
typedef std::shared_ptr<csModel_t> csModelPtr;

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
class OPENCS_MODELS_API csDifferentialEquationModel_t
{
public:
    virtual ~csDifferentialEquationModel_t(){}

    virtual void Load(int rank, const std::string& inputDirectory) = 0;
    virtual void Load(int rank, csModelPtr model) = 0;

    virtual void SetComputeStackEvaluator(csComputeStackEvaluatorPtr evaluator) = 0;

    virtual void Free() = 0;

    virtual csModelPtr GetModel() = 0;

    virtual void GetSparsityPattern(int& N,
                                    int& NNZ,
                                    std::vector<int>& IA,
                                    std::vector<int>& JA) = 0;

    virtual void EvaluateEquations(real_t time, real_t* equations) = 0;

    virtual void EvaluateJacobian(real_t time, real_t inverseTimeStep, csMatrixAccess_t* ma) = 0;

    virtual void SetDegreesOfFreedom(real_t* dofs) = 0;
    virtual void SetAndSynchroniseData(real_t  time,
                                       real_t* values,
                                       real_t* time_derivatives) = 0;

    virtual int NumberOfRoots() = 0;
    virtual void Roots(real_t time, real_t* values, real_t* time_derivatives, real_t* roots) = 0;
    virtual bool CheckForDiscontinuities(real_t time, real_t* values, real_t* time_derivatives) = 0;
    virtual csDiscontinuityType ExecuteActions(real_t time, real_t* values, real_t* time_derivatives) = 0;
};

class OPENCS_MODELS_API csLog_t
{
public:
    virtual ~csLog_t(){}

public:
    virtual std::string	GetName(void) const                    = 0;
    virtual bool        Connect(int rank)                      = 0;
    virtual void        Disconnect()                           = 0;
    virtual bool        IsConnected()                          = 0;
    virtual void		Message(const std::string& strMessage) = 0;
};
typedef std::shared_ptr<csLog_t> csLogPtr;

class OPENCS_MODELS_API csDataReporter_t
{
public:
    virtual ~csDataReporter_t(){}

public:
    virtual bool Connect(int rank)                                                = 0;
    virtual bool IsConnected()                                                    = 0;
    virtual bool Disconnect()                                                     = 0;
    virtual bool RegisterVariables(const std::vector<std::string>& variableNames) = 0;
    virtual bool StartNewResultSet(real_t time)	                                  = 0;
    virtual bool EndOfData()                                                      = 0;
    virtual bool SendVariables(const real_t* values, const size_t n)              = 0;
    virtual bool SendDerivatives(const real_t* derivatives, const size_t n)       = 0;
};
typedef std::shared_ptr<csDataReporter_t> csDataReporterPtr;

class OPENCS_MODELS_API csGraphPartitioner_t
{
public:
    virtual ~csGraphPartitioner_t(){}

    virtual std::string GetName() = 0;

/* Arguments:
 *   [in]  Npe:                                   Number of partitions to create
 *   [in]  Nvertices:                             Number of graph vertices
 *   [in]  Nconstraints:                          Number of balancing constraints
 *   [in]  rowIndices, colIndices:                The system's incidence matrix (in CRS format)
 *   [in]  vertexWeights[Nconstraints,Nvertices]: 2D array with weights for every vertex
 *   [out] partitions[Npe,]:                      2D array of Npe sets with equation indexes
 */
    virtual int Partition(int32_t                               Npe,           /* [in] */
                          int32_t                               Nvertices,     /* [in] */
                          int32_t                               Nconstraints,  /* [in] */
                          std::vector<uint32_t>&                rowIndices,    /* [in] */
                          std::vector<uint32_t>&                colIndices,    /* [in] */
                          std::vector< std::vector<int32_t> >&  vertexWeights, /* [in] */
                          std::vector< std::set<int32_t> >&     partitions     /* [out]*/) = 0;
};
typedef std::shared_ptr<csGraphPartitioner_t> csGraphPartitionerPtr;

}

#endif
