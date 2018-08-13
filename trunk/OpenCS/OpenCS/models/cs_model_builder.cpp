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
#include <omp.h>
#include <math.h>
#include <locale.h>
#include <set>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <functional>
#include <algorithm>
#include "cs_model_builder.h"
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;

namespace cs
{
csModelBuilder_t::csModelBuilder_t()
{
}

csModelBuilder_t::~csModelBuilder_t()
{
}

void csModelBuilder_t::Initialize_ODE_System(uint32_t           noVariables,
                                             uint32_t           noDofs,
                                             real_t             defaultVariableValue,
                                             real_t             defaultAbsoluteTolerance,
                                             const std::string& defaultVariableName)
{
    isODESystem = true;
    Initialize(noVariables,
               noDofs,
               defaultVariableValue,
               0.0, /* Not relevant in ODE systems. */
               defaultAbsoluteTolerance,
               defaultVariableName);
}

void csModelBuilder_t::Initialize_DAE_System(uint32_t           noVariables,
                                             uint32_t           noDofs,
                                             real_t             defaultVariableValue,
                                             real_t             defaultVariableTimeDerivative,
                                             real_t             defaultAbsoluteTolerance,
                                             const std::string& defaultVariableName)
{
    isODESystem = false;
    Initialize(noVariables,
               noDofs,
               defaultVariableValue,
               defaultVariableTimeDerivative,
               defaultAbsoluteTolerance,
               defaultVariableName);
}

void csModelBuilder_t::Initialize(uint32_t           noVariables,
                                  uint32_t           noDofs,
                                  real_t             defaultVariableValue,
                                  real_t             defaultVariableTimeDerivative,
                                  real_t             defaultAbsoluteTolerance,
                                  const std::string& defaultVariableName)
{
    if(noVariables == 0)
        throw std::runtime_error("Invalid number of variables specified");

    Nvariables = noVariables;
    Ndofs      = noDofs;

    if(Ndofs > 0)
        dofs_ptrs.resize(Ndofs);
    variables_ptrs.resize(Nvariables);
    timeDerivatives_ptrs.resize(Nvariables);
    equationNodes_ptrs.resize(Nvariables);

    if(Ndofs > 0)
        dofValues.resize(Ndofs, defaultVariableValue);
    variableValues.resize(Nvariables, defaultVariableValue);
    variableDerivatives.resize(Nvariables, defaultVariableTimeDerivative);
    //sensitivityValues.resize(Nvariables, 0.0);
    //sensitivityDerivatives.resize(Nvariables, 0.0);
    variableNames.resize(Nvariables);
    variableTypes.resize(Nvariables, csAlgebraicVariable);
    absoluteTolerances.resize(Nvariables, defaultAbsoluteTolerance);

    time_ptr.node.reset(new csTimeNode());

    const int bsize = 2048;
    char buffer[bsize];
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csVariableNode* var_node = new csVariableNode(i, i);
        csNumber_t& v = variables_ptrs[i];
        v.node.reset(var_node);

        csTimeDerivativeNode* tderiv_node = new csTimeDerivativeNode(i, i);
        csNumber_t& td = timeDerivatives_ptrs[i];
        td.node.reset(tderiv_node);

        std::snprintf(buffer, bsize, "%s[%d]", defaultVariableName.c_str(), i);
        variableNames[i] = buffer;
    }

    for(uint32_t i = 0; i < Ndofs; i++)
    {
        csDegreeOfFreedomNode* dof_node = new csDegreeOfFreedomNode(Nvariables+i, i);
        csNumber_t& dof = dofs_ptrs[i];
        dof.node.reset(dof_node);
    }
}

const csNumber_t& csModelBuilder_t::GetTime() const
{
    return time_ptr;
}

const std::vector<csNumber_t>& csModelBuilder_t::GetDegreesOfFreedom() const
{
    return dofs_ptrs;
}

const std::vector<csNumber_t>& csModelBuilder_t::GetVariables() const
{
    return variables_ptrs;
}

const std::vector<csNumber_t>& csModelBuilder_t::GetTimeDerivatives() const
{
    return timeDerivatives_ptrs;
}

void csModelBuilder_t::SetModelEquations(const std::vector<csNumber_t>& equations)
{
    if(equations.size() != Nvariables)
        throw std::runtime_error("Invalid equations size specified");

    equationNodes_ptrs = equations;

    /* Call CollectVariableTypes for every equation to set variable types. */
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr node = equationNodes_ptrs[i].node;
        node->CollectVariableTypes(variableTypes);
    }
}

void csModelBuilder_t::SetVariableValues(const std::vector<real_t>& values)
{
    if(values.size() != Nvariables)
        throw std::runtime_error("Invalid variable values size specified");
    variableValues = values;
}

void csModelBuilder_t::SetVariableTimeDerivatives(const std::vector<real_t>& timeDerivatives)
{
    if(timeDerivatives.size() != Nvariables)
        throw std::runtime_error("Invalid variable derivatives size specified");
    variableDerivatives = timeDerivatives;
}

void csModelBuilder_t::SetDegreeOfFreedomValues(const std::vector<real_t>& dofs)
{
    if(dofs.size() != Ndofs)
        throw std::runtime_error("Invalid dofs size specified");
    dofValues = dofs;
}

void csModelBuilder_t::SetVariableNames(const std::vector<std::string>& names)
{
    if(names.size() != Nvariables)
        throw std::runtime_error("Invalid variable names size specified");
    variableNames = names;
}

void csModelBuilder_t::SetVariableTypes(const std::vector<int32_t>& types)
{
    if(types.size() != Nvariables)
        throw std::runtime_error("Invalid variable types size specified");
    variableTypes = types;
}

void csModelBuilder_t::SetAbsoluteToleances(const std::vector<real_t>& absTolerances)
{
    if(absTolerances.size() != Nvariables)
        throw std::runtime_error("Invalid variable abs. tolerances size specified");
    absoluteTolerances = absTolerances;
}

void csModelBuilder_t::GetSparsityPattern(std::vector< std::map<uint32_t,uint32_t> >& incidenceMatrix)
{
    if(equationNodes_ptrs.size() != Nvariables)
        throw std::runtime_error("Model equations have not been set");

    incidenceMatrix.clear();
    incidenceMatrix.resize(Nvariables);
    std::map<uint32_t,uint32_t> indexes;
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr eqNode = equationNodes_ptrs[i].node;

        indexes.clear();
        eqNode->CollectVariableIndexes(indexes);

        /* Very important:
         *   ODE systems are defined as dx[i]/dt = rhs(x,y,p,t) and
         *   the RHS Jacobian matrix does not add indexes for x[i] (unless the RHS term depends on x[i]).
         *   However, the system Jacobian matrix is in the form: [M] = [I] - gamma * [Jrhs].
         *   Therefore, we must assure that the x[i] indexes are added to the sparsity pattern!!
         *   That effectively means that every row must have a diagonal item. */
        if(isODESystem)
        {
            // Add only if it is not already in the map.
            std::map<uint32_t,uint32_t>::const_iterator it = indexes.find(i);
            if(it == indexes.end())
                indexes[i] = i;
        }

        std::map<uint32_t,uint32_t>& indexes_i = incidenceMatrix[i];
        indexes_i.swap(indexes);
    }
}

void csModelBuilder_t::GetSparsityPattern(std::vector<uint32_t>& IA, std::vector<uint32_t>& JA)
{
    /* Get the total Nnz */
    uint32_t Nnz = 0;
    std::map<uint32_t,uint32_t> indexes;
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr eqNode = equationNodes_ptrs[i].node;

        indexes.clear();
        eqNode->CollectVariableIndexes(indexes);

        /* Very important:
         *   ODE systems are defined as dx[i]/dt = rhs(x,y,p,t) and
         *   the RHS Jacobian matrix does not add indexes for x[i] (unless the RHS term depends on x[i]).
         *   However, the system Jacobian matrix is in the form: [M] = [I] - gamma * [Jrhs].
         *   Therefore, we must assure that the x[i] indexes are added to the sparsity pattern!!
         *   That effectively means that every row must have a diagonal item. */
        if(isODESystem)
        {
            // Add only if it is not already in the map.
            std::map<uint32_t,uint32_t>::const_iterator it = indexes.find(i);
            if(it == indexes.end())
                indexes[i] = i;
        }

        Nnz += indexes.size();
    }

    IA.resize(Nvariables + 1);
    JA.reserve(Nnz);
    IA[0] = 0;
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr eqNode = equationNodes_ptrs[i].node;

        indexes.clear();
        eqNode->CollectVariableIndexes(indexes);

        /* The same logic as above:
         *   We must assure that the x[i] indexes are added to the sparsity pattern,
         *   which effectively means that every row must have a diagonal item. */
        if(isODESystem)
        {
            // Add only if it is not already in the map.
            std::map<uint32_t,uint32_t>::const_iterator it = indexes.find(i);
            if(it == indexes.end())
                indexes[i] = i;
        }

        uint32_t Nnz_i = indexes.size();
        IA[i+1] = IA[i] + Nnz_i;
        for(std::map<uint32_t,uint32_t>::const_iterator it = indexes.begin(); it != indexes.end(); it++)
        {
            uint32_t bi = it->first; /* Note: here, block indexes are the keys! (DOFs are skipped) */
            uint32_t oi = it->second;

            // block indexes must be sorted (they are automatically sorted in a std::map)
            JA.push_back(bi);
        }
    }
/*
    for(uint32_t row = 0; row < Nvariables; row++)
    {
        uint32_t colStart = IA[row];
        uint32_t colEnd   = IA[row+1];
        printf("%d - [ ", row);
        for(uint32_t col = colStart; col < colEnd; col++)
            printf("%d ", JA[col]);
        printf("]\n");
    }
*/
}

void csModelBuilder_t::EvaluateEquations(real_t currentTime, std::vector<real_t>& equations)
{
    csNodeEvaluationContext_t context;
    context.currentTime     = currentTime;
    context.inverseTimeStep = 0.0;
    context.jacobianIndex   = -1;
    context.dofs            = (dofValues.size() > 0              ? &dofValues[0]              : NULL);
    context.values          = (variableValues.size() > 0         ? &variableValues[0]         : NULL);
    context.timeDerivatives = (variableDerivatives.size() > 0    ? &variableDerivatives[0]    : NULL);
    context.svalues         = (sensitivityValues.size() > 0      ? &sensitivityValues[0]      : NULL);
    context.sdvalues        = (sensitivityDerivatives.size() > 0 ? &sensitivityDerivatives[0] : NULL);

    equations.resize(Nvariables);
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        adouble_t res = equationNodes_ptrs[i].node->Evaluate(context);
        equations[i] = adouble_getValue(&res);
    }
}

void csModelBuilder_t::EvaluateDerivatives(real_t                 currentTime,
                                           real_t                 timeStep,
                                           std::vector<uint32_t>& IA,
                                           std::vector<uint32_t>& JA,
                                           std::vector<real_t>&   A,
                                           bool                   generateIncidenceMatrix)
{
    csNodeEvaluationContext_t context;
    context.currentTime     = currentTime;
    context.inverseTimeStep = 1.0 / timeStep;
    context.jacobianIndex   = -1;
    context.dofs            = (dofValues.size() > 0              ? &dofValues[0]              : NULL);
    context.values          = (variableValues.size() > 0         ? &variableValues[0]         : NULL);
    context.timeDerivatives = (variableDerivatives.size() > 0    ? &variableDerivatives[0]    : NULL);
    context.svalues         = (sensitivityValues.size() > 0      ? &sensitivityValues[0]      : NULL);
    context.sdvalues        = (sensitivityDerivatives.size() > 0 ? &sensitivityDerivatives[0] : NULL);

    // Generate the incidence matrix, if requested.
    if(generateIncidenceMatrix)
    {
        GetSparsityPattern(IA, JA);
    }

    if(IA.size() != Nvariables+1 || JA.size() == 0)
        throw std::runtime_error("Invalid incidence matrix specified");

    /*
    printf("System incidence matrix (CRS):\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        printf(" row [%d]: ", i);
        for(uint32_t col = IA[i]; col < IA[i+1]; col++)
        {
            uint32_t bi = JA[col];

            printf("%d ", bi);
        }
        printf("\n");
    }
    */

    uint32_t Nnz = JA.size();
    A.resize(Nnz);

    for(uint32_t row = 0; row < Nvariables; row++)
    {
        for(uint32_t col = IA[row]; col < IA[row+1]; col++)
        {
            uint32_t bi = JA[col];
            context.jacobianIndex = bi;

            adouble_t res = equationNodes_ptrs[row].node->Evaluate(context);

            A[col] = adouble_getDerivative(&res);
        }
    }
}

static std::string formatFilenameJson(const std::string& inputDirectory, const std::string& inputFile)
{
    /* Compose the file path.
     * It is assumed that the inputDirectory is full path or relative path to the current diectory. */
    filesystem::path inputDirectoryPath = filesystem::absolute( filesystem::path(inputDirectory) );
    std::string filePath = (inputDirectoryPath / inputFile).string();

    return filePath;
}

void csModelBuilder_t::ExportModels(const std::vector<csModelPtr>& models, const std::string& outputDirectory, const std::string& simulationOptionsJSON)
{
    for(int i = 0; i < models.size(); i++)
    {
        csModelPtr model = models[i];
        model->SaveModel(outputDirectory);
    }

    /* Save simulation options. */
    std::ofstream f;
    std::string filePath = formatFilenameJson(outputDirectory, csModel_t::simulationOptionsFileName);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    f.write(simulationOptionsJSON.c_str(), simulationOptionsJSON.size());

    f.close();
}

static void GenerateModel(csModelPtr                               model,
                          csModelBuilder_t&                        modelBuilder,
                          std::set<int32_t>&                       all_oi_pe,
                          std::set<int32_t>&                       owned_oi_pe,
                          std::set<int32_t>&                       foreign_oi_pe,
                          std::map< uint32_t,std::set<int32_t> >&  send_to_oi_pe,
                          std::map< uint32_t,std::set<int32_t> >&  receive_from_oi_pe,
                          std::map< int32_t, int32_t >&            bi_to_bi_local_pe)
{
    uint32_t Ndofs            = modelBuilder.Ndofs;
    uint32_t Nvariables       = owned_oi_pe.size();
    uint32_t Nvariables_total = modelBuilder.Nvariables;

    std::vector<real_t>      dofValues(Ndofs);
    std::vector<real_t>      variableValues(Nvariables);
    std::vector<real_t>      variableDerivatives(Nvariables);
    std::vector<std::string> variableNames(Nvariables);
    std::vector<int32_t>     variableTypes(Nvariables);
    std::vector<real_t>      absoluteTolerances(Nvariables);
    std::vector<csNumber_t>  equationNodes_ptrs(Nvariables);

    dofValues = modelBuilder.dofValues;
    int32_t vi = 0;
    for(std::set<int32_t>::const_iterator it = owned_oi_pe.begin(); it != owned_oi_pe.end(); it++)
    {
        int32_t bi = *it;

        variableValues[vi]      = modelBuilder.variableValues[bi];
        variableDerivatives[vi] = modelBuilder.variableDerivatives[bi];
        variableNames[vi]       = modelBuilder.variableNames[bi];
        variableTypes[vi]       = modelBuilder.variableTypes[bi];
        absoluteTolerances[vi]  = modelBuilder.absoluteTolerances[bi];
        equationNodes_ptrs[vi]  = modelBuilder.equationNodes_ptrs[bi];

        vi++;
    }

    csModelType modelType;
    int32_t sumVarTypes = std::accumulate(variableTypes.begin(), variableTypes.end(), 0);
    if(sumVarTypes == Nvariables) // all equations are differential: it is probably an implicit ODE system
        modelType = eImplicitODE;
    else if(sumVarTypes == 0)     // all equations are algebraic
        modelType = eSteadyState;
    else                          // a mix of differential and algebraic equations
        modelType = eDAE;

    if(modelType == eSteadyState && !modelBuilder.isODESystem)
        printf("The generated model for pe = %u is Steady State (not DAE)\n", model->pe_rank);

    /* 1. Model structure. */
    model->structure.Ndofs               = Ndofs;
    model->structure.Nequations_total    = Nvariables_total;
    model->structure.Nequations          = Nvariables;
    model->structure.isODESystem         = modelBuilder.isODESystem;

    model->structure.dofValues           = dofValues;
    model->structure.variableValues      = variableValues;
    model->structure.variableDerivatives = variableDerivatives;
    model->structure.variableNames       = variableNames;
    model->structure.variableTypes       = variableTypes;
    model->structure.absoluteTolerances  = absoluteTolerances;

    /* 2. Partition data. */
    int Nforeign = foreign_oi_pe.size();
    model->partitionData.foreignIndexes.clear();
    model->partitionData.foreignIndexes.reserve(Nforeign);
    std::copy(foreign_oi_pe.begin(), foreign_oi_pe.end(), std::back_inserter(model->partitionData.foreignIndexes));

    model->partitionData.biToBiLocal = bi_to_bi_local_pe;
    for(std::map< uint32_t,std::set<int32_t> >::const_iterator it = receive_from_oi_pe.begin(); it != receive_from_oi_pe.end(); it++)
    {
        uint32_t                 pe     = it->first;
        const std::set<int32_t>& set_rf = it->second;

        std::vector<int32_t>& indexes = model->partitionData.receiveFromIndexes[pe];

        indexes.reserve( set_rf.size() );
        std::copy(set_rf.begin(), set_rf.end(), std::back_inserter(indexes));
    }
    for(std::map< uint32_t,std::set<int32_t> >::const_iterator it = send_to_oi_pe.begin(); it != send_to_oi_pe.end(); it++)
    {
        uint32_t                 pe     = it->first;
        const std::set<int32_t>& set_st = it->second;

        std::vector<int32_t>& indexes = model->partitionData.sendToIndexes[pe];

        indexes.reserve( set_st.size() );
        std::copy(set_st.begin(), set_st.end(), std::back_inserter(indexes));
    }

    /* 3. Model equations. */
    uint32_t Ncs = 0;
    std::map<uint32_t,uint32_t> indexes;

    /* Estimate the array sizes and populate the active equation set indexes. */
    model->equations.activeEquationSetIndexes.reserve(Nvariables);
    model->sparsityPattern.rowIndexes.reserve(Nvariables+1);

    uint32_t Nji_counter = 0;
    model->sparsityPattern.rowIndexes.push_back(Nji_counter); // first item is always 0

    std::vector<uint32_t> cs_sizes(Nvariables);
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr eqNode = equationNodes_ptrs[i].node;
        uint32_t Ncs_i = csNode_t::GetComputeStackSize(eqNode.get());
        cs_sizes[i] = Ncs_i;
        model->equations.activeEquationSetIndexes.push_back(Ncs);
        Ncs += Ncs_i;

        // Row indices
        indexes.clear();
        eqNode->CollectVariableIndexes(indexes);
        /* Very important:
         *   ODE systems are defined as dx[i]/dt = rhs(x,y,p,t) and
         *   the RHS Jacobian matrix does not add indexes for x[i] (unless the RHS term depends on x[i]).
         *   However, the system Jacobian matrix is in the form: [M] = [I] - gamma * [Jrhs].
         *   Therefore, we must assure that the x[i] indexes are added to the sparsity pattern!!
         *   That effectively means that every row must have a diagonal item. */
        if(modelBuilder.isODESystem)
        {
            // Add only if it is not already in the map.
            std::map<uint32_t,uint32_t>::const_iterator it = indexes.find(i);
            if(it == indexes.end())
                indexes[i] = i;
        }

        Nji_counter += indexes.size();
        model->sparsityPattern.rowIndexes.push_back(Nji_counter);
    }

    /* Reserve the memory in arrays. */
    model->sparsityPattern.Nnz        = Nji_counter;
    model->sparsityPattern.Nequations = Nvariables;
    model->equations.computeStacks.resize(Ncs);
    model->sparsityPattern.incidenceMatrixItems.resize(Nji_counter);

    /* Generate the computeStack and incidenceMatrixItems arrays. */
    std::map<int32_t, int32_t>::const_iterator bi_end = bi_to_bi_local_pe.end();
    std::map<int32_t, int32_t>::const_iterator bi_cit;
    std::vector<csComputeStackItem_t> computeStack_i;

    // The problem with this for loop was with push_back to the incidenceMatrixItems
    //   whose size was not reserved in advance. Resolved now.

    //#pragma omp parallel for private(indexes, computeStack_i)
    for(uint32_t ei = 0; ei < Nvariables; ei++)
    {
        //int omp_tid = omp_get_thread_num();
        //if(omp_tid == 0)
        //    printf("  processing equation %d\r", (int)ei);

        uint32_t firstIndex = model->equations.activeEquationSetIndexes[ei];
        uint32_t Ncs_i      = cs_sizes[ei];
        computeStack_i.clear();
        computeStack_i.reserve(Ncs_i);

        // Create compute stack and copy it to the global compute stack array.
        // Check the generated Ncs_i with the predicted Ncs_i (raise an exception if not equal).
        csNodePtr eqNode = equationNodes_ptrs[ei].node;
        csNode_t::CreateComputeStack(eqNode.get(), computeStack_i);
        if(Ncs_i != computeStack_i.size())
            throw std::runtime_error("Invalid size of the local compute stack");
        std::copy(computeStack_i.begin(), computeStack_i.end(), model->equations.computeStacks.begin()+firstIndex);

        // Update block indexes in the compute stack to mpi-node block indexes.
        csComputeStackItem_t* computeStack = &model->equations.computeStacks[firstIndex];
        uint32_t computeStackSize = computeStack->size;
        for(uint32_t csi = 0; csi < computeStackSize; csi++)
        {
            csComputeStackItem_t& item = computeStack[csi];

            if(item.opCode == eOP_Variable)
            {
                bi_cit = bi_to_bi_local_pe.find(item.data.indexes.blockIndex);
                if(bi_cit == bi_end)
                    throw std::runtime_error("Invalid index");
                item.data.indexes.blockIndex = bi_cit->second;
            }
            else if(item.opCode == eOP_DegreeOfFreedom)
            {
                // Leave dofIndex as it is for we have all dofs in the model
            }
            else if(item.opCode == eOP_TimeDerivative)
            {
                bi_cit = bi_to_bi_local_pe.find(item.data.indexes.blockIndex);
                if(bi_cit == bi_end)
                    throw std::runtime_error("Invalid index");
                item.data.indexes.blockIndex = bi_cit->second;
            }
        }

        indexes.clear();
        eqNode->CollectVariableIndexes(indexes);
        /* The same logic as above:
         *   We must assure that the x[i] indexes are added to the sparsity pattern,
         *   which effectively means that every row must have a diagonal item. */
        if(modelBuilder.isODESystem)
        {
            // Add only if it is not already in the map.
            std::map<uint32_t,uint32_t>::const_iterator it = indexes.find(ei);
            if(it == indexes.end())
                indexes[ei] = ei;
        }

        uint32_t jiStart = model->sparsityPattern.rowIndexes[ei];

        for(std::map<uint32_t,uint32_t>::const_iterator it = indexes.begin(); it != indexes.end(); it++)
        {
            uint32_t bi = it->first; /* Note: here, block indexes are the keys! (DOFs are skipped) */
            uint32_t oi = it->second;
            uint32_t bi_local = bi_to_bi_local_pe[bi];

            csIncidenceMatrixItem_t ji;
            ji.equationIndex = ei;
            ji.overallIndex  = oi;
            ji.blockIndex    = bi_local;
            model->sparsityPattern.incidenceMatrixItems[jiStart] = ji;
            jiStart++;
        }
    }
}

std::vector<csModelPtr> csModelBuilder_t::PartitionSystem(uint32_t                                    Npe,
                                                          csGraphPartitioner_t*                       graphPartitioner,
                                                          const std::vector<std::string>&             balancingConstraints,
                                                          bool                                        logPartitionResults,
                                                          const std::map<csUnaryFunctions,uint32_t>&  unaryOperationsFlops,
                                                          const std::map<csBinaryFunctions,uint32_t>& binaryOperationsFlops)
{
    std::vector<csModelPtr>  models;

    if(Npe == 0)
        throw std::runtime_error("Invalid number of processing elements (0)");
    if(!graphPartitioner)
        throw std::runtime_error("Invalid partitioner specified");

    std::vector<uint32_t> IA, JA;
    std::vector< std::vector<int32_t> >                     loads_oi;
    std::vector< std::set<int32_t> >                        partitions;
    std::vector< std::set<int32_t> >                        all_oi, owned_oi, foreign_oi;
    std::vector< std::map< uint32_t,std::set<int32_t> > >   send_to_oi, receive_from_oi;
    std::vector< std::map< int32_t, int32_t > >             bi2bi_local;

    /* Open the log file */
    std::string constraints_s;
    int32_t Nconstraints = balancingConstraints.size();
    for(uint32_t c = 0; c < Nconstraints; c++)
    {
        if(c == 0)
            constraints_s = balancingConstraints[c];
        else
            constraints_s += "," + balancingConstraints[c];
    }

    /* Initialise arrays/sets */
    //printf("Initialise arrays/sets\n");
    models.resize(Npe);
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        csModelPtr model = csModelPtr(new csModel_t);
        models[pe] = model;
        model->pe_rank = pe;
    }

    all_oi.resize(Npe);
    owned_oi.resize(Npe);
    foreign_oi.resize(Npe);
    partitions.resize(Npe);
    send_to_oi.resize(Npe);
    receive_from_oi.resize(Npe);
    bi2bi_local.resize(Npe);
    loads_oi.resize(4, std::vector<int32_t>(Npe)); // Ncs, Nflops, Nnz, Nflops_j

    /* Get the incidence matrix (in CRS format); */
    //printf("Get the incidence matrix\n");
    GetSparsityPattern(IA, JA);

    /* Partition the system into Npe parts */
    std::vector<int32_t> weights_Ncs     (Nvariables, 0);
    std::vector<int32_t> weights_Nnz     (Nvariables, 0);
    std::vector<int32_t> weights_Nflops  (Nvariables, 0);
    std::vector<int32_t> weights_Nflops_j(Nvariables, 0);

    std::vector<int32_t> total_Ncs     (Npe, 0);
    std::vector<int32_t> total_Nnz     (Npe, 0);
    std::vector<int32_t> total_Nflops  (Npe, 0);
    std::vector<int32_t> total_Nflops_j(Npe, 0);

    real_t Ncs_ave, Nflops_ave, Nnz_ave, Nflops_j_ave;
    std::vector<real_t> Ncs_dev(Npe, 0.0);
    std::vector<real_t> Nflops_dev(Npe, 0.0);
    std::vector<real_t> Nnz_dev(Npe, 0.0);
    std::vector<real_t> Nflops_j_dev(Npe, 0.0);

    /* Generate vertex weights. */
    //printf("Generate vertex weights\n");
    for(uint32_t i = 0; i < Nvariables; i++)
    {
        csNodePtr eqNode = equationNodes_ptrs[i].node;
        int32_t Nnz      = IA[i+1] - IA[i];
        int32_t Ncs      = csNode_t::GetComputeStackSize(eqNode.get());
        int32_t Nflops   = csNode_t::GetComputeStackFlops(eqNode.get(), unaryOperationsFlops, binaryOperationsFlops);
        int32_t Nflops_j = Nnz * Nflops;

        weights_Ncs[i]      = Ncs;
        weights_Nnz[i]      = Nnz;
        weights_Nflops[i]   = Nflops;
        weights_Nflops_j[i] = Nflops_j;
    }

    /* The result of the partitioning is the populated 'partitions' vector. */
    //printf("Partition the system\n");
    if(Npe == 1)
    {
        std::set<int32_t>& partition = partitions[0];
        for(int32_t ei = 0; ei < Nvariables; ei++)
           partition.insert(ei);
    }
    else
    {
        int Nconstraints = balancingConstraints.size();
        std::vector< std::vector<int32_t> > vweights(Nconstraints);

        for(uint32_t c = 0; c < Nconstraints; c++)
        {
            std::string constraint = balancingConstraints[c];
            if(constraint == "Ncs")
                vweights[c] = weights_Ncs;
            else if(constraint == "Nnz")
                vweights[c] = weights_Nnz;
            else if(constraint == "Nflops")
                vweights[c] = weights_Nflops;
            else if(constraint == "Nflops_j")
                vweights[c] = weights_Nflops_j;
            else
                throw std::runtime_error("Invalid balancing constraint requested: " + constraint);
        }

        graphPartitioner->Partition(Npe,
                                     Nvariables,
                                     Nconstraints,
                                     IA,
                                     JA,
                                     vweights,
                                     partitions);
    }

    /* Generate partitioning statistics. */
    //printf("Generate partitioning statistics\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        std::set<int32_t>& partition         = partitions[pe];
        int32_t&           total_Ncs_pe      = total_Ncs[pe];
        int32_t&           total_Nnz_pe      = total_Nnz[pe];
        int32_t&           total_Nflops_pe   = total_Nflops[pe];
        int32_t&           total_Nflops_j_pe = total_Nflops_j[pe];

        for(std::set<int32_t>::const_iterator it = partition.begin(); it != partition.end(); it++)
        {
            int32_t ei = *it;

            total_Ncs_pe      += weights_Ncs[ei];
            total_Nnz_pe      += weights_Nnz[ei];
            total_Nflops_pe   += weights_Nflops[ei];
            total_Nflops_j_pe += weights_Nflops_j[ei];
        }
    }

    Ncs_ave      = std::accumulate(total_Ncs.begin(),      total_Ncs.end(),      0) / Npe;
    Nflops_ave   = std::accumulate(total_Nflops.begin(),   total_Nflops.end(),   0) / Npe;
    Nnz_ave      = std::accumulate(total_Nnz.begin(),      total_Nnz.end(),      0) / Npe;
    Nflops_j_ave = std::accumulate(total_Nflops_j.begin(), total_Nflops_j.end(), 0) / Npe;

    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        Ncs_dev[pe]      = (total_Ncs[pe]      - Ncs_ave)      * 100 / Ncs_ave;
        Nflops_dev[pe]   = (total_Nflops[pe]   - Nflops_ave)   * 100 / Nflops_ave;
        Nnz_dev[pe]      = (total_Nnz[pe]      - Nnz_ave)      * 100 / Nnz_ave;
        Nflops_j_dev[pe] = (total_Nflops_j[pe] - Nflops_j_ave) * 100 / Nflops_j_ave;
    }

    /* Create index sets for every partition */
    //printf("All indexes\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        std::set<int32_t>& all_oi_pe   = all_oi[pe];
        std::set<int32_t>& owned_oi_pe = owned_oi[pe];
        std::set<int32_t>& partition   = partitions[pe];

        for(std::set<int32_t>::const_iterator it = partition.begin(); it != partition.end(); it++)
        {
            int32_t ei = *it;
            owned_oi_pe.insert(ei);

            for(uint32_t j = IA[ei]; j < IA[ei+1]; j++)
            {
                uint32_t bi = JA[j];
                all_oi_pe.insert(bi);
            }
        }
    }

    //printf("Owned indexes\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        std::set<int32_t>& all_oi_pe     = all_oi[pe];
        std::set<int32_t>& owned_oi_pe   = owned_oi[pe];
        std::set<int32_t>& foreign_oi_pe = foreign_oi[pe];

        std::set_difference(all_oi_pe.begin(),   all_oi_pe.end(),
                            owned_oi_pe.begin(), owned_oi_pe.end(),
                            std::inserter(foreign_oi_pe, foreign_oi_pe.begin()));
    }

    //printf("receiveFrom/sendTo indexes\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        std::set<int32_t>& foreign_oi_pe = foreign_oi[pe];

        for(uint32_t pe_other = 0; pe_other < Npe; pe_other++)
        {
            std::set<int32_t>& owned_oi_pe_other = owned_oi[pe_other];

            std::set<int32_t> intersection;
            std::set_intersection(foreign_oi_pe.begin(),     foreign_oi_pe.end(),
                                  owned_oi_pe_other.begin(), owned_oi_pe_other.end(),
                                  std::inserter(intersection, intersection.begin()));
            if(intersection.size() > 0)
            {
                std::map< uint32_t,std::set<int32_t> >& send_to_oi_pe      = send_to_oi[pe_other];
                std::map< uint32_t,std::set<int32_t> >& receive_from_oi_pe = receive_from_oi[pe];

                std::set<int32_t>& set_rf = receive_from_oi_pe[pe_other];
                std::set<int32_t>& set_st = send_to_oi_pe     [pe];

                set_rf.insert(intersection.begin(), intersection.end());
                set_st.insert(intersection.begin(), intersection.end());
            }
        }
    }

    //printf("bi_bi_local\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        std::map< int32_t, int32_t >& bi2bi_local_pe = bi2bi_local[pe];
        std::set<int32_t>&            owned_oi_pe    = owned_oi[pe];
        std::set<int32_t>&            foreign_oi_pe  = foreign_oi[pe];

        int32_t bi_local = 0;
        for(std::set<int32_t>::const_iterator it = owned_oi_pe.begin(); it != owned_oi_pe.end(); it++)
        {
            int32_t bi = *it;
            bi2bi_local_pe[bi] = bi_local;
            bi_local++;
        }

        for(std::set<int32_t>::const_iterator it = foreign_oi_pe.begin(); it != foreign_oi_pe.end(); it++)
        {
            int32_t bi = *it;
            bi2bi_local_pe[bi] = bi_local;
            bi_local++;
        }
    }

    printf("GenerateModel\n");
    for(uint32_t pe = 0; pe < Npe; pe++)
    {
        csModelPtr                               model_pe            = models[pe];
        std::set<int32_t>&                       all_oi_pe           = all_oi[pe];
        std::set<int32_t>&                       owned_oi_pe         = owned_oi[pe];
        std::set<int32_t>&                       foreign_oi_pe       = foreign_oi[pe];
        std::map< uint32_t,std::set<int32_t> >&  send_to_oi_pe       = send_to_oi[pe];
        std::map< uint32_t,std::set<int32_t> >&  receive_from_oi_pe  = receive_from_oi[pe];
        std::map< int32_t, int32_t >&            bi2bi_local_pe      = bi2bi_local[pe];

        GenerateModel(model_pe,
                      *this,
                      all_oi_pe,
                      owned_oi_pe,
                      foreign_oi_pe,
                      send_to_oi_pe,
                      receive_from_oi_pe,
                      bi2bi_local_pe);
    }

    /* Print partition info */
    if(logPartitionResults)
    {
        std::string log_filename;
        if(Nconstraints == 0)
            log_filename = "partition-Npe=" + std::to_string(Npe) + "-" + graphPartitioner->GetName() + ".log";
        else
            log_filename = "partition-Npe=" + std::to_string(Npe) + "-" + graphPartitioner->GetName() + "-[" + constraints_s + "].log";
        FILE* f = fopen(log_filename.c_str(), "w");
        fprintf(f, "Npe:         %u\n",   Npe);
        fprintf(f, "Partitioner: %s\n",   graphPartitioner->GetName().c_str());

        fprintf(f, "Balancing constraints (deviation from average, in %%):\n");

        std::string cNcs      = "Ncs";
        std::string cNnz      = "Nnz";
        std::string cNflops   = "Nflops";
        std::string cNflops_j = "Nflops_j";
        for(uint32_t c = 0; c < Nconstraints; c++)
        {
            if(balancingConstraints[c] == "Ncs")
                cNcs += "[*]";
            if(balancingConstraints[c] == "Nnz")
                cNnz += "[*]";
            if(balancingConstraints[c] == "Nflops")
                cNflops += "[*]";
            if(balancingConstraints[c] == "Nflops_j")
                cNflops_j += "[*]";
        }
        fprintf(f, "%7s%10s%10s%13s%13s%13s%13s\n",  "PE", "Neq", "Nadj", cNcs.c_str(), cNnz.c_str(), cNflops.c_str(), cNflops_j.c_str());
        fprintf(f, "------- --------- --------- ------------ ------------ ------------ ------------\n");
        for(uint32_t pe = 0; pe < Npe; pe++)
            fprintf(f, "%7u%10d%10d%13.2f%13.2f%13.2f%13.2f\n", pe, (int)owned_oi[pe].size(), (int)foreign_oi[pe].size(),
                                                                Ncs_dev[pe], Nnz_dev[pe], Nflops_dev[pe], Nflops_j_dev[pe]);

        //setlocale(LC_NUMERIC, "");
        for(uint32_t pe = 0; pe < Npe; pe++)
        {
            std::set<int32_t>&                       all_oi_pe           = all_oi[pe];
            std::set<int32_t>&                       owned_oi_pe         = owned_oi[pe];
            std::set<int32_t>&                       foreign_oi_pe       = foreign_oi[pe];
            std::map< uint32_t,std::set<int32_t> >&  send_to_oi_pe       = send_to_oi[pe];
            std::map< uint32_t,std::set<int32_t> >&  receive_from_oi_pe  = receive_from_oi[pe];

            fprintf(f, "\n");
            fprintf(f, "PE %d:\n", pe);
            fprintf(f, "  Nequations %12d\n", (int)owned_oi_pe.size());
            fprintf(f, "  Nadjacent  %12d\n", (int)foreign_oi_pe.size());

            fprintf(f, "  Weights (total):\n");
            fprintf(f, "    Ncs      %12d\n", total_Ncs[pe]);
            fprintf(f, "    Nnz      %12d\n", total_Nnz[pe]);
            fprintf(f, "    Nflops   %12d\n", total_Nflops[pe]);
            fprintf(f, "    Nflops_j %12d\n", total_Nflops_j[pe]);

            fprintf(f, "  Weights (deviation from average, in %%):\n");
            fprintf(f, "    Ncs      %12.2f\n", Ncs_dev[pe]);
            fprintf(f, "    Nnz      %12.2f\n", Nnz_dev[pe]);
            fprintf(f, "    Nflops   %12.2f\n", Nflops_dev[pe]);
            fprintf(f, "    Nflops_j %12.2f\n", Nflops_j_dev[pe]);

            //fprintf(f, "  All indexes:\n    [ ");
            //for(std::set<int32_t>::const_iterator it = all_oi_pe.begin(); it != all_oi_pe.end(); it++)
            //    fprintf(f, "%u ", *it);
            //fprintf(f, "]\n");

            fprintf(f, "  Owned indexes:\n    [ ");
            for(std::set<int32_t>::const_iterator it = owned_oi_pe.begin(); it != owned_oi_pe.end(); it++)
                fprintf(f, "%u ", *it);
            fprintf(f, "]\n");

            fprintf(f, "  Foreign indexes:\n    [ ");
            for(std::set<int32_t>::const_iterator it = foreign_oi_pe.begin(); it != foreign_oi_pe.end(); it++)
                fprintf(f, "%u ", *it);
            fprintf(f, "]\n");

            fprintf(f, "  Send to indexes:\n");
            for(std::map< uint32_t,std::set<int32_t> >::const_iterator mit = send_to_oi_pe.begin(); mit != send_to_oi_pe.end(); mit++)
            {
                uint32_t                 st_pe  = mit->first;
                const std::set<int32_t>& st_set = mit->second;

                fprintf(f, "    %d: [ ", st_pe);
                for(std::set<int32_t>::const_iterator it = st_set.begin(); it != st_set.end(); it++)
                    fprintf(f, "%u ", *it);
                fprintf(f, "]\n");
            }

            fprintf(f, "  Receive from indexes:\n");
            for(std::map< uint32_t,std::set<int32_t> >::const_iterator mit = receive_from_oi_pe.begin(); mit != receive_from_oi_pe.end(); mit++)
            {
                uint32_t                 st_pe  = mit->first;
                const std::set<int32_t>& st_set = mit->second;

                fprintf(f, "    %d: [ ", st_pe);
                for(std::set<int32_t>::const_iterator it = st_set.begin(); it != st_set.end(); it++)
                    fprintf(f, "%u ", *it);
                fprintf(f, "]\n");
            }
        }

        fclose(f);
    }

    return models;
}

}
