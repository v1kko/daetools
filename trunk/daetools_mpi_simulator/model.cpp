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
#include "cs_simulator.h"
#include "cs_evaluator_sequential.h"
#include "auxiliary.h"
#include "mpi_data_exchange.h"
#include <exception>
#include <fstream>
#include <iomanip>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/container/flat_map.hpp>
using boost::container::flat_map;
using namespace cs;

/* Internal storage containers for evaluation of equations. */
std::vector<real_t>                 g_values;
std::vector<real_t>                 g_timeDerivatives;
std::vector<real_t>                 g_dofs;
std::vector<real_t>                 g_jacobian;

/* Data related to MPI communication (include index mappings for both owned and foreign indexes). */
flat_map<int, real_t*>              g_mapValues;
flat_map<int, real_t*>              g_mapTimeDerivatives;

/* Initialisation data (loaded from files). */
csPartitionData_t                   g_partitionData;

/* ComputeStack-related data. */
std::vector<csComputeStackItem_t>   g_arrAllComputeStacks;
std::vector<csJacobianMatrixItem_t> g_arrJacobianMatrixItems;
std::vector<uint32_t>               g_arrActiveEquationSetIndexes;
uint32_t*                           g_activeEquationSetIndexes  = NULL;
csComputeStackItem_t*               g_computeStacks             = NULL;
csJacobianMatrixItem_t*             g_jacobianMatrixItems       = NULL;
uint32_t                            g_numberOfJacobianItems     = -1;
uint32_t                            g_numberOfComputeStackItems = -1;
csComputeStackEvaluator_t*          g_pEvaluator                = NULL;

namespace cs_simulator
{
daeModel_t::daeModel_t()
{
    mpi_world = NULL;
    mpi_rank  = -1;
    mpi_comm  = NULL;
/*
    Nequations_local  = 0;
    Ntotal_vars       = 0;
    Nequations        = 0;
    Ndofs             = 0;
    startTime         = 0;
    timeHorizon       = 0;
    reportingInterval = 0;
    relativeTolerance = 0;
    quasiSteadyState  = false;

    ids                = NULL;
    initValues         = NULL;
    initDerivatives    = NULL;
    absoluteTolerances = NULL;
    variableNames      = NULL;
*/
}

daeModel_t::~daeModel_t()
{
    Free();
}

void daeModel_t::Load(const std::string& input_directory)
{
    inputDirectory = input_directory;

    loadModelVariables(this, inputDirectory);

    loadPartitionData(this, inputDirectory);
/*
    Nequations_local   = g_rtData.Nequations_local;
    Ntotal_vars        = g_rtData.Ntotal_vars;
    Nequations         = g_rtData.Nequations;
    Ndofs              = g_rtData.Ndofs;
    startTime          = g_rtData.startTime;
    timeHorizon        = g_rtData.timeHorizon;
    reportingInterval  = g_rtData.reportingInterval;
    relativeTolerance  = g_rtData.relativeTolerance;
    quasiSteadyState   = g_rtData.quasiSteadyState;

    ids                = new int   [Nequations_local];
    initValues         = new real_t[Nequations_local];
    initDerivatives    = new real_t[Nequations_local];
    absoluteTolerances = new real_t[Nequations_local];
    variableNames      = new const char*[Nequations_local];
    for(int i = 0; i < Nequations_local; i++)
    {
        ids[i]                = g_rtData.ids[i];
        initValues[i]         = g_rtData.init_values[i];
        initDerivatives[i]    = g_rtData.init_derivatives[i];
        absoluteTolerances[i] = g_rtData.absolute_tolerances[i];
        variableNames[i]      = g_rtData.variable_names[i].c_str();
    }
*/

    g_dofs.resize(Ndofs, 0.0);
    for(int i = 0; i < Ndofs; i++)
        g_dofs[i] = dofs[i];

    loadModelEquations(this, inputDirectory);

    loadJacobianData(this, inputDirectory);

    g_pEvaluator = new cs::csComputeStackEvaluator_Sequential();
    g_pEvaluator->Initialize(false,
                             Nequations_local,
                             Nequations_local,
                             Ndofs,
                             g_numberOfComputeStackItems,
                             g_numberOfJacobianItems,
                             g_numberOfJacobianItems,
                             g_computeStacks,
                             g_activeEquationSetIndexes,
                             g_jacobianMatrixItems);

    InitializeValuesReferences();

/*
    if(mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modInitialize-node-") + std::to_string(mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out);

        ofs << "Nequations_local = "  << Nequations_local << std::endl;
        ofs << "Ntotal_vars = "       << Ntotal_vars << std::endl;
        ofs << "Nequations = "        << Nequations << std::endl;
        ofs << "Ndofs = "             << Ndofs << std::endl;
        ofs << "startTime = "         << startTime << std::endl;
        ofs << "timeHorizon = "       << timeHorizon << std::endl;
        ofs << "reportingInterval = " << reportingInterval << std::endl;
        ofs << "relativeTolerance = " << relativeTolerance << std::endl;
        ofs << "quasiSteadyState = "  << quasiSteadyState << std::endl;

        ofs << "g_numberOfComputeStackItems = "  << g_numberOfComputeStackItems << std::endl;

        ofs << "startEquationIndex = "  << startEquationIndex << std::endl;
        ofs << "startJacobianIndex = "  << startJacobianIndex << std::endl;
        ofs << "g_numberOfJacobianItems = "  << g_numberOfJacobianItems << std::endl;

        ofs << "ActiveEquationSetIndexes = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << g_activeEquationSetIndexes[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "g_jacobianMatrixItems equationIndexes = [";
        for(int ji = 0; ji < g_numberOfJacobianItems; ji++)
        {
            csJacobianMatrixItem_t& jd = g_jacobianMatrixItems[ji];
            ofs << jd.equationIndex << ", ";
        }
        ofs << "]" << std::endl;

        ofs << "dofs = [";
        for(int i = 0; i < Ndofs; i++)
            ofs << dofs[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "IDs = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << IDs[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "initValues = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << initValues[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "initDerivatives = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << initDerivatives[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "absoluteTolerances = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << absoluteTolerances[i] << ", ";
        ofs << "]" << std::endl;
    }
*/

    //%(stnActiveStates)s
}

void daeModel_t::Free()
{
    if(g_pEvaluator)
    {
        g_pEvaluator->FreeResources();
        delete g_pEvaluator;
        g_pEvaluator = NULL;
    }

/*
// These are not needed anymore
    if(ids)
        delete[] ids;
    if(initValues)
        delete[] initValues;
    if(initDerivatives)
        delete[] initDerivatives;
    if(absoluteTolerances)
        delete[] absoluteTolerances;
    if(variableNames)
        delete[] variableNames;

    ids                = NULL;
    initValues         = NULL;
    initDerivatives    = NULL;
    absoluteTolerances = NULL;
    variableNames      = NULL;
*/
}

void daeModel_t::InitializeValuesReferences()
{
    // Reserve the size for internal vectors/maps
    size_t Nforeign = g_partitionData.foreignIndexes.size();
    size_t Ntot     = Nequations_local + Nforeign;

    g_values.resize(Ntot, 0.0);
    g_timeDerivatives.resize(Ntot, 0.0);
    g_mapValues.reserve(Ntot);
    g_mapTimeDerivatives.reserve(Ntot);

    if(g_partitionData.biToBiLocal.size() != Ntot)
        throw std::runtime_error("Invalid number of items in bi_to_bi_local map");

    // Insert the pointers to the owned and the foreign values
    // Owned data are always in the range: [0, Nequations_local)
    for(std::map<int32_t,int32_t>::iterator iter = g_partitionData.biToBiLocal.begin(); iter != g_partitionData.biToBiLocal.end(); iter++)
    {
        int bi = iter->first;  // global block index
        int li = iter->second; // local index

        g_mapValues[bi]          = &g_values[li];
        g_mapTimeDerivatives[bi] = &g_timeDerivatives[li];
    }

    // Initialize pointer maps
    for(csPartitionIndexMap::iterator it = g_partitionData.sendToIndexes.begin(); it != g_partitionData.sendToIndexes.end(); it++)
    {
        // it->first is int (rank)
        // it->second is vector<int>
        int rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values_arr(i_size, 0.0),     derivs_arr(i_size, 0.0);
        std::vector<real_t*> pvalues_arr(i_size, nullptr), pderivs_arr(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues_arr[i] = g_mapValues.at         ( indexes[i] );
            pderivs_arr[i] = g_mapTimeDerivatives.at( indexes[i] );
        }

        mapValuesData.sendToIndexes[rank]   = make_pair(values_arr,  derivs_arr);
        mapPointersData.sendToIndexes[rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        // it->first is int (rank)
        // it->second is vector<int>
        int rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values_arr(i_size, 0.0),     derivs_arr(i_size, 0.0);
        std::vector<real_t*> pvalues_arr(i_size, nullptr), pderivs_arr(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues_arr[i] = g_mapValues.at         ( indexes[i] );
            pderivs_arr[i] = g_mapTimeDerivatives.at( indexes[i] );
        }

        mapValuesData.receiveFromIndexes[rank]   = make_pair(values_arr,  derivs_arr);
        mapPointersData.receiveFromIndexes[rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    CheckSynchronisationIndexes(mpi_world, mpi_rank);

/*
    printf("Nequations_local = %d\n", Nequations_local);

    printf("foreign_indexes:\n");
    for(size_t i = 0; i < g_partitionData.foreign_indexes.size(); i++)
    {
        int32_t fi = g_partitionData.foreign_indexes[i];
        printf(" %d", fi);
    }
    printf("\n");

    printf("bi_to_bi_local (local):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = g_partitionData.bi_to_bi_local.begin(); iter != g_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local < Nequations_local)
            printf(" (%d, %d)", bi, bi_local);
    }
    printf("\n");

    printf("bi_to_bi_local (foreign):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = g_partitionData.bi_to_bi_local.begin(); iter != g_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local >= Nequations_local)
            printf(" (%d, %d)", bi, bi_local);
    }
    printf("\n");
*/
}

int daeModel_t::EvaluateResiduals(real_t current_time,
                                  real_t* values,
                                  real_t* time_derivatives,
                                  real_t* residuals)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    call_stats::TimerCounter tc(tcs.ResidualsEvaluation);

    /* The values and timeDerivatives have been copied in modSynchroniseData function. */

    real_t* pdofs            = (g_dofs.size() > 0 ? &g_dofs[0] : NULL);
    real_t* pvalues          = &g_values[0];
    real_t* ptimeDerivatives = &g_timeDerivatives[0];

    csEvaluationContext_t EC;
    EC.equationEvaluationMode    = cs::eEvaluateResidual;
    EC.sensitivityParameterIndex = -1;
    EC.jacobianIndex             = -1;
    EC.numberOfVariables         = Nequations_local;
    EC.numberOfEquations         = Nequations_local; // ???
    EC.numberOfDOFs              = g_dofs.size();
    EC.numberOfComputeStackItems = g_numberOfComputeStackItems;
    EC.numberOfJacobianItems     = 0;
    EC.valuesStackSize           = 5;
    EC.lvaluesStackSize          = 20;
    EC.rvaluesStackSize          = 5;
    EC.currentTime               = current_time;
    EC.inverseTimeStep           = 0; // Should not be needed here. Double check...
    EC.startEquationIndex        = 0; // !!!
    EC.startJacobianIndex        = 0; // !!!

    g_pEvaluator->EvaluateResiduals(EC, pdofs, pvalues, ptimeDerivatives, residuals);

/*
    if(mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modResiduals-node-") + std::to_string(mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out|std::ofstream::app);

        ofs << std::setiosflags(std::ios_base::fixed);
        ofs << std::setprecision(15);

        ofs << "time = " << current_time << std::endl;
        flat_map<int, real_t*>::const_iterator iter;
        ofs << "g_mapValues " << g_mapValues.size() << std::endl;
        for(iter = g_mapValues.begin(); iter != g_mapValues.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "g_mapTimeDerivatives " << g_mapTimeDerivatives.size() << std::endl;
        for(iter = g_mapTimeDerivatives.begin(); iter != g_mapTimeDerivatives.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "residuals " << Nequations_local << std::endl;
        for(int i = 0; i < Nequations_local; i++)
            ofs << "["<< i << ":" << residuals[i] << "]";
        ofs << std::endl;
    }
*/
    return 0;
}

int daeModel_t::EvaluateJacobian(real_t             current_time,
                                 real_t             inverse_time_step,
                                 real_t*            values,
                                 real_t*            time_derivatives,
                                 real_t*            residuals,
                                 daeMatrixAccess_t* ma)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    call_stats::TimerCounter tc(tcs.JacobianEvaluation);

    /* The values and timeDerivatives have been copied in modSynchroniseData function. */

    real_t* pdofs            = (g_dofs.size() > 0 ? const_cast<real_t*>(&g_dofs[0]) : NULL);
    real_t* pvalues          = &g_values[0];
    real_t* ptimeDerivatives = &g_timeDerivatives[0];

    g_jacobian.resize(g_numberOfJacobianItems, 0.0);

    csEvaluationContext_t EC;
    EC.equationEvaluationMode    = cs::eEvaluateJacobian;
    EC.sensitivityParameterIndex = -1;
    EC.jacobianIndex             = -1;
    EC.numberOfVariables         = Nequations_local;
    EC.numberOfEquations         = Nequations_local; // ???
    EC.numberOfDOFs              = g_dofs.size();
    EC.numberOfComputeStackItems = g_numberOfComputeStackItems;
    EC.numberOfJacobianItems     = g_numberOfJacobianItems;
    EC.valuesStackSize           = 5;
    EC.lvaluesStackSize          = 20;
    EC.rvaluesStackSize          = 5;
    EC.currentTime               = current_time;
    EC.inverseTimeStep           = inverse_time_step;
    EC.startEquationIndex        = 0; // !!!
    EC.startJacobianIndex        = 0; // !!!

    g_pEvaluator->EvaluateJacobian(EC, pdofs, pvalues, ptimeDerivatives, &g_jacobian[0]);

    // Evaluated Jacobian values need to be copied to the Jacobian matrix.
    for(size_t ji = 0; ji < g_numberOfJacobianItems; ji++)
    {
        const csJacobianMatrixItem_t& jacobianItem = g_jacobianMatrixItems[ji];
        size_t ei_local = jacobianItem.equationIndex;
        size_t bi_local = jacobianItem.blockIndex; // it is already updated to local blockIndex in loadComputeStack

        //std::cout << "ei_local = "  << ei_local << std::endl;
        //std::cout << "bi_local = "  << bi_local << std::endl;

        // Some items contain block indexes that are foreign to this PE.
        // In this case skip the caculation.
        if(bi_local >= Nequations_local)
            continue;

        ma->SetItem(ei_local, bi_local, g_jacobian[ji]);
    }

/*
    if(mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modResiduals-node-") + std::to_string(mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out|std::ofstream::app);

        ofs << std::setiosflags(std::ios_base::fixed);
        ofs << std::setprecision(15);

        ofs << std::endl << "time = " << current_time << std::endl;
        flat_map<int, real_t*>::const_iterator iter;
        ofs << "g_mapValues " << g_mapValues.size() << std::endl;
        for(iter = g_mapValues.begin(); iter != g_mapValues.end(); iter++)
            if(iter->first < 5)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "g_mapTimeDerivatives " << g_mapTimeDerivatives.size() << std::endl;
        for(iter = g_mapTimeDerivatives.begin(); iter != g_mapTimeDerivatives.end(); iter++)
            if(iter->first < 5)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "Jacobian " << Nequations_local << std::endl;
        for(size_t jdi = 0; jdi < g_numberOfJacobianItems; jdi++)
        {
            int ji = startJacobianIndex + jdi;
            const csJacobianMatrixItem_t& jacobianItem = g_jacobianMatrixItems[ji];
            size_t ei_local = jacobianItem.equationIndex - startEquationIndex;
            size_t bi_local = g_partitionData.bi_to_bi_local[jacobianItem.blockIndex];
            if(bi_local >= Nequations_local)
                continue;
            ofs << ei_local << " " << bi_local << ": " << _jacobian_matrix_(ei_local,bi_local) << std::endl;
        }
    }
*/
    return 0;
}

int daeModel_t::NumberOfRoots()
{
//%(numberOfRoots)s

    return 0;
}

int daeModel_t::Roots(real_t current_time,
                      real_t* values,
                      real_t* time_derivatives,
                      real_t* roots)
{
//%(roots)s

    return 0;
}

bool daeModel_t::CheckForDiscontinuities(real_t current_time,
                                         real_t* values,
                                         real_t* time_derivatives)
{
//%(checkForDiscontinuities)s

    return false;
}

daeeDiscontinuityType daeModel_t::ExecuteActions(real_t current_time,
                                                 real_t* values,
                                                 real_t* time_derivatives)
{
//%(executeActions)s

    return eNoDiscontinuity;
}

// Variable local indexes in DAE system equations as a CSR matrix.
void daeModel_t::GetDAESystemStructure(int& N, int& NNZ, std::vector<int>& IA, std::vector<int>& JA)
{
    std::vector<size_t> numColumnsInRows;

    IA.reserve(Nequations_local + 1);
    JA.reserve(g_numberOfJacobianItems);
    numColumnsInRows.resize(Nequations_local, 0);

    int removed = 0;
    for(size_t ji = 0; ji < g_numberOfJacobianItems; ji++)
    {
        const csJacobianMatrixItem_t& jacobianItem = g_jacobianMatrixItems[ji];
        size_t ei_local = jacobianItem.equationIndex;
        size_t bi_local = jacobianItem.blockIndex;

        // Foreign indexes in this PE must be omitted from the matrix
        // for their indexes are out of the range [0, Nequations_local).
        //
        // VERY IMPORTANT!!
        //   This causes a very slow convergence if direct sparse solvers are used.
        if(bi_local >= Nequations_local)
        {
            removed++;
            continue;
        }

        numColumnsInRows[ei_local] += 1;
        JA.push_back(bi_local);
    }
    //printf("No. removed indexes = %d\n", removed);

    int endOfRow = 0;
    IA.push_back(0);
    for(size_t ri = 0; ri < Nequations_local; ri++)
    {
        endOfRow += numColumnsInRows[ri];
        IA.push_back(endOfRow);
    }
    N   = Nequations_local;
    NNZ = JA.size();
}

void daeModel_t::CheckSynchronisationIndexes(void* mpi_world, int mpi_rank)
{
    // Get synchronisation info for the current node
    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    std::vector<mpi::request> requests;
    csPartitionIndexMap received_indexes;

    // Send the data to the other nodes
    for(csPartitionIndexMap::iterator it = g_partitionData.sendToIndexes.begin(); it != g_partitionData.sendToIndexes.end(); it++)
    {
        requests.push_back( world.isend(it->first, 0, it->second) );
    }

    // Receive the data from the other nodes
    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        received_indexes[it->first] = std::vector<int>();
        requests.push_back( world.irecv(it->first, 0, received_indexes[it->first]) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Check if we received the correct indexes
    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        if(g_partitionData.receiveFromIndexes[it->first] != received_indexes[it->first])
            throw std::runtime_error(std::string("The received indexes do not match the requested ones, node: ") + std::to_string(mpi_rank));
    }

/*
    // Just for the debugging purposes print sent/received indexes to a file
    std::ofstream ofs;
    std::string filename = std::string("node-") + std::to_string(mpi_rank) + ".txt";
    ofs.open(filename, std::ofstream::out);

    ofs << "Node " << mpi_rank << std::endl;
    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        ofs << "Expected from " << it->first << ": " << std::endl;
        for(size_t i = 0; i < g_partitionData.receiveFromIndexes[it->first].size(); i++)
            ofs << g_partitionData.receiveFromIndexes[it->first][i] << ", ";
        ofs << std::endl;

        ofs << "Received from " << it->first << ": "  << std::endl;
        for(size_t i = 0; i < received_indexes[it->first].size(); i++)
            ofs << received_indexes[it->first][i] << ", ";
        ofs << std::endl << std::endl;
    }
    ofs.close();
*/
}

void daeModel_t::SynchroniseData(real_t time, real_t* daesolver_values, real_t* daesolver_time_derivatives)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    call_stats::TimerCounter tc(tcs.IPCDataExchange);

    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    // Copy values/derivatives from DAE solver into the local storage.
    for(int i = 0; i < Nequations_local; i++)
    {
        g_values[i]          = daesolver_values[i];
        g_timeDerivatives[i] = daesolver_time_derivatives[i];
    }

    // Get synchronisation info for the current node
    std::vector<mpi::request> requests;

    // Send the data to the other nodes
    for(csPartitionIndexMap::iterator it = g_partitionData.sendToIndexes.begin(); it != g_partitionData.sendToIndexes.end(); it++)
    {
        int               send_to_mpi_rank = it->first;
        std::vector<int>& indexes          = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = mapPointersData.sendToIndexes[send_to_mpi_rank].first;
        std::vector<real_t*>& pderivs = mapPointersData.sendToIndexes[send_to_mpi_rank].second;
        std::vector<real_t>&  values  = mapValuesData.sendToIndexes[send_to_mpi_rank].first;
        std::vector<real_t>&  derivs  = mapValuesData.sendToIndexes[send_to_mpi_rank].second;

        for(size_t i = 0; i < i_size; i++)
        {
            values[i] = *pvalues[i];
            derivs[i] = *pderivs[i];
        }

        requests.push_back( world.isend(send_to_mpi_rank, 1, values) );
        requests.push_back( world.isend(send_to_mpi_rank, 2, derivs) );
    }

    // Receive the data from the other nodes
    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        int receive_from_mpi_rank = it->first;

        std::vector<real_t>& values  = mapValuesData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t>& derivs  = mapValuesData.receiveFromIndexes[receive_from_mpi_rank].second;

        requests.push_back( world.irecv(receive_from_mpi_rank, 1, values) );
        requests.push_back( world.irecv(receive_from_mpi_rank, 2, derivs) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Copy the data from the pointer arrays to values arrays
    for(csPartitionIndexMap::iterator it = g_partitionData.receiveFromIndexes.begin(); it != g_partitionData.receiveFromIndexes.end(); it++)
    {
        int               receive_from_mpi_rank = it->first;
        std::vector<int>& indexes               = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = mapPointersData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t*>& pderivs = mapPointersData.receiveFromIndexes[receive_from_mpi_rank].second;
        std::vector<real_t>&  values  = mapValuesData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t>&  derivs  = mapValuesData.receiveFromIndexes[receive_from_mpi_rank].second;

        if(indexes.size() != values.size()  ||
           indexes.size() != derivs.size()  ||
           indexes.size() != pvalues.size() ||
           indexes.size() != pderivs.size())
            throw std::runtime_error(std::string("The received data do not match the requested ones, node: ") + std::to_string(mpi_rank));
        //else
        //    std::cout << "Node [" << mpi_rank << "] transferred " << values.size() << " values" << std::endl;

        for(size_t i = 0; i < i_size; i++)
        {
            *pvalues[i] = values[i];
            *pderivs[i] = derivs[i];
        }

        if(false /*mpi_rank == 0*/)
        {
            std::stringstream ss;
            ss << "Node [" << mpi_rank << "] values from node [" << receive_from_mpi_rank << "]:" << std::endl;
            for(size_t i = 0; i < i_size; i++)
                ss << *pvalues[i] << ", ";
            ss << std::endl;
            std::cout << ss.str();
        }
    }

    /*
    // Check the mapping
    for(flat_map<int32_t,int32_t>::iterator iter = g_partitionData.bi_to_bi_local.begin(); iter != g_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index

        double value       = g_values[bi_local];
        double deriv       = g_timeDerivatives[bi_local];
        double mappedValue = *g_mapValues[bi];
        double mappedDeriv = *g_mapTimeDerivatives[bi];

        if(value-mappedValue != 0.0)
            printf("  Value at (%d, %d) = does not match\n", bi, bi_local, value, mappedValue);
        if(deriv-mappedDeriv != 0.0)
            printf("  Value at (%d, %d) = does not match\n", bi, bi_local, deriv, mappedDeriv);
    }
    */

    /*
    // Now values and timeDerivatives are synchronised.

    std::string filename = inputDirectory + (boost::format("/mpi_sync-%05d.txt") % mpi_rank).str();
    FILE* f = fopen(filename.c_str(), "a");

    fprintf(f, "mpiSynchroniseData (%.15f):\n", time);
    fprintf(f, "bi_to_bi_local (local):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = g_partitionData.bi_to_bi_local.begin(); iter != g_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local < Nequations_local)
            fprintf(f, " (%d, %d) = %+.15f, %+.15f\n", bi, bi_local, *g_mapValues[bi], *g_mapTimeDerivatives[bi]);
    }
    fprintf(f, "\n");

    fprintf(f, "bi_to_bi_local (foreign):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = g_partitionData.bi_to_bi_local.begin(); iter != g_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local >= Nequations_local)
            fprintf(f, " (%d, %d) = %+.15f, %+.15f\n", bi, bi_local, *g_mapValues[bi], *g_mapTimeDerivatives[bi]);
    }
    fprintf(f, "\n");
    fprintf(f, "****************************************************************\n");

    fclose(f);
    */
}

/*
void loadSimulationData(daeSimulation_t* simulation, const std::string& inputDirectory)
{
    std::ifstream f;

    // Get runtime data for the current node.
    std::string filename = (boost::format("simulation_data-%05d.bin") % simulation->model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    f.read((char*)&simulation->startTime,          sizeof(real_t));
    f.read((char*)&simulation->timeHorizon,        sizeof(real_t));
    f.read((char*)&simulation->reportingInterval,  sizeof(real_t));
    f.read((char*)&simulation->relativeTolerance,  sizeof(real_t));

    f.close();
}
*/

void loadModelVariables(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    // Get runtime data for the current node.
    std::string filename = (boost::format("model_variables-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    f.read((char*)&model->Nequations,         sizeof(uint32_t));
    f.read((char*)&model->Nequations_local,   sizeof(uint32_t));
    f.read((char*)&model->Ndofs,              sizeof(uint32_t));
    f.read((char*)&model->quasiSteadyState,   sizeof(bool));

    int32_t Nitems;

    // dofs
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->dofs.resize(Nitems);
        f.read((char*)(&model->dofs[0]), sizeof(real_t) * Nitems);
    }

    // init_values
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->variableValues.resize(Nitems);
        f.read((char*)(&model->variableValues[0]), sizeof(real_t) * Nitems);
    }

    // init_derivatives
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->variableDerivatives.resize(Nitems);
        f.read((char*)(&model->variableDerivatives[0]), sizeof(real_t) * Nitems);
    }

    // absolute_tolerances
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->absoluteTolerances.resize(Nitems);
        f.read((char*)(&model->absoluteTolerances[0]), sizeof(real_t) * Nitems);
    }

    // ids
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->ids.resize(Nitems);
        f.read((char*)(&model->ids[0]), sizeof(int32_t) * Nitems);
    }

    // variable_names (skipped)
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems == 0)
    {
        model->variableNames.resize(model->Nequations_local);
        for(int i = 0; i < model->Nequations_local; i++)
        {
            model->variableNames[i] = (boost::format("y(%d)") % i).str();
        }
    }
    else
    {
        model->variableNames.resize(Nitems);

        int32_t length;
        char name[4096];
        for(int i = 0; i < Nitems; i++)
        {
            // Read string length
            f.read((char*)&length,  sizeof(int32_t));

            // Read string
            f.read((char*)(&name[0]), sizeof(char) * length);
            name[length] = '\0';

            model->variableNames[i] = std::string(name);
        }
    }

    f.close();
}

void loadPartitionData(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    // Get synchronisation data for the current node.
    std::string filename = (boost::format("partition_data-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    // MPI functions require integers so everything is saved as singed integers
    if(sizeof(int32_t) != sizeof(int))
        throw std::runtime_error("Invalid size of int (must be 4 bytes)");

    // foreign_indexes
    int32_t Nforeign;
    f.read((char*)&Nforeign,  sizeof(int32_t));
    if(Nforeign > 0)
    {
        g_partitionData.foreignIndexes.resize(Nforeign);
        f.read((char*)(&g_partitionData.foreignIndexes[0]), sizeof(int32_t) * Nforeign);
    }

    // bi_to_bi_local
    int32_t Nbi_to_bi_local_pairs, bi, bi_local;
    f.read((char*)&Nbi_to_bi_local_pairs,  sizeof(int32_t));
    //g_partitionData.bi_to_bi_local.reserve(Nbi_to_bi_local_pairs);
    for(int32_t i = 0; i < Nbi_to_bi_local_pairs; i++)
    {
        f.read((char*)&bi,       sizeof(int32_t));
        f.read((char*)&bi_local, sizeof(int32_t));
        g_partitionData.biToBiLocal[bi] = bi_local;
    }

    // sendToIndexes
    int32_t Nsend_to;
    f.read((char*)&Nsend_to,  sizeof(int32_t));
    for(int32_t i = 0; i < Nsend_to; i++)
    {
        int32_t rank, Nindexes;
        std::vector<int32_t> indexes;

        f.read((char*)&rank,     sizeof(int32_t));
        f.read((char*)&Nindexes, sizeof(int32_t));
        indexes.resize(Nindexes);
        f.read((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
        g_partitionData.sendToIndexes[rank] = indexes;

        if(!std::is_sorted(indexes.begin(), indexes.end()))
            throw std::runtime_error( (boost::format("sendToIndexes[%d][%d] indexes are not sorted") % model->mpi_rank % rank).str() );
    }

    // receiveFromIndexes
    int32_t Nreceive_from;
    f.read((char*)&Nreceive_from,  sizeof(int32_t));
    for(int32_t i = 0; i < Nreceive_from; i++)
    {
        int32_t rank, Nindexes;
        std::vector<int32_t> indexes;

        f.read((char*)&rank,     sizeof(int32_t));
        f.read((char*)&Nindexes, sizeof(int32_t));
        indexes.resize(Nindexes);
        f.read((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
        g_partitionData.receiveFromIndexes[rank] = indexes;

        if(!std::is_sorted(indexes.begin(), indexes.end()))
            throw std::runtime_error( (boost::format("receiveFromIndexes[%d][%d] indexes are not sorted") % model->mpi_rank % rank).str() );
    }
    f.close();
}

void loadModelEquations(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    std::string filename = (boost::format("model_equations-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    uint32_t Ncs  = 0;
    uint32_t Nasi = 0;

    f.read((char*)&Nasi, sizeof(uint32_t));
    f.read((char*)&Ncs,  sizeof(uint32_t));

    g_arrActiveEquationSetIndexes.resize(Nasi);
    g_arrAllComputeStacks.resize(Ncs);

    f.read((char*)(&g_arrActiveEquationSetIndexes[0]), sizeof(uint32_t)             * Nasi);
    f.read((char*)(&g_arrAllComputeStacks[0]),         sizeof(csComputeStackItem_t) * Ncs);
    f.close();

    g_activeEquationSetIndexes = &g_arrActiveEquationSetIndexes[0];
    g_computeStacks            = &g_arrAllComputeStacks[0];

    g_numberOfComputeStackItems = g_arrAllComputeStacks.size();
}

void loadJacobianData(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    std::string filename = (boost::format("jacobian_data-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    uint32_t Nji = 0;

    f.read((char*)&Nji, sizeof(uint32_t));

    g_arrJacobianMatrixItems.resize(Nji);

    f.read((char*)(&g_arrJacobianMatrixItems[0]), sizeof(csJacobianMatrixItem_t) * Nji);
    f.close();

    g_jacobianMatrixItems   = &g_arrJacobianMatrixItems[0];

    g_numberOfJacobianItems = g_arrJacobianMatrixItems.size();
}


}
