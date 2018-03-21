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
#include "runtime_information.h"
#include <exception>
#include <fstream>
#include <iomanip>
#include "compute_stack_openmp.h"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/container/flat_map.hpp>
using boost::container::flat_map;

/* Internal storage containers for evaluation of equations. */
std::vector<real_t> g_values;
std::vector<real_t> g_timeDerivatives;
std::vector<real_t> g_dofs;
std::vector<real_t> g_jacobian;

/* Data related to MPI communication (include index mappings for both owned and foreign indexes). */
flat_map<int, real_t*> g_mapValues;
flat_map<int, real_t*> g_mapTimeDerivatives;

/* Initialisation data (loaded from files). */
mpiIndexesData         g_syncData;
runtimeInformationData g_rtData;

/* ComputeStack-related data. */
std::vector<adComputeStackItem_t>   g_arrAllComputeStacks;
std::vector<adJacobianMatrixItem_t> g_arrComputeStackJacobianItems;
std::vector<uint32_t>               g_arrActiveEquationSetIndexes;
uint32_t*                           g_activeEquationSetIndexes  = NULL;
adComputeStackItem_t*               g_computeStacks             = NULL;
adJacobianMatrixItem_t*             g_jacobianMatrixItems       = NULL;
uint32_t                            g_numberOfJacobianItems     = -1;
uint32_t                            g_numberOfComputeStackItems = -1;
adComputeStackEvaluator_t*          g_pEvaluator                = NULL;

static void loadRuntimeData(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    // Get runtime data for the current node.
    std::string filename = (boost::format("runtime_data-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    f.read((char*)&g_rtData.i_start,            sizeof(int32_t));
    f.read((char*)&g_rtData.i_end,              sizeof(int32_t));
    f.read((char*)&g_rtData.Ntotal_vars,        sizeof(int32_t));
    f.read((char*)&g_rtData.Nequations,         sizeof(int32_t));
    f.read((char*)&g_rtData.Nequations_local,   sizeof(int32_t));
    f.read((char*)&g_rtData.Ndofs,              sizeof(int32_t));

    f.read((char*)&g_rtData.startTime,          sizeof(real_t));
    f.read((char*)&g_rtData.timeHorizon,        sizeof(real_t));
    f.read((char*)&g_rtData.reportingInterval,  sizeof(real_t));
    f.read((char*)&g_rtData.relativeTolerance,  sizeof(real_t));

    f.read((char*)&g_rtData.quasiSteadyState,  sizeof(bool));

    int32_t Nitems;

    // dofs
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        g_rtData.dofs.resize(Nitems);
        f.read((char*)(&g_rtData.dofs[0]), sizeof(real_t) * Nitems);
    }

    // init_values
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        g_rtData.init_values.resize(Nitems);
        f.read((char*)(&g_rtData.init_values[0]), sizeof(real_t) * Nitems);
    }

    // init_derivatives
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        g_rtData.init_derivatives.resize(Nitems);
        f.read((char*)(&g_rtData.init_derivatives[0]), sizeof(real_t) * Nitems);
    }

    // absolute_tolerances
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        g_rtData.absolute_tolerances.resize(Nitems);
        f.read((char*)(&g_rtData.absolute_tolerances[0]), sizeof(real_t) * Nitems);
    }

    // ids
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        g_rtData.ids.resize(Nitems);
        f.read((char*)(&g_rtData.ids[0]), sizeof(int32_t) * Nitems);
    }

    // variable_names (skipped)
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems == 0)
    {
        g_rtData.variable_names.resize(g_rtData.Nequations_local);
        for(int i = 0; i < g_rtData.Nequations_local; i++)
        {
            int index = g_rtData.i_start + i;
            g_rtData.variable_names[i] = (boost::format("y(%d)") % index).str();
        }
    }
    else
    {
        g_rtData.variable_names.resize(Nitems);

        int32_t length;
        char name[4096];
        for(int i = 0; i < Nitems; i++)
        {
            // Read string length
            f.read((char*)&length,  sizeof(int32_t));

            // Read string
            f.read((char*)(&name[0]), sizeof(char) * length);
            name[length] = '\0';

            g_rtData.variable_names[i] = std::string(name);
        }
    }

    f.close();
}

static void loadInterprocessCommunicationData(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    // Get synchronisation data for the current node.
    std::string filename = (boost::format("interprocess_comm_data-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    // MPI functions require integers so everything is saved as singed integers
    if(sizeof(int32_t) != sizeof(int))
        throw std::runtime_error("Invalid size of int (must be 4 bytes)");

    f.read((char*)&g_syncData.i_start,  sizeof(int32_t));
    f.read((char*)&g_syncData.i_end,    sizeof(int32_t));

    // foreign_indexes
    int32_t Nforeign;
    f.read((char*)&Nforeign,  sizeof(int32_t));
    if(Nforeign > 0)
    {
        g_syncData.foreign_indexes.resize(Nforeign);
        f.read((char*)(&g_syncData.foreign_indexes[0]), sizeof(int32_t) * Nforeign);
    }

    // bi_to_bi_local
    int32_t Nbi_to_bi_local_pairs, bi, bi_local;
    f.read((char*)&Nbi_to_bi_local_pairs,  sizeof(int32_t));
    g_syncData.bi_to_bi_local.reserve(Nbi_to_bi_local_pairs);
    for(int32_t i = 0; i < Nbi_to_bi_local_pairs; i++)
    {
        f.read((char*)&bi,       sizeof(int32_t));
        f.read((char*)&bi_local, sizeof(int32_t));
        g_syncData.bi_to_bi_local[bi] = bi_local;
    }

    // send_to
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
        g_syncData.send_to[rank] = indexes;
    }

    // receive_from
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
        g_syncData.receive_from[rank] = indexes;
    }
    f.close();
}

static void loadModelEquations(daeModel_t* model, const std::string& inputDirectory)
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
    f.read((char*)(&g_arrAllComputeStacks[0]),         sizeof(adComputeStackItem_t) * Ncs);
    f.close();

    g_activeEquationSetIndexes = &g_arrActiveEquationSetIndexes[0];
    g_computeStacks            = &g_arrAllComputeStacks[0];

    g_numberOfComputeStackItems = g_arrAllComputeStacks.size();
}

static void loadPreconditionerData(daeModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;

    std::string filename = (boost::format("preconditioner_data-%05d.bin") % model->mpi_rank).str();
    boost::filesystem::path inputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );
    std::string filePath = (inputDataPath / filename).string();
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    uint32_t Nji = 0;

    f.read((char*)&Nji, sizeof(uint32_t));

    g_arrComputeStackJacobianItems.resize(Nji);

    f.read((char*)(&g_arrComputeStackJacobianItems[0]), sizeof(adJacobianMatrixItem_t) * Nji);
    f.close();

    g_jacobianMatrixItems   = &g_arrComputeStackJacobianItems[0];

    g_numberOfJacobianItems = g_arrComputeStackJacobianItems.size();
}

void modInitialize(daeModel_t* model, const std::string& inputDirectory)
{
    model->inputDirectory = inputDirectory;

    loadRuntimeData(model, inputDirectory);

    loadInterprocessCommunicationData(model, inputDirectory);

    model->Nequations_local   = g_rtData.Nequations_local;
    model->Ntotal_vars        = g_rtData.Ntotal_vars;
    model->Nequations         = g_rtData.Nequations;
    model->Ndofs              = g_rtData.Ndofs;
    model->startTime          = g_rtData.startTime;
    model->timeHorizon        = g_rtData.timeHorizon;
    model->reportingInterval  = g_rtData.reportingInterval;
    model->relativeTolerance  = g_rtData.relativeTolerance;
    model->quasiSteadyState   = g_rtData.quasiSteadyState;

    model->ids                = new int   [model->Nequations_local];
    model->initValues         = new real_t[model->Nequations_local];
    model->initDerivatives    = new real_t[model->Nequations_local];
    model->absoluteTolerances = new real_t[model->Nequations_local];
    model->variableNames      = new const char*[model->Nequations_local];
    for(int i = 0; i < model->Nequations_local; i++)
    {
        model->ids[i]                = g_rtData.ids[i];
        model->initValues[i]         = g_rtData.init_values[i];
        model->initDerivatives[i]    = g_rtData.init_derivatives[i];
        model->absoluteTolerances[i] = g_rtData.absolute_tolerances[i];
        model->variableNames[i]      = g_rtData.variable_names[i].c_str();
    }

    g_dofs.resize(model->Ndofs, 0.0);
    for(int i = 0; i < model->Ndofs; i++)
        g_dofs[i] = g_rtData.dofs[i];

    loadModelEquations(model, inputDirectory);
    loadPreconditionerData(model, inputDirectory);

    g_pEvaluator = new daeComputeStackEvaluator_OpenMP();
    g_pEvaluator->Initialize(false,
                             model->Nequations_local,
                             model->Nequations_local,
                             model->Ndofs,
                             g_numberOfComputeStackItems,
                             g_numberOfJacobianItems,
                             g_numberOfJacobianItems,
                             g_computeStacks,
                             g_activeEquationSetIndexes,
                             g_jacobianMatrixItems);

    modInitializeValuesReferences(model);

/*
    if(model->mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modInitialize-node-") + std::to_string(model->mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out);

        ofs << "Nequations_local = "  << model->Nequations_local << std::endl;
        ofs << "Ntotal_vars = "       << model->Ntotal_vars << std::endl;
        ofs << "Nequations = "        << model->Nequations << std::endl;
        ofs << "Ndofs = "             << model->Ndofs << std::endl;
        ofs << "startTime = "         << model->startTime << std::endl;
        ofs << "timeHorizon = "       << model->timeHorizon << std::endl;
        ofs << "reportingInterval = " << model->reportingInterval << std::endl;
        ofs << "relativeTolerance = " << model->relativeTolerance << std::endl;
        ofs << "quasiSteadyState = "  << model->quasiSteadyState << std::endl;

        ofs << "g_numberOfComputeStackItems = "  << g_numberOfComputeStackItems << std::endl;

        ofs << "startEquationIndex = "  << startEquationIndex << std::endl;
        ofs << "startJacobianIndex = "  << startJacobianIndex << std::endl;
        ofs << "g_numberOfJacobianItems = "  << g_numberOfJacobianItems << std::endl;

        ofs << "ActiveEquationSetIndexes = [";
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << g_activeEquationSetIndexes[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "g_jacobianMatrixItems equationIndexes = [";
        for(int ji = 0; ji < g_numberOfJacobianItems; ji++)
        {
            adJacobianMatrixItem_t& jd = g_jacobianMatrixItems[ji];
            ofs << jd.equationIndex << ", ";
        }
        ofs << "]" << std::endl;

        ofs << "dofs = [";
        for(int i = 0; i < model->Ndofs; i++)
            ofs << dofs[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "IDs = [";
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << model->IDs[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "initValues = [";
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << model->initValues[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "initDerivatives = [";
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << model->initDerivatives[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "absoluteTolerances = [";
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << model->absoluteTolerances[i] << ", ";
        ofs << "]" << std::endl;
    }
*/

    //%(stnActiveStates)s
}

void modFinalize(daeModel_t* model)
{
// These are not needed anymore
    delete[] model->ids;
    delete[] model->initValues;
    delete[] model->initDerivatives;
    delete[] model->absoluteTolerances;
    delete[] model->variableNames;
    model->ids                = NULL;
    model->initValues         = NULL;
    model->initDerivatives    = NULL;
    model->absoluteTolerances = NULL;
    model->variableNames      = NULL;
}

void modInitializeValuesReferences(daeModel_t* model)
{
    // Reserve the size for internal vectors/maps
    size_t Nforeign = g_syncData.foreign_indexes.size();
    size_t Ntot     = g_rtData.Nequations_local + Nforeign;

    g_values.resize(Ntot, 0.0);
    g_timeDerivatives.resize(Ntot, 0.0);
    g_mapValues.reserve(Ntot);
    g_mapTimeDerivatives.reserve(Ntot);

    if(g_syncData.bi_to_bi_local.size() != Ntot)
        throw std::runtime_error("Invalid number of items in bi_to_bi_local map");

    // Insert the pointers to the owned and the foreign values
    // Owned data are always in the range: [0, Nequations_local)
    for(flat_map<int,int>::iterator iter = g_syncData.bi_to_bi_local.begin(); iter != g_syncData.bi_to_bi_local.end(); iter++)
    {
        int bi = iter->first;  // block index
        int li = iter->second; // local index

        g_mapValues[bi]          = &g_values[li];
        g_mapTimeDerivatives[bi] = &g_timeDerivatives[li];
    }

    // Initialize pointer maps
    for(mpiSyncMap::iterator it = g_syncData.send_to.begin(); it != g_syncData.send_to.end(); it++)
    {
        // it->first is int (mpi_rank)
        // it->second is vector<int>
        int mpi_rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values_arr(i_size, 0.0),     derivs_arr(i_size, 0.0);
        std::vector<real_t*> pvalues_arr(i_size, nullptr), pderivs_arr(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues_arr[i] = g_mapValues         [ indexes[i] ];
            pderivs_arr[i] = g_mapTimeDerivatives[ indexes[i] ];
        }

        mapValuesData.send_to[mpi_rank]   = make_pair(values_arr,  derivs_arr);
        mapPointersData.send_to[mpi_rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        // it->first is int (mpi_rank)
        // it->second is vector<int>
        int mpi_rank = it->first;
        std::vector<int>& indexes = it->second;
        int i_size = indexes.size();

        // Pointers to values/time_derivatives
        std::vector<real_t>   values_arr(i_size, 0.0),     derivs_arr(i_size, 0.0);
        std::vector<real_t*> pvalues_arr(i_size, nullptr), pderivs_arr(i_size, nullptr);

        for(int i = 0; i < i_size; i++)
        {
            pvalues_arr[i] = g_mapValues         [ indexes[i] ];
            pderivs_arr[i] = g_mapTimeDerivatives[ indexes[i] ];
        }

        mapValuesData.receive_from[mpi_rank]   = make_pair(values_arr,  derivs_arr);
        mapPointersData.receive_from[mpi_rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    mpiCheckSynchronisationIndexes(model, model->mpi_world, model->mpi_rank);
}

int modResiduals(daeModel_t* model,
                 real_t current_time,
                 real_t* values,
                 real_t* time_derivatives,
                 real_t* residuals)
{
    /* The values and timeDerivatives have been copied in modSynchroniseData function. */

    real_t* pdofs            = (g_dofs.size() > 0 ? &g_dofs[0] : NULL);
    real_t* pvalues          = &g_values[0];
    real_t* ptimeDerivatives = &g_timeDerivatives[0];

    daeComputeStackEvaluationContext_t EC;
    EC.equationCalculationMode                       = eCalculate;
    EC.sensitivityParameterIndex                     = -1;
    EC.jacobianIndex                                 = -1;
    EC.numberOfVariables                             = model->Nequations_local;
    EC.numberOfEquations                             = model->Nequations_local; // ???
    EC.numberOfDOFs                                  = g_dofs.size();
    EC.numberOfComputeStackItems                     = g_numberOfComputeStackItems;
    EC.numberOfJacobianItems                         = 0;
    EC.valuesStackSize                               = 5;
    EC.lvaluesStackSize                              = 20;
    EC.rvaluesStackSize                              = 5;
    EC.currentTime                                   = current_time;
    EC.inverseTimeStep                               = 0; // Should not be needed here. Double check...
    EC.startEquationIndex                            = 0; // !!!
    EC.startJacobianIndex                            = 0; // !!!

    g_pEvaluator->EvaluateResiduals(EC, pdofs, pvalues, ptimeDerivatives, residuals);

/*
    if(model->mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modResiduals-node-") + std::to_string(model->mpi_rank) + ".txt";
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
        ofs << "residuals " << model->Nequations_local << std::endl;
        for(int i = 0; i < model->Nequations_local; i++)
            ofs << "["<< i << ":" << residuals[i] << "]";
        ofs << std::endl;
    }
*/
    return 0;
}

int modJacobian(daeModel_t* model,
                long int number_of_equations,
                real_t current_time,
                real_t inverse_time_step,
                real_t* values,
                real_t* time_derivatives,
                real_t* residuals,
                void* matrix)
{
    /* The values and timeDerivatives have been copied in modSynchroniseData function. */

    real_t* pdofs            = (g_dofs.size() > 0 ? const_cast<real_t*>(&g_dofs[0]) : NULL);
    real_t* pvalues          = &g_values[0];
    real_t* ptimeDerivatives = &g_timeDerivatives[0];

    g_jacobian.resize(g_numberOfJacobianItems, 0.0);

    daeComputeStackEvaluationContext_t EC;
    EC.equationCalculationMode                       = eCalculateJacobian;
    EC.sensitivityParameterIndex                     = -1;
    EC.jacobianIndex                                 = -1;
    EC.numberOfVariables                             = model->Nequations_local;
    EC.numberOfEquations                             = model->Nequations_local; // ???
    EC.numberOfDOFs                                  = g_dofs.size();
    EC.numberOfComputeStackItems                     = g_numberOfComputeStackItems;
    EC.numberOfJacobianItems                         = g_numberOfJacobianItems;
    EC.valuesStackSize                               = 5;
    EC.lvaluesStackSize                              = 20;
    EC.rvaluesStackSize                              = 5;
    EC.currentTime                                   = current_time;
    EC.inverseTimeStep                               = inverse_time_step;
    EC.startEquationIndex                            = 0; // !!!
    EC.startJacobianIndex                            = 0; // !!!

    g_pEvaluator->EvaluateJacobian(EC, pdofs, pvalues, ptimeDerivatives, &g_jacobian[0]);

    // Evaluated Jacobian values need to be copied to the Jacobian matrix.
    for(size_t ji = 0; ji < g_numberOfJacobianItems; ji++)
    {
        const adJacobianMatrixItem_t& jacobianItem = g_jacobianMatrixItems[ji];
        size_t ei_local = jacobianItem.equationIndex;
        size_t bi_local = jacobianItem.blockIndex; // it is already updated to local blockIndex in loadComputeStack

        //std::cout << "ei_local = "  << ei_local << std::endl;
        //std::cout << "bi_local = "  << bi_local << std::endl;

        if(bi_local >= model->Nequations_local)
            continue;

        laSetMatrixItem(matrix, ei_local, bi_local, g_jacobian[ji]);
    }

/*
    if(model->mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modResiduals-node-") + std::to_string(model->mpi_rank) + ".txt";
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
        ofs << "Jacobian " << model->Nequations_local << std::endl;
        for(size_t jdi = 0; jdi < g_numberOfJacobianItems; jdi++)
        {
            int ji = startJacobianIndex + jdi;
            const adJacobianMatrixItem_t& jacobianItem = g_jacobianMatrixItems[ji];
            size_t ei_local = jacobianItem.equationIndex - startEquationIndex;
            size_t bi_local = g_syncData.bi_to_bi_local[jacobianItem.blockIndex];
            if(bi_local >= model->Nequations_local)
                continue;
            ofs << ei_local << " " << bi_local << ": " << _jacobian_matrix_(ei_local,bi_local) << std::endl;
        }
    }
*/
    return 0;
}

int modNumberOfRoots(daeModel_t* model)
{
//%(numberOfRoots)s

    return 0;
}

int modRoots(daeModel_t* model,
             real_t current_time,
             real_t* values,
             real_t* time_derivatives,
             real_t* roots)
{
//%(roots)s

    return 0;
}

bool modCheckForDiscontinuities(daeModel_t* model,
                                real_t current_time,
                                real_t* values,
                                real_t* time_derivatives)
{
//%(checkForDiscontinuities)s

    return false;
}

daeeDiscontinuityType modExecuteActions(daeModel_t* model,
                                        real_t current_time,
                                        real_t* values,
                                        real_t* time_derivatives)
{
//%(executeActions)s

    return eNoDiscontinuity;
}

void mpiCheckSynchronisationIndexes(daeModel_t* model, void* mpi_world, int mpi_rank)
{
    // Get synchronisation info for the current node
    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    std::vector<mpi::request> requests;
    mpiSyncMap received_indexes;

    // Send the data to the other nodes
    for(mpiSyncMap::iterator it = g_syncData.send_to.begin(); it != g_syncData.send_to.end(); it++)
    {
        requests.push_back( world.isend(it->first, 0, it->second) );
    }

    // Receive the data from the other nodes
    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        received_indexes[it->first] = std::vector<int>();
        requests.push_back( world.irecv(it->first, 0, received_indexes[it->first]) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Check if we received the correct indexes
    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        if(g_syncData.receive_from[it->first] != received_indexes[it->first])
            throw std::runtime_error(std::string("The received indexes do not match the requested ones, node: ") + std::to_string(mpi_rank));
    }

/*
    // Just for the debugging purposes print sent/received indexes to a file
    std::ofstream ofs;
    std::string filename = std::string("node-") + std::to_string(mpi_rank) + ".txt";
    ofs.open(filename, std::ofstream::out);

    ofs << "Node " << mpi_rank << std::endl;
    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        ofs << "Expected: " << std::endl;
        for(size_t i = 0; i < g_syncData.receive_from[it->first].size(); i++)
            ofs << g_syncData.receive_from[it->first][i] << ", ";
        ofs << std::endl;

        ofs << "Received: " << std::endl;
        for(size_t i = 0; i < received_indexes[it->first].size(); i++)
            ofs << received_indexes[it->first][i] << ", ";
        ofs << std::endl;
    }
    ofs.close();
*/
}

void mpiSynchroniseData(daeModel_t* model, real_t* values, real_t* time_derivatives)
{
    mpi::communicator& world = *(mpi::communicator*)model->mpi_world;
    int mpi_rank             = model->mpi_rank;

    for(int i = 0; i < model->Nequations_local; i++)
    {
        g_values[i]          = values[i];
        g_timeDerivatives[i] = time_derivatives[i];
    }

    // Get synchronisation info for the current node
    std::vector<mpi::request> requests;

    // Send the data to the other nodes
    for(mpiSyncMap::iterator it = g_syncData.send_to.begin(); it != g_syncData.send_to.end(); it++)
    {
        int               send_to_mpi_rank = it->first;
        std::vector<int>& indexes          = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = mapPointersData.send_to[send_to_mpi_rank].first;
        std::vector<real_t*>& pderivs = mapPointersData.send_to[send_to_mpi_rank].second;
        std::vector<real_t>&  values  = mapValuesData.send_to[send_to_mpi_rank].first;
        std::vector<real_t>&  derivs  = mapValuesData.send_to[send_to_mpi_rank].second;

        for(size_t i = 0; i < i_size; i++)
        {
            values[i] = *pvalues[i];
            derivs[i] = *pderivs[i];
        }

        requests.push_back( world.isend(send_to_mpi_rank, 1, values) );
        requests.push_back( world.isend(send_to_mpi_rank, 2, derivs) );
    }

    // Receive the data from the other nodes
    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        int receive_from_mpi_rank = it->first;

        std::vector<real_t>& values  = mapValuesData.receive_from[receive_from_mpi_rank].first;
        std::vector<real_t>& derivs  = mapValuesData.receive_from[receive_from_mpi_rank].second;

        requests.push_back( world.irecv(receive_from_mpi_rank, 1, values) );
        requests.push_back( world.irecv(receive_from_mpi_rank, 2, derivs) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Copy the data from the pointer arrays to values arrays
    for(mpiSyncMap::iterator it = g_syncData.receive_from.begin(); it != g_syncData.receive_from.end(); it++)
    {
        int               receive_from_mpi_rank = it->first;
        std::vector<int>& indexes               = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = mapPointersData.receive_from[receive_from_mpi_rank].first;
        std::vector<real_t*>& pderivs = mapPointersData.receive_from[receive_from_mpi_rank].second;
        std::vector<real_t>&  values  = mapValuesData.receive_from[receive_from_mpi_rank].first;
        std::vector<real_t>&  derivs  = mapValuesData.receive_from[receive_from_mpi_rank].second;

        if(indexes.size() != values.size() || indexes.size() != derivs.size())
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
}

