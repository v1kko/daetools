/***********************************************************************************
*                 OpenCS Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "cs_dae_model.h"
#include "../cs_model_io.h"
#include <exception>
#include <boost/format.hpp>
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;

#define daeThrowException(MSG) \
   throw std::runtime_error( (boost::format("Exception in %s (%s:%d):\n%s\n") % std::string(__FUNCTION__) % std::string(__FILE__) % __LINE__ % (MSG)).str() );

namespace cs
{
csDAEModelImplementation_t::csDAEModelImplementation_t()
{
    mpi_world = NULL;
    mpi_comm  = NULL;

    pe_rank           = -1;
    Nequations_local  = 0;
    Nequations        = 0;
    Ndofs             = 0;
    quasiSteadyState  = false;
    csEvaluator       = NULL;

    m_activeEquationSetIndexes  = NULL;
    m_computeStacks             = NULL;
    m_jacobianMatrixItems       = NULL;
    m_numberOfJacobianItems     = -1;
    m_numberOfComputeStackItems = -1;
}

csDAEModelImplementation_t::~csDAEModelImplementation_t()
{
    Free();
}

void csDAEModelImplementation_t::Load(const std::string& inputDirectory, csComputeStackEvaluator_t* csEvaluator_)
{
    cs::csLoadModel(this, inputDirectory);

    /* Finalize data initialisation. */
    m_dofs.resize(Ndofs, 0.0);
    for(int i = 0; i < Ndofs; i++)
        m_dofs[i] = dofs[i];

    m_activeEquationSetIndexes  = &activeEquationSetIndexes[0];
    m_computeStacks             = &computeStacks[0];
    m_numberOfComputeStackItems = computeStacks.size();

    m_jacobianMatrixItems   = &jacobianMatrixItems[0];
    m_numberOfJacobianItems = jacobianMatrixItems.size();

    /* If CS Evalator has not been specified throw an exception. */
    csEvaluator = csEvaluator_;
    if(!csEvaluator)
        daeThrowException( (boost::format("Invalid compute stack evaluator specified (node %d)") % pe_rank).str() );

    csEvaluator->Initialize(false,
                            Nequations_local,
                            Nequations_local,
                            Ndofs,
                            m_numberOfComputeStackItems,
                            m_numberOfJacobianItems,
                            m_numberOfJacobianItems,
                            m_computeStacks,
                            m_activeEquationSetIndexes,
                            m_jacobianMatrixItems);

    InitializeValuesReferences();

/*
    if(pe_rank == 0)
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

        ofs << "m_numberOfComputeStackItems = "  << m_numberOfComputeStackItems << std::endl;

        ofs << "startEquationIndex = "  << startEquationIndex << std::endl;
        ofs << "startJacobianIndex = "  << startJacobianIndex << std::endl;
        ofs << "m_numberOfJacobianItems = "  << m_numberOfJacobianItems << std::endl;

        ofs << "ActiveEquationSetIndexes = [";
        for(int i = 0; i < Nequations_local; i++)
            ofs << m_activeEquationSetIndexes[i] << ", ";
        ofs << "]" << std::endl;

        ofs << "m_jacobianMatrixItems equationIndexes = [";
        for(int ji = 0; ji < m_numberOfJacobianItems; ji++)
        {
            csJacobianMatrixItem_t& jd = m_jacobianMatrixItems[ji];
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
}

void csDAEModelImplementation_t::Free()
{
    /* CS Evaluator is instantiated outside the model, just call FreeResources - do not delete the pointer. */
    if(csEvaluator)
    {
        csEvaluator->FreeResources();
        csEvaluator = NULL;
    }
}

void csDAEModelImplementation_t::EvaluateResiduals(real_t time, real_t* residuals)
{
    /* The current time, values and timeDerivatives have already been copied in SetAndSynchroniseData function. */
    if(time != currentTime)
        daeThrowException( (boost::format("The current model time: %.15f does not match the time for which the residuals are requested: %.15f (node %d)") % currentTime % time % pe_rank).str() );
    if(!csEvaluator)
        daeThrowException( (boost::format("Invalid compute stack evaluator (node %d)") % pe_rank).str() );

    real_t* pdofs            = (m_dofs.size() > 0 ? &m_dofs[0] : NULL);
    real_t* pvalues          = &m_values[0];
    real_t* ptimeDerivatives = &m_timeDerivatives[0];

    csEvaluationContext_t EC;
    EC.equationEvaluationMode    = cs::eEvaluateResidual;
    EC.sensitivityParameterIndex = -1;
    EC.jacobianIndex             = -1;
    EC.numberOfVariables         = Nequations_local;
    EC.numberOfEquations         = Nequations_local; // ???
    EC.numberOfDOFs              = m_dofs.size();
    EC.numberOfComputeStackItems = m_numberOfComputeStackItems;
    EC.numberOfJacobianItems     = 0;
    EC.valuesStackSize           = 5;
    EC.lvaluesStackSize          = 20;
    EC.rvaluesStackSize          = 5;
    EC.currentTime               = currentTime;
    EC.inverseTimeStep           = 0; // Should not be needed here. Double check...
    EC.startEquationIndex        = 0; // !!!
    EC.startJacobianIndex        = 0; // !!!

    csEvaluator->EvaluateResiduals(EC, pdofs, pvalues, ptimeDerivatives, residuals);

/*
    if(mpi_rank == 0)
    {
        std::ofstream ofs;
        std::string filename = std::string("modResiduals-node-") + std::to_string(mpi_rank) + ".txt";
        ofs.open(filename, std::ofstream::out|std::ofstream::app);

        ofs << std::setiosflags(std::ios_base::fixed);
        ofs << std::setprecision(15);

        ofs << "time = " << currentTime << std::endl;
        flat_map<int, real_t*>::const_iterator iter;
        ofs << "m_mapValues " << m_mapValues.size() << std::endl;
        for(iter = m_mapValues.begin(); iter != m_mapValues.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "m_mapTimeDerivatives " << m_mapTimeDerivatives.size() << std::endl;
        for(iter = m_mapTimeDerivatives.begin(); iter != m_mapTimeDerivatives.end(); iter++)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "residuals " << Nequations_local << std::endl;
        for(int i = 0; i < Nequations_local; i++)
            ofs << "["<< i << ":" << residuals[i] << "]";
        ofs << std::endl;
    }
*/
}

void csDAEModelImplementation_t::EvaluateJacobian(real_t time, real_t inverseTimeStep, csMatrixAccess_t* ma)
{
    /* The current time, values and timeDerivatives have already been copied in SetAndSynchroniseData function. */
    if(time != currentTime)
        daeThrowException( (boost::format("The current model time: %.15f does not match the time for which the Jacobian is requested: %.15f (node %d)") % currentTime % time % pe_rank).str() );
    if(!csEvaluator)
        daeThrowException( (boost::format("Invalid compute stack evaluator (node %d)") % pe_rank).str() );

    real_t* pdofs            = (m_dofs.size() > 0 ? const_cast<real_t*>(&m_dofs[0]) : NULL);
    real_t* pvalues          = &m_values[0];
    real_t* ptimeDerivatives = &m_timeDerivatives[0];

    m_jacobian.resize(m_numberOfJacobianItems, 0.0);

    csEvaluationContext_t EC;
    EC.equationEvaluationMode    = cs::eEvaluateJacobian;
    EC.sensitivityParameterIndex = -1;
    EC.jacobianIndex             = -1;
    EC.numberOfVariables         = Nequations_local;
    EC.numberOfEquations         = Nequations_local; // ???
    EC.numberOfDOFs              = m_dofs.size();
    EC.numberOfComputeStackItems = m_numberOfComputeStackItems;
    EC.numberOfJacobianItems     = m_numberOfJacobianItems;
    EC.valuesStackSize           = 5;
    EC.lvaluesStackSize          = 20;
    EC.rvaluesStackSize          = 5;
    EC.currentTime               = currentTime;
    EC.inverseTimeStep           = inverseTimeStep;
    EC.startEquationIndex        = 0; // !!!
    EC.startJacobianIndex        = 0; // !!!

    csEvaluator->EvaluateJacobian(EC, pdofs, pvalues, ptimeDerivatives, &m_jacobian[0]);

    // Evaluated Jacobian values need to be copied to the Jacobian matrix.
    for(size_t ji = 0; ji < m_numberOfJacobianItems; ji++)
    {
        const csJacobianMatrixItem_t& jacobianItem = m_jacobianMatrixItems[ji];
        size_t ei_local = jacobianItem.equationIndex;
        size_t bi_local = jacobianItem.blockIndex; // it is already updated to local blockIndex during export

        //std::cout << "ei_local = "  << ei_local << std::endl;
        //std::cout << "bi_local = "  << bi_local << std::endl;

        // Some items contain block indexes that are foreign to this PE.
        // In this case skip the item (although the value has been calculated).
        // Important:
        //   Double check: is this a problem?
        //   Should the items that correspond to foreign varibles be removed from the csJacobianMatrixItem_t array?
        if(bi_local >= Nequations_local)
            continue;

        ma->SetItem(ei_local, bi_local, m_jacobian[ji]);
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
        ofs << "m_mapValues " << m_mapValues.size() << std::endl;
        for(iter = m_mapValues.begin(); iter != m_mapValues.end(); iter++)
            if(iter->first < 5)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "m_mapTimeDerivatives " << m_mapTimeDerivatives.size() << std::endl;
        for(iter = m_mapTimeDerivatives.begin(); iter != m_mapTimeDerivatives.end(); iter++)
            if(iter->first < 5)
            ofs << "[" << iter->first << ":" << *(iter->second) << "]";
        ofs << std::endl;
        ofs << "Jacobian " << Nequations_local << std::endl;
        for(size_t jdi = 0; jdi < m_numberOfJacobianItems; jdi++)
        {
            int ji = startJacobianIndex + jdi;
            const csJacobianMatrixItem_t& jacobianItem = m_jacobianMatrixItems[ji];
            size_t ei_local = jacobianItem.equationIndex - startEquationIndex;
            size_t bi_local = m_partitionData.bi_to_bi_local[jacobianItem.blockIndex];
            if(bi_local >= Nequations_local)
                continue;
            ofs << ei_local << " " << bi_local << ": " << _jacobian_matrix_(ei_local,bi_local) << std::endl;
        }
    }
*/
}

// Variable local indexes in DAE system equations as a CSR matrix.
void csDAEModelImplementation_t::GetDAESystemStructure(int& N, int& NNZ, std::vector<int>& IA, std::vector<int>& JA)
{
    std::vector<size_t> numColumnsInRows;

    IA.reserve(Nequations_local + 1);
    JA.reserve(m_numberOfJacobianItems);
    numColumnsInRows.resize(Nequations_local, 0);

    int removed = 0;
    for(size_t ji = 0; ji < m_numberOfJacobianItems; ji++)
    {
        const csJacobianMatrixItem_t& jacobianItem = m_jacobianMatrixItems[ji];
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

void csDAEModelImplementation_t::SetAndSynchroniseData(real_t time, real_t* daesolver_values, real_t* daesolver_time_derivatives)
{
    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    currentTime = time;

    // Copy values/derivatives from DAE solver into the local storage.
    for(int i = 0; i < Nequations_local; i++)
    {
        m_values[i]          = daesolver_values[i];
        m_timeDerivatives[i] = daesolver_time_derivatives[i];
    }

    // Get synchronisation info for the current node
    std::vector<mpi::request> requests;

    // Send the data to the other nodes
    for(csPartitionIndexMap::iterator it = partitionData.sendToIndexes.begin(); it != partitionData.sendToIndexes.end(); it++)
    {
        int               send_to_mpi_rank = it->first;
        std::vector<int>& indexes          = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = m_mapPointersData.sendToIndexes[send_to_mpi_rank].first;
        std::vector<real_t*>& pderivs = m_mapPointersData.sendToIndexes[send_to_mpi_rank].second;
        std::vector<real_t>&  values  = m_mapValuesData.sendToIndexes[send_to_mpi_rank].first;
        std::vector<real_t>&  derivs  = m_mapValuesData.sendToIndexes[send_to_mpi_rank].second;

        for(size_t i = 0; i < i_size; i++)
        {
            values[i] = *pvalues[i];
            derivs[i] = *pderivs[i];
        }

        requests.push_back( world.isend(send_to_mpi_rank, 1, values) );
        requests.push_back( world.isend(send_to_mpi_rank, 2, derivs) );
    }

    // Receive the data from the other nodes
    for(csPartitionIndexMap::iterator it = partitionData.receiveFromIndexes.begin(); it != partitionData.receiveFromIndexes.end(); it++)
    {
        int receive_from_mpi_rank = it->first;

        std::vector<real_t>& values  = m_mapValuesData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t>& derivs  = m_mapValuesData.receiveFromIndexes[receive_from_mpi_rank].second;

        requests.push_back( world.irecv(receive_from_mpi_rank, 1, values) );
        requests.push_back( world.irecv(receive_from_mpi_rank, 2, derivs) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Copy the data from the pointer arrays to values arrays
    for(csPartitionIndexMap::iterator it = partitionData.receiveFromIndexes.begin(); it != partitionData.receiveFromIndexes.end(); it++)
    {
        int               receive_from_mpi_rank = it->first;
        std::vector<int>& indexes               = it->second;
        size_t i_size = indexes.size();

        std::vector<real_t*>& pvalues = m_mapPointersData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t*>& pderivs = m_mapPointersData.receiveFromIndexes[receive_from_mpi_rank].second;
        std::vector<real_t>&  values  = m_mapValuesData.receiveFromIndexes[receive_from_mpi_rank].first;
        std::vector<real_t>&  derivs  = m_mapValuesData.receiveFromIndexes[receive_from_mpi_rank].second;

        if(indexes.size() != values.size()  ||
           indexes.size() != derivs.size()  ||
           indexes.size() != pvalues.size() ||
           indexes.size() != pderivs.size())
            throw std::runtime_error(std::string("The received data do not match the requested ones, node: ") + std::to_string(pe_rank));
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
            ss << "Node [" << pe_rank << "] values from node [" << receive_from_mpi_rank << "]:" << std::endl;
            for(size_t i = 0; i < i_size; i++)
                ss << *pvalues[i] << ", ";
            ss << std::endl;
            std::cout << ss.str();
        }
    }

    /*
    // Check the mapping
    for(flat_map<int32_t,int32_t>::iterator iter = m_partitionData.bi_to_bi_local.begin(); iter != m_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index

        double value       = m_values[bi_local];
        double deriv       = m_timeDerivatives[bi_local];
        double mappedValue = *m_mapValues[bi];
        double mappedDeriv = *m_mapTimeDerivatives[bi];

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
    for(flat_map<int32_t,int32_t>::iterator iter = m_partitionData.bi_to_bi_local.begin(); iter != m_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local < Nequations_local)
            fprintf(f, " (%d, %d) = %+.15f, %+.15f\n", bi, bi_local, *m_mapValues[bi], *m_mapTimeDerivatives[bi]);
    }
    fprintf(f, "\n");

    fprintf(f, "bi_to_bi_local (foreign):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = m_partitionData.bi_to_bi_local.begin(); iter != m_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local >= Nequations_local)
            fprintf(f, " (%d, %d) = %+.15f, %+.15f\n", bi, bi_local, *m_mapValues[bi], *m_mapTimeDerivatives[bi]);
    }
    fprintf(f, "\n");
    fprintf(f, "****************************************************************\n");

    fclose(f);
    */
}

int csDAEModelImplementation_t::NumberOfRoots()
{
    return 0;
}

void csDAEModelImplementation_t::Roots(real_t time, real_t* values, real_t* time_derivatives, real_t* roots)
{
}

bool csDAEModelImplementation_t::CheckForDiscontinuities(real_t time, real_t* values, real_t* time_derivatives)
{
    return false;
}

csDiscontinuityType csDAEModelImplementation_t::ExecuteActions(real_t time, real_t* values, real_t* time_derivatives)
{
    return eNoDiscontinuity;
}


void csDAEModelImplementation_t::InitializeValuesReferences()
{
    // Reserve the size for internal vectors/maps
    size_t Nforeign = partitionData.foreignIndexes.size();
    size_t Ntot     = Nequations_local + Nforeign;

    m_values.resize(Ntot, 0.0);
    m_timeDerivatives.resize(Ntot, 0.0);
    //m_mapValues.reserve(Ntot);
    //m_mapTimeDerivatives.reserve(Ntot);

    if(partitionData.biToBiLocal.size() != Ntot)
        throw std::runtime_error("Invalid number of items in bi_to_bi_local map");

    // Insert the pointers to the owned and the foreign values
    // Owned data are always in the range: [0, Nequations_local)
    for(std::map<int32_t,int32_t>::iterator iter = partitionData.biToBiLocal.begin(); iter != partitionData.biToBiLocal.end(); iter++)
    {
        int bi = iter->first;  // global block index
        int li = iter->second; // local index

        m_mapValues[bi]          = &m_values[li];
        m_mapTimeDerivatives[bi] = &m_timeDerivatives[li];
    }

    // Initialize pointer maps
    for(csPartitionIndexMap::iterator it = partitionData.sendToIndexes.begin(); it != partitionData.sendToIndexes.end(); it++)
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
            pvalues_arr[i] = m_mapValues.at         ( indexes[i] );
            pderivs_arr[i] = m_mapTimeDerivatives.at( indexes[i] );
        }

        m_mapValuesData.sendToIndexes[rank]   = make_pair(values_arr,  derivs_arr);
        m_mapPointersData.sendToIndexes[rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    for(csPartitionIndexMap::iterator it = partitionData.receiveFromIndexes.begin(); it != partitionData.receiveFromIndexes.end(); it++)
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
            pvalues_arr[i] = m_mapValues.at         ( indexes[i] );
            pderivs_arr[i] = m_mapTimeDerivatives.at( indexes[i] );
        }

        m_mapValuesData.receiveFromIndexes[rank]   = make_pair(values_arr,  derivs_arr);
        m_mapPointersData.receiveFromIndexes[rank] = make_pair(pvalues_arr, pderivs_arr);
    }

    CheckSynchronisationIndexes();

/*
    printf("Nequations_local = %d\n", Nequations_local);

    printf("foreign_indexes:\n");
    for(size_t i = 0; i < m_partitionData.foreign_indexes.size(); i++)
    {
        int32_t fi = m_partitionData.foreign_indexes[i];
        printf(" %d", fi);
    }
    printf("\n");

    printf("bi_to_bi_local (local):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = m_partitionData.bi_to_bi_local.begin(); iter != m_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local < Nequations_local)
            printf(" (%d, %d)", bi, bi_local);
    }
    printf("\n");

    printf("bi_to_bi_local (foreign):\n");
    for(flat_map<int32_t,int32_t>::iterator iter = m_partitionData.bi_to_bi_local.begin(); iter != m_partitionData.bi_to_bi_local.end(); iter++)
    {
        int32_t bi       = iter->first;  // global block index
        int32_t bi_local = iter->second; // local index
        if(bi_local >= Nequations_local)
            printf(" (%d, %d)", bi, bi_local);
    }
    printf("\n");
*/
}

void csDAEModelImplementation_t::CheckSynchronisationIndexes()
{
    // Get synchronisation info for the current node
    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    std::vector<mpi::request> requests;
    csPartitionIndexMap received_indexes;

    // Send the data to the other nodes
    for(csPartitionIndexMap::iterator it = partitionData.sendToIndexes.begin(); it != partitionData.sendToIndexes.end(); it++)
    {
        requests.push_back( world.isend(it->first, 0, it->second) );
    }

    // Receive the data from the other nodes
    for(csPartitionIndexMap::iterator it = partitionData.receiveFromIndexes.begin(); it != partitionData.receiveFromIndexes.end(); it++)
    {
        received_indexes[it->first] = std::vector<int>();
        requests.push_back( world.irecv(it->first, 0, received_indexes[it->first]) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Check if we received the correct indexes
    for(csPartitionIndexMap::iterator it = partitionData.receiveFromIndexes.begin(); it != partitionData.receiveFromIndexes.end(); it++)
    {
        if(partitionData.receiveFromIndexes[it->first] != received_indexes[it->first])
            throw std::runtime_error(std::string("The received indexes do not match the requested ones, node: ") + std::to_string(pe_rank));
    }

/*
    // Just for the debugging purposes print sent/received indexes to a file
    std::ofstream ofs;
    std::string filename = std::string("node-") + std::to_string(pe_rank) + ".txt";
    ofs.open(filename, std::ofstream::out);

    ofs << "Node " << pe_rank << std::endl;
    for(csPartitionIndexMap::iterator it = m_partitionData.receiveFromIndexes.begin(); it != m_partitionData.receiveFromIndexes.end(); it++)
    {
        ofs << "Expected from " << it->first << ": " << std::endl;
        for(size_t i = 0; i < m_partitionData.receiveFromIndexes[it->first].size(); i++)
            ofs << m_partitionData.receiveFromIndexes[it->first][i] << ", ";
        ofs << std::endl;

        ofs << "Received from " << it->first << ": "  << std::endl;
        for(size_t i = 0; i < received_indexes[it->first].size(); i++)
            ofs << received_indexes[it->first][i] << ", ";
        ofs << std::endl << std::endl;
    }
    ofs.close();
*/
}

}
