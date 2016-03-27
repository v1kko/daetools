/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2016
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_MPI_SYNC_H
#define DAE_MPI_SYNC_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
namespace mpi = boost::mpi;
using namespace std;

typedef std::map< int, std::vector<int> > mpiSyncMap;
struct mpiIndexesData
{
    int            i_start;
    int            i_end;
    vector<int>    foreign_indexes;
    mpiSyncMap     send_to;
    mpiSyncMap     receive_from;
};

typedef std::map< int, std::pair< std::vector<real_t>, std::vector<real_t> > >  mpiSyncValuesMap;
struct mpiValuesData
{
    mpiSyncValuesMap send_to;
    mpiSyncValuesMap receive_from;
};

typedef std::map< int, std::pair< std::vector<real_t*>,std::vector<real_t*> > > mpiSyncPointersMap;
struct mpiPointersData
{
    mpiSyncPointersMap send_to;
    mpiSyncPointersMap receive_from;
};

mpiValuesData   mapValuesData;
mpiPointersData mapPointersData;

%(mpi_synchronise_data)s

void CheckSynchronisationIndexes(daeModel_t* model, void* mpi_world, int mpi_rank)
{
    // Get synchronisation info for the current node
    mpiIndexesData indData = mapIndexesData.at(mpi_rank);

    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    std::vector<mpi::request> requests;
    mpiSyncMap received_indexes;

    // Send the data to the other nodes
    for(mpiSyncMap::iterator it = indData.send_to.begin(); it != indData.send_to.end(); it++)
    {
        requests.push_back( world.isend(it->first, 0, it->second) );
    }

    // Receive the data from the other nodes
    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
    {
        received_indexes[it->first] = std::vector<int>();
        requests.push_back( world.irecv(it->first, 0, received_indexes[it->first]) );
    }

    // Wait until all mpi send/receive requests are done
    mpi::wait_all(requests.begin(), requests.end());

    // Check if we received the correct indexes
    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
    {
        if(indData.receive_from[it->first] != received_indexes[it->first])
            throw std::runtime_error(std::string("The received indexes do not match the requested ones, node: ") + std::to_string(mpi_rank));
    }

    // Just for the debugging purposes print sent/received indexes to a file
    std::ofstream ofs;
    std::string filename = std::string("node-") + std::to_string(mpi_rank) + ".txt";
    ofs.open(filename, std::ofstream::out);

    ofs << "Node " << mpi_rank << std::endl;
    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
    {
        ofs << "Expected: " << std::endl;
        for(size_t i = 0; i < indData.receive_from[it->first].size(); i++)
            ofs << indData.receive_from[it->first][i] << ", ";
        ofs << std::endl;

        ofs << "Received: " << std::endl;
        for(size_t i = 0; i < received_indexes[it->first].size(); i++)
            ofs << received_indexes[it->first][i] << ", ";
        ofs << std::endl;
    }
    ofs.close();
}

int SynchroniseData(daeModel_t* model, void* mpi_world, int mpi_rank)
{
    // Get synchronisation info for the current node
    mpiIndexesData indData = mapIndexesData.at(mpi_rank);

    mpi::communicator& world = *(mpi::communicator*)mpi_world;

    std::vector<mpi::request> requests;

    // Send the data to the other nodes
    for(mpiSyncMap::iterator it = indData.send_to.begin(); it != indData.send_to.end(); it++)
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
    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
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
    for(mpiSyncMap::iterator it = indData.receive_from.begin(); it != indData.receive_from.end(); it++)
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
        else
            std::cout << "Node [" << mpi_rank << "] transferred " << values.size() << " values" << std::endl;

        for(size_t i = 0; i < i_size; i++)
        {
            *pvalues[i] = values[i];
            *pderivs[i] = derivs[i];
        }
    }

    return 0;
}


#endif
