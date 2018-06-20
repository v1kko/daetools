/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_SIMULATOR_MPI_DATA_EXCHANGE_H
#define CS_SIMULATOR_MPI_DATA_EXCHANGE_H

#include <string>
#include <vector>
#include <map>
#include <stdint.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/container/flat_map.hpp>
using boost::container::flat_map;
namespace mpi = boost::mpi;

typedef flat_map< int32_t, std::pair< std::vector<real_t>, std::vector<real_t> > >  mpiSyncValuesMap;
struct mpiValuesData
{
    mpiSyncValuesMap  sendToIndexes;
    mpiSyncValuesMap  receiveFromIndexes;
};

typedef flat_map< int32_t, std::pair< std::vector<real_t*>,std::vector<real_t*> > > mpiSyncPointersMap;
struct mpiPointersData
{
    mpiSyncPointersMap sendToIndexes;
    mpiSyncPointersMap receiveFromIndexes;
};

mpiValuesData   mapValuesData;
mpiPointersData mapPointersData;


#endif
