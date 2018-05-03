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
#ifndef DAE_MPI_RUNTIME_INFORMATION_H
#define DAE_MPI_RUNTIME_INFORMATION_H

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

/* Runtime information data structure. */
typedef struct runtimeInformationData_
{
    uint32_t Ntotal_vars;
    uint32_t Nequations;
    uint32_t Nequations_local;
    uint32_t Ndofs;
    real_t   startTime;
    real_t   timeHorizon;
    real_t   reportingInterval;
    real_t   relativeTolerance;
    bool     quasiSteadyState;

    std::vector<real_t>      dofs;
    std::vector<real_t>      init_values;
    std::vector<real_t>      init_derivatives;
    std::vector<real_t>      absolute_tolerances;
    std::vector<int32_t>     ids;
    std::vector<std::string> variable_names;
} runtimeInformationData_t;

/* Partinioning data (mostly used for inter-process communication). */
typedef std::map< int32_t, std::vector<int32_t> > nodeIndexesMap;
typedef struct partitionData_
{
    std::vector<int32_t>      foreign_indexes;
    flat_map<int32_t,int32_t> bi_to_bi_local;
    nodeIndexesMap            send_to;
    nodeIndexesMap            receive_from;
} partitionData_t;

typedef flat_map< int32_t, std::pair< std::vector<real_t>, std::vector<real_t> > >  mpiSyncValuesMap;
struct mpiValuesData
{
    mpiSyncValuesMap  send_to;
    mpiSyncValuesMap  receive_from;
};

typedef flat_map< int32_t, std::pair< std::vector<real_t*>,std::vector<real_t*> > > mpiSyncPointersMap;
struct mpiPointersData
{
    mpiSyncPointersMap send_to;
    mpiSyncPointersMap receive_from;
};

mpiValuesData   mapValuesData;
mpiPointersData mapPointersData;


#endif
