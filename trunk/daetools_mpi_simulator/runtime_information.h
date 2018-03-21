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
#ifndef DAE_RUNTIME_INFORMATION_H
#define DAE_RUNTIME_INFORMATION_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <boost/mpi.hpp>
#include <boost/container/flat_map.hpp>
using boost::container::flat_map;
namespace mpi = boost::mpi;

struct runtimeInformationData
{
    int     i_start;
    int     i_end;
    int     Ntotal_vars;
    int     Nequations;
    int     Nequations_local;
    int     Ndofs;
    real_t  startTime;
    real_t  timeHorizon;
    real_t  reportingInterval;
    real_t  relativeTolerance;        
    bool    quasiSteadyState;

    std::vector<real_t>      dofs;
    std::vector<real_t>      init_values;
    std::vector<real_t>      init_derivatives;
    std::vector<real_t>      absolute_tolerances;
    std::vector<int>         ids;
    std::vector<std::string> variable_names;
};


typedef std::map< int, std::vector<int> > mpiSyncMap;
struct mpiIndexesData
{
    int               i_start;
    int               i_end;
    std::vector<int>  foreign_indexes;
    flat_map<int,int> bi_to_bi_local;
    mpiSyncMap        send_to;
    mpiSyncMap        receive_from;
};

typedef flat_map< int, std::pair< std::vector<real_t>, std::vector<real_t> > >  mpiSyncValuesMap;
struct mpiValuesData
{
    mpiSyncValuesMap  send_to;
    mpiSyncValuesMap  receive_from;
};

typedef flat_map< int, std::pair< std::vector<real_t*>,std::vector<real_t*> > > mpiSyncPointersMap;
struct mpiPointersData
{
    mpiSyncPointersMap send_to;
    mpiSyncPointersMap receive_from;
};

mpiValuesData   mapValuesData;
mpiPointersData mapPointersData;


#endif
