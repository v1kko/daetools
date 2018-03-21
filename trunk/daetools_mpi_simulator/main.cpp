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
#include "model.h"
#include "simulation.h"
#include "daesolver.h"

#include <boost/mpi.hpp>
namespace mpi = boost::mpi;

/* Run using:
    $ mpirun -np 4 ./daetools_mpi_simulator inputDataDir outputDataDir
    $ mpirun -np 4 --hostfile mpi-hosts ./daetools_mpi_simulator inputDataDir outputDataDir
    $ mpirun -np 4 konsole -e ./daetools_mpi_simulator inputDataDir outputDataDir
    $ mpirun -np 4 konsole -e gdb -ex run --args ./daetools_mpi_simulator inputDataDir outputDataDir
*/
int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        printf("Usage: daetools_mpi_simulator inputDataDirectory outputDataDirectory n\n");
        return -1;
    }

    std::string inputDirectory  = argv[1];
    std::string outputDirectory = argv[2];
    //printf("inputDirectory = %s, outputDirectory = %s\n", inputDirectory.c_str(), outputDirectory.c_str());

    mpi::environment env;
    mpi::communicator world;

    daeModel_t       model;
    daeIDASolver_t   daesolver;
    daeLASolver_t    lasolver;
    daeSimulation_t  simulation;

    /* Initialize all data structures to zero */
    memset(&model,      0, sizeof(daeModel_t));
    memset(&daesolver,  0, sizeof(daeIDASolver_t));
    memset(&lasolver,   0, sizeof(daeLASolver_t));
    memset(&simulation, 0, sizeof(daeSimulation_t));

    /* Initialize the model */
    model.mpi_world = &world;
    model.mpi_rank  = world.rank();
    model.mpi_comm  = (MPI_Comm)world;
    modInitialize(&model, inputDirectory);

    /* Initialize the simulation and the dae solver */
    simInitialize(&simulation, &model, &daesolver, &lasolver, outputDirectory, false);

    /* Solve the system at time = 0.0 */
    simSolveInitial(&simulation);

    /* Run the simulation */
    simRun(&simulation);

    /* Finalize the simulation */
    simFinalize(&simulation);
    modFinalize(&model);

    return 0;
}


