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
#include "typedefs.h"
using namespace daetools_mpi;

#include <boost/mpi.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
namespace mpi = boost::mpi;

/* Run using:
    $ mpirun -np 4 ./daetools_mpi_simulator inputDataDir
    $ mpirun -np 4 --hostfile mpi-hosts ./daetools_mpi_simulator inputDataDir
    $ mpirun -np 4 konsole -e ./daetools_mpi_simulator inputDataDir
    $ mpirun -np 4 konsole -e gdb -ex run --args ./daetools_mpi_simulator inputDataDir
*/
int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: daetools_mpi_simulator inputDataDir n\n");
        return -1;
    }

    mpi::environment env;
    mpi::communicator world;

    try
    {
        printf("Runing on MPI node: %d\n", world.rank());

        std::string inputDirectory = argv[1];
        boost::filesystem::path inputDirectoryPath = boost::filesystem::weakly_canonical( boost::filesystem::path(inputDirectory) );

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        boost::filesystem::path solverOptionsPath = inputDirectoryPath / "solver_options.json";
        cfg.Load(solverOptionsPath.string());
        printf("Loaded options from '%s':\n%s\n", cfg.configFile.c_str(), cfg.ToString().c_str());

        // Results are stored in: inputDirectory/results
        boost::filesystem::path outputDirectoryPath = inputDirectoryPath / cfg.GetString("OutputDirectory");

        daeModel_t      model;
        daeIDASolver_t  daesolver;
        daeSimulation_t simulation;

        /* Initialize the model */
        model.mpi_world = &world;
        model.mpi_rank  = world.rank();
        model.mpi_comm  = (MPI_Comm)world;
        model.Load(inputDirectoryPath.string());

        /* Initialize the simulation and the dae solver */
        simulation.Initialize(&model, &daesolver, outputDirectoryPath.string());

        /* Solve the system at time = 0.0 */
        simulation.SolveInitial();

        /* Run the simulation */
        simulation.Run();

        /* Finalize the simulation */
        simulation.Finalize();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}


