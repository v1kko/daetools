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
#include "cs_simulator.h"
using namespace cs_dae_simulator;
#include "../opencs/sequential/cs_evaluator_sequential.h"
#include "../opencs/openmp/cs_evaluator_openmp.h"
#include "../opencs/opencl/cs_evaluator_opencl_factory.h"

#include <boost/mpi.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
namespace mpi = boost::mpi;

/* Run using:
    $ mpirun -np 4 ./cs_dae_simulator inputDataDir
    $ mpirun -np 4 --hostfile mpi-hosts ./cs_dae_simulator inputDataDir
    $ mpirun -np 4 konsole -e ./cs_dae_simulator inputDataDir
    $ mpirun -np 4 konsole -e gdb -ex run --args ./cs_dae_simulator inputDataDir
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
        boost::filesystem::path simulationOptionsPath = inputDirectoryPath / "simulation_options.json";
        cfg.Load(simulationOptionsPath.string());
        printf("Loaded options from '%s':\n%s\n", cfg.configFile.c_str(), cfg.ToString().c_str());

        // Results are stored in: inputDirectory/results
        boost::filesystem::path outputDirectoryPath = inputDirectoryPath / cfg.GetString("OutputDirectory");

        real_t startTime         = cfg.GetFloat("StartTime");
        real_t timeHorizon       = cfg.GetFloat("TimeHorizon");
        real_t reportingInterval = cfg.GetFloat("ReportingInterval");

        std::string csEvaluatorLibrary = cfg.GetString("DAEModel.ComputeStackEvaluator.Library", "Unknown");
        csComputeStackEvaluator_t* csEvaluator = NULL;
        if(csEvaluatorLibrary == "Sequential")
        {
            csEvaluator = new cs::csComputeStackEvaluator_Sequential();
        }
        else if(csEvaluatorLibrary == "OpenMP")
        {
            int numThreads = cfg.GetInteger("DAEModel.ComputeStackEvaluator.Parameters.numThreads", 0);
            csEvaluator = new cs::csComputeStackEvaluator_OpenMP(numThreads);
        }
        else if(csEvaluatorLibrary == "OpenCL")
        {
            std::string csEvaluatorName = cfg.GetString("DAEModel.ComputeStackEvaluator.Name", "Unknown");
            if(csEvaluatorName == "Single-device")
            {
                int platformID                  = cfg.GetInteger("DAEModel.ComputeStackEvaluator.Parameters.platformID");
                int deviceID                    = cfg.GetInteger("DAEModel.ComputeStackEvaluator.Parameters.deviceID");
                std::string buildProgramOptions = cfg.GetString ("DAEModel.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator = cs::CreateComputeStackEvaluator(platformID, deviceID, buildProgramOptions);
            }
            else if(csEvaluatorName == "Multi-device")
            {
                std::vector<int>    platforms;
                std::vector<int>    devices;
                std::vector<double> taskPortions;

                for(int i = 0; i < 10; i++)
                {
                    std::string device = (boost::format("DAEModel.ComputeStackEvaluator.Parameters.Device_%d") % i).str();
                    int    platformID  = cfg.GetInteger(device + ".platformID",    -1);
                    int    deviceID    = cfg.GetInteger(device + ".deviceID",      -1);
                    double taskPortion = cfg.GetFloat  (device + ".taskPortion", -1.0);

                    if(platformID != -1 && deviceID != -1 && taskPortion != -1.0)
                    {
                        platforms.push_back(platformID);
                        devices.push_back(deviceID);
                        taskPortions.push_back(taskPortion);
                    }
                }

                std::string buildProgramOptions = cfg.GetString ("DAEModel.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator = CreateComputeStackEvaluator_MultiDevice(platforms, devices, taskPortions, buildProgramOptions);
            }
        }

        daeModel_t      model;
        daeIDASolver_t  daesolver;
        daeSimulation_t simulation;

        /* Initialize the model */
        model.mpi_world = &world;
        model.pe_rank  = world.rank();
        model.mpi_comm  = (MPI_Comm)world;
        model.Load(inputDirectoryPath.string(), csEvaluator);

        /* Initialize the simulation and the dae solver */
        simulation.Initialize(&model, &daesolver, startTime, timeHorizon, reportingInterval, outputDirectoryPath.string());

        /* Solve the system at time = 0.0 */
        simulation.SolveInitial();

        /* Run the simulation */
        simulation.Run();

        /* Finalize the simulation */
        simulation.PrintStats();
        simulation.Finalize();
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}


