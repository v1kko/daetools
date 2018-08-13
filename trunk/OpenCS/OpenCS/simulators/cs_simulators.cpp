/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <mpi.h>
#include "cs_simulators.h"
#include "daesimulator.h"
using namespace cs_dae_simulator;
#include <OpenCS/evaluators/cs_evaluator_sequential.h>
#include <OpenCS/evaluators/cs_evaluator_openmp.h>
#include <OpenCS/evaluators/cs_evaluator_opencl_factory.h>

#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;

namespace cs
{
void csSimulate_DAE(const std::string& inputDirectory)
{
    try
    {
        /* Do not call MPI_Initialize here since we do not know the argc/argv arguments.
         * It must be called by the executable linking this library. */
        int rank;
        MPI_Comm mpi_world = MPI_COMM_WORLD;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        printf("Running on MPI node: %d\n", rank);

        filesystem::path inputDirectoryPath = filesystem::absolute( filesystem::path(inputDirectory) );

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        filesystem::path simulationOptionsPath = inputDirectoryPath / "simulation_options.json";
        cfg.Load(simulationOptionsPath.string());
        //printf("Loaded options from '%s':\n%s\n", cfg.configFile.c_str(), cfg.ToString().c_str());

        // Results are stored in: inputDirectory/results
        filesystem::path outputDirectoryPath = inputDirectoryPath / cfg.GetString("Simulation.OutputDirectory");

        real_t startTime         = cfg.GetFloat("Simulation.StartTime");
        real_t timeHorizon       = cfg.GetFloat("Simulation.TimeHorizon");
        real_t reportingInterval = cfg.GetFloat("Simulation.ReportingInterval");

        std::string csEvaluatorLibrary = cfg.GetString("Model.ComputeStackEvaluator.Library", "Unknown");
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator;
        if(csEvaluatorLibrary == "Sequential")
        {
            csEvaluator.reset( new cs::csComputeStackEvaluator_Sequential() );
        }
        else if(csEvaluatorLibrary == "OpenMP")
        {
            int numThreads = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.numThreads", 0);
            csEvaluator.reset( new cs::csComputeStackEvaluator_OpenMP(numThreads) );
        }
        else if(csEvaluatorLibrary == "OpenCL")
        {
            std::string csEvaluatorName = cfg.GetString("Model.ComputeStackEvaluator.Name", "Unknown");
            if(csEvaluatorName == "Single-device")
            {
                int platformID                  = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.platformID");
                int deviceID                    = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.deviceID");
                std::string buildProgramOptions = cfg.GetString ("Model.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator.reset( cs::csCreateOpenCLEvaluator(platformID, deviceID, buildProgramOptions) );
            }
            else if(csEvaluatorName == "Multi-device")
            {
                std::vector<int>    platforms;
                std::vector<int>    devices;
                std::vector<double> taskPortions;

                for(int i = 0; i < 10; i++)
                {
                    std::string device = (boost::format("Model.ComputeStackEvaluator.Parameters.Device_%d") % i).str();
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

                std::string buildProgramOptions = cfg.GetString ("Model.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator.reset( csCreateOpenCLEvaluator_MultiDevice(platforms, devices, taskPortions, buildProgramOptions) );
            }
        }

        daeModel_t      model;
        daeSolver_t     daesolver;
        daeSimulation_t simulation;

        /* Initialize the model */
        model.pe_rank  = rank;
        model.Load(inputDirectoryPath.string(), csEvaluator.get());
        if(model.structure.isODESystem)
            daeThrowException("Simulation of the DAE system is requested but the system is ODE");

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
}

void csSimulate_ODE(const std::string& inputDirectory)
{
    try
    {
        /* Do not call MPI_Initialize here since we do not know the argc/argv arguments.
         * It must be called by the executable linking this library. */
        int rank;
        MPI_Comm mpi_world = MPI_COMM_WORLD;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        printf("Running on MPI node: %d\n", rank);

        filesystem::path inputDirectoryPath = filesystem::absolute( filesystem::path(inputDirectory) );

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        filesystem::path simulationOptionsPath = inputDirectoryPath / "simulation_options.json";
        cfg.Load(simulationOptionsPath.string());
        //printf("Loaded options from '%s':\n%s\n", cfg.configFile.c_str(), cfg.ToString().c_str());

        // Results are stored in: inputDirectory/results
        filesystem::path outputDirectoryPath = inputDirectoryPath / cfg.GetString("Simulation.OutputDirectory");

        real_t startTime         = cfg.GetFloat("Simulation.StartTime");
        real_t timeHorizon       = cfg.GetFloat("Simulation.TimeHorizon");
        real_t reportingInterval = cfg.GetFloat("Simulation.ReportingInterval");

        std::string csEvaluatorLibrary = cfg.GetString("Model.ComputeStackEvaluator.Library", "Unknown");
        std::shared_ptr<csComputeStackEvaluator_t> csEvaluator;
        if(csEvaluatorLibrary == "Sequential")
        {
            csEvaluator.reset( new cs::csComputeStackEvaluator_Sequential() );
        }
        else if(csEvaluatorLibrary == "OpenMP")
        {
            int numThreads = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.numThreads", 0);
            csEvaluator.reset( new cs::csComputeStackEvaluator_OpenMP(numThreads) );
        }
        else if(csEvaluatorLibrary == "OpenCL")
        {
            std::string csEvaluatorName = cfg.GetString("Model.ComputeStackEvaluator.Name", "Unknown");
            if(csEvaluatorName == "Single-device")
            {
                int platformID                  = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.platformID");
                int deviceID                    = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.deviceID");
                std::string buildProgramOptions = cfg.GetString ("Model.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator.reset( cs::csCreateOpenCLEvaluator(platformID, deviceID, buildProgramOptions) );
            }
            else if(csEvaluatorName == "Multi-device")
            {
                std::vector<int>    platforms;
                std::vector<int>    devices;
                std::vector<double> taskPortions;

                for(int i = 0; i < 10; i++)
                {
                    std::string device = (boost::format("Model.ComputeStackEvaluator.Parameters.Device_%d") % i).str();
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

                std::string buildProgramOptions = cfg.GetString ("Model.ComputeStackEvaluator.Parameters.buildProgramOptions");

                csEvaluator.reset( csCreateOpenCLEvaluator_MultiDevice(platforms, devices, taskPortions, buildProgramOptions) );
            }
        }

        daeModel_t      model;
        odeiSolver_t    daesolver;
        daeSimulation_t simulation;

        /* Initialize the model */
        model.pe_rank  = rank;
        model.Load(inputDirectoryPath.string(), csEvaluator.get());
        if(!model.structure.isODESystem)
            daeThrowException("Simulation of the ODE system is requested but the system is DAE");

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
}

}
