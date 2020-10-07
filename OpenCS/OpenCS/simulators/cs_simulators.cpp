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
#include <ctime>
#include <iostream>
#include <locale>
#include <mpi.h>
#include <boost/format.hpp>
#include "cs_simulators.h"
#include "daesimulator.h"
#include "cs_logs.h"
#include "cs_data_reporters.h"
#include "../evaluators/cs_evaluators.h"
using namespace cs_dae_simulator;

#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;

#if defined(__MACH__) || defined(__APPLE__) || __linux__ == 1
#include <dlfcn.h>
#endif

namespace cs
{
/* Internal functions. */
static void simulate_DAE(daeModel_t&                        model,
                         filesystem::path                   inputDirectoryPath,
                         daeSimulationOptions&              cfg,
                         std::shared_ptr<csLog_t>           log,
                         std::shared_ptr<csDataReporter_t>  datareporter);
static void simulate_ODE(daeModel_t& model,
                         filesystem::path                   inputDirectoryPath,
                         daeSimulationOptions&              cfg,
                         std::shared_ptr<csLog_t>           log,
                         std::shared_ptr<csDataReporter_t>  datareporter);

/* Auxiliary class that manages MPI initialisation/finalisation.
 * In GNU/Linux and for some MPI implementations the symbols from MPI libary are not resolved because of the RTLD_LAZY flag.
 * In these cases the MPI library must be manually loaded using dlopen with RTLD_NOW flag
 * and initialisation performed using the function pointers to MPI_Init/MPI_Finalize from the shared library. */
class MPI_Handler
{
public:
    MPI_Handler()
    {
#if __linux__ == 1
        handle         = 0;
        fnMPI_Init     = NULL;
        fnMPI_Finalize = NULL;

        int mode = RTLD_NOW | RTLD_GLOBAL;
#ifdef RTLD_NOLOAD
        mode |= RTLD_NOLOAD;
#endif
        if(!handle) handle = dlopen("libmpi.so.20", mode);
        if(!handle) handle = dlopen("libmpi.so.12", mode);
        if(!handle) handle = dlopen("libmpi.so.1",  mode);
        if(!handle) handle = dlopen("libmpi.so.0",  mode);
        if(!handle) handle = dlopen("libmpi.so",    mode);

        if(handle)
        {
            fnMPI_Init     = (int (*)(int*, char***))dlsym(handle, "MPI_Init");
            fnMPI_Finalize = (int (*)(void))         dlsym(handle, "MPI_Finalize");
        }

        if(fnMPI_Init)
            fnMPI_Init(NULL,NULL);
#elif defined(__MACH__) || defined(__APPLE__)
        MPI_Init(NULL,NULL);
#else
        MPI_Init(NULL,NULL);
#endif

        int mpiInitialised;
        MPI_Initialized(&mpiInitialised);
        if(!mpiInitialised)
            csThrowException("MPI cannot be initialised as requested.");
    }

    ~MPI_Handler()
    {
#if __linux__ == 1
        if(fnMPI_Finalize)
            fnMPI_Finalize();
        if(handle)
            dlclose(handle);
#elif defined(__MACH__) || defined(__APPLE__)
        MPI_Finalize();
#else
        MPI_Finalize();
#endif
    }
protected:
#if __linux__ == 1
    int (*fnMPI_Init)(int*, char***);
    int (*fnMPI_Finalize)(void);
    void* handle;
#endif
};

void csSimulate(const std::string&                inputDirectory,
                std::shared_ptr<csLog_t>          log,
                std::shared_ptr<csDataReporter_t> datareporter,
                bool                              initMPI)
{
    /* MPI_Init is typically called by the executable linking this library.
     * However, it can be performed by setting initMPI to true.
     * This is mostly used for single processor simulations (i.e. in pyOpenCS). */
    std::shared_ptr<MPI_Handler> mpiHandler;
    int mpiInitialised = false;
    MPI_Initialized(&mpiInitialised);
    if(!mpiInitialised)
    {
        if(initMPI)
            mpiHandler.reset(new MPI_Handler);
        else
            csThrowException("MPI is not initialised.");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try
    {
        daeModel_t       model;
        filesystem::path inputDirectoryPath = filesystem::absolute( filesystem::path(inputDirectory) );

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        filesystem::path simulationOptionsPath = inputDirectoryPath / "simulation_options.json";
        cfg.Load(simulationOptionsPath.string());

        /* Load the model from the specified directory. */
        model.Load(rank, inputDirectoryPath.string());

        /* Run simulation based on the type of the model. */
        if(model.GetModel()->structure.isODESystem)
            simulate_ODE(model, inputDirectoryPath, cfg, log, datareporter);
        else
            simulate_DAE(model, inputDirectoryPath, cfg, log, datareporter);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

void csSimulate(csModelPtr                        csModel,
                const std::string&                jsonOptions,
                const std::string&                simulationDirectory,
                std::shared_ptr<csLog_t>          log,
                std::shared_ptr<csDataReporter_t> datareporter,
                bool                              initMPI)
{
    /* MPI_Init is typically called by the executable linking this library.
     * However, it can be performed by setting initMPI to true.
     * This is mostly used for single processor simulations (i.e. in pyOpenCS). */
    std::shared_ptr<MPI_Handler> mpiHandler;
    int mpiInitialised = false;
    MPI_Initialized(&mpiInitialised);
    if(!mpiInitialised)
    {
        if(initMPI)
            mpiHandler.reset(new MPI_Handler);
        else
            csThrowException("MPI is not initialised.");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try
    {
        daeModel_t       model;
        filesystem::path simulationDirectoryPath = filesystem::absolute( filesystem::path(simulationDirectory) );

        daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();
        cfg.LoadString(jsonOptions);
        //printf("Loaded options from string:\n%s\n", cfg.ToString().c_str());

        /* Load the model using the specified Compute Stack Model. */
        model.Load(rank, csModel);

        /* Run simulation based on the type of the model. */
        if(model.GetModel()->structure.isODESystem)
            simulate_ODE(model, simulationDirectoryPath, cfg, log, datareporter);
        else
            simulate_DAE(model, simulationDirectoryPath, cfg, log, datareporter);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
}

static std::string getFormattedDateTime()
{
    std::locale::global(std::locale());
    std::time_t t = std::time(nullptr);
    char time_s[100];
    std::strftime(time_s, sizeof(time_s), "[%d.%m.%Y-%H.%M.%S]", std::localtime(&t));
    return std::string(time_s);
}

static void printInfo(csModelPtr model, csLog_t* log)
{
    std::string sysType             = (model->structure.isODESystem ? "ODE" : "DAE");
    unsigned long Nequations        = model->structure.Nequations;
    unsigned long Nequations_total  = model->structure.Nequations_total;
    unsigned long Ndofs             = model->structure.Ndofs;
    unsigned long Ncs               = model->equations.computeStacks.size();
    double Ncs_per_eq               = double(Ncs)/Nequations;
    unsigned long Nnz               = model->sparsityPattern.Nnz;
    double Nnz_per_eq               = double(Nnz)/Nequations;
    double Ncs_jacob                = double(Nnz) * Ncs_per_eq;
    double Ncs_jacob_per_eq         = double(Ncs_jacob)/Nequations;

    const int msgBufferSize = 2048;
    char msgBuffer[msgBufferSize];
    std::string message;

    std::snprintf(msgBuffer, msgBufferSize, "Model information:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Type of the system:               %s\n",   sysType.c_str());
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. equations:                    %lu\n",  Nequations);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. equations (total):            %lu\n",  Nequations_total);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. dofs:                         %lu\n",  Ndofs);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. non-zero items:               %lu\n",  Nnz);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. non-zero items/equation:      %.2f\n", Nnz_per_eq);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. CS items:                     %lu\n",  Ncs);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. CS items/equation:            %.2f\n", Ncs_per_eq);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. CS items (Jacobian):          %.2f (est.)\n", Ncs_jacob);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  No. CS items (Jacobian)/equation: %.2f (est.)\n", Ncs_jacob_per_eq);
    message += msgBuffer;

    log->Message(message);
}

static void instantiateObjects(daeModel_t& model,
                               filesystem::path& simulationDirectoryPath,
                               daeSimulationOptions& cfg,
                               std::shared_ptr<csLog_t>& log,
                               std::shared_ptr<csDataReporter_t>& datareporter,
                               std::shared_ptr<csComputeStackEvaluator_t>& csEvaluator,
                               filesystem::path& outputDirectoryPath,
                               real_t& startTime,
                               real_t& timeHorizon,
                               real_t& reportingInterval)
{
    /* Store results and stats in the specified OutputDirectory */
    filesystem::path outDir = cfg.GetString("Simulation.OutputDirectory", "");
    /* If OutputDirectory is empty use the default 'results' and append the current date/time. */
    if(outDir.string().empty())
        outDir += "results-" + getFormattedDateTime();
    if(outDir.is_absolute())
        outputDirectoryPath = outDir;
    else
        outputDirectoryPath = simulationDirectoryPath / outDir;

    /* Create output directory. */
    filesystem::path odpath = filesystem::absolute( filesystem::path(outputDirectoryPath) );
    if(!filesystem::is_directory(odpath))
        filesystem::create_directories(odpath);

    startTime         = cfg.GetFloat("Simulation.StartTime");
    timeHorizon       = cfg.GetFloat("Simulation.TimeHorizon");
    reportingInterval = cfg.GetFloat("Simulation.ReportingInterval");

    std::string csEvaluatorLibrary = cfg.GetString("Model.ComputeStackEvaluator.Library", "Unknown");
    if(csEvaluatorLibrary == "Sequential")
    {
        csEvaluator = std::shared_ptr<csComputeStackEvaluator_t>(createEvaluator_Sequential());
    }
    else if(csEvaluatorLibrary == "OpenMP")
    {
        int numThreads = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.numThreads", 0);
        csEvaluator = std::shared_ptr<csComputeStackEvaluator_t>(createEvaluator_OpenMP(numThreads));
    }
    else if(csEvaluatorLibrary == "OpenCL")
    {
        std::string csEvaluatorName = cfg.GetString("Model.ComputeStackEvaluator.Name", "Unknown");
        if(csEvaluatorName == "Single-device")
        {
            int platformID                  = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.platformID");
            int deviceID                    = cfg.GetInteger("Model.ComputeStackEvaluator.Parameters.deviceID");
            std::string buildProgramOptions = cfg.GetString ("Model.ComputeStackEvaluator.Parameters.buildProgramOptions");

            csEvaluator = std::shared_ptr<csComputeStackEvaluator_t>(createEvaluator_OpenCL(platformID, deviceID, buildProgramOptions));
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

            csEvaluator = std::shared_ptr<csComputeStackEvaluator_t>(createEvaluator_OpenCL_MultiDevice(platforms,
                                                                                                        devices,
                                                                                                        taskPortions,
                                                                                                        buildProgramOptions));
        }
    }
    else
    {
        csThrowException("Invalid Compute Stack evaluator type specified");
    }

    /* Set the Compute Stack Evaluator. */
    model.SetComputeStackEvaluator(csEvaluator);

    /* Instantiate the log object. */
    if(!log)
    {
        std::string csLogName = cfg.GetString("Simulation.Log.Name", "StdOut");
        if(csLogName == "StdOut")
        {
            log = createLog_StdOut();
        }
        else if(csLogName == "TextFile")
        {
            filesystem::path path_filename_ext = cfg.GetString("Simulation.Log.Parameters.fileName", "");

            std::string logFilePath;
            if(!path_filename_ext.empty())
            {
                std::string logFileName = path_filename_ext.filename().stem().string() + "-" + std::to_string(model.GetModel()->pe_rank) + path_filename_ext.filename().extension().string();
                logFilePath = (outputDirectoryPath / logFileName).string();
            }

            log = createLog_TextFile(logFilePath);
        }
        else
        {
            csThrowException("Invalid log type specified");
        }
    }

    /* Instantiate the data reporter object. */
    if(!datareporter)
    {
        std::string csDataReporterName = cfg.GetString("Simulation.DataReporter.Name", "CSV");
        if(csDataReporterName == "CSV")
        {
            filesystem::path path_filename_res = cfg.GetString("Simulation.DataReporter.Parameters.fileNameResults",     "");
            filesystem::path path_filename_der = cfg.GetString("Simulation.DataReporter.Parameters.fileNameDerivatives", "");
            std::string outputformat           = cfg.GetString("Simulation.DataReporter.Parameters.outputFormat",        "fixed");
            std::string delim                  = cfg.GetString("Simulation.DataReporter.Parameters.delimiter",           ";");
            int precision                      = cfg.GetInteger("Simulation.DataReporter.Parameters.precision",          15);

            char delimiter = (delim.size() > 0 ? delim[0] : ';');

            std::string csvFilePath;
            std::string csvFilePath_der;
            if(!path_filename_res.empty())
            {
                std::string csvFileName = path_filename_res.filename().stem().string() + "-" + std::to_string(model.GetModel()->pe_rank) + path_filename_res.filename().extension().string();
                csvFilePath = (outputDirectoryPath / csvFileName).string();

            }
            if(!path_filename_der.empty())
            {
                std::string csvFileName_der = path_filename_der.filename().stem().string() + "-" + std::to_string(model.GetModel()->pe_rank) + path_filename_der.filename().extension().string();
                csvFilePath_der = (outputDirectoryPath / csvFileName_der).string();
            }

            datareporter = createDataReporter_CSV(csvFilePath, csvFilePath_der, delimiter, outputformat, precision);
        }
        else if(csDataReporterName == "HDF5")
        {
            filesystem::path path_filename_res = cfg.GetString("Simulation.DataReporter.Parameters.fileNameResults",     "");
            filesystem::path path_filename_der = cfg.GetString("Simulation.DataReporter.Parameters.fileNameDerivatives", "");

            std::string csvFilePath;
            std::string csvFilePath_der;
            if(!path_filename_res.empty())
            {
                std::string csvFileName = path_filename_res.filename().stem().string() + "-" + std::to_string(model.GetModel()->pe_rank) + path_filename_res.filename().extension().string();
                csvFilePath = (outputDirectoryPath / csvFileName).string();

            }
            if(!path_filename_der.empty())
            {
                std::string csvFileName_der = path_filename_der.filename().stem().string() + "-" + std::to_string(model.GetModel()->pe_rank) + path_filename_der.filename().extension().string();
                csvFilePath_der = (outputDirectoryPath / csvFileName_der).string();
            }

            datareporter = createDataReporter_HDF5(csvFilePath, csvFilePath_der);
        }
        else
        {
            csThrowException("Invalid data reporter type specified");
        }
    }

    /* Connect log and data reporter. */
    log->Connect(model.GetModel()->pe_rank);
    if(!log->IsConnected())
        csThrowException("Cannot connect log");

    datareporter->Connect(model.GetModel()->pe_rank);
    if(!datareporter->IsConnected())
        csThrowException("Cannot connect data reporter");

    /* Print model info. */
    if(model.GetModel()->pe_rank == 0)
        printInfo(model.GetModel(), log.get());
}


static void simulate_DAE(daeModel_t&                        model,
                         filesystem::path                   simulationDirectoryPath,
                         daeSimulationOptions&              cfg,
                         std::shared_ptr<csLog_t>           log_,
                         std::shared_ptr<csDataReporter_t>  datareporter_)
{
    daeSolver_t     daesolver;
    daeSimulation_t simulation;

    /* Double-check the model type. */
    if(model.GetModel()->structure.isODESystem)
        csThrowException("Simulation of the DAE system is requested but the system is ODE");

    /* Instatiate objects using the data from simulation_options.json file. */
    std::shared_ptr<csLog_t>                    log          = log_;
    std::shared_ptr<csDataReporter_t>           datareporter = datareporter_;
    std::shared_ptr<csComputeStackEvaluator_t>  evaluator;

    filesystem::path outputDirectoryPath;
    real_t startTime, timeHorizon, reportingInterval;

    instantiateObjects(model, simulationDirectoryPath, cfg, /* inputs */
                       log, datareporter, evaluator, outputDirectoryPath, /* outputs */
                       startTime, timeHorizon, reportingInterval);

    /* Initialize the simulation and the dae solver */
    simulation.Initialize(&model, log.get(), datareporter.get(), &daesolver, startTime, timeHorizon, reportingInterval, outputDirectoryPath.string());

    /* Calculate corrected initial conditions at t = 0. */
    simulation.SolveInitial();

    /* Run the simulation. */
    simulation.Run();

    /* Finalize the simulation. */
    simulation.PrintStats();
    simulation.Finalize();
}

static void simulate_ODE(daeModel_t&                        model,
                         filesystem::path                   simulationDirectoryPath,
                         daeSimulationOptions&              cfg,
                         std::shared_ptr<csLog_t>           log_,
                         std::shared_ptr<csDataReporter_t>  datareporter_)
{
    odeiSolver_t    odesolver;
    daeSimulation_t simulation;

    /* Initialize the model */
    if(!model.GetModel()->structure.isODESystem)
        csThrowException("Simulation of the ODE system is requested but the system is DAE");

    /* Instatiate objects using the data from simulation_options.json file. */
    std::shared_ptr<csLog_t>                    log          = log_;
    std::shared_ptr<csDataReporter_t>           datareporter = datareporter_;
    std::shared_ptr<csComputeStackEvaluator_t>  evaluator;

    filesystem::path outputDirectoryPath;
    real_t startTime, timeHorizon, reportingInterval;

    instantiateObjects(model, simulationDirectoryPath, cfg, /* inputs */
                       log, datareporter, evaluator, outputDirectoryPath, /* outputs */
                       startTime, timeHorizon, reportingInterval);

    /* Initialize the simulation and the dae solver. */
    simulation.Initialize(&model, log.get(), datareporter.get(), &odesolver, startTime, timeHorizon, reportingInterval, outputDirectoryPath.string());

    /* Calculate corrected initial conditions at t = 0 (not required here, and does nothing). */
    simulation.SolveInitial();

    /* Run the simulation. */
    simulation.Run();

    /* Finalize the simulation. */
    simulation.PrintStats();
    simulation.Finalize();
}

}
