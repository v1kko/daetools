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
#include "auxiliary.h"
#include "cs_simulator.h"
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

namespace cs_dae_simulator
{
daeSimulation_t::daeSimulation_t()
{

}

daeSimulation_t::~daeSimulation_t()
{

}

void daeSimulation_t::Initialize(csDAEModel_t* pmodel,
                                 daeIDASolver_t* pdae_solver,
                                 real_t dStartTime,
                                 real_t dTimeHorizon,
                                 real_t dReportingInterval,
                                 const std::string& strOutputDirectory,
                                 bool bCalculateSensitivities)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationInitialise);

    outputDirectory        = strOutputDirectory;
    calculateSensitivities = bCalculateSensitivities;
    model                  = pmodel;
    daesolver              = pdae_solver;
    isInitialized          = true;

    startTime              = dStartTime;
    timeHorizon            = dTimeHorizon;
    reportingInterval      = dReportingInterval;
    currentTime            = startTime;

    daesolver->Initialize(model, this, model->Nequations,
                                       model->Nequations_local,
                                       &model->variableValues[0],
                                       &model->variableDerivatives[0],
                                       &model->absoluteTolerances[0],
                                       &model->ids[0]);
    real_t t = 0;
    reportingTimes.clear();
    while(t < timeHorizon)
    {
        t += reportingInterval;
        if(t > timeHorizon)
            t = timeHorizon;
        reportingTimes.push_back(t);
    }
}

void daeSimulation_t::Finalize()
{
    SaveStats();
    daesolver->Free();
    model->Free();
}

void daeSimulation_t::Reinitialize()
{
    daesolver->Reinitialize(true, false);
}

void daeSimulation_t::SolveInitial()
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationSolveInitial);

    daesolver->SolveInitial();
    ReportData();
    printf("System successfuly initialised\n");
}

void daeSimulation_t::Run()
{
    real_t t;
    int step = 0;
    while(currentTime < timeHorizon)
    {
        t = reportingTimes[step];

        /* If discontinuity is found, loop until the end of the integration period
         * The data will be reported around discontinuities! */
        while(t > currentTime)
        {
            printf("Integrating from [%f] to [%f]...\n", currentTime, t);
            IntegrateUntilTime(t, eStopAtModelDiscontinuity, true);
        }

        ReportData();
        step++;
    }
    printf("The simulation has finished successfully!\n");
}

real_t daeSimulation_t::Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(timeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(currentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(time, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

void daeSimulation_t::PrintStats()
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();

    double totalRunTime = tcs.SimulationInitialise.duration + tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double residuals = tcs.ResidualsEvaluation.duration;
    double total   = tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double latotal = 0.0;
    if(daesolver->lasolver)
        latotal = tcs.LASetup.duration + tcs.LASolve.duration;
    else if(daesolver->preconditioner)
        latotal = tcs.PSetup.duration + tcs.PSolve.duration;

    printf("\n");
    printf("-------------------------------------------------------------------\n");
    printf("                          %15s %14s %8s\n", "Time (s)",  "Rel.time (%)", "Count");
    printf("-------------------------------------------------------------------\n");
    printf("Simulation stats:\n");
    printf("  Initialisation          %15.3f %14s %8s\n",   tcs.SimulationInitialise.duration,   "-", "-");
    printf("  Solve initial           %15.3f %14s %8s\n",   tcs.SimulationSolveInitial.duration, "-", "-");
    printf("  Integration             %15.3f %14s %8s\n",   tcs.SimulationIntegration.duration,  "-", "-");
    if(daesolver->lasolver)
        printf("  Lin.solver + residuals  %15.3f %14.2f %8s\n", (latotal+residuals), 100.0*(latotal+residuals) / total, "-");
    else if(daesolver->preconditioner)
        printf("  Preconditioner + residuals%13.3f %14.2f %8s\n", (latotal+residuals), 100.0*(latotal+residuals) / total, "-");
    printf("  Total                   %15.3f %14s %8s\n",   totalRunTime,                        "-", "-");
    printf("-------------------------------------------------------------------\n");

    if(!daesolver)
        return;
    daesolver->CollectSolverStats();
    std::map<std::string, double>& stats = daesolver->stats;
    if(stats.empty())
        return;

    printf("DAE solver:\n");
    printf("  Steps                   %15s %14s %8d\n",    "-", "-", (int)stats["NumSteps"]);
    printf("  Error Test Fails        %15s %14s %8d\n",    "-", "-", (int)stats["NumErrTestFails"]);
    printf("  Residuals (DAE)         %15s %14s %8d\n",    "-", "-", (int)stats["NumResEvals"]);
    printf("  Residuals (total)       %15.3f %14.2f %8d\n", tcs.ResidualsEvaluation.duration, 100.0*tcs.ResidualsEvaluation.duration / total, tcs.ResidualsEvaluation.count);
    printf("  IPC data exchange       %15.3f %14.2f %8d\n", tcs.IPCDataExchange.duration,     100.0*tcs.IPCDataExchange.duration     / total, tcs.IPCDataExchange.count);
    printf("-------------------------------------------------------------------\n");

    printf("Non-linear solver:\n");
    printf("  Iterations              %15s %14s %8d\n", "-", "-", (int)stats["NumNonlinSolvIters"]);
    printf("  Convergence fails       %15s %14s %8d\n", "-", "-", (int)stats["NumNonlinSolvIters"]);
    printf("-------------------------------------------------------------------\n");

    printf("Linear solver:\n");
    printf("  Iterations              %15s %14s %8d\n", "-", "-", (int)stats["NumLinIters"]);
    if(daesolver->lasolver)
    {
        printf("  Setup                   %15.3f %14.2f %8d\n", tcs.LASetup.duration,            100.0*tcs.LASetup.duration / total, tcs.LASetup.count);
        printf("  Jacobian                %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration, 100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
        printf("  Solve                   %15.3f %14.2f %8d\n", tcs.LASolve.duration,            100.0*tcs.LASolve.duration / total, tcs.LASolve.count);
    }
    else if(daesolver->preconditioner)
    {
        printf("  Preconditioner setup    %15.3f %14.2f %8d\n", tcs.PSetup.duration, 100.0*tcs.PSetup.duration / total, tcs.PSetup.count);
        printf("  Jacobian                %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration,  100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
        printf("  Preconditioner solve    %15.3f %14.2f %8d\n", tcs.PSolve.duration, 100.0*tcs.PSolve.duration / total, tcs.PSolve.count);
    }
    printf("  Total (setup+solve)     %15.3f %14.2f %8s\n", latotal,                100.0*latotal / total,                "-");
    printf("  Jv multiply             %15s %14s %8d\n",     "-",                    "-",                                  (int)stats["NumJtimesEvals"]);
    printf("  Jv multiply DQ          %15.3f %14.2f %8d\n", tcs.JvtimesDQ.duration, 100.0*tcs.JvtimesDQ.duration / total, tcs.JvtimesDQ.count);
    printf("-------------------------------------------------------------------\n");

    /*
    printf("  Linear solver setup:    %15.3f %14.2f %8d\n", tcs.LASetup.duration,             100.0*tcs.LASetup.duration             / total, tcs.LASetup.count);
    printf("  Linear solver solve:    %15.3f %14.2f %8d\n", tcs.LASolve.duration,             100.0*tcs.LASolve.duration             / total, tcs.LASolve.count);
    printf("  Preconditioner setup    %15s %14s %8d\n", "-", "-", (int)stats["NumPrecEvals"]);
    printf("  Preconditioner solve    %15s %14s %8d\n", "-", "-", (int)stats["NumPrecSolves"]);
    printf("  Jvtimes:                %15.3f %14.2f %8d\n", tcs.Jvtimes.duration,             100.0*tcs.Jvtimes.duration             / total, tcs.Jvtimes.count);
    printf("  JvtimesDQ:              %15.3f %14.2f %8d\n", tcs.JvtimesDQ.duration,           100.0*tcs.JvtimesDQ.duration           / total, tcs.JvtimesDQ.count);
    printf("  Jacobian evaluation:    %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration,  100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
    printf("  IPC data exchange:      %15.3f %14.2f %8d\n", tcs.IPCDataExchange.duration,     100.0*tcs.IPCDataExchange.duration     / total, tcs.IPCDataExchange.count);
    printf("-------------------------------------------------------------------\n");


    printf("DAE solver stats:\n");
    printf("  NumSteps:               %15s %14s %8d\n",    "-", "-", (int)stats["NumSteps"]);
    printf("  NumResEvals:            %15s %14s %8d\n",    "-", "-", (int)stats["NumResEvals"]);
    printf("  NumErrTestFails:        %15s %14s %8d\n",    "-", "-", (int)stats["NumErrTestFails"]);
    printf("  LastOrder:              %15s %14s %8d\n",    "-", "-", (int)stats["LastOrder"]);
    printf("  CurrentOrder:           %15s %14s %8d\n",    "-", "-", (int)stats["CurrentOrder"]);
    printf("  LastStep:               %15.12f %14s %8s\n", stats["LastStep"], "-", "-");
    printf("  CurrentStep:            %15.12f %14s %8s\n", stats["CurrentStep"], "-", "-");
    printf("-------------------------------------------------------------------\n");
    */
}

std::string toString(std::vector<double>& durations)
{
    std::string result = "[";
    for(int i = 0; i < durations.size(); i++)
    {
        if(i != 0)
            result += ",";
        result += (boost::format("%.15f") % durations[i]).str();
    }
    result += "]";
    return result;
}

void daeSimulation_t::SaveStats()
{
    if(!daesolver)
        return;

    daesolver->CollectSolverStats();
    std::map<std::string, double>& stats = daesolver->stats;
    if(stats.empty())
        return;

    std::ofstream ofs;
    std::string filename = std::string("stats-") + std::to_string(model->pe_rank) + ".json";
    boost::filesystem::path outputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(outputDirectory) );
    if(!boost::filesystem::is_directory(outputDataPath))
        boost::filesystem::create_directories(outputDataPath);
    std::string filePath = (outputDataPath / filename).string();
    ofs.open(filePath, std::ofstream::out);
    if(!ofs.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();

    double totalRunTime = tcs.SimulationInitialise.duration + tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double total   = tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double residuals = tcs.ResidualsEvaluation.duration;
    double latotal = 0.0;
    if(daesolver->lasolver)
        latotal = tcs.LASetup.duration + tcs.LASolve.duration;
    else if(daesolver->preconditioner)
        latotal = tcs.PSetup.duration + tcs.PSolve.duration;

    double LA_residuals_perc = 100.0*(latotal+residuals) / total;
    double LASetup_perc      = 100.0*tcs.LASetup.duration / total;
    double LASolve_perc      = 100.0*tcs.LASolve.duration / total;
    double PSetup_perc       = 100.0*tcs.PSetup.duration / total;
    double PSolve_perc       = 100.0*tcs.PSolve.duration / total;
    double Residuals_perc    = 100.0*tcs.ResidualsEvaluation.duration / total;
    double Jacobian_perc     = 100.0*tcs.JacobianEvaluation.duration / total;

    ofs << "{" << std::endl;
    ofs << "  \"Timings\": {" << std::endl;
    ofs << (boost::format("    \"Initialization\":            %25.15f,\n") % tcs.SimulationInitialise.duration).str();
    ofs << (boost::format("    \"Integration\":               %25.15f,\n") % tcs.SimulationIntegration.duration).str();
    ofs << (boost::format("    \"Total\":                     %25.15f,\n") % totalRunTime).str();
    ofs << (boost::format("    \"LinearSolver_Residuals\":    %25.15f,\n") % (latotal + residuals)).str();
    if(daesolver->lasolver)
    {
    ofs << (boost::format("    \"LASetup\":                   %25.15f,\n") % tcs.LASetup.duration).str();
    ofs << (boost::format("    \"LASolve\":                   %25.15f,\n") % tcs.LASolve.duration).str();
    }
    else if(daesolver->preconditioner)
    {
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % tcs.PSetup.duration).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % tcs.PSolve.duration).str();
    }
    ofs << (boost::format("    \"Jvtimes\":                   %25.15f,\n") % tcs.Jvtimes.duration).str();
    ofs << (boost::format("    \"JvtimesDQ\":                 %25.15f,\n") % tcs.JvtimesDQ.duration).str();
    ofs << (boost::format("    \"Residuals\":                 %25.15f,\n") % tcs.ResidualsEvaluation.duration).str();
    ofs << (boost::format("    \"Jacobian\":                  %25.15f,\n") % tcs.JacobianEvaluation.duration).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %25.15f\n")  % tcs.IPCDataExchange.duration).str();
    ofs << "  }," << std::endl;

#ifdef STEP_DURATIONS
    ofs << "  \"Durations\": {" << std::endl;
    if(daesolver->lasolver)
    {
    ofs << (boost::format("    \"LASetup\":                   %s,\n") % toString(tcs.LASetup.durations)).str();
    ofs << (boost::format("    \"LASolve\":                   %s,\n") % toString(tcs.LASolve.durations)).str();
    }
    else if(daesolver->preconditioner)
    {
    ofs << (boost::format("    \"PreconditionerSetup\":       %s,\n") % toString(tcs.PSetup.durations)).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %s,\n") % toString(tcs.PSolve.durations)).str();
    }
    ofs << (boost::format("    \"JvtimesDQ\":                 %s,\n") % toString(tcs.JvtimesDQ.durations)).str();
    ofs << (boost::format("    \"Residuals\":                 %s,\n") % toString(tcs.ResidualsEvaluation.durations)).str();
    ofs << (boost::format("    \"Jacobian\":                  %s,\n") % toString(tcs.JacobianEvaluation.durations)).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %s\n")  % toString(tcs.IPCDataExchange.durations)).str();
    ofs << "  }," << std::endl;
#endif

    ofs << "  \"Counts\": {" << std::endl;
    if(daesolver->lasolver)
    {
    ofs << (boost::format("    \"LASetup\":                   %25d,\n") % tcs.LASetup.count).str();
    ofs << (boost::format("    \"LASolve\":                   %25d,\n") % tcs.LASolve.count).str();
    }
    else if(daesolver->preconditioner)
    {
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % tcs.PSetup.count).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % tcs.PSolve.count).str();
    }
    ofs << (boost::format("    \"Jvtimes\":                   %25d,\n") % tcs.Jvtimes.count).str();
    ofs << (boost::format("    \"JvtimesDQ\":                 %25d,\n") % tcs.JvtimesDQ.count).str();
    ofs << (boost::format("    \"Residuals\":                 %25d,\n") % tcs.ResidualsEvaluation.count).str();
    ofs << (boost::format("    \"Jacobian\":                  %25d,\n") % tcs.JacobianEvaluation.count).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %25d\n")  % tcs.IPCDataExchange.count).str();
    ofs << "  }," << std::endl;

    ofs << "  \"Percents\": {" << std::endl;
    ofs << (boost::format("    \"LinearSolver_Residuals\":    %25.15f,\n") % LA_residuals_perc).str();
    if(daesolver->lasolver)
    {
    ofs << (boost::format("    \"LAsetup\":                   %25.15f,\n") % LASetup_perc).str();
    ofs << (boost::format("    \"LAsolve\":                   %25.15f,\n") % LASolve_perc).str();
    }
    else if(daesolver->preconditioner)
    {
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % PSetup_perc).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % PSolve_perc).str();
    }
    ofs << (boost::format("    \"Residuals\":                 %25.15f,\n") % Residuals_perc).str();
    ofs << (boost::format("    \"Jacobian\":                  %25.15f\n") % Jacobian_perc).str();
    ofs << "  }," << std::endl;

    ofs << "  \"DAESolverStats\": {" << std::endl;
    ofs << (boost::format("    \"NumSteps\":                  %25d,\n")    % (int)stats["NumSteps"]).str();
    ofs << (boost::format("    \"NumResEvals\":               %25d,\n")    % (int)stats["NumResEvals"]).str();
    ofs << (boost::format("    \"NumErrTestFails\":           %25d,\n")    % (int)stats["NumErrTestFails"]).str();
    ofs << (boost::format("    \"LastOrder\":                 %25d,\n")    % (int)stats["LastOrder"]).str();
    ofs << (boost::format("    \"CurrentOrder\":              %25d,\n")    % (int)stats["CurrentOrder"]).str();
    ofs << (boost::format("    \"LastStep\":                  %25.15f,\n") % stats["LastStep"]).str();
    ofs << (boost::format("    \"CurrentStep\":               %25.15f\n")  % stats["CurrentStep"]).str();
    ofs << "  }," << std::endl;

    ofs << "  \"LinearSolverStats\": {" << std::endl;
    ofs << (boost::format("    \"NumLinIters\":               %25d,\n") % (int)stats["NumLinIters"]).str();
    ofs << (boost::format("    \"NumResEvals\":               %25d,\n") % (int)stats["NumResEvals"]).str();
    ofs << (boost::format("    \"NumPrecEvals\":              %25d,\n") % (int)stats["NumPrecEvals"]).str();
    ofs << (boost::format("    \"NumPrecSolves\":             %25d,\n") % (int)stats["NumPrecSolves"]).str();
    ofs << (boost::format("    \"NumJtimesEvals\":            %25d\n")  % (int)stats["NumJtimesEvals"]).str();
    ofs << "  }," << std::endl;

    ofs << "  \"NonLinarSolverStats\": {" << std::endl;
    ofs << (boost::format("    \"NumNonlinSolvIters\":        %25d,\n") % (int)stats["NumNonlinSolvIters"]).str();
    ofs << (boost::format("    \"NumNonlinSolvConvFails\":    %25d\n")  % (int)stats["NumNonlinSolvConvFails"]).str();
    ofs << "  }" << std::endl;
    ofs << "}" << std::endl;
}

void daeSimulation_t::ReportData_dx()
{
    static int counter_dx = 0;

    std::ofstream ofs;
    std::string filename = std::string("derivatives-node-") + std::to_string(model->pe_rank) + ".csv";
    boost::filesystem::path outputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(outputDirectory) );
    if(!boost::filesystem::is_directory(outputDataPath))
        boost::filesystem::create_directories(outputDataPath);
    std::string filePath = (outputDataPath / filename).string();

    ofs << std::setiosflags(std::ios_base::fixed);
    ofs << std::setprecision(15);

    if(counter_dx == 0)
    {
        ofs.open(filePath, std::ofstream::out);
        if(!ofs.is_open())
            throw std::runtime_error("Cannot open " + filePath + " file");

        ofs << " ";
        for(int i = 0; i < daesolver->Nequations; i++)
            ofs << "; " << i;
        ofs << std::endl;

        ofs << "time";
        for(int i = 0; i < daesolver->Nequations; i++)
            ofs << ";d" << model->variableNames[i] << "/dt";
        ofs << std::endl;
    }
    else
    {
        ofs.open(filePath, std::ofstream::out|std::ofstream::app);
        if(!ofs.is_open())
            throw std::runtime_error("Cannot open " + filePath + " file");
    }

    ofs << daesolver->currentTime;
    for(int i = 0; i < daesolver->Nequations; i++)
        ofs << ";" << daesolver->ypval[i];
    ofs << std::endl;

    counter_dx++;
}

void daeSimulation_t::ReportData()
{
    static int counter = 0;

    std::ofstream ofs;
    std::string filename = std::string("results-node-") + std::to_string(model->pe_rank) + ".csv";
    boost::filesystem::path outputDataPath = boost::filesystem::weakly_canonical( boost::filesystem::path(outputDirectory) );
    if(!boost::filesystem::is_directory(outputDataPath))
        boost::filesystem::create_directories(outputDataPath);
    std::string filePath = (outputDataPath / filename).string();

    ofs << std::setiosflags(std::ios_base::fixed);
    ofs << std::setprecision(15);

    if(counter == 0)
    {
        ofs.open(filePath, std::ofstream::out);
        if(!ofs.is_open())
            throw std::runtime_error("Cannot open " + filePath + " file");

        ofs << " ";
        for(int i = 0; i < daesolver->Nequations; i++)
            ofs << ";" << i;
        ofs << std::endl;

        ofs << "time";
        for(int i = 0; i < daesolver->Nequations; i++)
            ofs << ";" << model->variableNames[i];
        ofs << std::endl;
    }
    else

    {
        ofs.open(filePath, std::ofstream::out|std::ofstream::app);
        if(!ofs.is_open())
            throw std::runtime_error("Cannot open " + filePath + " file");
    }

    ofs << daesolver->currentTime;
    for(int i = 0; i < daesolver->Nequations; i++)
        ofs << ";" << daesolver->yval[i];
    ofs << std::endl;

    counter++;

    ReportData_dx();

/*
    int i;
    char* out;
    int* lengths = malloc(daesolver->Nequations * sizeof(int));

    for(i = 0; i < daesolver->Nequations; i++)
        lengths[i] = MAX(strlen(model->variableNames[i]) + 1, 21);

    printf("\n");
    printf("Results at time: %.7f\n", daesolver->currentTime);
    for(i = 0; i < daesolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");
    for(i = 0; i < daesolver->Nequations; i++)
        printf("| %-*s ", lengths[i], model->variableNames[i]);
    printf("|\n");
    for(i = 0; i < daesolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");

    for(i = 0; i < daesolver->Nequations; i++)
        printf("| %-*.14e ", lengths[i], daesolver->yval[i]);
    printf("|\n");
    for(i = 0; i < daesolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");

    free(out);
    free(lengths);
*/
}

void daeSimulation_t::StoreInitializationValues(const char* strFileName)
{
}

void daeSimulation_t::LoadInitializationValues(const char* strFileName)
{
}

}
