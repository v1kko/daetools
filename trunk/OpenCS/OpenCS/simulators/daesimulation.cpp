/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "auxiliary.h"
#include "daesimulator.h"
#include <fstream>
#include <iomanip>
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
using namespace cs;

namespace cs_dae_simulator
{
daeSimulation_t::daeSimulation_t()
{
    reportData    = true;
    model         = NULL;
    log           = NULL;
    data_reporter = NULL;
    daesolver     = NULL;
}

daeSimulation_t::~daeSimulation_t()
{

}

void daeSimulation_t::Initialize(csDifferentialEquationModel_t* pmodel,
                                 csLog_t* plog,
                                 csDataReporter_t* pdata_reporter,
                                 csDifferentialEquationSolver_t* pdae_solver,
                                 real_t dStartTime,
                                 real_t dTimeHorizon,
                                 real_t dReportingInterval,
                                 const std::string& strOutputDirectory,
                                 bool bCalculateSensitivities)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationInitialise);

    if(!plog)
        csThrowException("Invalid Log specified");
    if(!pdata_reporter)
        csThrowException("Invalid Data Reporter specified");
    if(!pdae_solver)
        csThrowException("Invalid DifferentialEquation Solver specified");
    if(!pmodel)
        csThrowException("Invalid Compute Stack Differential Equation Model specified");

    csModelPtr csModel = pmodel->GetModel();
    if(!csModel)
        csThrowException("The Compute Stack Model is not loaded");

    outputDirectory        = strOutputDirectory;
    calculateSensitivities = bCalculateSensitivities;
    model                  = pmodel;
    log                    = plog;
    data_reporter          = pdata_reporter;
    daesolver              = pdae_solver;
    isInitialized          = true;

    startTime              = dStartTime;
    timeHorizon            = dTimeHorizon;
    reportingInterval      = dReportingInterval;
    currentTime            = startTime;

    daesolver->Initialize(model, this, csModel->structure.Nequations_total,
                                       csModel->structure.Nequations,
                                       &csModel->structure.variableValues[0],
                                       &csModel->structure.variableDerivatives[0],
                                       &csModel->structure.absoluteTolerances[0],
                                       &csModel->structure.variableTypes[0]);

    csModelPtr cs_model = model->GetModel();
    data_reporter->RegisterVariables(cs_model->structure.variableNames);

    reportingTimes.clear();
    if(reportingInterval == 0.0) // do not report data, simulate from 0 to time horizon
    {
        reportingInterval = 0.0;
        reportingTimes.resize(1);
        reportingTimes[0] = timeHorizon;
        reportData = false;
    }
    else if(reportingInterval < 0.0) // simulate using the specified steps, but do not report data
    {
        reportingInterval = std::fabs(reportingInterval);
        int Nt = std::ceil(timeHorizon / reportingInterval);
        if(Nt <= 0)
            Nt = 1;
        reportingTimes.resize(Nt);
        for(int i = 0; i < Nt; i++)
        {
            real_t t = (i+1)*reportingInterval;
            if(i == Nt-1)
                t = timeHorizon;
            reportingTimes[i] = t;
        }
        reportData = false;
    }
    else // simulate using the specified steps and report data
    {
        int Nt = std::ceil(timeHorizon / reportingInterval);
        if(Nt <= 0)
            Nt = 1;
        reportingTimes.resize(Nt);
        for(int i = 0; i < Nt; i++)
        {
            real_t t = (i+1)*reportingInterval;
            if(i == Nt-1)
                t = timeHorizon;
            reportingTimes[i] = t;
        }
        reportData = true;
    }

    // Create the output directory
    try
    {
        filesystem::path outputDataPath = filesystem::absolute( filesystem::path(outputDirectory) );
        if(!filesystem::is_directory(outputDataPath))
            filesystem::create_directories(outputDataPath);
    }
    catch(std::exception& e)
    {
        std::snprintf(msgBuffer, msgBufferSize, "Cannot create output directory %s:\n%s\n", outputDirectory.c_str(), e.what());
        log->Message(msgBuffer);
    }
}

void daeSimulation_t::Finalize()
{
    SaveStats();
    if(daesolver)
        daesolver->Free();
    if(model)
        model->Free();
    if(data_reporter)
    {
        data_reporter->EndOfData();
        data_reporter->Disconnect();
    }
}

void daeSimulation_t::Reinitialize()
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    daesolver->Reinitialize(true, false);
}

void daeSimulation_t::SolveInitial()
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationSolveInitial);

    daesolver->SolveInitial();

    if(reportData)
        ReportData(0.0);

    int pe_rank = model->GetModel()->pe_rank;
    if(pe_rank == 0)
    {
        std::snprintf(msgBuffer, msgBufferSize, "System successfuly initialised");
        log->Message(msgBuffer);
    }
}

void daeSimulation_t::Run()
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    real_t t;
    int step = 0;
    int pe_rank = model->GetModel()->pe_rank;

    while(currentTime < timeHorizon)
    {
        t = reportingTimes[step];

        /* If discontinuity is found, loop until the end of the integration period
         * The data will be reported around discontinuities! */
        while(t > currentTime)
        {
            if(pe_rank == 0)
            {
                std::snprintf(msgBuffer, msgBufferSize, "Integrating from [%.15f] to [%.15f]...", currentTime, t);
                log->Message(msgBuffer);
            }
            IntegrateUntilTime(t, eStopAtModelDiscontinuity, true);
        }

        if(reportData)
            ReportData(currentTime);

        step++;
    }
    if(pe_rank == 0)
    {
        std::snprintf(msgBuffer, msgBufferSize, "The simulation has finished successfully!");
        log->Message(msgBuffer);
    }
}

real_t daeSimulation_t::Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(timeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities)
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(currentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.SimulationIntegration);

    currentTime = daesolver->Solve(time, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

void daeSimulation_t::PrintStats()
{
    if(!daesolver)
        csThrowException("Invalid DifferentialEquation Solver");

    int pe_rank = model->GetModel()->pe_rank;
    if(pe_rank != 0)
        return;

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();

    daesolver->CollectSolverStats();
    std::map<std::string, double>& stats = daesolver->stats;
    if(stats.empty())
        return;

    double totalRunTime = tcs.SimulationInitialise.duration + tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    // equations: time for equations evaluations that originates from the ODE/DAE solver.
    int EquationsEvaluation_daesolver_count = (int)stats["NumEquationEvals"];
    double dae_portion = double(EquationsEvaluation_daesolver_count) / tcs.EquationsEvaluation.count;
    double equations = tcs.EquationsEvaluation.duration * dae_portion;
    double total   = tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double latotal = 0.0;
    //if(daesolver->lasolver)
    //    latotal = tcs.LASetup.duration + tcs.LASolve.duration;
    //else if(daesolver->preconditioner)
    //    latotal = tcs.PSetup.duration + tcs.PSolve.duration;
    latotal = tcs.LASetup.duration + tcs.LASolve.duration;

    std::string message;
    std::snprintf(msgBuffer, msgBufferSize, "\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "                          %15s %14s %8s\n", "Time (s)",  "Rel.time (%)", "Count");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "Simulation stats:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Initialisation          %15.3f %14s %8s\n",   tcs.SimulationInitialise.duration,   "-", "-");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Solve initial           %15.3f %14s %8s\n",   tcs.SimulationSolveInitial.duration, "-", "-");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Integration             %15.3f %14s %8s\n",   tcs.SimulationIntegration.duration,  "-", "-");
    message += msgBuffer;
    //if(daesolver->lasolver)
    //    std::snprintf(msgBuffer, msgBufferSize, "  Lin.solver + equations  %15.3f %14.2f %8s\n", (latotal+equations), 100.0*(latotal+equations) / total, "-");
    //else if(daesolver->preconditioner)
    //    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner + equations%13.3f %14.2f %8s\n", (latotal+equations), 100.0*(latotal+equations) / total, "-");
    std::snprintf(msgBuffer, msgBufferSize, "  Lin.solver + equations  %15.3f %14.2f %8s\n", (latotal+equations), 100.0*(latotal+equations) / total, "-");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Total                   %15.3f %14s %8s\n",   totalRunTime,        "-",                               "-");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "Solver:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Steps                   %15s %14s %8d\n",    "-", "-", (int)stats["NumSteps"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Error Test Fails        %15s %14s %8d\n",    "-", "-", (int)stats["NumErrTestFails"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Equations (solver)      %15.3f %14.2f %8d\n", equations,                        100.0*equations                        / total, (int)stats["NumEquationEvals"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Equations (total)       %15.3f %14.2f %8d\n", tcs.EquationsEvaluation.duration, 100.0*tcs.EquationsEvaluation.duration / total, tcs.EquationsEvaluation.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  IPC data exchange       %15.3f %14.2f %8d\n", tcs.IPCDataExchange.duration,     100.0*tcs.IPCDataExchange.duration     / total, tcs.IPCDataExchange.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "Non-linear solver:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Iterations              %15s %14s %8d\n", "-", "-", (int)stats["NumNonlinSolvIters"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Convergence fails       %15s %14s %8d\n", "-", "-", (int)stats["NumNonlinSolvIters"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "Linear solver:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Iterations              %15s %14s %8d\n", "-", "-", (int)stats["NumLinIters"]);
    message += msgBuffer;
    //if(daesolver->lasolver)
    //{
    //    std::snprintf(msgBuffer, msgBufferSize, "  Setup                   %15.3f %14.2f %8d\n", tcs.LASetup.duration,            100.0*tcs.LASetup.duration / total, tcs.LASetup.count);
    //    std::snprintf(msgBuffer, msgBufferSize, "  Jacobian                %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration, 100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
    //    std::snprintf(msgBuffer, msgBufferSize, "  Solve                   %15.3f %14.2f %8d\n", tcs.LASolve.duration,            100.0*tcs.LASolve.duration / total, tcs.LASolve.count);
    //}
    //else if(daesolver->preconditioner)
    //{
    //    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner setup    %15.3f %14.2f %8d\n", tcs.PSetup.duration, 100.0*tcs.PSetup.duration / total, tcs.PSetup.count);
    //    std::snprintf(msgBuffer, msgBufferSize, "  Jacobian                %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration,  100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
    //    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner solve    %15.3f %14.2f %8d\n", tcs.PSolve.duration, 100.0*tcs.PSolve.duration / total, tcs.PSolve.count);
    //}
    std::snprintf(msgBuffer, msgBufferSize, "  LASetup                 %15.3f %14.2f %8d\n", tcs.LASetup.duration,            100.0*tcs.LASetup.duration             / total, tcs.LASetup.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  LASolve                 %15.3f %14.2f %8d\n", tcs.LASolve.duration,            100.0*tcs.LASolve.duration             / total, tcs.LASolve.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Jacobian                %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration, 100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner setup    %15.3f %14.2f %8d\n", tcs.PSetup.duration,             100.0*tcs.PSetup.duration              / total, tcs.PSetup.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner solve    %15.3f %14.2f %8d\n", tcs.PSolve.duration,             100.0*tcs.PSolve.duration              / total, tcs.PSolve.count);
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "  Total (LA setup+solve)  %15.3f %14.2f %8s\n", latotal,                100.0*latotal / total,                "-");
    message += msgBuffer;
    if(tcs.Jvtimes.count > 0)
        std::snprintf(msgBuffer, msgBufferSize, "  Jv multiply             %15.3f %14.2f %8d\n", tcs.Jvtimes.duration,   100.0*tcs.Jvtimes.duration   / total, tcs.Jvtimes.count);
    else
        std::snprintf(msgBuffer, msgBufferSize, "  Jv multiply DQ          %15.3f %14.2f %8d\n", tcs.JvtimesDQ.duration, 100.0*tcs.JvtimesDQ.duration / total, tcs.JvtimesDQ.count);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
    message += msgBuffer;

    log->Message(message);

    /*
    std::snprintf(msgBuffer, msgBufferSize, "  Linear solver setup:    %15.3f %14.2f %8d\n", tcs.LASetup.duration,             100.0*tcs.LASetup.duration             / total, tcs.LASetup.count);
    std::snprintf(msgBuffer, msgBufferSize, "  Linear solver solve:    %15.3f %14.2f %8d\n", tcs.LASolve.duration,             100.0*tcs.LASolve.duration             / total, tcs.LASolve.count);
    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner setup    %15s %14s %8d\n", "-", "-", (int)stats["NumPrecEvals"]);
    std::snprintf(msgBuffer, msgBufferSize, "  Preconditioner solve    %15s %14s %8d\n", "-", "-", (int)stats["NumPrecSolves"]);
    std::snprintf(msgBuffer, msgBufferSize, "  Jvtimes:                %15.3f %14.2f %8d\n", tcs.Jvtimes.duration,             100.0*tcs.Jvtimes.duration             / total, tcs.Jvtimes.count);
    std::snprintf(msgBuffer, msgBufferSize, "  JvtimesDQ:              %15.3f %14.2f %8d\n", tcs.JvtimesDQ.duration,           100.0*tcs.JvtimesDQ.duration           / total, tcs.JvtimesDQ.count);
    std::snprintf(msgBuffer, msgBufferSize, "  Jacobian evaluation:    %15.3f %14.2f %8d\n", tcs.JacobianEvaluation.duration,  100.0*tcs.JacobianEvaluation.duration  / total, tcs.JacobianEvaluation.count);
    std::snprintf(msgBuffer, msgBufferSize, "  IPC data exchange:      %15.3f %14.2f %8d\n", tcs.IPCDataExchange.duration,     100.0*tcs.IPCDataExchange.duration     / total, tcs.IPCDataExchange.count);
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");


    std::snprintf(msgBuffer, msgBufferSize, "DAE solver stats:\n");
    std::snprintf(msgBuffer, msgBufferSize, "  NumSteps:               %15s %14s %8d\n",    "-", "-", (int)stats["NumSteps"]);
    std::snprintf(msgBuffer, msgBufferSize, "  NumEquationEvals:            %15s %14s %8d\n",    "-", "-", (int)stats["NumEquationEvals"]);
    std::snprintf(msgBuffer, msgBufferSize, "  NumErrTestFails:        %15s %14s %8d\n",    "-", "-", (int)stats["NumErrTestFails"]);
    std::snprintf(msgBuffer, msgBufferSize, "  LastOrder:              %15s %14s %8d\n",    "-", "-", (int)stats["LastOrder"]);
    std::snprintf(msgBuffer, msgBufferSize, "  CurrentOrder:           %15s %14s %8d\n",    "-", "-", (int)stats["CurrentOrder"]);
    std::snprintf(msgBuffer, msgBufferSize, "  LastStep:               %15.12f %14s %8s\n", stats["LastStep"], "-", "-");
    std::snprintf(msgBuffer, msgBufferSize, "  CurrentStep:            %15.12f %14s %8s\n", stats["CurrentStep"], "-", "-");
    std::snprintf(msgBuffer, msgBufferSize, "-------------------------------------------------------------------\n");
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
        csThrowException("Invalid DifferentialEquation Solver");
    if(!model || !model->GetModel())
        csThrowException("Invalid Compute Stack Model");

    daesolver->CollectSolverStats();
    std::map<std::string, double>& stats = daesolver->stats;
    if(stats.empty())
        return;

    std::ofstream ofs;
    std::string filename = std::string("stats-") + std::to_string(model->GetModel()->pe_rank) + ".json";
    filesystem::path outputDataPath = filesystem::canonical( filesystem::path(outputDirectory) );
    if(!filesystem::is_directory(outputDataPath))
        filesystem::create_directories(outputDataPath);
    std::string filePath = (outputDataPath / filename).string();
    ofs.open(filePath, std::ofstream::out);
    if(!ofs.is_open())
        csThrowException("Cannot open " + filePath + " file");

    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();

    double totalRunTime = tcs.SimulationInitialise.duration + tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    double total   = tcs.SimulationSolveInitial.duration + tcs.SimulationIntegration.duration;
    int EquationsEvaluation_daesolver_count = (int)stats["NumEquationEvals"];
    double dae_portion = double(EquationsEvaluation_daesolver_count) / tcs.EquationsEvaluation.count;
    double equations = tcs.EquationsEvaluation.duration * dae_portion;
    double latotal = 0.0;
    //if(daesolver->lasolver)
    //    latotal = tcs.LASetup.duration + tcs.LASolve.duration;
    //else if(daesolver->preconditioner)
    //    latotal = tcs.PSetup.duration + tcs.PSolve.duration;
    latotal = tcs.LASetup.duration + tcs.LASolve.duration;

    double LA_equations_perc     = 100.0*(latotal+equations) / total;
    double LASetup_perc          = 100.0*tcs.LASetup.duration / total;
    double LASolve_perc          = 100.0*tcs.LASolve.duration / total;
    double PSetup_perc           = 100.0*tcs.PSetup.duration / total;
    double PSolve_perc           = 100.0*tcs.PSolve.duration / total;
    double Equations_solver_perc = 100.0*equations / total;
    double Equations_perc        = 100.0*tcs.EquationsEvaluation.duration / total;
    double Jacobian_perc         = 100.0*tcs.JacobianEvaluation.duration / total;

    ofs << "{" << std::endl;
    ofs << "  \"Timings\": {" << std::endl;
    ofs << (boost::format("    \"Initialization\":            %25.15f,\n") % tcs.SimulationInitialise.duration).str();
    ofs << (boost::format("    \"Integration\":               %25.15f,\n") % tcs.SimulationIntegration.duration).str();
    ofs << (boost::format("    \"Total\":                     %25.15f,\n") % totalRunTime).str();
    ofs << (boost::format("    \"LinearSolver_and_Equations\":%25.15f,\n") % (latotal + equations)).str();
    //if(daesolver->lasolver)
    //{
    ofs << (boost::format("    \"LASetup\":                   %25.15f,\n") % tcs.LASetup.duration).str();
    ofs << (boost::format("    \"LASolve\":                   %25.15f,\n") % tcs.LASolve.duration).str();
    //}
    //else if(daesolver->preconditioner)
    //{
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % tcs.PSetup.duration).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % tcs.PSolve.duration).str();
    //}
    ofs << (boost::format("    \"Jvtimes\":                   %25.15f,\n") % tcs.Jvtimes.duration).str();
    ofs << (boost::format("    \"JvtimesDQ\":                 %25.15f,\n") % tcs.JvtimesDQ.duration).str();
    ofs << (boost::format("    \"Equations_solver\":          %25.15f,\n") % equations).str();
    ofs << (boost::format("    \"Equations\":                 %25.15f,\n") % tcs.EquationsEvaluation.duration).str();
    ofs << (boost::format("    \"Jacobian\":                  %25.15f,\n") % tcs.JacobianEvaluation.duration).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %25.15f\n")  % tcs.IPCDataExchange.duration).str();
    ofs << "  }," << std::endl;

#ifdef STEP_DURATIONS
    ofs << "  \"Durations\": {" << std::endl;
    //if(daesolver->lasolver)
    //{
    ofs << (boost::format("    \"LASetup\":                   %s,\n") % toString(tcs.LASetup.durations)).str();
    ofs << (boost::format("    \"LASolve\":                   %s,\n") % toString(tcs.LASolve.durations)).str();
    //}
    //else if(daesolver->preconditioner)
    //{
    ofs << (boost::format("    \"PreconditionerSetup\":       %s,\n") % toString(tcs.PSetup.durations)).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %s,\n") % toString(tcs.PSolve.durations)).str();
    //}
    ofs << (boost::format("    \"JvtimesDQ\":                 %s,\n") % toString(tcs.JvtimesDQ.durations)).str();
    ofs << (boost::format("    \"Equations\":                 %s,\n") % toString(tcs.EquationsEvaluation.durations)).str();
    ofs << (boost::format("    \"Jacobian\":                  %s,\n") % toString(tcs.JacobianEvaluation.durations)).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %s\n")  % toString(tcs.IPCDataExchange.durations)).str();
    ofs << "  }," << std::endl;
#endif

    ofs << "  \"Counts\": {" << std::endl;
    //if(daesolver->lasolver)
    //{
    ofs << (boost::format("    \"LASetup\":                   %25d,\n") % tcs.LASetup.count).str();
    ofs << (boost::format("    \"LASolve\":                   %25d,\n") % tcs.LASolve.count).str();
    //}
    //else if(daesolver->preconditioner)
    //{
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % tcs.PSetup.count).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % tcs.PSolve.count).str();
    //}
    ofs << (boost::format("    \"Jvtimes\":                   %25d,\n") % tcs.Jvtimes.count).str();
    ofs << (boost::format("    \"JvtimesDQ\":                 %25d,\n") % tcs.JvtimesDQ.count).str();
    ofs << (boost::format("    \"Equations_solver\":          %25d,\n") % EquationsEvaluation_daesolver_count).str();
    ofs << (boost::format("    \"Equations\":                 %25d,\n") % tcs.EquationsEvaluation.count).str();
    ofs << (boost::format("    \"Jacobian\":                  %25d,\n") % tcs.JacobianEvaluation.count).str();
    ofs << (boost::format("    \"IPCDataExchange\":           %25d\n")  % tcs.IPCDataExchange.count).str();
    ofs << "  }," << std::endl;

    ofs << "  \"Percents\": {" << std::endl;
    ofs << (boost::format("    \"LinearSolver_Equations\":    %25.15f,\n") % LA_equations_perc).str();
    //if(daesolver->lasolver)
    //{
    ofs << (boost::format("    \"LAsetup\":                   %25.15f,\n") % LASetup_perc).str();
    ofs << (boost::format("    \"LAsolve\":                   %25.15f,\n") % LASolve_perc).str();
    //}
    //else if(daesolver->preconditioner)
    //{
    ofs << (boost::format("    \"PreconditionerSetup\":       %25.15f,\n") % PSetup_perc).str();
    ofs << (boost::format("    \"PreconditionerSolve\":       %25.15f,\n") % PSolve_perc).str();
    //}
    ofs << (boost::format("    \"Equations_solver\":          %25.15f,\n") % Equations_solver_perc).str();
    ofs << (boost::format("    \"Equations\":                 %25.15f,\n") % Equations_perc).str();
    ofs << (boost::format("    \"Jacobian\":                  %25.15f\n")  % Jacobian_perc).str();
    ofs << "  }," << std::endl;

    ofs << "  \"SolverStats\": {" << std::endl;
    ofs << (boost::format("    \"NumSteps\":                  %25d,\n")    % (int)stats["NumSteps"]).str();
    ofs << (boost::format("    \"NumEquationEvals\":          %25d,\n")    % (int)stats["NumEquationEvals"]).str();
    ofs << (boost::format("    \"NumErrTestFails\":           %25d,\n")    % (int)stats["NumErrTestFails"]).str();
    ofs << (boost::format("    \"LastOrder\":                 %25d,\n")    % (int)stats["LastOrder"]).str();
    ofs << (boost::format("    \"CurrentOrder\":              %25d,\n")    % (int)stats["CurrentOrder"]).str();
    ofs << (boost::format("    \"LastStep\":                  %25.15f,\n") % stats["LastStep"]).str();
    ofs << (boost::format("    \"CurrentStep\":               %25.15f\n")  % stats["CurrentStep"]).str();
    ofs << "  }," << std::endl;

    ofs << "  \"LinearSolverStats\": {" << std::endl;
    ofs << (boost::format("    \"NumLinIters\":               %25d,\n") % (int)stats["NumLinIters"]).str();
    ofs << (boost::format("    \"NumEquationEvals\":          %25d,\n") % (int)stats["NumEquationEvals"]).str();
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

void daeSimulation_t::ReportData(real_t currentTime)
{
    data_reporter->StartNewResultSet(currentTime);
    data_reporter->SendVariables(daesolver->yval,    daesolver->Nequations);
    data_reporter->SendDerivatives(daesolver->ypval, daesolver->Nequations);
}

void daeSimulation_t::StoreInitializationValues(const char* strFileName)
{
}

void daeSimulation_t::LoadInitializationValues(const char* strFileName)
{
}

}
