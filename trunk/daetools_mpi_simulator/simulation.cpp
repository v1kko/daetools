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
#include "typedefs.h"
#include <fstream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>

namespace daetools_mpi
{
daeSimulation_t::daeSimulation_t()
{

}

daeSimulation_t::~daeSimulation_t()
{

}

void daeSimulation_t::Initialize(daeModel_t* pmodel,
                                 daeIDASolver_t* pdae_solver,
                                 const std::string& strOutputDirectory,
                                 bool bCalculateSensitivities)
{
    double start = auxiliary::get_time_in_seconds();

    outputDirectory        = strOutputDirectory;
    calculateSensitivities = bCalculateSensitivities;
    model                  = pmodel;
    daesolver              = pdae_solver;
    isInitialized          = true;
    currentTime            = model->startTime;
    timeHorizon            = model->timeHorizon;
    reportingInterval      = model->reportingInterval;

    daesolver->Initialize(model, this, model->Nequations,
                                       model->Nequations_local,
                                       model->initValues,
                                       model->initDerivatives,
                                       model->absoluteTolerances,
                                       model->ids,
                                       model->relativeTolerance);
    real_t t = 0;
    reportingTimes.clear();
    while(t < timeHorizon)
    {
        t += reportingInterval;
        if(t > timeHorizon)
            t = timeHorizon;
        reportingTimes.push_back(t);
    }

    initDuration = auxiliary::get_time_in_seconds() - start;
}

void daeSimulation_t::Finalize()
{
    daesolver->PrintSolverStats();
    daesolver->Free();
    model->Free();
}

void daeSimulation_t::Reinitialize()
{
    daesolver->Reinitialize(true, false);
}

void daeSimulation_t::SolveInitial()
{
    double start = auxiliary::get_time_in_seconds();

    daesolver->SolveInitial();
    ReportData();
    printf("System successfuly initialised\n");

    solveInitDuration = auxiliary::get_time_in_seconds() - start;
}

void daeSimulation_t::Run()
{
    real_t t;

    double start = auxiliary::get_time_in_seconds();

    int step = 0;
    while(currentTime < timeHorizon)
    {
        t = reportingTimes[step];
        //t = currentTime + reportingInterval;
        //if(t > timeHorizon)
        //    t = timeHorizon;

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

    integrationDuration = auxiliary::get_time_in_seconds() - start;

    printf("\n");
    printf("The simulation has finished successfully!\n");
    printf("Initialization time = %.3f\n", initDuration);
    printf("Integration time = %.3f\n", integrationDuration);
    printf("Total run time = %.3f\n", initDuration + solveInitDuration + integrationDuration);
}

real_t daeSimulation_t::Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    currentTime = daesolver->Solve(timeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities)
{
    currentTime = daesolver->Solve(currentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
    return currentTime;
}

real_t daeSimulation_t::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    currentTime = daesolver->Solve(time, eStopCriterion, bReportDataAroundDiscontinuities);
    return currentTime;
}

void daeSimulation_t::ReportData_dx()
{
    static int counter_dx = 0;

    std::ofstream ofs;
    std::string filename = std::string("derivatives-node-") + std::to_string(model->mpi_rank) + ".csv";
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
    std::string filename = std::string("results-node-") + std::to_string(model->mpi_rank) + ".csv";
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
