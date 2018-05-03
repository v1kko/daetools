#include "stdafx.h"
#include "simulation.h"
//#include "../Core/nodes.h"
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <limits>
#include <boost/foreach.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
//#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <boost/format.hpp>

namespace dae
{
namespace activity
{
daeSimulation::daeSimulation(void)
{
    m_dCurrentTime					= 0;
    m_dTimeHorizon					= 0;
    m_dReportingInterval			= 0;
    m_pDAESolver					= NULL;
    m_pDataReporter					= NULL;
    m_pModel						= NULL;
    m_pLog							= NULL;
    m_eActivityAction				= eAAUnknown;
    m_eSimulationMode				= eSimulation;
    m_bConditionalIntegrationMode	= false;
    m_bIsInitialized				= false;
    m_bIsSolveInitial				= false;
    m_bCalculateSensitivities		= false;
    m_iNoSensitivityFiles           = 0;
    m_nNumberOfObjectiveFunctions	= 1;
    m_pCurrentSensitivityMatrix     = NULL;

    m_bEvaluationModeSet    = false;
    m_computeStackEvaluator = NULL;

    m_InitializationDuration  = 0;
    m_SolveInitalDuration     = 0;
    m_IntegrationDuration     = 0;
    m_ProblemCreationStart    = 0;
    m_ProblemCreationEnd      = 0;
    m_InitializationStart     = 0;
    m_InitializationEnd       = 0;
    m_IntegrationStart        = 0;
    m_IntegrationEnd          = 0;

    daeConfig& cfg = daeConfig::GetConfig();
    m_bReportTimeDerivatives = cfg.GetBoolean("daetools.activity.reportTimeDerivatives", false);
    m_bReportSensitivities   = cfg.GetBoolean("daetools.activity.reportSensitivities",   false);

    if( cfg.GetBoolean("daetools.activity.stopAtModelDiscontinuity", true) )
        m_eStopAtModelDiscontinuity = eStopAtModelDiscontinuity;
    else
        m_eStopAtModelDiscontinuity = eDoNotStopAtDiscontinuity;

    m_bReportDataAroundDiscontinuities = cfg.GetBoolean("daetools.activity.reportDataAroundDiscontinuities", true);
}

daeSimulation::~daeSimulation(void)
{
}

void daeSimulation::SetUpParametersAndDomains()
{
}

void daeSimulation::SetUpVariables()
{
}

void daeSimulation::SetUpOptimization(void)
{
}

void daeSimulation::SetUpParameterEstimation(void)
{
}

void daeSimulation::SetUpSensitivityAnalysis(void)
{
}

void daeSimulation::DoDataPartitioning(daeEquationsIndexes& equationsIndexes, std::map<size_t,size_t>& mapVariableIndexes)
{
}

void daeSimulation::DoPostProcessing(void)
{
}

void daeSimulation::Resume(void)
{
    if(!m_bIsInitialized)
        return;
    if(!m_bIsSolveInitial)
        return;

// Prevents resuming of the already running simulation
// or multiple Resume() calls
    if(m_eActivityAction == eRunActivity)
        return;

// Prevents resuming if the simulation has been finished
    if(m_dCurrentTime >= m_dTimeHorizon)
        return;

    if(m_pLog)
        m_pLog->Message(string("Trying to resume activity..."), 0);

    m_eActivityAction = eRunActivity;

    Run();
}

void daeSimulation::Pause(void)
{
    if(!m_bIsInitialized)
        return;
    if(!m_bIsSolveInitial)
        return;

// Prevents multiple Pause() calls
    if(m_eActivityAction == ePauseActivity)
        return;
// Prevents pausing if the simulation has been finished
    if(m_dCurrentTime == 0 || m_dCurrentTime >= m_dTimeHorizon)
        return;

    if(m_pLog)
        m_pLog->Message(string("Trying to pause activity..."), 0);

    m_eActivityAction = ePauseActivity;
}

void daeSimulation::Initialize(daeDAESolver_t* pDAESolver,
                               daeDataReporter_t* pDataReporter,
                               daeLog_t* pLog,
                               bool bCalculateSensitivities,
                               const std::string& strJSONRuntimeSettings)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pLog)
        daeDeclareAndThrowException(exInvalidPointer);

    if(m_bIsInitialized)
    {
        daeDeclareException(exInvalidCall);
        e << "Simulation has already been initialized";
        throw e;
    }

    if(!strJSONRuntimeSettings.empty())
        SetJSONRuntimeSettings(strJSONRuntimeSettings);

    m_bCalculateSensitivities     = bCalculateSensitivities;
    m_nNumberOfObjectiveFunctions = 0;

    m_pDAESolver    = pDAESolver;
    m_pDataReporter	= pDataReporter;
    m_pLog			= pLog;

    m_ProblemCreationStart = dae::GetTimeInSeconds();

    daeConfig& cfg = daeConfig::GetConfig();
    bool bPrintHeader = cfg.GetBoolean("daetools.activity.printHeader", true);
    bool bPrintInfo   = cfg.GetBoolean("daetools.core.printInfo",       false);

    if(bPrintHeader)
    {
        m_pLog->Message(string("***********************************************************************"), 0);
        m_pLog->Message(string("                                 Version:   ") + daeVersion(true),         0);
        m_pLog->Message(string("                                 Copyright: Dragan Nikolic, 2017       "), 0);
        m_pLog->Message(string("                                 Homepage:  http://www.daetools.com    "), 0);
        m_pLog->Message(string("       @                       @                                       "), 0);
        m_pLog->Message(string("       @   @@@@@     @@@@@   @@@@@    @@@@@    @@@@@   @      @@@@@    "), 0);
        m_pLog->Message(string("  @@@@@@        @   @     @    @     @     @  @     @  @     @         "), 0);
        m_pLog->Message(string(" @     @   @@@@@@   @@@@@@     @     @     @  @     @  @      @@@@@    "), 0);
        m_pLog->Message(string(" @     @  @     @   @          @     @     @  @     @  @           @   "), 0);
        m_pLog->Message(string("  @@@@@@   @@@@@@    @@@@@      @@@   @@@@@    @@@@@    @@@@  @@@@@    "), 0);
        m_pLog->Message(string("                                                                       "), 0);
        m_pLog->Message(string("***********************************************************************"), 0);
        m_pLog->Message(string(" DAE Tools is free software: you can redistribute it and/or modify     "), 0);
        m_pLog->Message(string(" it under the terms of the GNU General Public License version 3        "), 0);
        m_pLog->Message(string(" as published by the Free Software Foundation.                         "), 0);
        m_pLog->Message(string(" DAE Tools is distributed in the hope that it will be useful,          "), 0);
        m_pLog->Message(string(" but WITHOUT ANY WARRANTY; without even the implied warranty of        "), 0);
        m_pLog->Message(string(" MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      "), 0);
        m_pLog->Message(string(" General Public License for more details.                              "), 0);
        m_pLog->Message(string(" You should have received a copy of the GNU General Public License     "), 0);
        m_pLog->Message(string(" along with this program. If not, see <http://www.gnu.org/licenses>.   "), 0);
        m_pLog->Message(string("***********************************************************************"), 0);
    }

    m_pLog->Message(string("Creating the system... "), 0);

    if(bPrintInfo)
        m_pLog->Message(string("  Initializing the simulation... "), 0);

// Create data proxy and propagate it
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage1"), 0);
    m_pModel->InitializeStage1();

// Initialize params and domains
    if(bPrintInfo)
        m_pLog->Message(string("    SetUpParametersAndDomains"), 0);

    if(m_strJSONRuntimeSettings.empty())
        SetUpParametersAndDomains();
    else
        SetUpParametersAndDomains_RuntimeSettings();

// Define the optimization problem: objective function and constraints
    if(bPrintInfo)
        m_pLog->Message(string("    Setup optimization/sensitivity analysis"), 0);

    if(m_eSimulationMode == eOptimization)
    {
        m_bCalculateSensitivities = true;

        SetNumberOfObjectiveFunctions(1);

    // Call SetUpOptimization to define obj. function, constraints and opt. variables
        SetUpOptimization();
    }
    else if(m_eSimulationMode == eParameterEstimation)
    {
        m_bCalculateSensitivities = true;

        SetNumberOfObjectiveFunctions(0);

    // Call SetUpParameterEstimation to define obj. function(s), constraints and opt. variables
        SetUpParameterEstimation();
    }
    else
    {
        if(m_bCalculateSensitivities)
        {
        // There are no optmisation functions by default.
        // They can be setup using the SetNumberOfObjectiveFunctions function that can be
        // called from the SetUpSensitivityAnalysis function
        //    SetNumberOfObjectiveFunctions(1);

        // Call SetUpSensitivityAnalysis to define sensitivity parameters using SetSensitivityParameter function
        // which is just an alias for SetContinuousOptimizationVariable(variable, LB=0.0, UB=1.0, defaultValue=1.0).
            SetUpSensitivityAnalysis();
        }
    }

    if(m_bCalculateSensitivities)
    {
    // Check if there are any opt. variable set up; if not, raise an exception
        if(m_arrOptimizationVariables.empty())
        {
            daeDeclareException(exInvalidCall);
            e << "Sensitivity calculation is enabled but no optimization variables have been defined";
            throw e;
        }

        m_iNoSensitivityFiles = 0;

        // If sensitivity data folder is specified, check if it exists and create it if necessary
        if(!m_strSensitivityDataDirectory.empty())
        {
            boost::filesystem::path sensitivityDataDirectory = m_strSensitivityDataDirectory;
            if(!boost::filesystem::is_directory(sensitivityDataDirectory))
            {
                if(!boost::filesystem::create_directories(sensitivityDataDirectory))
                {
                    daeDeclareException(exInvalidCall);
                    e << "An invalid sensitivity data directory specified: cannot create the " << m_strSensitivityDataDirectory << " directory";
                    throw e;
                }
            }
        }
    }

// Create model/port arrays and initialize variable indexes
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage2"), 0);
    m_pModel->InitializeStage2();

// Create data storage for variables, derivatives, var. types, tolerances, etc
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage3"), 0);
    m_pModel->InitializeStage3(m_pLog);

// Set initial values, initial conditions, fix variables, set initial guesses, abs tolerances, etc
    if(bPrintInfo)
        m_pLog->Message(string("    SetUpVariables"), 0);

    if(m_strJSONRuntimeSettings.empty())
        SetUpVariables();
    else
        SetUpVariables_RuntimeSettings();

// Create equation execution infos in models and stns
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage4"), 0);
    m_pModel->InitializeStage4();

// Now we have everything set up and we should check for inconsistences
    if(bPrintInfo)
        m_pLog->Message(string("    CheckSystem"), 0);
    CheckSystem();

// Do the block decomposition if needed (at the moment only one block is created)
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage5"), 0);
    m_ptrBlock = m_pModel->InitializeStage5();

    daeBlock* pBlock = dynamic_cast<daeBlock*>(m_ptrBlock);

    // Set the computeStackEvaluator (if specified)
    if(m_computeStackEvaluator)
        pBlock->SetComputeStackEvaluator(m_computeStackEvaluator);
    else if(m_bEvaluationModeSet)
        m_pModel->GetDataProxy()->SetEvaluationMode(m_evaluationMode);

    // If required manipulate the block indexes (i.e. used in C++(MPI) code generator)
    if(bPrintInfo)
        m_pLog->Message(string("    DoDataPartitioning"), 0);
    DoDataPartitioning(pBlock->m_EquationsIndexes, pBlock->m_mapVariableIndexes);

    // Use the block indexes from the daeBlock to populate EquationExecutionInfos,
    // build Jacobian expressions (if required; also uses the block indexes),
    // and initialize the block.
    if(bPrintInfo)
        m_pLog->Message(string("    InitializeStage6"), 0);
    m_pModel->InitializeStage6(m_ptrBlock);

// Setup DAE solver and sensitivities
    if(bPrintInfo)
        m_pLog->Message(string("    Setup DAE Solver"), 0);
    SetupSolver();

// Collect variables to report
    if(bPrintInfo)
        m_pLog->Message(string("    Collect variables to report"), 0);
    m_dCurrentTime = 0;
    CollectVariables(m_pModel);

// Set the IsInitialized flag to true
    m_bIsInitialized = true;

// Announce success
    m_ProblemCreationEnd = dae::GetTimeInSeconds();
    m_pLog->Message(string("The system created successfully in: ") +
                    toStringFormatted<real_t>(m_ProblemCreationEnd - m_ProblemCreationStart, -1, 3) +
                    string(" s"), 0);
}

std::map<std::string, real_t> daeSimulation::GetEvaluationCallsStats()
{
    std::map<std::string, real_t> stats;

    daeBlock* pBlock = dynamic_cast<daeBlock*>(m_ptrBlock);
    if(!pBlock)
        daeDeclareAndThrowException(exInvalidPointer);
    stats["nuberOfResidualsCalls"]              = pBlock->m_nNuberOfResidualsCalls;
    stats["nuberOfJacobianCalls"]               = pBlock->m_nNuberOfJacobianCalls;
    stats["nuberOfSensitivityResidualsCalls"]   = pBlock->m_nNuberOfSensitivityResidualsCalls;
    stats["totalTimeForResiduals"]              = pBlock->m_dTotalTimeForResiduals;
    stats["totalTimeForJacobian"]               = pBlock->m_dTotalTimeForJacobian;
    stats["totalTimeForSensitivityResiduals"]   = pBlock->m_dTotalTimeForSensitivityResiduals;
    stats["initializationTime"]                 = m_InitializationDuration;
    stats["solveInitialTime"]                   = m_SolveInitalDuration;
    stats["integrationTime"]                    = m_IntegrationDuration;
    return stats;
}

std::vector<daeEquationExecutionInfo*> daeSimulation::GetEquationExecutionInfos(void) const
{
    daeBlock* pBlock = dynamic_cast<daeBlock*>(m_ptrBlock);
    if(!pBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    std::vector<daeEquationExecutionInfo*>& arrEEI_ActiveSet = pBlock->GetEquationExecutionInfos_ActiveSet();
    return arrEEI_ActiveSet;
}

size_t daeSimulation::GetNumberOfEquations(void) const
{
    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);
    return m_ptrBlock->GetNumberOfEquations();
}

size_t daeSimulation::GetTotalNumberOfVariables(void) const
{
    return m_pModel->GetDataProxy()->GetTotalNumberOfVariables();
}

void daeSimulation::SetupSolver(void)
{
    size_t i;
    vector<size_t> narrParametersIndexes;
    boost::shared_ptr<daeOptimizationVariable> pOptVariable;
    boost::shared_ptr<daeOptimizationConstraint> pConstraint;
    boost::shared_ptr<daeObjectiveFunction> pObjectiveFunction;
    boost::shared_ptr<daeMeasuredVariable> pMeasuredVariable;
    vector<string> strarrErrors;

    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    if(m_bCalculateSensitivities)
    {
    // 1. First check obj. functions, Constraints, optimization variables and measured variables
        for(i = 0; i < m_arrOptimizationVariables.size(); i++)
        {
            pOptVariable = m_arrOptimizationVariables[i];
            if(!pOptVariable)
                daeDeclareAndThrowException(exInvalidPointer);

            if(!pOptVariable->CheckObject(strarrErrors))
            {
                daeDeclareException(exRuntimeCheck);
                for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
                    e << *it << "\n";
                throw e;
            }
        }

        for(i = 0; i < m_arrObjectiveFunctions.size(); i++)
        {
            pObjectiveFunction = m_arrObjectiveFunctions[i];
            if(!pObjectiveFunction)
                daeDeclareAndThrowException(exInvalidPointer);

            if(!pObjectiveFunction->CheckObject(strarrErrors))
            {
                daeDeclareException(exRuntimeCheck);
                for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
                    e << *it << "\n";
                throw e;
            }
        }

        for(i = 0; i < m_arrConstraints.size(); i++)
        {
            pConstraint = m_arrConstraints[i];
            if(!pConstraint)
                daeDeclareAndThrowException(exInvalidPointer);

            if(!pConstraint->CheckObject(strarrErrors))
            {
                daeDeclareException(exRuntimeCheck);
                for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
                    e << *it << "\n";
                throw e;
            }
        }

        for(i = 0; i < m_arrMeasuredVariables.size(); i++)
        {
            pMeasuredVariable = m_arrMeasuredVariables[i];
            if(!pMeasuredVariable)
                daeDeclareAndThrowException(exInvalidPointer);

            if(!pMeasuredVariable->CheckObject(strarrErrors))
            {
                daeDeclareException(exRuntimeCheck);
                for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
                    e << *it << "\n";
                throw e;
            }
        }
    }

    if(m_eSimulationMode == eOptimization)
    {
    // 2. Fill the parameters indexes (optimization variables)
        for(i = 0; i < m_arrOptimizationVariables.size(); i++)
        {
            pOptVariable = m_arrOptimizationVariables[i];
            narrParametersIndexes.push_back(pOptVariable->GetOverallIndex());
        }

    // 3. Initialize the objective functions
        for(i = 0; i < m_arrObjectiveFunctions.size(); i++)
        {
            pObjectiveFunction = m_arrObjectiveFunctions[i];
            pObjectiveFunction->Initialize(m_arrOptimizationVariables, m_ptrBlock);
        }

    // 4. Initialize the constraints
        for(i = 0; i < m_arrConstraints.size(); i++)
        {
            pConstraint = m_arrConstraints[i];
            pConstraint->Initialize(m_arrOptimizationVariables, m_ptrBlock);
        }
    }
    else if(m_eSimulationMode == eParameterEstimation)
    {
    // 2. Fill the parameters indexes (optimization variables)
        for(i = 0; i < m_arrOptimizationVariables.size(); i++)
        {
            pOptVariable = m_arrOptimizationVariables[i];
            narrParametersIndexes.push_back(pOptVariable->GetOverallIndex());
        }

    // 3. Initialize objective functions
        for(i = 0; i < m_arrObjectiveFunctions.size(); i++)
        {
            pObjectiveFunction = m_arrObjectiveFunctions[i];
            pObjectiveFunction->Initialize(m_arrOptimizationVariables, m_ptrBlock);
        }

    // 4. Initialize measured variables
        for(i = 0; i < m_arrMeasuredVariables.size(); i++)
        {
            pMeasuredVariable = m_arrMeasuredVariables[i];
            pMeasuredVariable->Initialize(m_arrOptimizationVariables, m_ptrBlock);
        }
    }
    else
    {
        if(m_bCalculateSensitivities)
        {
        // 2. Fill the parameters indexes (optimization variables)
            for(i = 0; i < m_arrOptimizationVariables.size(); i++)
            {
                pOptVariable = m_arrOptimizationVariables[i];
                narrParametersIndexes.push_back(pOptVariable->GetOverallIndex());
            }

        // 3. Initialize the objective functions
            for(i = 0; i < m_arrObjectiveFunctions.size(); i++)
            {
                pObjectiveFunction = m_arrObjectiveFunctions[i];
                pObjectiveFunction->Initialize(m_arrOptimizationVariables, m_ptrBlock);
            }

        // 4. Initialize the constraints
            for(i = 0; i < m_arrConstraints.size(); i++)
            {
                pConstraint = m_arrConstraints[i];
                pConstraint->Initialize(m_arrOptimizationVariables, m_ptrBlock);
            }
        }
    }

    m_pDAESolver->Initialize(m_ptrBlock, m_pLog, this, m_pModel->GetInitialConditionMode(), m_bCalculateSensitivities, narrParametersIndexes);
}

void daeSimulation::SolveInitial(void)
{
// Check if initialized
    if(!m_bIsInitialized)
    {
        daeDeclareException(exInvalidCall);
        e << "Simulation ignobly refused to solve initial: the simulation has not been initialized";
        throw e;
    }

// Check pointers
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDataReporter)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pLog)
        daeDeclareAndThrowException(exInvalidPointer);

// Start initialization
    m_InitializationStart = dae::GetTimeInSeconds();

    daeConfig& cfg = daeConfig::GetConfig();
    bool bPrintInfo = cfg.GetBoolean("daetools.core.printInfo", false);

// Register model if in simulation mode; otherwise it will be done later by the optimization/param.estimation
    if(!m_pDataReporter->IsConnected())
    {
        daeDeclareException(exInvalidCall);
        e << "Simulation cowardly refused to solve initial: the data reporter is not connected";
        throw e;
    }
    if(m_eSimulationMode == eSimulation)
        RegisterData("");

// Ask DAE solver to initialize the system
    if(bPrintInfo)
        m_pLog->Message(string("    Trying to solve the system initially"), 0);
    m_pDAESolver->SolveInitial();

// Report data at TIME=0
    if(bPrintInfo)
        m_pLog->Message(string("    Report data at the initial time"), 0);
    ReportData(m_dCurrentTime);

// Set the SolveInitial flag to true
    m_bIsSolveInitial = true;

// Announce success
    m_InitializationEnd = dae::GetTimeInSeconds();
    m_pLog->Message(string("Starting the initialization of the system... Done."), 0);
    m_pLog->SetProgress(0);
}

void daeSimulation::Run(void)
{
// Check if initialized
    if(!m_bIsInitialized)
    {
        daeDeclareException(exInvalidCall);
        e << "Simulation ignobly refuses to run: the simulation has not been initialized";
        throw e;
    }
    if(!m_bIsSolveInitial)
    {
        daeDeclareException(exInvalidCall);
        e << "Simulation ignobly refuses to run: the function SolveInitial() must be called before a call to Run()";
        throw e;
    }

// Check pointers
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDataReporter)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pLog)
        daeDeclareAndThrowException(exInvalidPointer);

// If the model is not dynamic there is no point in running
    if(!m_pModel->IsModelDynamic())
        return;

// Check for some mistakes
    if(m_dTimeHorizon <= 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid time horizon (less than or equal to zero); did you forget to set the TimeHorizon?";
        throw e;
    }
    if(m_darrReportingTimes.empty())
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid reporting times; did you forget to set the ReportingInterval or ReportingTimes array?";
        throw e;
    }

    m_IntegrationStart = dae::GetTimeInSeconds();
    m_pLog->Message(string("m_IntegrationStart time = ")    + toStringFormatted<real_t>(m_IntegrationStart,     -1, 3) + string(" s"), 0);

    if(m_dCurrentTime >= m_dTimeHorizon)
    {
        m_pLog->Message(string("The time horizon has been riched; exiting."), 0);
        return;
    }
    m_eActivityAction = eRunActivity;


    real_t t;
    while(m_dCurrentTime < m_dTimeHorizon)
    {
        t = GetNextReportingTime();
        if(t > m_dTimeHorizon)
            t = m_dTimeHorizon;

    // If the flag is set - terminate
        if(m_eActivityAction == ePauseActivity)
        {
            m_pLog->Message(string("Activity paused by the user"), 0);
            return;
        }
/*
    // Integrate until the first discontinuity or until the end of the integration period
    // Report current time period in low precision
        m_pLog->Message(string("Integrating from [") + toString<real_t>(m_dCurrentTime) +
                        string("] to [")             + toString<real_t>(t)              +
                        string("] ..."), 0);
        m_dCurrentTime = IntegrateUntilTime(t, eStopAtModelDiscontinuity);
        ReportData();
*/

    // If discontinuity is found, loop until the end of the integration period
    // The data will be reported around discontinuities!
        while(t > m_dCurrentTime)
        {
            m_pLog->Message(string("Integrating from [") + toStringFormatted<real_t>(m_dCurrentTime, -1, 15, false, true) +
                            string("] to [")             + toStringFormatted<real_t>(t, -1, 15, false, true)              +
                            string("] ..."), 0);
            m_dCurrentTime = IntegrateUntilTime(t, m_eStopAtModelDiscontinuity, m_bReportDataAroundDiscontinuities);
        }

        // Purpose of this line?
        // m_dCurrentTime is already set to the current time returned by IDA solver in Integrate... functions.
        //m_dCurrentTime = t;

        ReportData(m_dCurrentTime);
        real_t newProgress = ceil(100.0 * m_dCurrentTime/m_dTimeHorizon);
        if(newProgress > m_pLog->GetProgress())
            m_pLog->SetProgress(newProgress);
    }

// Print the end of the simulation info if not in the optimization mode
    if(m_eSimulationMode == eSimulation)
    {
        m_IntegrationEnd = dae::GetTimeInSeconds();

        m_InitializationDuration = m_ProblemCreationEnd - m_ProblemCreationStart;
        m_SolveInitalDuration    = m_InitializationEnd  - m_InitializationStart;
        m_IntegrationDuration    = m_IntegrationEnd     - m_IntegrationStart;
        double totalTime = m_InitializationDuration + m_SolveInitalDuration + m_IntegrationDuration;

        m_pLog->Message(string(" "), 0);
        m_pLog->Message(string("The simulation has finished successfully!"), 0);
        m_pLog->Message(string("Initialization time = ") + toStringFormatted<real_t>(m_InitializationDuration,  -1, 3) + string(" s"), 0);
        m_pLog->Message(string("Integration time = ")    + toStringFormatted<real_t>(m_IntegrationDuration,     -1, 3) + string(" s"), 0);
        m_pLog->Message(string("Total run time = ")      + toStringFormatted<real_t>(totalTime,                 -1, 3) + string(" s"), 0);
    }
}

void daeSimulation::CleanUpSetupData(void)
{
// Clean up what can be cleaned up in the Models/DataProxy
    m_pModel->GetDataProxy()->CleanUpSetupData();
}

void daeSimulation::Finalize(void)
{
// Notify the receiver that there is no more data, and disconnect it
    if(m_pDataReporter)
    {
        m_pDataReporter->EndOfData();
        m_pDataReporter->Disconnect();
    }
    if(m_computeStackEvaluator)
    {
        m_computeStackEvaluator->FreeResources();
    }

    if(m_pDAESolver)
    {
        m_pDAESolver->Finalize();
    }

    m_pModel		= NULL;
    m_pDAESolver	= NULL;
    m_pDataReporter = NULL;
    m_pLog			= NULL;
    m_computeStackEvaluator = NULL;

    m_bIsInitialized	 = false;
    m_bIsSolveInitial	 = false;
}

void daeSimulation::Reset(void)
{
    m_dCurrentTime			= 0;
    m_eActivityAction		= eAAUnknown;

    m_pDAESolver->Reset();
}

void daeSimulation::ReRun(void)
{
    SetUpVariables();
    Reset();
    SolveInitial();
    Run();
}

daeOptimizationConstraint* daeSimulation::CreateInequalityConstraint(string strDescription)
{
    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.constraintsAbsoluteTolerance", 1E-8);

    boost::shared_ptr<daeOptimizationConstraint> pConstraint(new daeOptimizationConstraint(m_pModel, m_pDAESolver, true, dAbsTolerance, m_arrConstraints.size(), strDescription));
    m_arrConstraints.push_back(pConstraint);
    return pConstraint.get();
}

daeOptimizationConstraint* daeSimulation::CreateEqualityConstraint(string strDescription)
{
    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.constraintsAbsoluteTolerance", 1E-8);

    boost::shared_ptr<daeOptimizationConstraint> pConstraint(new daeOptimizationConstraint(m_pModel, m_pDAESolver, false, dAbsTolerance, m_arrConstraints.size(), strDescription));
    m_arrConstraints.push_back(pConstraint);
    return pConstraint.get();
}

daeMeasuredVariable* daeSimulation::SetMeasuredVariable(daeVariable& variable)
{
    if(variable.GetNumberOfPoints() > 1)
    {
        daeDeclareException(exInvalidCall);
        e << "The measured variable [" << variable.GetCanonicalName() << "] is distributed but domain indexes have not been provided.";
        throw e;
    }

    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.measuredVariableAbsoluteTolerance", 1E-8);
    size_t nIndex  = m_arrMeasuredVariables.size();

    boost::shared_ptr<daeMeasuredVariable> measvar(new daeMeasuredVariable(m_pModel, m_pDAESolver, dAbsTolerance, nIndex, "Measured variable"));
    measvar->SetResidual(variable());
    m_arrMeasuredVariables.push_back(measvar);

    return measvar.get();
}

daeVariableWrapper* daeSimulation::SetInputVariable(daeVariable& variable)
{
    boost::shared_ptr<daeVariableWrapper> var(new daeVariableWrapper(variable, ""));
    m_arrInputVariables.push_back(var);

    return var.get();
}

daeOptimizationVariable* daeSimulation::SetModelParameter(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue)
{
    return daeSimulation::SetContinuousOptimizationVariable(variable, LB, UB, defaultValue);
}

daeMeasuredVariable* daeSimulation::SetMeasuredVariable(adouble a)
{
    daeVariable* variable;
    std::vector<size_t> narrDomainIndexes;

    daeGetVariableAndIndexesFromNode(a, &variable, narrDomainIndexes);

    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.measuredVariableAbsoluteTolerance", 1E-8);
    size_t nIndex  = m_arrMeasuredVariables.size();

    boost::shared_ptr<daeMeasuredVariable> measvar(new daeMeasuredVariable(m_pModel, m_pDAESolver, dAbsTolerance, nIndex, "Measured variable"));
    measvar->SetResidual(a);
    m_arrMeasuredVariables.push_back(measvar);

    return measvar.get();
}

daeVariableWrapper* daeSimulation::SetInputVariable(adouble a)
{
    boost::shared_ptr<daeVariableWrapper> var(new daeVariableWrapper(a, ""));
    m_arrInputVariables.push_back(var);
    return var.get();
}

daeOptimizationVariable* daeSimulation::SetModelParameter(adouble a, real_t LB, real_t UB, real_t defaultValue)
{
    return daeSimulation::SetContinuousOptimizationVariable(a, LB, UB, defaultValue);
}

daeOptimizationVariable* daeSimulation::SetContinuousOptimizationVariable(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue)
{
    std::vector<size_t> narrDomainIndexes;
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(&variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetContinuousOptimizationVariable(daeVariable& variable, quantity qLB, quantity qUB, quantity qdefaultValue)
{
    unit u = variable.GetVariableType()->GetUnits();
    real_t LB = qLB.scaleTo(u).getValue();
    real_t UB = qUB.scaleTo(u).getValue();
    real_t defaultValue = qdefaultValue.scaleTo(u).getValue();

    std::vector<size_t> narrDomainIndexes;
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(&variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetBinaryOptimizationVariable(daeVariable& variable, bool defaultValue)
{
    std::vector<size_t> narrDomainIndexes;
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(&variable, nOptVarIndex, narrDomainIndexes, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetIntegerOptimizationVariable(daeVariable& variable, int LB, int UB, int defaultValue)
{
    std::vector<size_t> narrDomainIndexes;
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(&variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetContinuousOptimizationVariable(adouble a, real_t LB, real_t UB, real_t defaultValue)
{
    daeVariable* variable;
    std::vector<size_t> narrDomainIndexes;

    daeGetVariableAndIndexesFromNode(a, &variable, narrDomainIndexes);
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetContinuousOptimizationVariable(adouble a, quantity qLB, quantity qUB, quantity qdefaultValue)
{
    daeVariable* variable;
    std::vector<size_t> narrDomainIndexes;

    daeGetVariableAndIndexesFromNode(a, &variable, narrDomainIndexes);
    size_t nOptVarIndex = m_arrOptimizationVariables.size();

    unit u = variable->GetVariableType()->GetUnits();
    real_t LB = qLB.scaleTo(u).getValue();
    real_t UB = qUB.scaleTo(u).getValue();
    real_t defaultValue = qdefaultValue.scaleTo(u).getValue();

    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetBinaryOptimizationVariable(adouble a, bool defaultValue)
{
    daeVariable* variable;
    std::vector<size_t> narrDomainIndexes;

    daeGetVariableAndIndexesFromNode(a, &variable, narrDomainIndexes);
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(variable, nOptVarIndex, narrDomainIndexes, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

daeOptimizationVariable* daeSimulation::SetIntegerOptimizationVariable(adouble a, int LB, int UB, int defaultValue)
{
    daeVariable* variable;
    std::vector<size_t> narrDomainIndexes;

    daeGetVariableAndIndexesFromNode(a, &variable, narrDomainIndexes);
    size_t nOptVarIndex = m_arrOptimizationVariables.size();
    boost::shared_ptr<daeOptimizationVariable> pVar(new daeOptimizationVariable(variable, nOptVarIndex, narrDomainIndexes, LB, UB, defaultValue));
    m_arrOptimizationVariables.push_back(pVar);
    return pVar.get();
}

//void daeSimulation::SetInitialConditionsToZero(void)
//{
//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer)
//	m_pModel->SetInitialConditions(0);
//}

void daeSimulation::CheckSystem(void) const
{
// The most important thing is to check:
//	- total number of variables
//	- total number of equations
//	- number of fixed variables
//	- number of time derivatives and initial conditions

    daeModelInfo mi;
    m_pModel->GetModelInfo(mi);

    if(mi.m_nNumberOfVariables == 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "The system cowardly refused to initialize:\n The number of variables is equal to zero\n";
        throw e;
    }

    if(mi.m_nNumberOfEquations == 0)
    {
        daeDeclareException(exRuntimeCheck);
        e << "The system cowardly refused to initialize:\n The number of equations is equal to zero\n";
        throw e;
    }

    if((mi.m_nNumberOfVariables - mi.m_nNumberOfFixedVariables) != mi.m_nNumberOfEquations)
    {
        daeDeclareException(exRuntimeCheck);
        if(mi.m_nNumberOfEquations < mi.m_nNumberOfVariables)
            e << "The system cowardly refused to initialize:\n The number of equations is lower than the number of variables \n";
        else
            e << "The system cowardly refused to initialize:\n The number of variables is lower than then number of equations \n";
        e << string("Number of equations: ")       + toString(mi.m_nNumberOfEquations)      + string("\n");
        e << string("Number of variables: ")       + toString(mi.m_nNumberOfVariables)      + string("\n");
        e << string("Number of fixed variables: ") + toString(mi.m_nNumberOfFixedVariables) + string("\n");
        throw e;
    }

    if(GetInitialConditionMode() == eAlgebraicValuesProvided)
    {
        if(mi.m_nNumberOfInitialConditions != mi.m_nNumberOfDifferentialVariables)
        {
            daeDeclareException(exRuntimeCheck);
            e << "Simulation cowardly refused to initialize the problem:\n The number of differential variables is not equal to the number of initial conditions \n";
            e << string("Number of differential variables: ") + toString(mi.m_nNumberOfDifferentialVariables) + string("\n");
            e << string("Number of initial conditions: ")     + toString(mi.m_nNumberOfInitialConditions)     + string("\n");
            throw e;
        }
    }
}

void daeSimulation::GetOptimizationConstraints(std::vector<daeOptimizationConstraint_t*>& ptrarrConstraints) const
{
    for(size_t i = 0; i < m_arrConstraints.size(); i++)
        ptrarrConstraints.push_back(m_arrConstraints[i].get());
}

void daeSimulation::GetOptimizationVariables(std::vector<daeOptimizationVariable_t*>& ptrarrOptVariables) const
{
    for(size_t i = 0; i < m_arrOptimizationVariables.size(); i++)
        ptrarrOptVariables.push_back(m_arrOptimizationVariables[i].get());
}

void daeSimulation::GetObjectiveFunctions(std::vector<daeObjectiveFunction_t*>& ptrarrObjectiveFunctions) const
{
    for(size_t i = 0; i < m_arrObjectiveFunctions.size(); i++)
        ptrarrObjectiveFunctions.push_back(m_arrObjectiveFunctions[i].get());
}

void daeSimulation::GetMeasuredVariables(std::vector<daeMeasuredVariable_t*>& ptrarrMeasuredVariables) const
{
    for(size_t i = 0; i < m_arrMeasuredVariables.size(); i++)
        ptrarrMeasuredVariables.push_back(m_arrMeasuredVariables[i].get());
}

daeObjectiveFunction* daeSimulation::GetObjectiveFunction(void) const
{
    if(m_arrObjectiveFunctions.empty())
        daeDeclareAndThrowException(exInvalidCall);

    return m_arrObjectiveFunctions[0].get();
}

boost::shared_ptr<daeObjectiveFunction> daeSimulation::AddObjectiveFunction(void)
{
    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.objFunctionAbsoluteTolerance", 1E-8);
    size_t nObjFunIndex  = m_arrObjectiveFunctions.size();

    boost::shared_ptr<daeObjectiveFunction> objfun(new daeObjectiveFunction(m_pModel, m_pDAESolver, dAbsTolerance, nObjFunIndex, "Objective function"));
    m_arrObjectiveFunctions.push_back(objfun);

    return objfun;
}

void daeSimulation::SetNumberOfObjectiveFunctions(size_t n)
{
    size_t i;

    if(!m_bCalculateSensitivities)
        daeDeclareAndThrowException(exInvalidCall);

    daeConfig& cfg = daeConfig::GetConfig();
    real_t dAbsTolerance = cfg.GetFloat("daetools.activity.objFunctionAbsoluteTolerance", 1E-8);

// Remove all equations from the model and delete all obj. functions
    for(i = 0; i < m_arrObjectiveFunctions.size(); i++)
    {
        boost::shared_ptr<daeObjectiveFunction> objfun = m_arrObjectiveFunctions[i];
        objfun->RemoveEquationFromModel();
    }
    m_arrObjectiveFunctions.clear();

    if(n == 0)
        return;

// Create new set of n objective functions
    m_nNumberOfObjectiveFunctions = n;
    for(size_t i = 0; i < m_nNumberOfObjectiveFunctions; i++)
        AddObjectiveFunction();
}

size_t daeSimulation::GetNumberOfObjectiveFunctions(void) const
{
    if(!m_bCalculateSensitivities)
        daeDeclareAndThrowException(exInvalidCall);

    return m_arrObjectiveFunctions.size();
}

daeeActivityAction daeSimulation::GetActivityAction(void) const
{
    return m_eActivityAction;
}

daeeSimulationMode daeSimulation::GetSimulationMode(void) const
{
    return m_eSimulationMode;
}

void daeSimulation::SetSimulationMode(daeeSimulationMode eMode)
{
    m_eSimulationMode = eMode;
}

void daeSimulation::GetReportingTimes(std::vector<real_t>& darrReportingTimes) const
{
    darrReportingTimes = m_darrReportingTimes;
}

void daeSimulation::SetReportingTimes(const std::vector<real_t>& darrReportingTimes)
{
    if(darrReportingTimes.empty())
        daeDeclareAndThrowException(exInvalidCall);

// Copy the array
    m_darrReportingTimes = darrReportingTimes;

// Sort the array
    std::sort(m_darrReportingTimes.begin(), m_darrReportingTimes.end());

// Remove duplicates and resize the array accordingly
    std::vector<real_t> arrRepTimes = m_darrReportingTimes;
    std::vector<real_t>::iterator iter  = std::unique_copy(arrRepTimes.begin(), arrRepTimes.end(), m_darrReportingTimes.begin());
    m_darrReportingTimes.resize(iter - m_darrReportingTimes.begin());

// Check the first element
    std::vector<real_t>::iterator first = m_darrReportingTimes.begin();
    if(*first <= 0)
    {
        daeDeclareException(exInvalidCall);
        e << "Invalid reporting time: " << *first << "  (less than or equal to zero)";
        throw e;
    }

// Set the TimeHorizon to the last (the highest) value
    std::vector<real_t>::iterator last  = m_darrReportingTimes.end() - 1;
    if(*last != m_dTimeHorizon)
        m_dTimeHorizon = *last;
}

real_t daeSimulation::GetNextReportingTime(void) const
{
    std::vector<real_t>::const_iterator next = std::upper_bound(m_darrReportingTimes.begin(), m_darrReportingTimes.end(), m_dCurrentTime);

    if(next == m_darrReportingTimes.end())
        return m_dTimeHorizon;
    if(*next > m_dTimeHorizon)
        return m_dTimeHorizon;

    return (*next);
}

real_t daeSimulation::GetCurrentTime_() const
{
    return m_dCurrentTime;
}

void daeSimulation::SetComputeStackEvaluator(computestack::adComputeStackEvaluator_t* computeStackEvaluator)
{
    m_computeStackEvaluator = computeStackEvaluator;
}

daeeEvaluationMode daeSimulation::GetEvaluationMode()
{
    return m_evaluationMode;
}

void daeSimulation::SetEvaluationMode(daeeEvaluationMode evaluationMode)
{
    if(evaluationMode == eComputeStack_External)
    {
        daeDeclareException(exInvalidCall);
        e << "EvaluationMode mode can be only set to eEvaluationTree_OpenMP and eComputeStack_OpenMP. "
          << "For external evaluators use SetComputeStackEvaluator function.";
        throw e;
    }

    m_bEvaluationModeSet = true;
    m_evaluationMode     = evaluationMode;
}

void daeSimulation::SetInitialConditionMode(daeeInitialConditionMode eMode)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);

    m_pModel->SetInitialConditionMode(eMode);
    m_pDAESolver->SetInitialConditionMode(eMode);
}

daeeInitialConditionMode daeSimulation::GetInitialConditionMode(void) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(m_pModel->GetInitialConditionMode() != m_pDAESolver->GetInitialConditionMode())
        daeDeclareAndThrowException(exRuntimeCheck);

    return m_pModel->GetInitialConditionMode();
}

void daeSimulation::StoreInitializationValues(const std::string& strFileName) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    m_pModel->StoreInitializationValues(strFileName);
}

void daeSimulation::LoadInitializationValues(const std::string& strFileName) const
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);

// Load values into the m_pDataProxy->m_pdarrValuesReferences
    m_pModel->LoadInitializationValues(strFileName);
}

void daeSimulation::SetTimeHorizon(real_t dTimeHorizon)
{
    if(dTimeHorizon <= 0)
        return;

    m_dTimeHorizon = dTimeHorizon;

    m_darrReportingTimes.clear();
    if(m_dReportingInterval <= 0)
        return;

    real_t t = 0;
    while(t < m_dTimeHorizon)
    {
        t += m_dReportingInterval;
        if(t > m_dTimeHorizon)
            t = m_dTimeHorizon;
        m_darrReportingTimes.push_back(t);
    }
}

real_t daeSimulation::GetTimeHorizon(void) const
{
    return m_dTimeHorizon;
}

void daeSimulation::SetReportingInterval(real_t dReportingInterval)
{
    if(dReportingInterval <= 0)
        return;
    m_dReportingInterval = dReportingInterval;

    m_darrReportingTimes.clear();

    real_t t = 0;
    while(t < m_dTimeHorizon)
    {
        t += m_dReportingInterval;
        if(t > m_dTimeHorizon)
            t = m_dTimeHorizon;
        m_darrReportingTimes.push_back(t);
    }
}

daeCondition* daeSimulation::GetLastSatisfiedCondition(void) const
{
    return m_pModel->GetDataProxy()->GetLastSatisfiedCondition();
}

real_t daeSimulation::GetReportingInterval(void) const
{
    return m_dReportingInterval;
}

// Integrates until the stopping criterion is reached or the time horizon of simulation
real_t daeSimulation::Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(m_dCurrentTime >= m_dTimeHorizon)
        daeDeclareAndThrowException(exInvalidCall);

    // Reset the last satisfied condition pointer
    m_pModel->GetDataProxy()->SetLastSatisfiedCondition(NULL);

    // Inform the DAE solver about the time past which the solution is not to proceed.
    // For instance, IDA solver can take time steps past the time horizon since the
    // default stop time is infinity and that can cause errors sometimes.
    m_pDAESolver->SetTimeHorizon(m_dTimeHorizon);

    m_dCurrentTime = m_pDAESolver->Solve(m_dTimeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return m_dCurrentTime;
}

// Integrates for the given time interval
real_t daeSimulation::IntegrateForTimeInterval(real_t time_interval, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if((m_dCurrentTime + time_interval) > m_dTimeHorizon)
        daeDeclareAndThrowException(exInvalidCall);

    // Reset the last satisfied condition pointer
    m_pModel->GetDataProxy()->SetLastSatisfiedCondition(NULL);

    // Inform the DAE solver about the time past which the solution is not to proceed.
    // For instance, IDA solver can take time steps past the time horizon since the
    // default stop time is infinity and that can cause errors sometimes.
    m_pDAESolver->SetTimeHorizon(m_dTimeHorizon);

    m_dCurrentTime = m_pDAESolver->Solve(m_dCurrentTime + time_interval, eStopCriterion, bReportDataAroundDiscontinuities);
    return m_dCurrentTime;
}

// Integrates until the stopping criterion or time is reached
real_t daeSimulation::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(time > m_dTimeHorizon)
        daeDeclareAndThrowException(exInvalidCall);

    // Reset the last satisfied condition pointer
    m_pModel->GetDataProxy()->SetLastSatisfiedCondition(NULL);

    // Inform the DAE solver about the time past which the solution is not to proceed.
    // For instance, IDA solver can take time steps past the time horizon since the
    // default stop time is infinity and that can cause errors sometimes.
    m_pDAESolver->SetTimeHorizon(m_dTimeHorizon);

    m_dCurrentTime = m_pDAESolver->Solve(time, eStopCriterion, bReportDataAroundDiscontinuities);
    return m_dCurrentTime;
}

void daeSimulation::Reinitialize(void)
{
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(m_dCurrentTime >= m_dTimeHorizon)
        daeDeclareAndThrowException(exInvalidCall);

    m_pDAESolver->Reinitialize(true, false);
}


std::string daeSimulation::GetSensitivityDataDirectory(void) const
{
    return m_strSensitivityDataDirectory;
}

void daeSimulation::SetSensitivityDataDirectory(const std::string strSensitivityDataDirectory)
{
    m_strSensitivityDataDirectory = strSensitivityDataDirectory;
}

daeeStopCriterion daeSimulation::GetStopAtModelDiscontinuity(void) const
{
    return m_eStopAtModelDiscontinuity;
}

void daeSimulation::SetStopAtModelDiscontinuity(daeeStopCriterion eStopAtModelDiscontinuity)
{
    m_eStopAtModelDiscontinuity = eStopAtModelDiscontinuity;
}

bool daeSimulation::GetReportDataAroundDiscontinuities(void) const
{
    return m_bReportDataAroundDiscontinuities;
}

void daeSimulation::SetReportDataAroundDiscontinuities(bool bReportDataAroundDiscontinuities)
{
    m_bReportDataAroundDiscontinuities = bReportDataAroundDiscontinuities;
}

std::vector<size_t> daeSimulation::GetActiveEquationSetMemory() const
{
    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    return m_ptrBlock->GetActiveEquationSetMemory();
}

std::map<std::string, size_t> daeSimulation::GetActiveEquationSetNodeCount() const
{
    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    return m_ptrBlock->GetActiveEquationSetNodeCount();
}

void daeSimulation::ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                              const std::string& filenameJacobianIndexes,
                                              int startEquationIndex,
                                              int endEquationIndex,
                                              const std::map<int,int>& bi_to_bi_local)
{
    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    m_ptrBlock->ExportComputeStackStructs(filenameComputeStacks, filenameJacobianIndexes, startEquationIndex, endEquationIndex, bi_to_bi_local);
}

void daeSimulation::ExportComputeStackStructs(const std::string& filenameComputeStacks,
                                              const std::string& filenameJacobianIndexes,
                                              const std::vector<uint32_t>& equationIndexes,
                                              const std::map<uint32_t,uint32_t>& bi_to_bi_local)
{
    if(!m_ptrBlock)
        daeDeclareAndThrowException(exInvalidPointer);

    m_ptrBlock->ExportComputeStackStructs(filenameComputeStacks, filenameJacobianIndexes, equationIndexes, bi_to_bi_local);
}

void daeSimulation::EnterConditionalIntegrationMode(void)
{
/**************************************************************/
    daeDeclareAndThrowException(exNotImplemented)
/**************************************************************/

//	m_bConditionalIntegrationMode = true;
//	daeModel* pModel = dynamic_cast<daeModel*>(m_pModel);
//	if(!pModel)
//		daeDeclareAndThrowException(exInvalidPointer);
//	pModel->SetGlobalConditionContext();
}

// Integrates until the stopping condition or final time is reached
real_t daeSimulation::IntegrateUntilConditionSatisfied(daeCondition rCondition, daeeStopCriterion eStopCriterion)
{
/**************************************************************/
    daeDeclareAndThrowException(exNotImplemented);
/**************************************************************/
/*
    if(!m_bConditionalIntegrationMode)
    {
        daeDeclareException(exInvalidCall);
        e << string("Simulation spinelessly failed to integrate until condition satisfied: "
                    "the function EnterConditionalIntegrationMode() should be called prior the call to IntegrateUntilConditionSatisfied()");
        throw e;
    }

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pDAESolver)
        daeDeclareAndThrowException(exInvalidPointer);
    if(m_dCurrentTime >= m_dTimeHorizon)
        daeDeclareAndThrowException(exInvalidCall);

    daeModel* pModel = dynamic_cast<daeModel*>(m_pModel);
    if(!pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    pModel->SetGlobalCondition(rCondition);
    m_bConditionalIntegrationMode = false;
    pModel->UnsetGlobalConditionContext();

    m_pDAESolver->RefreshRootFunctions();
    m_dCurrentTime = m_pDAESolver->Solve(m_dTimeHorizon, eStopCriterion);
    pModel->ResetGlobalCondition();
*/
    return m_dCurrentTime;
}

bool daeSimulation::GetCalculateSensitivities() const
{
    return m_bCalculateSensitivities;
}

void daeSimulation::SetCalculateSensitivities(bool bCalculateSensitivities)
{
    m_bCalculateSensitivities = bCalculateSensitivities;
}

void CollectAllDomains(daeModel* pModel, std::map<string, daeDomain*>& mapDomains);
void CollectAllParameters(daeModel* pModel, std::map<string, daeParameter*>& mapParameters);
void CollectAllVariables(daeModel* pModel, std::map<string, daeVariable*>& mapVariables);
void CollectAllSTNs(daeModel* pModel, std::map<string, daeSTN*>& mapSTNs);
void ProcessListOfValues(boost::property_tree::ptree& pt, std::vector<quantity>& values, unit& Units, std::vector<size_t>& Shape,
                         int currentDimension, bool allowNULL, bool bPrintInfo);
void ProcessSTN(daeSTN* pSTN, std::map<string, daeSTN*>& mapSTNs);

void CollectAllDomains(daeModel* pModel, std::map<string, daeDomain*>& mapDomains)
{
    // Insert objects from the model
    for(std::vector<daeDomain*>::const_iterator iter = pModel->Domains().begin(); iter != pModel->Domains().end(); iter++)
        mapDomains[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = pModel->Ports().begin(); piter != pModel->Ports().end(); piter++)
        for(std::vector<daeDomain*>::const_iterator citer = (*piter)->Domains().begin(); citer != (*piter)->Domains().end(); citer++)
            mapDomains[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllDomains(*miter, mapDomains);

    if(pModel->ModelArrays().size() > 0)
    {
        string msg = "Number of model arrays is not zero (collect domains from them too)";
        throw std::runtime_error(msg);
    }
    if(pModel->PortArrays().size() > 0)
    {
        string msg = "Number of ports arrays is not zero (collect domains from them too)";
        throw std::runtime_error(msg);
    }
}

void CollectAllParameters(daeModel* pModel, std::map<string, daeParameter*>& mapParameters)
{
    // Insert objects from the model
    for(std::vector<daeParameter*>::const_iterator iter = pModel->Parameters().begin(); iter != pModel->Parameters().end(); iter++)
        mapParameters[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = pModel->Ports().begin(); piter != pModel->Ports().end(); piter++)
        for(std::vector<daeParameter*>::const_iterator citer = (*piter)->Parameters().begin(); citer != (*piter)->Parameters().end(); citer++)
            mapParameters[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllParameters(*miter, mapParameters);

    if(pModel->ModelArrays().size() > 0)
    {
        string msg = "Number of model arrays is not zero (collect parameters from them too)";
        throw std::runtime_error(msg);
    }
    if(pModel->PortArrays().size() > 0)
    {
        string msg = "Number of ports arrays is not zero (collect parameters from them too)";
        throw std::runtime_error(msg);
    }
}

void CollectAllVariables(daeModel* pModel, std::map<string, daeVariable*>& mapVariables)
{
    // Insert objects from the model
    for(std::vector<daeVariable*>::const_iterator iter = pModel->Variables().begin(); iter != pModel->Variables().end(); iter++)
        mapVariables[(*iter)->GetCanonicalName()] = *iter;

    // Insert objects from the ports
    for(std::vector<daePort*>::const_iterator piter = pModel->Ports().begin(); piter != pModel->Ports().end(); piter++)
        for(std::vector<daeVariable*>::const_iterator citer = (*piter)->Variables().begin(); citer != (*piter)->Variables().end(); citer++)
            mapVariables[(*citer)->GetCanonicalName()] = *citer;

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllVariables(*miter, mapVariables);

    if(pModel->ModelArrays().size() > 0)
    {
        string msg = "Number of model arrays is not zero (collect variables from them too)";
        throw std::runtime_error(msg);
    }
    if(pModel->PortArrays().size() > 0)
    {
        string msg = "Number of ports arrays is not zero (collect variables from them too)";
        throw std::runtime_error(msg);
    }
}

void ProcessSTN(daeSTN* pSTN, std::map<string, daeSTN*>& mapSTNs)
{
    // Add only daeSTN type of STN
    if(pSTN->GetType() == eSTN)
        mapSTNs[pSTN->GetCanonicalName()] = pSTN;

    // Iterate over states and then over nested STNs within each state and recursively process nested STNs
    for(std::vector<daeState*>::const_iterator siter = pSTN->States().begin(); siter != pSTN->States().end(); siter++)
        for(std::vector<daeSTN*>::const_iterator iter = (*siter)->NestedSTNs().begin(); iter != (*siter)->NestedSTNs().end(); iter++)
            ProcessSTN(*iter, mapSTNs);
}

void CollectAllSTNs(daeModel* pModel, std::map<string, daeSTN*>& mapSTNs)
{
    // Recursively process STNs from the model
    for(std::vector<daeSTN*>::const_iterator iter = pModel->STNs().begin(); iter != pModel->STNs().end(); iter++)
        ProcessSTN(*iter, mapSTNs);

    // Insert objects from the child models (units)
    for(std::vector<daeModel*>::const_iterator miter = pModel->Models().begin(); miter != pModel->Models().end(); miter++)
        CollectAllSTNs(*miter, mapSTNs);

    if(pModel->ModelArrays().size() > 0)
    {
        string msg = "Number of model arrays is not zero (collect STNs from them too)";
        throw std::runtime_error(msg);
    }
    if(pModel->PortArrays().size() > 0)
    {
        string msg = "Number of ports arrays is not zero (collect STNs from them too)";
        throw std::runtime_error(msg);
    }
}

// Iterates over property_tree that represents a (multi-dimensional) array of floats (may contain null values, though):
void ProcessListOfValues(boost::property_tree::ptree& pt,
                         std::vector<quantity>& values,
                         unit& Units,
                         std::vector<size_t>& Shape,
                         int currentDimension,
                         bool allowNULL,
                         bool bPrintInfo)
{
    // Check the number of items in the current item by comparing to the current dimension in the Shape array
    if(pt.size() != Shape[currentDimension])
    {
        string msg = "Invalid number of values (" + toString(pt.size()) + ") in dimension " + toString(currentDimension) +
                     " (required is " + toString(Shape[currentDimension]) + ")";
        throw std::runtime_error(msg);
    }

    real_t Value;
    BOOST_FOREACH(boost::property_tree::ptree::value_type& pt_child, pt)
    {
        if(pt_child.second.size() == 0)
        {
        // If the number of children is zero we bumped into a leaf; therefore, process its value

            // If the item raw data is 'null' consider it as unset by design; thus add an item with the value set to cnUnsetValue (DOUBLE_MAX)
            if(boost::lexical_cast<string>(pt_child.second.data()) == "null")
            {
                // If null value is not allowed throw an exception, otherwise set it to cnUnsetValue (DOUBLE_MAX)
                if(!allowNULL)
                {
                    string msg = "Invalid value found (null)";
                    throw std::runtime_error(msg);
                }

                Value = cnUnsetValue;
                if(bPrintInfo)
                    std::cout << "          null data found" << std::endl;
            }
            else
            {
                Value = boost::lexical_cast<real_t>(pt_child.second.data());
            }

            values.push_back(quantity(Value, Units));
        }
        else
        {
         // Item contains children - therefore process them recursively
            ProcessListOfValues(pt_child.second, values, Units, Shape, currentDimension + 1, allowNULL, bPrintInfo);
        }
    }
}

void daeSimulation::SetUpParametersAndDomains_RuntimeSettings()
{
    std::map<string, daeDomain*> mapDomains;
    std::map<string, daeParameter*> mapParameters;
    std::map<string, daeDomain*>::iterator domain_iter;
    std::map<string, daeParameter*>::iterator param_iter;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    daeConfig& cfg = daeConfig::GetConfig();
    bool bPrintInfo = cfg.GetBoolean("daetools.core.printInfo", false);

    if(bPrintInfo)
        std::cout << "      SetUpParametersAndDomains for RuntimeSettings" << std::endl;

    CollectAllDomains(m_pModel, mapDomains);
    CollectAllParameters(m_pModel, mapParameters);

    // v.first is the name of the child.
    // v.second is the child tree
    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_domain, m_ptreeRuntimeSettings.get_child("Domains"))
    {
        // Get the domain name
        string strDomainName = v_domain.first;
        if(bPrintInfo)
            std::cout << "      Processing domain " << strDomainName << " ..." << std::endl;

        // Find the domain in the map of all domains in the simulation
        domain_iter = mapDomains.find(strDomainName);
        if(domain_iter == mapDomains.end())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot find domain " << strDomainName << " while setting the runtime settings";
            throw e;
        }
        daeDomain* pDomain = domain_iter->second;

        // Get the domain type
        string strType;
        try
        {
            strType = v_domain.second.get<string>("Type");
        }
        catch(std::exception& ex)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot find domain Type in domain " << strDomainName << " in the runtime settings";
            throw e;
        }

        if(strType == "eArray")
        {
            try
            {
                size_t NumberOfPoints = v_domain.second.get<size_t>("NumberOfPoints");
                if(bPrintInfo)
                {
                    std::cout << "          Type           = " << strType << std::endl;
                    std::cout << "          NumberOfPoints = " << NumberOfPoints << std::endl;
                }

                pDomain->CreateArray(NumberOfPoints);
            }
            catch(std::exception& ex)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot find NumberOfPoints in domain " << strDomainName << " in the runtime settings";
                throw e;
            }
        }
        else if(strType == "eStructuredGrid")
        {
            try
            {
                //string method               = v_domain.second.get<string>("DiscretizationMethod");
                //size_t DiscretizationOrder  = v_domain.second.get<size_t>("DiscretizationOrder");
                size_t NumberOfIntervals    = v_domain.second.get<size_t>("NumberOfIntervals");
                real_t LowerBound           = v_domain.second.get<real_t>("LowerBound");
                real_t UpperBound           = v_domain.second.get<real_t>("UpperBound");

                //daeeDiscretizationMethod DiscretizationMethod;
                //if(method == "eCFDM")
                //    DiscretizationMethod = eCFDM;
                //else if(method == "eFFDM")
                //    DiscretizationMethod = eFFDM;
                //else if(method == "eBFDM")
                //    DiscretizationMethod = eBFDM;
                //else if(method == "eUpwindCCFV")
                //    DiscretizationMethod = eUpwindCCFV;
                //else
                //    DiscretizationMethod = eDMUnknown;

                if(bPrintInfo)
                {
                    std::cout << "          Type                  = " << strType << std::endl;
                    //std::cout << "          DiscretizationMethod  = " << method << std::endl;
                    //std::cout << "          DiscretizationOrder   = " << DiscretizationOrder << std::endl;
                    std::cout << "          NumberOfIntervals     = " << NumberOfIntervals << std::endl;
                    std::cout << "          LowerBound            = " << LowerBound << std::endl;
                    std::cout << "          UpperBound            = " << UpperBound << std::endl;
                }

                pDomain->CreateStructuredGrid(/*DiscretizationMethod, DiscretizationOrder,*/ NumberOfIntervals, LowerBound, UpperBound);
            }
            catch(std::exception& ex)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot process structured grid domain " << strDomainName << " in the runtime settings: " << ex.what();
                throw e;
            }
        }
        else if(strType == "eUnstructuredGrid")
        {
            try
            {
                size_t NumberOfPoints = v_domain.second.get<size_t>("NumberOfPoints");
                if(bPrintInfo)
                {
                    std::cout << "          Type           = " << strType << std::endl;
                    std::cout << "          NumberOfPoints = " << NumberOfPoints << std::endl;
                }
                std::vector<daePoint> arrPoints(NumberOfPoints);
                pDomain->CreateUnstructuredGrid(arrPoints);
            }
            catch(std::exception& ex)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot find NumberOfPoints in domain " << strDomainName << " in the runtime settings";
                throw e;
            }
        }
        else
        {
            daeDeclareException(exInvalidCall);
            e << "Invalid domain type " << strType << " in domain " << strDomainName << " in the runtime settings";
            throw e;
        }
    }

    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_parameter, m_ptreeRuntimeSettings.get_child("Parameters"))
    {
        // Get the parameter name
        string strParameterName = v_parameter.first;
        if(bPrintInfo)
            std::cout << "      Processing parameter " << strParameterName << " ..." << std::endl;

        // Find the domain in the map of all domains in the simulation
        param_iter = mapParameters.find(strParameterName);
        if(param_iter == mapParameters.end())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot find parameter " << strParameterName << " while setting the runtime settings";
            throw e;
        }
        daeParameter* pParameter = param_iter->second;

        try
        {
            std::map<std::string, double> mapUnits;
            // Get units and their exponents
            BOOST_FOREACH(boost::property_tree::ptree::value_type& v_units, v_parameter.second.get_child("Units"))
            {
                string u    = v_units.first;
                real_t exp  = boost::lexical_cast<real_t>(v_units.second.data());
                mapUnits[u] = exp;
            }

            // Create unit object from the map {"unit" : exponent}
            unit Units = unit(mapUnits);

            // Look for the Shape; if missing it has only a single value
            if(v_parameter.second.find("Shape") == v_parameter.second.not_found())
            {
            // Single value
                real_t Value = v_parameter.second.get<real_t>("Value");

                if(bPrintInfo)
                    std::cout << "          Value = " << quantity(Value, Units) << std::endl;

                // Use SetValues because the intention could have been to use a single value for all points
                pParameter->SetValues(quantity(Value, Units));
            }
            else
            {
            // List of values
                std::vector<size_t> Shape;

                // Get the shape of the array
                BOOST_FOREACH(boost::property_tree::ptree::value_type& v_shape, v_parameter.second.get_child("Shape"))
                {
                    Shape.push_back(boost::lexical_cast<size_t>(v_shape.second.data()));
                }

                // Check if the shape matches the shape in the parameter
                if(Shape.size() != pParameter->GetNumberOfDomains())
                {
                    string msg = "Invalid number of dimensions (" + toString(Shape.size()) + ") of the array of values (required is " +
                                 toString(pParameter->GetNumberOfDomains()) + ")";
                    throw std::runtime_error(msg);
                }
                for(size_t k = 0; k < Shape.size(); k++)
                {
                    size_t dim_avail = Shape[k];
                    size_t dim_req   = pParameter->GetDomain(k)->GetNumberOfPoints();
                    if(dim_req != dim_avail)
                    {
                        string msg = "Dimension " + toString(k) + " of the array of values has " + toString(dim_avail) + " points (required is " + toString(dim_req) + ")";
                        throw std::runtime_error(msg);
                    }
                }

                // Parse the array into a flat (1-D) array of quantity objects
                std::vector<quantity> values;
                ProcessListOfValues(v_parameter.second.get_child("Value"), values, Units, Shape, 0, false, bPrintInfo);

                if(bPrintInfo)
                {
                    std::cout << "          Shape  = (" << toString(Shape) << ")" << std::endl;
                    std::cout << "          Values = [" << toString(values) << "]" << std::endl;
                }

                // Finally set the values
                pParameter->SetValues(values);
            }
        }
        catch(std::exception& ex)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot process parameter " << strParameterName << " in the runtime settings: " << ex.what();
            throw e;
        }
    }
}

void daeSimulation::SetUpVariables_RuntimeSettings()
{
    std::map<string, daeParameter*> mapParameters;
    std::map<string, daeVariable*> mapVariables;
    std::map<string, daeSTN*> mapSTNs;
    std::map<string, daeParameter*>::iterator param_iter;
    std::map<string, daeVariable*>::iterator var_iter;
    std::map<string, daeSTN*>::iterator stn_iter;

    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    daeConfig& cfg = daeConfig::GetConfig();
    bool bPrintInfo = cfg.GetBoolean("daetools.core.printInfo", false);

    if(bPrintInfo)
        std::cout << "      SetUpVariables for RuntimeSettings" << std::endl;

    CollectAllParameters(m_pModel, mapParameters);
    CollectAllVariables(m_pModel, mapVariables);
    CollectAllSTNs(m_pModel, mapSTNs);

    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_stn, m_ptreeRuntimeSettings.get_child("STNs"))
    {
        // Get the STN name
        string strSTNName = v_stn.first;
        if(bPrintInfo)
            std::cout << "      Processing STN " << strSTNName << " ..." << std::endl;

        // Try to find it among available STNS
        stn_iter = mapSTNs.find(strSTNName);
        if(stn_iter == mapSTNs.end())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot find STN " << strSTNName << " while setting the runtime settings";
            throw e;
        }
        daeSTN* pSTN = stn_iter->second;

        // Get the ActiveState
        string strActiveState = v_stn.second.get<string>("ActiveState");

        // Set the active state
        pSTN->SetActiveState(strActiveState);

        if(bPrintInfo)
            std::cout << "          ActiveState = " << strActiveState << std::endl;
    }

    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_output, m_ptreeRuntimeSettings.get_child("Outputs"))
    {
        // Get the Output name
        string strOutputName = v_output.first;
        if(bPrintInfo)
            std::cout << "      Processing Output " << strOutputName << " ..." << std::endl;

        // Get the flag
        bool on = false;
        if(boost::lexical_cast<string>(v_output.second.data()) == "true" ||
           boost::lexical_cast<string>(v_output.second.data()) == "True")
        {
            on = true;
        }

        // Try to find it among variables and set the ReportingOn flag
        var_iter = mapVariables.find(strOutputName);
        if(var_iter != mapVariables.end())
        {
            daeVariable* pVariable = var_iter->second;
            pVariable->SetReportingOn(on);
            if(bPrintInfo)
                std::cout << "          ReportingOn = " << (on ? "true" : "false") << std::endl;
            continue;
        }

        // Try to find it among parameters and set the ReportingOn flag
        param_iter = mapParameters.find(strOutputName);
        if(param_iter != mapParameters.end())
        {
            daeParameter* pParameter = param_iter->second;
            pParameter->SetReportingOn(on);
            if(bPrintInfo)
                std::cout << "          ReportingOn = " << (on ? "true" : "false") << std::endl;
            continue;
        }

        // Raise an exception if not found
        daeDeclareException(exInvalidCall);
        e << "Cannot find variable " << strOutputName << " marked to be an output while setting the runtime settings";
        throw e;
    }

    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_variable, m_ptreeRuntimeSettings.get_child("DOFs"))
    {
        // Get the DOF name
        string strVariableName = v_variable.first;
        if(bPrintInfo)
            std::cout << "      Processing DOF " << strVariableName << " ..." << std::endl;

        // Find the domain in the map of all domains in the simulation
        var_iter = mapVariables.find(strVariableName);
        if(var_iter == mapVariables.end())
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot find variable " << strVariableName << " while setting the runtime settings";
            throw e;
        }
        daeVariable* pVariable = var_iter->second;

        try
        {
            std::map<std::string, double> mapUnits;
            // Get units and their exponents
            BOOST_FOREACH(boost::property_tree::ptree::value_type& v_units, v_variable.second.get_child("Units"))
            {
                string u    = v_units.first;
                real_t exp  = boost::lexical_cast<real_t>(v_units.second.data());
                mapUnits[u] = exp;
            }

            // Create unit object from the map {"unit" : exponent}
            unit Units = unit(mapUnits);

            // Look for the Shape; if missing it has only a single value
            if(v_variable.second.find("Shape") == v_variable.second.not_found())
            {
            // Single value
                real_t Value = v_variable.second.get<real_t>("Value");

                if(bPrintInfo)
                    std::cout << "          Value = " << quantity(Value, Units) << std::endl;

                // Use AssignValues because the intention could have been to use a single value for all points
                pVariable->AssignValues(quantity(Value, Units));
            }
            else
            {
                // List of values
                    std::vector<size_t> Shape;

                    // Get the shape of the array
                    BOOST_FOREACH(boost::property_tree::ptree::value_type& v_shape, v_variable.second.get_child("Shape"))
                    {
                        Shape.push_back(boost::lexical_cast<size_t>(v_shape.second.data()));
                    }

                    // Check if the shape matches the shape in the parameter
                    if(Shape.size() != pVariable->GetNumberOfDomains())
                    {
                        string msg = "Invalid number of dimensions (" + toString(Shape.size()) + ") of the array of values (required is " +
                                     toString(pVariable->GetNumberOfDomains()) + ")";
                        throw std::runtime_error(msg);
                    }
                    for(size_t k = 0; k < Shape.size(); k++)
                    {
                        size_t dim_avail = Shape[k];
                        size_t dim_req   = pVariable->GetDomain(k)->GetNumberOfPoints();
                        if(dim_req != dim_avail)
                        {
                            string msg = "Dimension " + toString(k) + " of the array of values has " + toString(dim_avail) + " points (required is " + toString(dim_req) + ")";
                            throw std::runtime_error(msg);
                        }
                    }

                    // Parse the array into a flat (1-D) array of quantity objects
                    std::vector<quantity> values;
                    ProcessListOfValues(v_variable.second.get_child("Value"), values, Units, Shape, 0, true, bPrintInfo);

                    if(bPrintInfo)
                    {
                        std::cout << "          Shape  = (" << toString(Shape) << ")" << std::endl;
                        std::cout << "          Values = [" << toString(values) << "]" << std::endl;
                    }

                    // Finally assign the values
                    pVariable->AssignValues(values);
            }
        }
        catch(std::exception& ex)
        {
            daeDeclareException(exInvalidCall);
            e << "Cannot process degrees of freedom for variable " << strVariableName << " in the runtime settings: " << ex.what();
            throw e;
        }
    }

    bool QuasiSteadyState = m_ptreeRuntimeSettings.get<bool>("QuasiSteadyState");
    if(QuasiSteadyState)
    {
        SetInitialConditionMode(eQuasiSteadyState);
    }
    else
    {
        SetInitialConditionMode(eAlgebraicValuesProvided);

        BOOST_FOREACH(boost::property_tree::ptree::value_type& v_variable, m_ptreeRuntimeSettings.get_child("InitialConditions"))
        {
            // Get the variable name
            string strVariableName = v_variable.first;
            if(bPrintInfo)
                std::cout << "      Processing InitialConditions for " << strVariableName << " ..." << std::endl;

            // Find the domain in the map of all domains in the simulation
            var_iter = mapVariables.find(strVariableName);
            if(var_iter == mapVariables.end())
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot find variable " << strVariableName << " while setting the runtime settings";
                throw e;
            }
            daeVariable* pVariable = var_iter->second;

            try
            {
                std::map<std::string, double> mapUnits;
                // Get units and their exponents
                BOOST_FOREACH(boost::property_tree::ptree::value_type& v_units, v_variable.second.get_child("Units"))
                {
                    string u    = v_units.first;
                    real_t exp  = boost::lexical_cast<real_t>(v_units.second.data());
                    mapUnits[u] = exp;
                }

                // Create unit object from the map {"unit" : exponent}
                unit Units = unit(mapUnits);

                // Look for the Shape; if missing it has only a single value
                if(v_variable.second.find("Shape") == v_variable.second.not_found())
                {
                // Single value
                    real_t Value = v_variable.second.get<real_t>("Value");

                    if(bPrintInfo)
                        std::cout << "          Value = " << quantity(Value, Units) << std::endl;

                    // Use SetInitialConditions because the intention could have been to use a single value for all points
                    pVariable->SetInitialConditions(quantity(Value, Units));
                }
                else
                {
                    // List of values
                        std::vector<size_t> Shape;

                        // Get the shape of the array
                        BOOST_FOREACH(boost::property_tree::ptree::value_type& v_shape, v_variable.second.get_child("Shape"))
                        {
                            Shape.push_back(boost::lexical_cast<size_t>(v_shape.second.data()));
                        }

                        // Check if the shape matches the shape in the parameter
                        if(Shape.size() != pVariable->GetNumberOfDomains())
                        {
                            string msg = "Invalid number of dimensions (" + toString(Shape.size()) + ") of the array of values (required is " +
                                         toString(pVariable->GetNumberOfDomains()) + ")";
                            throw std::runtime_error(msg);
                        }
                        for(size_t k = 0; k < Shape.size(); k++)
                        {
                            size_t dim_avail = Shape[k];
                            size_t dim_req   = pVariable->GetDomain(k)->GetNumberOfPoints();
                            if(dim_req != dim_avail)
                            {
                                string msg = "Dimension " + toString(k) + " of the array of values has " + toString(dim_avail) + " points (required is " + toString(dim_req) + ")";
                                throw std::runtime_error(msg);
                            }
                        }

                        // Parse the array into a flat (1-D) array of quantity objects
                        std::vector<quantity> values;
                        ProcessListOfValues(v_variable.second.get_child("Value"), values, Units, Shape, 0, true, bPrintInfo);

                        if(bPrintInfo)
                        {
                            std::cout << "          Shape  = (" << toString(Shape) << ")" << std::endl;
                            std::cout << "          Values = [" << toString(values) << "]" << std::endl;
                        }

                        // Finally assign the values
                        pVariable->SetInitialConditions(values);
                }
            }
            catch(std::exception& ex)
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot process initial conditions for variable " << strVariableName << " in the runtime settings: " << ex.what();
                throw e;
            }
        }
    }

    SetTimeHorizon(m_ptreeRuntimeSettings.get<real_t>("TimeHorizon"));
    SetReportingInterval(m_ptreeRuntimeSettings.get<real_t>("ReportingInterval"));
    if(bPrintInfo)
    {
        std::cout << "      TimeHorizon       = " << m_dTimeHorizon << std::endl;
        std::cout << "      ReportingInterval = " << m_dReportingInterval  << std::endl;
    }

    if(m_pDAESolver)
    {
        real_t RelativeTolerance = m_ptreeRuntimeSettings.get<real_t>("RelativeTolerance");
        m_pDAESolver->SetRelativeTolerance(RelativeTolerance);
        if(bPrintInfo)
            std::cout << "      RelativeTolerance = " << RelativeTolerance  << std::endl;
    }
}

bool daeSimulation::GetIsInitialized(void) const
{
    return m_bIsInitialized;
}

bool daeSimulation::GetIsSolveInitial(void) const
{
    return m_bIsSolveInitial;
}

bool daeSimulation::GetReportTimeDerivatives(void) const
{
    return m_bReportTimeDerivatives;
}

void daeSimulation::SetReportTimeDerivatives(bool bReportTimeDerivatives)
{
    m_bReportTimeDerivatives = bReportTimeDerivatives;
}

bool daeSimulation::GetReportSensitivities(void) const
{
    return m_bReportSensitivities;
}

void daeSimulation::SetReportSensitivities(bool bReportSensitivities)
{
    m_bReportSensitivities = bReportSensitivities;
}

void daeSimulation::SetJSONRuntimeSettings(const std::string& strJSONRuntimeSettings)
{
    try
    {
        std::stringstream ss(strJSONRuntimeSettings);
        boost::property_tree::json_parser::read_json(ss, m_ptreeRuntimeSettings);

        // Only at the end of parsing set the m_strJSONRuntimeSettings
        m_strJSONRuntimeSettings = strJSONRuntimeSettings;
    }
    catch(std::exception& ex)
    {
        daeDeclareException(exRuntimeCheck);
        e << "Cannot set JSON Runtime settings: " << ex.what();
        throw e;
    }
}

std::string daeSimulation::GetJSONRuntimeSettings() const
{
    return m_strJSONRuntimeSettings;
}

template<class T>
class daeCollectObject : public std::unary_function<T, void>
{
public:
    daeCollectObject(daeSimulation& rSimulation)
        : m_Simulation(rSimulation)
    {
    }

    void operator() (T obj)
    {
        m_Simulation.CollectVariables(obj);
    }

    daeSimulation& m_Simulation;
};

template<class T>
class daeRegisterObject : public std::unary_function<T, void>
{
public:
    daeRegisterObject(daeSimulation& rSimulation)
        : m_Simulation(rSimulation)
    {
    }

    void operator() (T obj)
    {
        m_Simulation.Register(obj);
    }

    daeSimulation& m_Simulation;
};

template<class T>
class daeReportObject : public std::unary_function<T, void>
{
public:
    daeReportObject(daeSimulation& rSimulation, real_t time)
        : m_time(time), m_Simulation(rSimulation)
    {
    }

    void operator() (T obj)
    {
        m_Simulation.Report(obj, m_time);
    }

    real_t         m_time;
    daeSimulation& m_Simulation;
};

void daeSimulation::RegisterData(const std::string& strIteration)
{
    if(strIteration.empty())
        m_strIteration = strIteration;
    else
        m_strIteration = strIteration + ".";

    m_pDataReporter->StartRegistration();
    Register(m_pModel);
    m_pDataReporter->EndRegistration();
}

void daeSimulation::CollectVariables(daeModel* pModel)
{
    if(!pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    daeCollectObject<daeParameter*> regParameter(*this);
    daeCollectObject<daeVariable*>  regVariable(*this);
    daeCollectObject<daePort*>      regPort(*this);
    daeCollectObject<daeModel*>     regModel(*this);

    std::for_each(pModel->Parameters().begin(), pModel->Parameters().end(), regParameter);
    std::for_each(pModel->Variables().begin(),  pModel->Variables().end(),  regVariable);
    std::for_each(pModel->Ports().begin(),      pModel->Ports().end(),      regPort);
    std::for_each(pModel->Models().begin(),     pModel->Models().end(),     regModel);

    if(pModel->ModelArrays().size() > 0)
    {
        string msg = "Number of model arrays is not zero (collect parameters/variables from them too)";
        throw std::runtime_error(msg);
    }
    if(pModel->PortArrays().size() > 0)
    {
        string msg = "Number of ports arrays is not zero (collect parametersvariables from them too)";
        throw std::runtime_error(msg);
    }
}

void daeSimulation::CollectVariables(daePort* pPort)
{
    if(!pPort)
        daeDeclareAndThrowException(exInvalidPointer);

    daeCollectObject<daeParameter*> regParameter(*this);
    daeCollectObject<daeVariable*>  regVariable(*this);

    std::for_each(pPort->Parameters().begin(), pPort->Parameters().end(), regParameter);
    std::for_each(pPort->Variables().begin(),  pPort->Variables().end(),  regVariable);
}

void daeSimulation::CollectVariables(daeParameter* pParameter)
{
    if(pParameter->GetReportingOn())
        dae_push_back(m_ptrarrReportParameters, pParameter);
}

void daeSimulation::CollectVariables(daeVariable* pVariable)
{
    if(pVariable->GetReportingOn())
        dae_push_back(m_ptrarrReportVariables, pVariable);
}

void daeSimulation::Register(daeModel* pModel)
{
    if(!pModel)
        daeDeclareAndThrowException(exInvalidPointer);

    daeRegisterObject<daeDomain*>	 regDomain(*this);
    daeRegisterObject<daeParameter*> regParameter(*this);
    daeRegisterObject<daeVariable*>  regVariable(*this);
    daeRegisterObject<daePort*>      regPort(*this);
    daeRegisterObject<daeModel*>     regModel(*this);

    std::for_each(pModel->Domains().begin(),    pModel->Domains().end(),    regDomain);
    std::for_each(pModel->Parameters().begin(), pModel->Parameters().end(), regParameter);
    std::for_each(pModel->Variables().begin(),  pModel->Variables().end(),  regVariable);
    std::for_each(pModel->Ports().begin(),      pModel->Ports().end(),      regPort);
    std::for_each(pModel->Models().begin(),     pModel->Models().end(),     regModel);
}

void daeSimulation::Register(daePort* pPort)
{
    if(!pPort)
        daeDeclareAndThrowException(exInvalidPointer);

    daeRegisterObject<daeDomain*>	 regDomain(*this);
    daeRegisterObject<daeParameter*> regParameter(*this);
    daeRegisterObject<daeVariable*>  regVariable(*this);

    std::for_each(pPort->Domains().begin(),    pPort->Domains().end(),    regDomain);
    std::for_each(pPort->Parameters().begin(), pPort->Parameters().end(), regParameter);
    std::for_each(pPort->Variables().begin(),  pPort->Variables().end(),  regVariable);
}

void daeSimulation::Register(daeVariable* pVariable)
{
    if(!pVariable)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pVariable->GetReportingOn())
        return;

    size_t i;
    daeDomain_t* pDomain;
    vector<daeDomain_t*> arrDomains;

    boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();

    daeDataReporterVariable var;
    var.m_strName  = m_strIteration + pVariable->GetCanonicalName();
    var.m_strUnits = pVariable->GetVariableType()->GetUnits().toString();

    var.m_nNumberOfPoints = pVariable->GetNumberOfPoints();
    pVariable->GetDomains(arrDomains);
    for(i = 0; i < arrDomains.size(); i++)
    {
        pDomain = arrDomains[i];
        var.m_strarrDomains.push_back(m_strIteration + pDomain->GetCanonicalName());
    }

    if(!m_pDataReporter->RegisterVariable(&var))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to register variable [" << var.m_strName << "]";
        throw e;
    }

    // Register a variable for a time derivative
    // Reuse the daeDataReporterVariable!!
    if(m_bReportTimeDerivatives)
    {
        var.m_strName = m_strIteration + pVariable->GetCanonicalNameAndPrepend("time_derivatives.d(");
        var.m_strName += ")_dt";

        if(!m_pDataReporter->RegisterVariable(&var))
        {
            daeDeclareException(exDataReportingError);
            e << "Simulation dastardly failed to register the time derivative for the variable [" << var.m_strName << "]";
            throw e;
        }
    }

    // Register a variable for a sensitivity
    // Reuse the daeDataReporterVariable!!
    if(m_bReportSensitivities && m_bCalculateSensitivities)
    {
        const std::vector<size_t>& bis = pVariable->GetBlockIndexes();

        bool isAssigned = false;
        if(bis.size() == 1)
        {
            if(pDataProxy->GetVariableType(pVariable->GetOverallIndex()) == cnAssigned || bis[0] == ULONG_MAX)
                isAssigned = true;
        }

        if(!isAssigned)
        {
            for(size_t pi = 0; pi < m_arrOptimizationVariables.size(); pi++)
            {
                boost::shared_ptr<daeOptimizationVariable> optVar = m_arrOptimizationVariables[pi];

                var.m_strName = m_strIteration + pVariable->GetCanonicalNameAndPrepend("sensitivities.d(");
                var.m_strName += ")_d(" + optVar->GetName() + ")";

                if(!m_pDataReporter->RegisterVariable(&var))
                {
                    daeDeclareException(exDataReportingError);
                    e << "Simulation dastardly failed to register the sensitivity for the variable [" << var.m_strName << "]"
                      << " per parameter [" << optVar->GetName() << "]";
                    throw e;
                }
            }
        }
    }
}

void daeSimulation::Register(daeParameter* pParameter)
{
    if(!pParameter)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pParameter->GetReportingOn())
        return;

    size_t i;
    daeDomain_t* pDomain;
    vector<daeDomain_t*> arrDomains;

    daeDataReporterVariable var;
    var.m_strName         = m_strIteration + pParameter->GetCanonicalName();
    var.m_strUnits        = pParameter->GetUnits().toString();
    var.m_nNumberOfPoints = pParameter->GetNumberOfPoints();
    pParameter->GetDomains(arrDomains);
    for(i = 0; i < arrDomains.size(); i++)
    {
        pDomain = arrDomains[i];
        var.m_strarrDomains.push_back(m_strIteration + pDomain->GetCanonicalName());
    }

    if(!m_pDataReporter->RegisterVariable(&var))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to register parameter [" << var.m_strName << "]";
        throw e;
    }
}

void daeSimulation::Register(daeDomain* pDomain)
{
    if(!pDomain)
        daeDeclareAndThrowException(exInvalidPointer);

    daeDataReporterDomain domain;
    domain.m_strName            = m_strIteration + pDomain->GetCanonicalName();
    domain.m_strUnits           = pDomain->GetUnits().toString();
    domain.m_eType				= pDomain->GetType();
    domain.m_nNumberOfPoints	= pDomain->GetNumberOfPoints();
    if(pDomain->GetNumberOfPoints() == 0)
        daeDeclareAndThrowException(exInvalidCall);

    if(pDomain->GetType() == eUnstructuredGrid)
        domain.m_arrCoordinates = pDomain->GetCoordinates();
    else
        pDomain->GetPoints(domain.m_arrPoints);

    if(!m_pDataReporter->RegisterDomain(&domain))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to register domain [" << domain.m_strName << "]";
        throw e;
    }
}

void daeSimulation::ReportData(real_t dCurrentTime)
{
    if(!m_pDataReporter)
        daeDeclareAndThrowException(exInvalidPointer);

    // First, do post-processing to calculate some variables that depend on the current solution.
    // DoPostProcessing function is overloaded in the derived-daeSimulation classes.
    // This function can be also called in daesolver.Solve() function but it is better to do it here.
    DoPostProcessing();

    if(!m_pDataReporter->StartNewResultSet(dCurrentTime))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to start new result set at TIME: [" << dCurrentTime << "]";
        throw e;
    }

    // Get the current sensitivity matrix and save it if the output directory is set.
    bool bSaveSensitivityMatrix = !m_strSensitivityDataDirectory.empty();
    m_pCurrentSensitivityMatrix = NULL;
    if(m_bCalculateSensitivities && (bSaveSensitivityMatrix || m_bReportSensitivities))
    {
    // Call m_pDAESolver->GetSensitivities() only once since it is very expensive!
        daeMatrix<real_t>& sm = m_pDAESolver->GetSensitivities();
        daeDenseMatrix& dsm = dynamic_cast<daeDenseMatrix&>(sm);

        // Set the current sensitivity matrix for use in reporting sensitivities.
        m_pCurrentSensitivityMatrix = &dsm;
    }

    if(m_bCalculateSensitivities && bSaveSensitivityMatrix)
    {
        std::string filename = (boost::format("%06d-%.12f.mmx") % m_iNoSensitivityFiles % dCurrentTime).str();
        boost::filesystem::path sensitivityDataDirectory = m_strSensitivityDataDirectory;
        if(m_iNoSensitivityFiles == 0) // test the validity of the directory only at the beginning of the simulation
        {
            if(!boost::filesystem::is_directory(sensitivityDataDirectory))
            {
                daeDeclareException(exDataReportingError);
                e << "An invalid sensitivity data directory specified: " << m_strSensitivityDataDirectory;
                throw e;
            }
        }
        boost::filesystem::path mmxFile = sensitivityDataDirectory / filename;
        if(m_pCurrentSensitivityMatrix)
            m_pCurrentSensitivityMatrix->SaveAsMatrixMarketFile(mmxFile.string(), "Sensitivity Matrix", "Sensitivity[Nparams,Nvariables]");
        m_iNoSensitivityFiles++;
    }

    daeReportObject<daeVariable*>  repVariables(*this, dCurrentTime);
    daeReportObject<daeParameter*> repParameters(*this, dCurrentTime);

    std::for_each(m_ptrarrReportVariables.begin(), m_ptrarrReportVariables.end(), repVariables);
    if(dCurrentTime == 0)
        std::for_each(m_ptrarrReportParameters.begin(), m_ptrarrReportParameters.end(), repParameters);

    // Reset the current sensitivity matrix to NULL again.
    m_pCurrentSensitivityMatrix = NULL;

    // OLD:
    //Report(m_pModel, dCurrentTime);
}

//void daeSimulation::Report(daeModel* pModel, real_t time)
//{
//	if(!pModel)
//		daeDeclareAndThrowException(exInvalidPointer);

//	daeReportObject<daeParameter*>  repParameter(*this, time);
//	daeReportObject<daeVariable*>   repVariable(*this, time);
//	daeReportObject<daePort*>       repPort(*this, time);
//	daeReportObject<daeModel*>      repModel(*this, time);

//	if(time == 0)
//		std::for_each(pModel->Parameters().begin(), pModel->Parameters().end(), repParameter);
//	std::for_each(pModel->Variables().begin(),  pModel->Variables().end(),  repVariable);
//	std::for_each(pModel->Ports().begin(),      pModel->Ports().end(),      repPort);
//	std::for_each(pModel->Models().begin(),     pModel->Models().end(),     repModel);
//}

//void daeSimulation::Report(daePort* pPort, real_t time)
//{
//	if(!pPort)
//		daeDeclareAndThrowException(exInvalidPointer);

//	daeReportObject<daeParameter*>  repParameter(*this, time);
//	daeReportObject<daeVariable*>   repVariable(*this, time);

//	if(time == 0)
//		std::for_each(pPort->Parameters().begin(), pPort->Parameters().end(), repParameter);
//	std::for_each(pPort->Variables().begin(),  pPort->Variables().end(),  repVariable);
//}

void daeSimulation::Report(daeVariable* pVariable, real_t time)
{
    if(!pVariable)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pVariable->GetReportingOn())
        return;

    daeDataReporterVariableValue var;
    size_t i, k, nEnd, nStart, nPoints, bi;

    var.m_strName = m_strIteration + pVariable->GetCanonicalName();
    nPoints = pVariable->GetNumberOfPoints();
    nStart  = pVariable->GetOverallIndex();
    nEnd    = pVariable->GetOverallIndex() + nPoints;
    var.m_nNumberOfPoints = nPoints;
    var.m_pValues = new real_t[nPoints];
    boost::shared_ptr<daeDataProxy_t> pDataProxy = m_pModel->GetDataProxy();
    for(k = 0, i = nStart; i < nEnd; i++, k++)
        var.m_pValues[k] = pDataProxy->GetValue(i);

    if(!m_pDataReporter->SendVariable(&var))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to report variable [" << var.m_strName << "] at TIME: [" << time << "]";
        throw e;
    }

    // Report a time dervative for a variable
    // Reuse the daeDataReporterVariableValue!!
    if(m_bReportTimeDerivatives)
    {
        var.m_strName = m_strIteration + pVariable->GetCanonicalNameAndPrepend("time_derivatives.d(");
        var.m_strName += ")_dt";

        for(k = 0, i = nStart; i < nEnd; i++, k++)
            var.m_pValues[k] = pDataProxy->GetTimeDerivative_or_zero(i);

        if(!m_pDataReporter->SendVariable(&var))
        {
            daeDeclareException(exDataReportingError);
            e << "Simulation dastardly failed to report the time derivative for the variable [" << var.m_strName << "]";
            throw e;
        }
    }

    // Report sensitivities of a variable per every sensitivity parameter
    // Reuse the daeDataReporterVariableValue!!
    if(m_bReportSensitivities && m_bCalculateSensitivities)
    {
        const std::vector<size_t>& bis = pVariable->GetBlockIndexes();

        bool isAssigned = false;
        if(bis.size() == 1)
        {
            if(pDataProxy->GetVariableType(pVariable->GetOverallIndex()) == cnAssigned || bis[0] == ULONG_MAX)
                isAssigned = true;
        }

        if(!isAssigned)
        {
            for(size_t pi = 0; pi < m_arrOptimizationVariables.size(); pi++)
            {
                boost::shared_ptr<daeOptimizationVariable> optVar = m_arrOptimizationVariables[pi];

                var.m_strName = m_strIteration + pVariable->GetCanonicalNameAndPrepend("sensitivities.d(");
                var.m_strName += ")_d(" + optVar->GetName() + ")";

                for(k = 0; k < bis.size(); k++)
                {
                    bi = bis[k];
                    if(bi != ULONG_MAX)
                        var.m_pValues[k] = m_pCurrentSensitivityMatrix->GetItem(pi, bi);
                    else
                        var.m_pValues[k] = 0;
                }

                if(!m_pDataReporter->SendVariable(&var))
                {
                    daeDeclareException(exDataReportingError);
                    e << "Simulation dastardly failed to report the sensitivity for the variable [" << var.m_strName << "]"
                      << " per parameter [" << optVar->GetName() << "]";
                    throw e;
                }
            }
        }
    }
}

void daeSimulation::Report(daeParameter* pParameter, real_t time)
{
    if(!pParameter)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!pParameter->GetReportingOn())
        return;

    real_t* pd;
    daeDataReporterVariableValue var;
    size_t i, nSize;

    var.m_strName = m_strIteration + pParameter->GetCanonicalName();
    pd = pParameter->GetValuePointer();
    nSize = pParameter->GetNumberOfPoints();
    var.m_nNumberOfPoints = nSize;
    var.m_pValues = new real_t[nSize];
    for(i = 0; i < nSize; i++)
        var.m_pValues[i] = pd[i];

    if(!m_pDataReporter->SendVariable(&var))
    {
        daeDeclareException(exDataReportingError);
        e << "Simulation dastardly failed to report parameter [" << var.m_strName << "]";
        throw e;
    }
}

daeModel_t* daeSimulation::GetModel(void) const
{
    return m_pModel;
}

void daeSimulation::SetModel(daeModel_t* pModel)
{
    if(!pModel)
        daeDeclareAndThrowException(exInvalidPointer)

    m_pModel = dynamic_cast<daeModel*>(pModel);
    if(!m_pModel)
        daeDeclareAndThrowException(exInvalidPointer)
}

daeDataReporter_t* daeSimulation::GetDataReporter(void) const
{
    return m_pDataReporter;
}

daeLog_t* daeSimulation::GetLog(void) const
{
    return m_pLog;
}

daeDAESolver_t* daeSimulation::GetDAESolver(void) const
{
    return m_pDAESolver;
}



}
}

