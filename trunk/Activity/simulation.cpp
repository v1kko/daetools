#include "stdafx.h"
#include "simulation.h"
//#include "../Core/nodes.h"
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <math.h>

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
	m_nNumberOfObjectiveFunctions	= 1;
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
							   bool bCalculateSensitivities)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer);
	
	if(m_bIsInitialized)
	{
		daeDeclareException(exInvalidCall);
		e << "Simulation has already been initialized";
		throw e;
	}
	
	m_bCalculateSensitivities     = bCalculateSensitivities;
	m_nNumberOfObjectiveFunctions = 0;
	
	if(!pDataReporter->IsConnected())
	{
		daeDeclareException(exInvalidCall);
		e << "Simulation ignobly refused to initialize: the data reporter is not connected";
		throw e;
	}

	m_pDAESolver    = pDAESolver;
	m_pDataReporter	= pDataReporter;
	m_pLog			= pLog;

	m_ProblemCreationStart = dae::GetTimeInSeconds();
	
	daeConfig& cfg = daeConfig::GetConfig();
	bool bPrintInfo = cfg.Get<bool>("daetools.activity.printHeader", true);
	if(bPrintInfo)
	{
		m_pLog->Message(string("***********************************************************************"), 0);
		m_pLog->Message(string("                          @@@@@                                        "), 0);
		m_pLog->Message(string("       @                    @                                          "), 0);
		m_pLog->Message(string("       @   @@@@@     @@@@@  @                DAE Tools                 "), 0);
		m_pLog->Message(string("  @@@@@@        @   @     @       Version:   ") + daeVersion(true),        0);
		m_pLog->Message(string(" @     @   @@@@@@   @@@@@@        Copyright: Dragan Nikolic, 2011      "), 0);
		m_pLog->Message(string(" @     @  @     @   @             E-mail:    dnikolic at daetools.com  "), 0);
		m_pLog->Message(string("  @@@@@    @@@@@@    @@@@@        Homepage:  daetools.sourceforge.net  "), 0);
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
		m_pLog->Message(string(" along with this program. If not, see <http://www.gnu.org/licenses/>.  "), 0);
		m_pLog->Message(string("***********************************************************************"), 0);
	}

	m_pLog->Message(string("Creating the system... "), 0);
	
// Create data proxy and propagate it
	m_pModel->InitializeStage1();

// Initialize params and domains
	SetUpParametersAndDomains();

// Define the optimization problem: objective function and constraints
	if(m_eSimulationMode == eOptimization)
	{
		if(!m_bCalculateSensitivities)
			m_bCalculateSensitivities = true;
		
		SetNumberOfObjectiveFunctions(1);
		
	// Call SetUpOptimization to define obj. function, constraints and opt. variables
		SetUpOptimization();
	}
	else if(m_eSimulationMode == eParameterEstimation)
	{
		if(!m_bCalculateSensitivities)
			m_bCalculateSensitivities = true;
		
		SetNumberOfObjectiveFunctions(0);
		
	// Call SetUpParameterEstimation to define obj. function(s), constraints and opt. variables
		SetUpParameterEstimation();
	}
	else
	{
		if(m_bCalculateSensitivities)
		{
			SetNumberOfObjectiveFunctions(1);
		
		// Call SetUpSensitivityAnalysis to define obj.function(s), constraint(s) and opt.variable(s)
		// and whatever else is needed
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
	}
	
// Create model/port arrays and initialize variable indexes
	m_pModel->InitializeStage2();

// Create data storage for variables, derivatives, var. types, tolerances, etc
	m_pModel->InitializeStage3(m_pLog);

// Set initial values, initial conditions, fix variables, set initial guesses, abs tolerances, etc
	SetUpVariables();
	
// Create equation execution infos in models and stns
	m_pModel->InitializeStage4();

//// Set the solver's InitialConditionMode
//	daeeInitialConditionMode eMode = GetInitialConditionMode();
//	m_pDAESolver->SetInitialConditionMode(eMode);
//	if(eMode == eQuasySteadyState)
//		SetInitialConditionsToZero();

// Now I have everything set up and I should check for inconsistences
	CheckSystem();

// Do the block decomposition if needed (at the moment only one block is created)
	m_ptrarrBlocks.EmptyAndFreeMemory();
	m_pModel->InitializeStage5(false, m_ptrarrBlocks);

// Setup DAE solver and sensitivities
	SetupSolver();
	
// Register model
	m_dCurrentTime = 0;
	m_pDataReporter->StartRegistration();
	Register(m_pModel);
	m_pDataReporter->EndRegistration();
	
// Set the IsInitialized flag to true
	m_bIsInitialized = true;

// Announce success	
	m_ProblemCreationEnd = dae::GetTimeInSeconds();
	m_pLog->Message(string("The system created successfully in: ") + 
					toStringFormatted<real_t>(m_ProblemCreationEnd - m_ProblemCreationStart, -1, 3) + 
					string(" s"), 0);
}

void daeSimulation::SetupSolver(void)
{
	size_t i;
	vector<size_t> narrParametersIndexes;
	daeBlock_t* pBlock;
	boost::shared_ptr<daeOptimizationVariable> pOptVariable;
	boost::shared_ptr<daeOptimizationConstraint> pConstraint;
	boost::shared_ptr<daeObjectiveFunction> pObjectiveFunction;
	boost::shared_ptr<daeMeasuredVariable> pMeasuredVariable;
	vector<string> strarrErrors;

	if(m_ptrarrBlocks.size() != 1)
		daeDeclareAndThrowException(exInvalidCall);
	pBlock = m_ptrarrBlocks[0];
	
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
			pObjectiveFunction->Initialize(m_arrOptimizationVariables, pBlock);
		}
		
	// 4. Initialize the constraints
		for(i = 0; i < m_arrConstraints.size(); i++)
		{  
			pConstraint = m_arrConstraints[i];
			pConstraint->Initialize(m_arrOptimizationVariables, pBlock);
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
			pObjectiveFunction->Initialize(m_arrOptimizationVariables, pBlock);
		}
		
	// 4. Initialize measured variables
		for(i = 0; i < m_arrMeasuredVariables.size(); i++)
		{  
			pMeasuredVariable = m_arrMeasuredVariables[i];
			pMeasuredVariable->Initialize(m_arrOptimizationVariables, pBlock);
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
				pObjectiveFunction->Initialize(m_arrOptimizationVariables, pBlock);
			}
			
		// 4. Initialize the constraints
			for(i = 0; i < m_arrConstraints.size(); i++)
			{  
				pConstraint = m_arrConstraints[i];
				pConstraint->Initialize(m_arrOptimizationVariables, pBlock);
			}
		}
	}
	
	m_pDAESolver->Initialize(pBlock, m_pLog, this, m_pModel->GetInitialConditionMode(), m_bCalculateSensitivities, narrParametersIndexes);
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
	m_IntegrationStart    = dae::GetTimeInSeconds();
	m_IntegrationEnd      = dae::GetTimeInSeconds();

// Ask DAE solver to initialize the system
	m_pDAESolver->SolveInitial();

// Report data at TIME=0
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
	
	if(m_dCurrentTime == 0)
	{
		m_IntegrationStart = dae::GetTimeInSeconds();
	}
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
			m_dCurrentTime = IntegrateUntilTime(t, eStopAtModelDiscontinuity, true);
		}
		
		m_dCurrentTime = t;
		ReportData(m_dCurrentTime);
		real_t newProgress = ceil(100.0 * m_dCurrentTime/m_dTimeHorizon); 
		if(newProgress > m_pLog->GetProgress())
			m_pLog->SetProgress(newProgress);	
	}

// Print the end of the simulation info if not in the optimization mode		
	if(m_eSimulationMode == eSimulation)
	{
		m_IntegrationEnd = dae::GetTimeInSeconds();
		
		double creation       = m_ProblemCreationEnd - m_ProblemCreationStart;
		double initialization = m_InitializationEnd  - m_InitializationStart;
		double integration    = m_IntegrationEnd     - m_IntegrationStart;
		
		m_pLog->Message(string(" "), 0);
		m_pLog->Message(string("The simulation has finished successfuly!"), 0);
		m_pLog->Message(string("Initialization time = ") + toStringFormatted<real_t>(initialization,                          -1, 3) + string(" s"), 0);
		m_pLog->Message(string("Integration time = ")    + toStringFormatted<real_t>(integration,                             -1, 3) + string(" s"), 0);
		m_pLog->Message(string("Total run time = ")      + toStringFormatted<real_t>(creation + initialization + integration, -1, 3) + string(" s"), 0);
	}
}

void daeSimulation::CleanUpSetupData(void)
{
//std::cout << "daeSimulation::CleanUpSetupData" << std::endl;

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
	
	m_pModel		= NULL;
	m_pDAESolver	= NULL;
	m_pDataReporter = NULL;
	m_pLog			= NULL;
	
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
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.constraintsAbsoluteTolerance", 1E-8);

    boost::shared_ptr<daeOptimizationConstraint> pConstraint(new daeOptimizationConstraint(m_pModel, m_pDAESolver, true, dAbsTolerance, m_arrConstraints.size(), strDescription));
	m_arrConstraints.push_back(pConstraint);
	return pConstraint.get();
}

daeOptimizationConstraint* daeSimulation::CreateEqualityConstraint(string strDescription) 
{
	daeConfig& cfg = daeConfig::GetConfig();
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.constraintsAbsoluteTolerance", 1E-8);

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
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.measuredVariableAbsoluteTolerance", 1E-8);
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
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.measuredVariableAbsoluteTolerance", 1E-8);
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
	
	if(mi.m_nNumberOfInitialConditions != mi.m_nNumberOfDifferentialVariables)
	{
		daeDeclareException(exRuntimeCheck);
		e << "Simulation cowardly refused to initialize the problem:\n The number of differential variables is not equal to the number of initial conditions \n";
		e << string("Number of differential variables: ") + toString(mi.m_nNumberOfDifferentialVariables) + string("\n");
		e << string("Number of initial conditions: ")     + toString(mi.m_nNumberOfInitialConditions)     + string("\n");
		throw e;
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

daeObjectiveFunction_t* daeSimulation::GetObjectiveFunction(void) const
{
	if(m_arrObjectiveFunctions.empty())
		daeDeclareAndThrowException(exInvalidCall);
	
	return m_arrObjectiveFunctions[0].get();
}

boost::shared_ptr<daeObjectiveFunction> daeSimulation::AddObjectiveFunction(void)
{
	daeConfig& cfg = daeConfig::GetConfig();
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.objFunctionAbsoluteTolerance", 1E-8);
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
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.objFunctionAbsoluteTolerance", 1E-8);
	
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

real_t daeSimulation::GetCurrentTime(void) const
{
	return m_dCurrentTime;
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

	m_dCurrentTime = m_pDAESolver->Solve(m_dTimeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
	return m_dCurrentTime;
}

// Integrates for the given time interval
real_t daeSimulation::IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if((m_dCurrentTime + time_interval) > m_dTimeHorizon)
		daeDeclareAndThrowException(exInvalidCall);

	m_dCurrentTime = m_pDAESolver->Solve(m_dCurrentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
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

	daeeInitialConditionMode eMode = m_pModel->GetInitialConditionMode();
	m_pDAESolver->SetInitialConditionMode(eMode);
	
	m_pDAESolver->Reinitialize(true);
}

void daeSimulation::EnterConditionalIntegrationMode(void)
{
/**************************************************************/
	daeDeclareAndThrowException(exNotImplemented)
/**************************************************************/
	
	m_bConditionalIntegrationMode = true;
	daeModel* pModel = dynamic_cast<daeModel*>(m_pModel);
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	pModel->SetGlobalConditionContext();
}

// Integrates until the stopping condition or final time is reached
real_t daeSimulation::IntegrateUntilConditionSatisfied(daeCondition rCondition, daeeStopCriterion eStopCriterion)
{
/**************************************************************/
	daeDeclareAndThrowException(exNotImplemented);
/**************************************************************/

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
	
	return m_dCurrentTime;
}

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

	dae_push_back(m_ptrarrReportVariables, pVariable);
	
	size_t i; 
	daeDomain_t* pDomain;
	vector<daeDomain_t*> arrDomains;
	
	daeDataReporterVariable var;
	var.m_strName = pVariable->GetCanonicalName();
	var.m_nNumberOfPoints = pVariable->GetNumberOfPoints();
	pVariable->GetDomains(arrDomains);
	for(i = 0; i < arrDomains.size(); i++)
	{
		pDomain = arrDomains[i];
		var.m_strarrDomains.push_back(pDomain->GetCanonicalName());
	}

	if(!m_pDataReporter->RegisterVariable(&var))
	{
		daeDeclareException(exDataReportingError);
		e << "Simulation dastardly failed to register variable [" << var.m_strName << "]";
		throw e;
	}
}

void daeSimulation::Register(daeParameter* pParameter)
{
	if(!pParameter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pParameter->GetReportingOn())
		return;
	
	dae_push_back(m_ptrarrReportParameters, pParameter);

	size_t i; 
	daeDomain_t* pDomain;
	vector<daeDomain_t*> arrDomains;

	daeDataReporterVariable var;
	var.m_strName = pParameter->GetCanonicalName();
	var.m_nNumberOfPoints = pParameter->GetNumberOfPoints();
	pParameter->GetDomains(arrDomains);
	for(i = 0; i < arrDomains.size(); i++)
	{
		pDomain = arrDomains[i];
		var.m_strarrDomains.push_back(pDomain->GetCanonicalName());
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
	domain.m_strName			= pDomain->GetCanonicalName();
	domain.m_eType				= pDomain->GetType();
	domain.m_nNumberOfPoints	= pDomain->GetNumberOfPoints();
	if(pDomain->GetNumberOfPoints() == 0)
		daeDeclareAndThrowException(exInvalidCall);
	
	domain.m_pPoints = new real_t[domain.m_nNumberOfPoints];
	for(size_t i = 0; i < domain.m_nNumberOfPoints; i++)
		domain.m_pPoints[i] = *pDomain->GetPoint(i);

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

	if(!m_pDataReporter->StartNewResultSet(dCurrentTime))
	{
		daeDeclareException(exDataReportingError);
		e << "Simulation dastardly failed to start new result set at TIME: [" << dCurrentTime << "]";
		throw e;
	}
	
	daeReportObject<daeVariable*>  repVariables(*this, dCurrentTime);
	daeReportObject<daeParameter*> repParameters(*this, dCurrentTime);
	
	std::for_each(m_ptrarrReportVariables.begin(), m_ptrarrReportVariables.end(), repVariables);
	if(dCurrentTime == 0)
		std::for_each(m_ptrarrReportParameters.begin(), m_ptrarrReportParameters.end(), repParameters);
	
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
	size_t i, k, nEnd, nStart, nPoints;
	
	var.m_strName = pVariable->GetCanonicalName();
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

	var.m_strName = pParameter->GetCanonicalName();
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

