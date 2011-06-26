#include "stdafx.h"
#include "simulation.h"
//#include "../Core/nodes.h"
#include <stdio.h>
#include <time.h>

namespace dae
{
namespace activity
{
daeSimulation::daeSimulation(void)
{
	m_dCurrentTime		 = 0;
	m_dTimeHorizon		 = 0;
	m_dReportingInterval = 0;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pModel		     = NULL;
	m_pLog			     = NULL;
	m_eActivityAction    = eAAUnknown;	
	m_bConditionalIntegrationMode = false;
	m_bIsInitialized	 = false;
	m_bIsSolveInitial	 = false;
	m_bSetupOptimization = false;

	daeConfig& cfg = daeConfig::GetConfig();
	m_dTimeHorizon       = cfg.Get<real_t>("daetools.activity.timeHorizon", 100);
	m_dReportingInterval = cfg.Get<real_t>("daetools.activity.reportingInterval", 10);
}

daeSimulation::~daeSimulation(void)
{
	Finalize();
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

daeeActivityAction daeSimulation::GetActivityAction(void) const
{
	return m_eActivityAction;
}

//void daeSimulation::Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog, bool bCalculateSensitivities)
//{
//	if(m_bIsInitialized)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Simulation has already been initialized";
//		throw e;
//	}
//		
//	m_bSetupOptimization = bCalculateSensitivities;
//	Init(pDAESolver, pDataReporter, pLog);
//}
//
//void daeSimulation::InitializeOptimization(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
//{
//	if(m_bIsInitialized)
//	{
//		daeDeclareException(exInvalidCall);
//		e << "Simulation has already been initialized";
//		throw e;
//	}
//	
//	m_bSetupOptimization = true;
//	Init(pDAESolver, pDataReporter, pLog);
//}

void daeSimulation::Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog, bool bCalculateSensitivities)
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
	
	m_bSetupOptimization = bCalculateSensitivities;

// Check data reporter
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
	
// Create params, domains, vars, ports, child models
	m_pModel->InitializeStage1();

// Initialize params and domains
	SetUpParametersAndDomains();

// Define the optimization problem: objective function and constraints
	if(m_bSetupOptimization)
	{
		daeConfig& cfg = daeConfig::GetConfig();
		real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.objFunctionAbsoluteTolerance", 1E-8);
		m_pObjectiveFunction = boost::shared_ptr<daeObjectiveFunction>(new daeObjectiveFunction(this, dAbsTolerance));
		
		SetUpOptimization();
	}
	
// Create params, domains, vars, ports
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
	RegisterModel(m_pModel);
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
	vector<string> strarrErrors;

	if(m_ptrarrBlocks.size() != 1)
		daeDeclareAndThrowException(exInvalidCall);
	pBlock = m_ptrarrBlocks[0];
	
	if(m_bSetupOptimization)
	{
	// 1. First check ObjFunction, Constraints and optimization variables
		for(i = 0; i < m_arrOptimizationVariables.size(); i++)
		{
			pOptVariable = m_arrOptimizationVariables[i];
			if(!pOptVariable)
				daeDeclareAndThrowException(exInvalidPointer)
				
			if(!pOptVariable->CheckObject(strarrErrors))
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
				daeDeclareAndThrowException(exInvalidPointer)
				
			if(!pConstraint->CheckObject(strarrErrors))
			{
				daeDeclareException(exRuntimeCheck);
				for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
					e << *it << "\n";
				throw e;
			}
		}
		
		if(!m_pObjectiveFunction)
			daeDeclareAndThrowException(exInvalidPointer)
		if(!m_pObjectiveFunction->CheckObject(strarrErrors))
		{
			daeDeclareException(exRuntimeCheck);
			for(vector<string>::iterator it = strarrErrors.begin(); it != strarrErrors.end(); it++)
				e << *it << "\n";
			throw e;
		}
		
	// 2. Fill the parameters indexes (optimization variables)
		for(i = 0; i < m_arrOptimizationVariables.size(); i++)
		{
			pOptVariable = m_arrOptimizationVariables[i];
			if(!pOptVariable)
				daeDeclareAndThrowException(exInvalidPointer)
				
			narrParametersIndexes.push_back(pOptVariable->GetOverallIndex());
		}
		
	// 3. Initialize the objective function
		m_pObjectiveFunction->Initialize(m_arrOptimizationVariables, pBlock);
		
	// 4. Initialize the constraints
		for(i = 0; i < m_arrConstraints.size(); i++)
		{  
			pConstraint = m_arrConstraints[i];
			if(!pConstraint)
				daeDeclareAndThrowException(exInvalidPointer)
				
			pConstraint->Initialize(m_arrOptimizationVariables, pBlock);
		}
	}

	m_pDAESolver->Initialize(pBlock, m_pLog, m_pModel->GetInitialConditionMode(), m_bSetupOptimization, narrParametersIndexes);
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
	ReportData();

// Set the SolveInitial flag to true
	m_bIsSolveInitial = true;

// Announce success	
	m_InitializationEnd = dae::GetTimeInSeconds();
	m_pLog->Message(string("Starting the initialization of the system... Done."), 0);
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
		daeDeclareAndThrowException(exInvalidCall);
	if(m_dReportingInterval <= 0)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_dReportingInterval > m_dTimeHorizon)
		m_dReportingInterval = m_dTimeHorizon;
	
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
		t = m_dCurrentTime + m_dReportingInterval;
		if(t > m_dTimeHorizon)
			t = m_dTimeHorizon;
		
	// If the flag is set - terminate
		if(m_eActivityAction == ePauseActivity)
		{
			m_pLog->Message(string("Activity paused by the user"), 0);
			return;
		}

	// Integrate until the first discontinuity or until the end of the integration period
	// Report current time period in low precision
		m_pLog->Message(string("Integrating from [") + toString<real_t>(m_dCurrentTime) + 
						string("] to [")             + toString<real_t>(t)              +  
						string("] ..."), 0);
		m_dCurrentTime = IntegrateUntilTime(t, eStopAtModelDiscontinuity);
		ReportData();

	// If discontinuity is found, loop until the end of the integration period
		while(t > m_dCurrentTime)
		{
		// Report current time period in high precision
			m_pLog->Message(string("Integrating from [") + toStringFormatted<real_t>(m_dCurrentTime, -1, 15) + 
							string("] to [")             + toStringFormatted<real_t>(t, -1, 15)              +  
							string("] ..."), 0);
			m_dCurrentTime = IntegrateUntilTime(t, eStopAtModelDiscontinuity);
			ReportData();
		}
		
		m_dCurrentTime = t;
	}

// Print the ned of the simulation info if not in the optimization mode		
	if(!m_bSetupOptimization)
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
	m_dCurrentTime		 = 0;
	m_eActivityAction    = eAAUnknown;	

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

    boost::shared_ptr<daeOptimizationConstraint> pConstraint(new daeOptimizationConstraint(this, true, dAbsTolerance, m_arrConstraints.size(), strDescription));
	m_arrConstraints.push_back(pConstraint);
	return pConstraint.get();
}

daeOptimizationConstraint* daeSimulation::CreateEqualityConstraint(string strDescription)
{
	daeConfig& cfg = daeConfig::GetConfig();
	real_t dAbsTolerance = cfg.Get<real_t>("daetools.activity.constraintsAbsoluteTolerance", 1E-8);

    boost::shared_ptr<daeOptimizationConstraint> pConstraint(new daeOptimizationConstraint(this, false, dAbsTolerance, m_arrConstraints.size(), strDescription));
	m_arrConstraints.push_back(pConstraint);
	return pConstraint.get();
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

daeObjectiveFunction_t* daeSimulation::GetObjectiveFunction(void) const
{
	return m_pObjectiveFunction.get();
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
}

real_t daeSimulation::GetReportingInterval(void) const
{
	return m_dReportingInterval;
}

// Integrates until the stopping criterion is reached or the time horizon of simulation
real_t daeSimulation::Integrate(daeeStopCriterion eStopCriterion)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(m_dCurrentTime >= m_dTimeHorizon)
		daeDeclareAndThrowException(exInvalidCall);

	m_dCurrentTime = m_pDAESolver->Solve(m_dTimeHorizon, eStopCriterion);
	return m_dCurrentTime;
}

// Integrates for the given time interval
real_t daeSimulation::IntegrateForTimeInterval(real_t time_interval)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if((m_dCurrentTime + time_interval) > m_dTimeHorizon)
		daeDeclareAndThrowException(exInvalidCall);

	m_dCurrentTime = m_pDAESolver->Solve(m_dCurrentTime + time_interval, eDoNotStopAtDiscontinuity);
	return m_dCurrentTime;
}

// Integrates until the stopping criterion or time is reached
real_t daeSimulation::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(time > m_dTimeHorizon)
		daeDeclareAndThrowException(exInvalidCall);

	m_dCurrentTime = m_pDAESolver->Solve(time, eStopCriterion);
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
	daeDeclareAndThrowException(exNotImplemented)
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

void daeSimulation::RegisterModel(daeModel_t* pModel)
{
	size_t i;
	daeVariable_t* pVariable;
	daeDomain_t* pDomain;
	daePort_t* pPort;
	daeModel_t* pChildModel;
	vector<daeVariable_t*> arrVars;
	vector<daePort_t*> arrPorts;
	vector<daeDomain_t*> arrDomains;
	vector<daeModel_t*> arrModels;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	pModel->GetDomains(arrDomains);
	for(i = 0; i < arrDomains.size(); i++)
	{
		pDomain = arrDomains[i];
		RegisterDomain(pDomain);
	}

	pModel->GetVariables(arrVars);
	for(i = 0; i < arrVars.size(); i++)
	{
		pVariable = arrVars[i];
		RegisterVariable(pVariable);
	}

	pModel->GetPorts(arrPorts);
	for(i = 0; i < arrPorts.size(); i++)
	{
		pPort = arrPorts[i];
		RegisterPort(pPort);
	}

	pModel->GetModels(arrModels);
	for(i = 0; i < arrModels.size(); i++)
	{
		pChildModel = arrModels[i];
		RegisterModel(pChildModel);
	}
}

void daeSimulation::RegisterPort(daePort_t* pPort)
{
	size_t i;
	daeVariable_t* pVariable;
	daeDomain_t* pDomain;
	vector<daeVariable_t*> arrVars;
	vector<daeDomain_t*> arrDomains;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);

	pPort->GetDomains(arrDomains);
	for(i = 0; i < arrDomains.size(); i++)
	{
		pDomain = arrDomains[i];
		RegisterDomain(pDomain);
	}

	pPort->GetVariables(arrVars);
	for(i = 0; i < arrVars.size(); i++)
	{
		pVariable = arrVars[i];
		RegisterVariable(pVariable);
	}
}

void daeSimulation::RegisterVariable(daeVariable_t* pVariable)
{
	size_t i;
	daeDomain_t* pDomain;
	vector<daeDomain_t*> arrDomains;

	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer);

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

void daeSimulation::RegisterDomain(daeDomain_t* pDomain)
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
		domain.m_pPoints[i] = pDomain->GetPoint(i);

	if(!m_pDataReporter->RegisterDomain(&domain))
	{
		daeDeclareException(exDataReportingError);
		e << "Simulation dastardly failed to register domain [" << domain.m_strName << "]";
		throw e;
	}
}

void daeSimulation::ReportData(void)
{
	if(!m_pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);

	if(!m_pDataReporter->StartNewResultSet(m_dCurrentTime))
	{
		daeDeclareException(exDataReportingError);
		e << "Simulation dastardly failed to start new result set at TIME: [" << m_dCurrentTime << "]";
		throw e;
	}

	ReportModel(m_pModel, m_dCurrentTime);
}

void daeSimulation::ReportModel(daeModel_t* pModel, real_t time)
{
	size_t i;
	daeVariable_t* pVariable;
	daePort_t* pPort;
	daeModel_t* pChildModel;
	vector<daeVariable_t*> arrVars;
	vector<daePort_t*> arrPorts;
	vector<daeModel_t*> arrModels;

	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	pModel->GetVariables(arrVars);
	for(i = 0; i < arrVars.size(); i++)
	{
		pVariable = arrVars[i];
		ReportVariable(pVariable, time);
	}

	pModel->GetPorts(arrPorts);
	for(i = 0; i < arrPorts.size(); i++)
	{
		pPort = arrPorts[i];
		ReportPort(pPort, time);
	}

	pModel->GetModels(arrModels);
	for(i = 0; i < arrModels.size(); i++)
	{
		pChildModel = arrModels[i];
		ReportModel(pChildModel, time);
	}
}

void daeSimulation::ReportPort(daePort_t* pPort, real_t time)
{
	size_t i;
	daeVariable_t* pVariable;
	vector<daeVariable_t*> arrVars;

	if(!pPort)
		daeDeclareAndThrowException(exInvalidPointer);

	pPort->GetVariables(arrVars);
	for(i = 0; i < arrVars.size(); i++)
	{
		pVariable = arrVars[i];
		ReportVariable(pVariable, time);
	}
}

void daeSimulation::ReportVariable(daeVariable_t* pVariable, real_t time)
{
	real_t* pd;
	daeDataReporterVariableValue var;
	size_t i, nSize;

	if(!pVariable)
		daeDeclareAndThrowException(exInvalidPointer);

	if(!pVariable->GetReportingOn())
		return;
	
	var.m_strName = pVariable->GetCanonicalName();
	pd = pVariable->GetValuePointer();
	nSize = pVariable->GetNumberOfPoints();
	var.m_nNumberOfPoints = nSize;
	var.m_pValues = new real_t[nSize];
	for(i = 0; i < nSize; i++)
		var.m_pValues[i] = pd[i];

	if(!m_pDataReporter->SendVariable(&var))
	{
		daeDeclareException(exDataReportingError);
		e << "Simulation dastardly failed to report variable [" << var.m_strName << "] at TIME: [" << time << "]";
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

