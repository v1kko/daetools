#include "stdafx.h"
#include "dyn_simulation.h"
#include <stdio.h>
#include <time.h>

namespace dae
{
namespace activity
{
daeDynamicSimulation::daeDynamicSimulation(void)
{
	m_dCurrentTime		 = 0;
	m_dTimeHorizon		 = 0;
	m_dReportingInterval = 0;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pModel		     = NULL;
	m_pLog			     = NULL;
	m_ProblemCreation    = 0;
	m_Initialization     = 0;
	m_Integration        = 0;
	m_eActivityAction    = eAAUnknown;	
	m_bConditionalIntegrationMode = false;
	m_bIsInitialized	 = false;
	m_bIsSolveInitial	 = false;
	

	daeConfig& cfg = daeConfig::GetConfig();
	m_dTimeHorizon       = cfg.Get<real_t>("daetools.activity.timeHorizon", 100);
	m_dReportingInterval = cfg.Get<real_t>("daetools.activity.reportingInterval", 10);
}

daeDynamicSimulation::~daeDynamicSimulation(void)
{
}

void daeDynamicSimulation::SetUpParametersAndDomains()
{
}

void daeDynamicSimulation::SetUpVariables()
{
}

void daeDynamicSimulation::Resume(void)
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

void daeDynamicSimulation::Pause(void)
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

daeeActivityAction daeDynamicSimulation::GetActivityAction(void) const
{
	return m_eActivityAction;
}

void daeDynamicSimulation::Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
{
	time_t start, end;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pLog)
		daeDeclareAndThrowException(exInvalidPointer);

// Check data reporter
	if(!pDataReporter->IsConnected())
	{
		daeDeclareException(exInvalidCall);
		e << "Data Reporter is not connected \n";
		throw e;
	}

	m_pDAESolver    = pDAESolver;
	m_pDataReporter	= pDataReporter;
	m_pLog			= pLog;

	start = time(NULL);

	m_pLog->Message(string("*************************************************************************"), 0);
	m_pLog->Message(string("*                          @@@@@                                        *"), 0);
	m_pLog->Message(string("*       @                    @                                          *"), 0);
	m_pLog->Message(string("*       @   @@@@@     @@@@@  @                    DAE Tools             *"), 0);
	m_pLog->Message(string("*  @@@@@@        @   @     @           Version:   ") + daeVersion() + string("                 *"), 0);
	m_pLog->Message(string("* @     @   @@@@@@   @@@@@@            Copyright: Dragan Nikolic, 2010  *"), 0);
	m_pLog->Message(string("* @     @  @     @   @                 E-mail:    dnikolic@daetools.com *"), 0);
	m_pLog->Message(string("*  @@@@@    @@@@@@    @@@@@            Homepage:  www.daetools.com      *"), 0);
	m_pLog->Message(string("*                                                                       *"), 0);
	m_pLog->Message(string("*************************************************************************"), 0);
	m_pLog->Message(string("* DAE Tools is free software: you can redistribute it and/or modify     *"), 0);
	m_pLog->Message(string("* it under the terms of the GNU General Public License as published     *"), 0);
	m_pLog->Message(string("* by the Free Software Foundation; either version 3 of the License,     *"), 0);
	m_pLog->Message(string("* or (at your option) any later version.                                *"), 0);
	m_pLog->Message(string("* This program is distributed in the hope that it will be useful,       *"), 0);
	m_pLog->Message(string("* but WITHOUT ANY WARRANTY; without even the implied warranty of        *"), 0);
	m_pLog->Message(string("* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          *"), 0);
	m_pLog->Message(string("* GNU General Public License for more details.                          *"), 0);
	m_pLog->Message(string("* You should have received a copy of the GNU General Public License     *"), 0);
	m_pLog->Message(string("* along with this program. If not, see <http://www.gnu.org/licenses/>.  *"), 0);
	m_pLog->Message(string("*************************************************************************"), 0);
	m_pLog->Message(string("  "), 0);
	m_pLog->Message(string("Creating the system... "), 0);

// Create params, domains, vars, ports, child models
	m_pModel->InitializeStage1();

// Initialize params and domains
	SetUpParametersAndDomains();
 
// Create params, domains, vars, ports
	m_pModel->InitializeStage2();

// Create data storage for variables, derivatives, var. types, tolerances, etc
	m_pModel->InitializeStage3(m_pLog);

// Set initial values, initial conditions, fix variables, set initial guesses, abs tolerances, etc
	SetUpVariables();
	
// Create equation execution infos in models and stns
	m_pModel->InitializeStage4();

// Set the solver's InitialConditionMode
	daeeInitialConditionMode eMode = GetInitialConditionMode();
	m_pDAESolver->SetInitialConditionMode(eMode);
	if(eMode == eSteadyState)
		SetInitialConditionsToZero();

// Now I have everything set up and I should check for inconsistences
	CheckSystem();

// Do the block decomposition if needed (at the moment only one block is created)
	m_ptrarrBlocks.EmptyAndFreeMemory();
	m_pModel->InitializeStage5(false, m_ptrarrBlocks);

// Initialize solver
	if(m_ptrarrBlocks.size() != 1)
		daeDeclareAndThrowException(exInvalidCall);
	daeBlock_t* pBlock = m_ptrarrBlocks[0];
	m_pDAESolver->Initialize(pBlock, m_pLog, m_pModel->GetInitialConditionMode());

// Register model
	m_dCurrentTime = 0;
	m_pDataReporter->StartRegistration();
	RegisterModel(m_pModel);
	m_pDataReporter->EndRegistration();
	
// Set the IsInitialized flag to true
	m_bIsInitialized = true;

// Announce success	
	end = time(NULL);
	m_ProblemCreation = difftime(end, start);
	m_pLog->Message(string("The system created successfully in: ") + toStringFormatted<real_t>(real_t(m_ProblemCreation), -1, 3) + string(" s"), 0);
	m_pLog->Message(string(""), 0);
}

void daeDynamicSimulation::SolveInitial(void)
{
	clock_t start, end;

// Check if initialized
	if(!m_bIsInitialized)
		daeDeclareAndThrowException(exInvalidCall);
	
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
	start = time(NULL);
	m_pLog->Message(string("Starting initialization of the system..."), 0);

// Ask DAE solver to initialize the system
	m_pDAESolver->SolveInitial();

// Report data at TIME=0
	ReportData();

// Set the SolveInitial flag to true
	m_bIsSolveInitial = true;

// Announce success	
	end = time(NULL);
	m_Initialization = difftime(end, start);
	m_pLog->Message(string("Initialization completed. Initialization time: ") + toStringFormatted<real_t>(real_t(m_Initialization), -1, 0) + string(" s"), 0);
	m_pLog->Message(string("  "), 0);
}

void daeDynamicSimulation::Run(void)
{
// Once simulation has started one can change time horizon or reporting interval
// Is it a good idea?

// Check if initialized and solved initially
	if(!m_bIsInitialized)
		daeDeclareAndThrowException(exInvalidCall);
	if(!m_bIsSolveInitial)
		daeDeclareAndThrowException(exInvalidCall);
	
// Check pointers
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pLog)
		daeDeclareAndThrowException(exInvalidPointer);
	
// Check for some mistakes
	if(m_dTimeHorizon <= 0)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_dReportingInterval <= 0)
		daeDeclareAndThrowException(exInvalidCall);
	if(m_dReportingInterval > m_dTimeHorizon)
		m_dReportingInterval = m_dTimeHorizon;
	
	if(m_dCurrentTime == 0)
	{
		m_pLog->Message(string("Starting dynamic simulation..."), 0);
		m_Integration = time(NULL);
	}
	if(m_dCurrentTime >= m_dTimeHorizon)
	{
		m_pLog->Message(string("The time domain has been riched; exiting."), 0);
		return;
	}
	m_eActivityAction = eRunActivity;

	for(real_t t = m_dCurrentTime + m_dReportingInterval; t <= m_dTimeHorizon; t += m_dReportingInterval)
	{
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
	}

// Finalize the simulation		
	clock_t end = time(NULL);
	m_Integration = difftime(end, m_Integration);
	m_pLog->Message(string("Dynamic simulation has finished successfuly"), 0);
	m_pLog->Message(string("Integration time = ") + toStringFormatted<real_t>(real_t(m_Integration), -1, 0) + string(" s"), 0);
	m_pLog->Message(string("Total run time = ") + toStringFormatted<real_t>(real_t(m_ProblemCreation + m_Initialization + m_Integration), -1, 0) + string(" s"), 0);
}

void daeDynamicSimulation::Finalize(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pLog)
		daeDeclareAndThrowException(exInvalidPointer);
	
// Notify the receiver that there is no more data, and disconnect it		
	m_pDataReporter->EndOfData();
	m_pDataReporter->Disconnect();

	m_pModel		= NULL;
	m_pDAESolver	= NULL;
	m_pDataReporter = NULL;
	m_pLog			= NULL;
	
	m_ProblemCreation    = 0;
	m_Initialization     = 0;
	m_Integration        = 0;
	m_bIsInitialized	 = false;
	m_bIsSolveInitial	 = false;
}

void daeDynamicSimulation::Reset(void)
{
	m_dCurrentTime		 = 0;
	m_ProblemCreation    = 0;
	m_Initialization     = 0;
	m_Integration        = 0;
	m_eActivityAction    = eAAUnknown;	

// Set again the initial conditions, values, tolerances, active states etc
	SetUpVariables();
		
// Reset the DAE solver
	m_pDAESolver->Reset();

// Set the solver's InitialConditionMode
//	daeeInitialConditionMode eMode = GetInitialConditionMode();
//	m_pDAESolver->SetInitialConditionMode(eMode);
//	if(eMode == eSteadyState)
//		SetInitialConditionsToZero();
}

void daeDynamicSimulation::SetInitialConditionsToZero(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer)
	m_pModel->SetInitialConditions(0);
}

void daeDynamicSimulation::CheckSystem(void) const
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
		e << "Number of variables is equal zero\n";
		throw e;
	}

	if(mi.m_nNumberOfEquations == 0)
	{
		daeDeclareException(exRuntimeCheck);
		e << "Number of equations is equal zero\n";
		throw e;
	}

	if((mi.m_nNumberOfVariables - mi.m_nNumberOfFixedVariables) != mi.m_nNumberOfEquations)
	{
		daeDeclareException(exRuntimeCheck);
		if(mi.m_nNumberOfEquations < mi.m_nNumberOfVariables)
			e << "Number of equations is lower than number of variables \n";
		else
			e << "Number of variables is lower than number of equations \n";
		e << string("Number of equations: ")       + toString(mi.m_nNumberOfEquations)      + string("\n");
		e << string("Number of variables: ")       + toString(mi.m_nNumberOfVariables)      + string("\n");
		e << string("Number of fixed variables: ") + toString(mi.m_nNumberOfFixedVariables) + string("\n");
		throw e;
	}
	
	if(mi.m_nNumberOfInitialConditions != mi.m_nNumberOfDifferentialVariables)
	{
		daeDeclareException(exRuntimeCheck);
		e << "Number of differential variables is not equal to number of initial conditions \n";
		e << string("Number of differential variables: ") + toString(mi.m_nNumberOfDifferentialVariables) + string("\n");
		e << string("Number of initial conditions: ")     + toString(mi.m_nNumberOfInitialConditions)     + string("\n");
		throw e;
	}
}

real_t daeDynamicSimulation::GetCurrentTime(void) const
{
	return m_dCurrentTime;
}

void daeDynamicSimulation::SetInitialConditionMode(daeeInitialConditionMode eMode)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer)
	m_pModel->SetInitialConditionMode(eMode);
}

daeeInitialConditionMode daeDynamicSimulation::GetInitialConditionMode(void) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer)
	return m_pModel->GetInitialConditionMode();
}

void daeDynamicSimulation::StoreInitializationValues(const std::string& strFileName) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pModel->StoreInitializationValues(strFileName);
}

void daeDynamicSimulation::LoadInitializationValues(const std::string& strFileName) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pModel->LoadInitializationValues(strFileName);
}

void daeDynamicSimulation::SetTimeHorizon(real_t dTimeHorizon)
{
	if(dTimeHorizon <= 0)
		return;
	m_dTimeHorizon = dTimeHorizon;
}

real_t daeDynamicSimulation::GetTimeHorizon(void) const
{
	return m_dTimeHorizon;
}

void daeDynamicSimulation::SetReportingInterval(real_t dReportingInterval)
{
	if(dReportingInterval <= 0)
		return;
	m_dReportingInterval = dReportingInterval;
}

real_t daeDynamicSimulation::GetReportingInterval(void) const
{
	return m_dReportingInterval;
}

// Integrates until the stopping criterion is reached or the time horizon of simulation
real_t daeDynamicSimulation::Integrate(daeeStopCriterion eStopCriterion)
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
real_t daeDynamicSimulation::IntegrateForTimeInterval(real_t time_interval)
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
real_t daeDynamicSimulation::IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion)
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

void daeDynamicSimulation::Reinitialize(void)
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

void daeDynamicSimulation::EnterConditionalIntegrationMode(void)
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
real_t daeDynamicSimulation::IntegrateUntilConditionSatisfied(daeCondition rCondition, daeeStopCriterion eStopCriterion)
{
/**************************************************************/
	daeDeclareAndThrowException(exNotImplemented)
/**************************************************************/

	if(!m_bConditionalIntegrationMode)
	{
		daeDeclareException(exInvalidCall);
		e << string("the function EnterConditionalIntegrationMode() should be called prior the call to IntegrateUntilConditionSatisfied()");
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

void daeDynamicSimulation::RegisterModel(daeModel_t* pModel)
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

void daeDynamicSimulation::RegisterPort(daePort_t* pPort)
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

void daeDynamicSimulation::RegisterVariable(daeVariable_t* pVariable)
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
		e << "Failed to register variable [" << var.m_strName << "]";
		throw e;
	}
}

void daeDynamicSimulation::RegisterDomain(daeDomain_t* pDomain)
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
		e << "Failed to register domain [" << domain.m_strName << "]";
		throw e;
	}
}

void daeDynamicSimulation::ReportData(void)
{
	if(!m_pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);

	if(!m_pDataReporter->StartNewResultSet(m_dCurrentTime))
	{
		daeDeclareException(exDataReportingError);
		e << "Failed to start new result set at TIME: [" << m_dCurrentTime << "]";
		throw e;
	}

	ReportModel(m_pModel, m_dCurrentTime);
}

void daeDynamicSimulation::ReportModel(daeModel_t* pModel, real_t time)
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

void daeDynamicSimulation::ReportPort(daePort_t* pPort, real_t time)
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

void daeDynamicSimulation::ReportVariable(daeVariable_t* pVariable, real_t time)
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
		e << "Failed to report variable [" << var.m_strName << "] at TIME: [" << time << "]";
		throw e;
	}
}

daeModel_t* daeDynamicSimulation::GetModel(void) const
{
	return m_pModel;
}

void daeDynamicSimulation::SetModel(daeModel_t* pModel)
{
	if(!pModel)
		return;
	m_pModel = pModel;
}

daeDataReporter_t* daeDynamicSimulation::GetDataReporter(void) const
{
	return m_pDataReporter;
}

daeLog_t* daeDynamicSimulation::GetLog(void) const
{
	return m_pLog;
}

daeDAESolver_t* daeDynamicSimulation::GetDAESolver(void) const
{
	return m_pDAESolver;
}



}
}

