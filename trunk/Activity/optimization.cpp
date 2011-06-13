#include "stdafx.h"
#include "simulation.h"
#include <stdio.h>
#include <time.h>

namespace dae
{
namespace activity
{
daeOptimization::daeOptimization(void)
{
	m_pSimulation		 = NULL;
	m_pNLPSolver	     = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;
	m_bIsInitialized     = false;
	m_Initialization     = 0;
	m_Optimization       = 0;

	daeConfig& cfg = daeConfig::GetConfig();
}

daeOptimization::~daeOptimization(void)
{
	Finalize();
}

void daeOptimization::Initialize(daeSimulation_t*   pSimulation,
								 daeNLPSolver_t*    pNLPSolver, 
								 daeDAESolver_t*    pDAESolver, 
								 daeDataReporter_t* pDataReporter, 
								 daeLog_t*          pLog)
{
	if(m_bIsInitialized)
	{
		daeDeclareException(exInvalidCall);
		e << "Optimization has already been initialized";
		throw e;
	}

	if(!pSimulation)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pNLPSolver)
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

	m_Initialization = 0;
	m_Optimization   = 0;
	clock_t start    = time(NULL);

	m_pSimulation		 = pSimulation;
	m_pNLPSolver	     = pNLPSolver;
	m_pDAESolver		 = pDAESolver;
	m_pDataReporter		 = pDataReporter;
	m_pLog			     = pLog;
	
	m_pSimulation->InitializeOptimization(m_pDAESolver, m_pDataReporter, m_pLog);
	m_pNLPSolver->Initialize(m_pSimulation, m_pDAESolver, m_pDataReporter, m_pLog);

	m_bIsInitialized = true;
	
	clock_t end = time(NULL);
	m_Initialization = difftime(end, start);
}

void daeOptimization::Run(void)
{
	if(!m_bIsInitialized)
	{
		daeDeclareException(exInvalidCall);
		e << "Optimization has not been initialized";
		throw e;
	}
	
	m_Optimization = time(NULL);
	
	if(!m_pSimulation)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pNLPSolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDAESolver)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDataReporter)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pLog)
		daeDeclareAndThrowException(exInvalidPointer);
	
	m_pNLPSolver->Solve();
	
	clock_t end = time(NULL);
	m_Optimization = difftime(end, m_Optimization);

	m_pLog->Message(string(" "), 0);
	m_pLog->Message(string("The optimization finished."), 0);
	m_pLog->Message(string("Total run time = ") + toStringFormatted<real_t>(real_t(m_Initialization + m_Optimization), -1, 0) + string(" s"), 0);
}

void daeOptimization::Finalize(void)
{
	if(m_pSimulation)
		m_pSimulation->Finalize();
	
	m_pSimulation	= NULL;
	m_pLog			= NULL;
	m_pNLPSolver	= NULL;
	m_pDataReporter = NULL;
	m_pDAESolver	= NULL;
	
	m_bIsInitialized = false;
}

}
}

