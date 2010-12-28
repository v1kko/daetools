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

	daeConfig& cfg = daeConfig::GetConfig();
}

daeOptimization::~daeOptimization(void)
{
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

	m_pSimulation		 = pSimulation;
	m_pNLPSolver	     = pNLPSolver;
	m_pDAESolver		 = pDAESolver;
	m_pDataReporter		 = pDataReporter;
	m_pLog			     = pLog;
	
	m_pSimulation->InitializeOptimization(m_pDAESolver, m_pDataReporter, m_pLog);
	m_pNLPSolver->Initialize(m_pSimulation, m_pDAESolver, m_pDataReporter, m_pLog);

	m_bIsInitialized = true;
}

void daeOptimization::Run(void)
{
	if(!m_bIsInitialized)
	{
		daeDeclareException(exInvalidCall);
		e << "Optimization has not been initialized";
		throw e;
	}
	
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
}

void daeOptimization::Finalize(void)
{
	if(!m_bIsInitialized)
	{
		daeDeclareException(exInvalidCall);
		e << "Optimization has not been initialized";
		throw e;
	}
	
	if(!m_pSimulation)
		daeDeclareAndThrowException(exInvalidPointer);

	m_pSimulation->Finalize();
}



}
}

