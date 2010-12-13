#include "stdafx.h"
#include "bd_dyn_simulation.h"

namespace dae
{
namespace activity
{
//daeBDDynamicSimulation::daeBDDynamicSimulation(void)
//{
//}
//
//daeBDDynamicSimulation::~daeBDDynamicSimulation(void)
//{
//}
//
//void daeBDDynamicSimulation::Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
//{
//	daeBlock_t* pBlock;
//	daeDAESolver_t* pSolver;
//
//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);
//	if(!pDAESolver)
//		daeDeclareAndThrowException(exInvalidPointer);
//	if(!pDataReporter)
//		daeDeclareAndThrowException(exInvalidPointer);
//	if(!pLog)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	m_pDAESolver    = pDAESolver;
//	m_pDataReporter	= pDataReporter;
//	m_pLog			= pLog;
//
//// Create params, domains, vars, ports, child models
//	m_pModel->InitializeStage1();
//
//// Initialize params and domains
//	SetUpParametersAndDomains();
//
//// Create params, domains, vars, ports
//	m_pModel->InitializeStage2();
//
//// Create data storage for variables, derivatives, var. types, tolerances, etc
//	size_t n = m_pModel->GetTotalNumberOfVariables();
//	if(n == 0)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	m_pDataProxy = new daeDataProxy_t(pLog, n);
//	if(!m_pDataProxy)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	m_pModel->InitializeStage3(m_pDataProxy);
//
//// Set initial values, initial conditions, fix variables, set initial guesses, abs tolerances, etc
//	SetUpVariables();
//
//// Create and initialize equations
//	m_pModel->InitializeStage4(false, m_ptrarrBlocks);
//
//	if(!m_pDataReporter->IsConnected())
//	{
//		daeDeclareException(exMiscellanous);
//		e << "Data reporter is not connected";
//		throw e;
//	}
//
//// Create solver for each block
//	for(size_t i = 0; i < m_ptrarrBlocks.size(); i++)
//	{
//		pBlock = m_ptrarrBlocks[i];
//		if(!pBlock)
//			daeDeclareAndThrowException(exInvalidPointer);
//
//		pSolver = new daeIDASolver;
//		pSolver->Initialize(pBlock, NULL);
//
//		m_ptrarrDAEBlockSolvers.push_back(pSolver);
//	}
//
//	RegisterModel(m_pModel);
//	ReportData(0);
//}
//
//void daeBDDynamicSimulation::Run(void)
//{
//	double dCurrentTime;
//
//	dCurrentTime = 0;
//	for(real_t t = m_dReportingInterval; t <= m_dTimeHorizon; t += m_dReportingInterval)
//	{
//		m_pLog->Message(string("Integrating from [") + toString<double>(t-m_dReportingInterval) + 
//					    string("] to [") +  toString<double>(t) +  string("] ..."), 0);
//
//		dCurrentTime = IntegrateUntilDiscontinuity(t);
//		while(t > dCurrentTime)
//		{
//			dCurrentTime = IntegrateUntilDiscontinuity(t);
//		}
//	}
//}
//
//void daeBDDynamicSimulation::ContinueTo(real_t& time, bool bStopAtDiscontinuity)
//{
//	if(!m_pModel)
//		daeDeclareAndThrowException(exInvalidPointer);
//	if(time > m_dTimeHorizon)
//		daeDeclareAndThrowException(exInvalidCall);
//
//	daeBlockSolverData BlockSolverData;
//	size_t i, nNoProcessors;
//	pthread_t* pthreads;
//	int  iret;
//
//	if(m_ptrarrDAEBlockSolvers.size() == 1)
//		nNoProcessors = 1;
//	else
//		nNoProcessors = 2;
//
//	if(nNoProcessors == 1)
//	{
//		daeDAESolver_t* pSolver = m_ptrarrDAEBlockSolvers[0];
//		if(!pSolver)
//			daeDeclareAndThrowException(exInvalidPointer);
//
//		pSolver->Solve(time, bStopAtDiscontinuity);
//	}
//	else
//	{
//		pthreads = new pthread_t[nNoProcessors];
//
//		BlockSolverData.m_dTimeHorizon = time;
//		BlockSolverData.m_ptrarrDAEBlockSolvers = m_ptrarrDAEBlockSolvers;
//		BlockSolverData.m_pModel = m_pModel;
//		BlockSolverData.m_pMutex = PTHREAD_MUTEX_INITIALIZER;
//
//		for(i = 0; i < nNoProcessors; i++)
//			iret = pthread_create( &pthreads[i], NULL, BlockSolver, (void*)&BlockSolverData);
//
//		for(i = 0; i < nNoProcessors; i++)
//			pthread_join(pthreads[i], NULL);
//
//		delete[] pthreads;
//	}
//}
//
//void daeBDDynamicSimulation::Integrate(real_t time)
//{
//	ContinueTo(time, false);
//	ReportData(time);
//}
//
//real_t daeBDDynamicSimulation::IntegrateUntilDiscontinuity(real_t time)
//{
//	ContinueTo(time, true);
//	ReportData(time);
//	return time;
//}
//
//real_t daeBDDynamicSimulation::IntegrateUntil(daeCondition& rCondition)
//{
//	return 0;
//}

/*********************************************************************
	daeBlockSolverData
*********************************************************************/
daeBlockSolverData::daeBlockSolverData(void)
{
	m_pModel				= NULL;
	m_nCurrentBlockIndex	= 0;
#ifdef DAE_MUTEX
	m_pMutex				= NULL;
#endif
	m_bIntegrationFailure	= false;
}

daeBlockSolverData::~daeBlockSolverData(void)
{
}

daeDAESolver_t* daeBlockSolverData::GetNextBlockSolver()
{
	daeDAESolver_t* pBlockSolver;

	pthread_mutex_lock(&m_pMutex);
		if(m_bIntegrationFailure || m_nCurrentBlockIndex >= m_ptrarrDAEBlockSolvers.size())
		{
			pBlockSolver = NULL;
		}
		else
		{
			pBlockSolver = m_ptrarrDAEBlockSolvers[m_nCurrentBlockIndex];
			m_nCurrentBlockIndex++;
		}
	pthread_mutex_unlock(&m_pMutex);
	return pBlockSolver;
}

void daeBlockSolverData::SetFailed()
{
	m_bIntegrationFailure = true;
}

void* BlockSolver(void* pThreadData)
{
	daeDAESolver_t* pSolver;
	daeBlockSolverData* pBSData;
	try
	{
		pBSData = static_cast<daeBlockSolverData*>(pThreadData);
		if(!pBSData)
			return NULL;

		pSolver = pBSData->GetNextBlockSolver();
		while(pSolver)
		{
			pSolver->Solve(pBSData->m_dTimeHorizon, false);
			pSolver = pBSData->GetNextBlockSolver();
		}
	}
	catch(std::exception& e)
	{
		pBSData->SetFailed();
		throw e;
	}
	catch(...)
	{
		pBSData->SetFailed();
		throw;
	}
	return pThreadData;
}

}
}

