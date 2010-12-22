#include "stdafx.h"
#include "optimization.h"
#include <stdio.h>
#include <time.h>

namespace dae
{
namespace activity
{

daeIPOPT::daeIPOPT(void)
{
	m_pSimulation	     = NULL;
	m_pNLPSolver		 = NULL;
	m_pDAESolver		 = NULL;
	m_pDataReporter		 = NULL;
	m_pLog			     = NULL;
}

daeIPOPT::~daeIPOPT(void)
{
}

void daeIPOPT::Initialize(daeSimulation_t* pSimulation, 
                          daeNLPSolver_t* pNLPSolver, 
                          daeDAESolver_t* pDAESolver, 
                          daeDataReporter_t* pDataReporter, 
                          daeLog_t* pLog)
{
	time_t start, end;

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

	m_pSimulation   = pSimulation;
	m_pNLPSolver    = pNLPSolver;
	m_pDAESolver    = pDAESolver;
	m_pDataReporter	= pDataReporter;
	m_pLog			= pLog;

}



}
}
