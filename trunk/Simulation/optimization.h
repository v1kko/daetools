#ifndef DAE_OPTIMIZATION_H
#define DAE_OPTIMIZATION_H

#include "stdafx.h"
#include "dyn_simulation.h"
#include <stdio.h>
#include <time.h>
using namespace dae::core;
using namespace dae::solver;

namespace dae
{
namespace activity
{

class DAE_ACTIVITY_API daeIPOPT : public daeOptimization_t
{
public:
	daeIPOPT(void);
	virtual ~daeIPOPT(void);

public:
	virtual void Initialize(daeSimulation_t* pSimulation, 
							daeNLPSolver_t* pNLPSolver, 
							daeDAESolver_t* pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t* pLog);
	
	
protected:
	daeSimulation_t*	m_pSimulation;
	daeNLPSolver_t*		m_pNLPSolver;
	daeDAESolver_t*		m_pDAESolver;
	daeLog_t*			m_pLog;
	daeDataReporter_t*	m_pDataReporter;
};

}
}
#endif
