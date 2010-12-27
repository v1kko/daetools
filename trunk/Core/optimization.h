#ifndef DAE_OPTIMIZATION_H
#define DAE_OPTIMIZATION_H

#include "definitions.h"
#include "core.h"
#include "log.h"
#include "activity.h"
#include "datareporting.h"

namespace dae
{
namespace nlpsolver
{
/*********************************************************************************************
	daeNLPSolver_t
**********************************************************************************************/
class daeNLPSolver_t
{
public:
	virtual ~daeNLPSolver_t(void){}

public:
	virtual void Initialize(daeSimulation_t*   pSimulation,
						    daeDAESolver_t*    pDAESolver,
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog) = 0;
	virtual void Solve() = 0;
};

}
}

using namespace dae::nlpsolver;

namespace dae
{
namespace activity
{
/*********************************************************************
	daeOptimization_t
*********************************************************************/
class daeOptimization_t
{
public:
	virtual void Initialize(daeSimulation_t*   pSimulation,
					        daeNLPSolver_t*    pNLPSolver, 
							daeDAESolver_t*    pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog)			= 0;
	virtual void Run(void)										= 0;
	virtual void Finalize(void)									= 0;
};

}
}



#endif
