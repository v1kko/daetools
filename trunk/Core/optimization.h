/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the 
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
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
