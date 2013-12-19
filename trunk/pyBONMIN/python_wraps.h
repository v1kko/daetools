#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporting/datareporters.h"
#include "../BONMIN_MINLPSolver/nlpsolver.h"
#include "../BONMIN_MINLPSolver/base_solvers.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

using namespace dae::nlpsolver;

namespace daepython
{
daeNLPSolver_t* daeCreateNLPSolver(void);

class daeNLPSolverWrapper : public daeNLPSolver_t,
					        public boost::python::wrapper<daeNLPSolver_t>
{
public:
	void Initialize(daeSimulation_t*   pSimulation,
			        daeNLPSolver_t*    pNLPSolver, 
					daeDAESolver_t*    pDAESolver, 
					daeDataReporter_t* pDataReporter, 
					daeLog_t*          pLog)
	{
		this->get_override("Initialize")(pSimulation, pNLPSolver, pDAESolver, pDataReporter, pLog);
	}
	
	void Solve(void)
	{
		this->get_override("Solve")();
	}
};


void SetOptionS(daeBONMINSolver& self, const string& strOptionName, const string& strValue);
void SetOptionF(daeBONMINSolver& self, const string& strOptionName, real_t dValue);
void SetOptionI(daeBONMINSolver& self, const string& strOptionName, int iValue);

}

#endif
