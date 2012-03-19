#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#include <python.h>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporting/datareporters.h"
#include "../BONMIN_MINLPSolver/nlpsolver.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

using namespace dae::nlpsolver;

namespace daepython
{
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


class daeBONMINWrapper : public daeBONMINSolver,
                         public boost::python::wrapper<daeBONMINSolver>
{
public:
    daeBONMINWrapper(void)
    {
    }
	
	void SetOptionS(const string& strOptionName, const string& strValue)
	{
		daeBONMINSolver::SetOption(strOptionName, strValue);
	}
	    
	void SetOptionN(const string& strOptionName, real_t dValue)
	{
		daeBONMINSolver::SetOption(strOptionName, dValue);
	}
    
	void SetOptionI(const string& strOptionName, int iValue)
	{
		daeBONMINSolver::SetOption(strOptionName, iValue);
	}   
};


}

#endif
