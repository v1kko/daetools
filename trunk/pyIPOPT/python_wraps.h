#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#include <string>
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../DataReporters/datareporters.h"
#include "../NLPSolver/nlpsolver.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

using namespace dae::nlpsolver;

namespace daepython
{
//class daeSimulationWrapper : public daeSimulation_t,
//							 public boost::python::wrapper<daeSimulation_t>
//{
//public:
//	daeModel_t* GetModel(void) const
//	{
//		return this->get_override("GetModel")();
//	}

//	void SetModel(daeModel_t* pModel)
//	{
//		this->get_override("SetModel")(pModel);
//	}

//	daeDataReporter_t* GetDataReporter(void) const
//	{
//		return this->get_override("GetDataReporter")();
//	}

//	daeLog_t* GetLog(void) const
//	{
//		return this->get_override("GetLog")();
//	}

//	void Run(void)
//	{
//		this->get_override("Run")();
//	}

//	void ReportData(void) const
//	{
//		this->get_override("ReportData")();
//	}

//	void SetTimeHorizon(real_t dTimeHorizon)
//	{
//		this->get_override("SetTimeHorizon")(dTimeHorizon);
//	}
	
//	real_t GetTimeHorizon(void) const
//	{
//		return this->get_override("GetTimeHorizon")();
//	}
	
//	void SetReportingInterval(real_t dReportingInterval)
//	{
//		this->get_override("SetReportingInterval")(dReportingInterval);
//	}
	
//	real_t GetReportingInterval(void) const
//	{
//		return this->get_override("GetReportingInterval")();
//	}
	
//	void Pause(void)
//	{
//		this->get_override("Pause")();
//	}

//	void Resume(void)
//	{
//		this->get_override("Resume")();
//	}

//	void Stop(void)
//	{
//		this->get_override("Stop")();
//	}

//	void Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
//	{
//		this->get_override("Initialize")(pDAESolver, pDataReporter, pLog);
//	}
	
//	void Reinitialize(void)
//	{
//		this->get_override("Reinitialize")();
//	}

//	void SolveInitial(void)
//	{
//		this->get_override("SolveInitial")();
//	}
	
//	daeDAESolver_t* GetDAESolver(void) const
//	{
//		return this->get_override("GetDAESolver")();
//	}
	
//	void SetUpParametersAndDomains(void)
//	{
//		this->get_override("SetUpParametersAndDomains")();
//	}

//	void SetUpVariables(void)
//	{
//		this->get_override("SetUpVariables")();
//	}
	
//	real_t Integrate(daeeStopCriterion eStopCriterion)
//	{
//		return this->get_override("Integrate")(eStopCriterion);
//	}
	
//	real_t IntegrateForTimeInterval(real_t time_interval)
//	{
//		return this->get_override("IntegrateForTimeInterval")(time_interval);
//	}
	
//	real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion)
//	{
//		return this->get_override("IntegrateUntilTime")(time, eStopCriterion);
//	}
	
//};

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


class daeIPOPTWrapper : public daeIPOPTSolver,
                        public boost::python::wrapper<daeIPOPTSolver>
{
public:
    daeIPOPTWrapper(void)
    {
    }
	
	void SetOptionS(const string& strOptionName, const string& strValue)
	{
		daeIPOPTSolver::SetOption(strOptionName, strValue);
	}
	    
	void SetOptionN(const string& strOptionName, real_t dValue)
	{
		daeIPOPTSolver::SetOption(strOptionName, dValue);
	}
    
	void SetOptionI(const string& strOptionName, int iValue)
	{
		daeIPOPTSolver::SetOption(strOptionName, iValue);
	}   
};


}

#endif
