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
#include "../Simulation/dyn_simulation.h"
#include "../Solver/ida_solver.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

namespace daepython
{
/*******************************************************
	daeDynamicSimulation
*******************************************************/
class daeActivityWrapper : public daeActivity_t,
	                       public boost::python::wrapper<daeActivity_t>
{
public:
	daeModel_t* GetModel(void) const
	{
		return this->get_override("GetModel")();
	}

	void SetModel(daeModel_t* pModel)
	{
		this->get_override("SetModel")(pModel);
	}

	daeDataReporter_t* GetDataReporter(void) const
	{
		return this->get_override("GetDataReporter")();
	}

	daeLog_t* GetLog(void) const
	{
		return this->get_override("GetLog")();
	}

	void Run(void)
	{
		this->get_override("Run")();
	}
};

class daeDynamicActivityWrapper : public daeDynamicActivity_t,
	                              public boost::python::wrapper<daeDynamicActivity_t>
{
public:
	void ReportData(void) const
	{
        this->get_override("ReportData")();
	}

	void SetTimeHorizon(real_t dTimeHorizon)
	{
        this->get_override("SetTimeHorizon")(dTimeHorizon);
	}
	
	real_t GetTimeHorizon(void) const
	{
        return this->get_override("GetTimeHorizon")();
	}
	
	void SetReportingInterval(real_t dReportingInterval)
	{
        this->get_override("SetReportingInterval")(dReportingInterval);
	}
	
	real_t GetReportingInterval(void) const
	{
        return this->get_override("GetReportingInterval")();
	}
	
	void Pause(void)
	{
		this->get_override("Pause")();
	}

	void Resume(void)
	{
		this->get_override("Resume")();
	}

	void Stop(void)
	{
		this->get_override("Stop")();
	}
};

class daeDynamicSimulationWrapper : public daeDynamicSimulation_t,
	                                public boost::python::wrapper<daeDynamicSimulation_t>
{
public:
	void Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
	{
        this->get_override("Initialize")(pDAESolver, pDataReporter, pLog);
	}
	
	void Reinitialize(void)
	{
        this->get_override("Reinitialize")();
	}

	void SolveInitial(void)
	{
        this->get_override("SolveInitial")();
	}
	
	daeDAESolver_t* GetDAESolver(void) const
	{
        return this->get_override("GetDAESolver")();
	}
	
	void SetUpParametersAndDomains(void)
	{
        this->get_override("SetUpParametersAndDomains")();
	}

	void SetUpVariables(void)
	{
        this->get_override("SetUpVariables")();
	}
	
	real_t Integrate(daeeStopCriterion eStopCriterion)
	{
        return this->get_override("Integrate")(eStopCriterion);
	}
	
	real_t IntegrateForTimeInterval(real_t time_interval)
	{
        return this->get_override("IntegrateForTimeInterval")(time_interval);
	}
	
	real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion)
	{
        return this->get_override("IntegrateUntilTime")(time, eStopCriterion);
	}
	
};

class daeDefaultDynamicSimulationWrapper : public daeDynamicSimulation,
	                                       public boost::python::wrapper<daeDynamicSimulation>
{
public:
	daeDefaultDynamicSimulationWrapper()
	{
	}

	boost::python::object GetModel_(void) const
	{
		return model;
	}

	void SetModel_(boost::python::object Model)
	{
		model = Model;
		daeModel* pModel = boost::python::extract<daeModel*>(Model);
		this->daeDynamicSimulation::SetModel(pModel);
	}

	void SetUpParametersAndDomains(void)
	{
        if(boost::python::override f = this->get_override("SetUpParametersAndDomains"))
			f();
		else
			this->daeDynamicSimulation::SetUpParametersAndDomains();
	}
	void def_SetUpParametersAndDomains(void)
	{
		this->daeDynamicSimulation::SetUpParametersAndDomains();
	}

	void SetUpVariables(void)
	{
        if(boost::python::override f = this->get_override("SetUpVariables"))
            f();
		else
			this->daeDynamicSimulation::SetUpVariables();
	}
	void def_SetUpVariables(void)
	{
		this->daeDynamicSimulation::SetUpVariables();
	}

	void Run(void)
	{
        if(boost::python::override f = this->get_override("Run"))
			f();
 		else
	       return daeDynamicSimulation::Run();
	}
	void def_Run(void)
	{
        this->daeDynamicSimulation::Run();
	}

public:
	boost::python::object model;	
};

}

#endif
