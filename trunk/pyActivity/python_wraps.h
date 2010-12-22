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
#include "../Simulation/optimization.h"
#include "../Solver/ida_solver.h"
#include "../Core/base_logging.h"
#include "../Core/tcpiplog.h"

namespace daepython
{
class daeDefaultSimulationWrapper : public daeSimulation,
	                                public boost::python::wrapper<daeSimulation>
{
public:
	daeDefaultSimulationWrapper(void)
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
		this->daeSimulation::SetModel(pModel);
	}

	void SetUpParametersAndDomains(void)
	{
        if(boost::python::override f = this->get_override("SetUpParametersAndDomains"))
			f();
		else
			this->daeSimulation::SetUpParametersAndDomains();
	}
	void def_SetUpParametersAndDomains(void)
	{
		this->daeSimulation::SetUpParametersAndDomains();
	}

	void SetUpVariables(void)
	{
        if(boost::python::override f = this->get_override("SetUpVariables"))
            f();
		else
			this->daeSimulation::SetUpVariables();
	}
	void def_SetUpVariables(void)
	{
		this->daeSimulation::SetUpVariables();
	}

	void Run(void)
	{
        if(boost::python::override f = this->get_override("Run"))
			f();
 		else
	       return daeSimulation::Run();
	}
	void def_Run(void)
	{
        this->daeSimulation::Run();
	}

    void SetUpOptimization(void)
    {
        if(boost::python::override f = this->get_override("SetUpOptimization"))
            f();
        else
            this->daeSimulation::SetUpOptimization();
    }
    void def_SetUpOptimization(void)
    {
        this->daeSimulation::SetUpOptimization();
    }
    
public:
	boost::python::object model;	
};



class daeIPOPTWrapper : public daeIPOPT,
                        public boost::python::wrapper<daeIPOPT>
{
public:
    daeIPOPTWrapper(void)
    {
    }
    
   
};

}

#endif
