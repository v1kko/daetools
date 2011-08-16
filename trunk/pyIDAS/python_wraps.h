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
#include "../IDAS_DAESolver/ida_solver.h"

namespace daepython
{
/*******************************************************
	daeDAESolver
*******************************************************/
class daeDAESolverWrapper : public daeDAESolver_t,
	                        public boost::python::wrapper<daeDAESolver_t>
{
public:
	daeDAESolverWrapper(void){}

	void Initialize(daeBlock_t* pBlock, daeLog_t* pLog)
	{
		this->get_override("Initialize")(pBlock, pLog);
	}
	
	real_t Solve(real_t dTime, bool bStopAtDiscontinuity)
	{
		return this->get_override("Solve")(dTime, bStopAtDiscontinuity);
	}
	
	daeBlock_t* GetBlock(void) const
	{
		return this->get_override("GetBlock")();
	}
	
	daeLog_t* GetLog(void) const
	{
		return this->get_override("GetLog")();
	}
};


class daeIDASolverWrapper : public daeIDASolver,
	                        public boost::python::wrapper<daeIDASolver>
{
public:
	daeIDASolverWrapper(void)
	{
	}

	void Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeSimulation_t* pSimulation, daeeInitialConditionMode eMode, bool bCalculateSensitivities, boost::python::list l)
	{
		size_t index;
		std::vector<size_t> narrParametersIndexes;
		boost::python::ssize_t n = boost::python::len(l);
		for(boost::python::ssize_t i = 0; i < n; i++) 
		{
			index = boost::python::extract<size_t>(l[i]);
			narrParametersIndexes.push_back(index);
		}
		
		daeIDASolver::Initialize(pBlock, pLog, pSimulation, eMode, bCalculateSensitivities, narrParametersIndexes);
	}

	void SetLASolver1(daeeIDALASolverType eLASolverType)
	{
		daeIDASolver::SetLASolver(eLASolverType);
	}

	void SetLASolver2(daeIDALASolver_t* pLASolver)
	{
		daeIDASolver::SetLASolver(pLASolver);
	}
	

};

}

#endif
