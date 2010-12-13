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
#include "../Solver/ida_solver.h"

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

	void Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeeInitialConditionMode eMode)
	{
        if(boost::python::override f = this->get_override("Initialize"))
            f(pBlock, pLog, eMode);
		else
			this->daeIDASolver::Initialize(pBlock, pLog, eMode);
	}
	void def_Initialize(daeBlock_t* pBlock, daeLog_t* pLog, daeeInitialConditionMode eMode)
	{
        this->daeIDASolver::Initialize(pBlock, pLog, eMode);
	}
	
	real_t Solve(real_t dTime, daeeStopCriterion eStop)
	{
        if(boost::python::override f = this->get_override("Solve"))
            return f(dTime, eStop);
		else
			return this->daeIDASolver::Solve(dTime, eStop);
	}
	real_t def_Solve(real_t dTime, daeeStopCriterion eStop)
	{
        return this->daeIDASolver::Solve(dTime, eStop);
	}
	
//	boost::python::tuple GetSparseMatrixData(void)
//	{
//		boost::python::list ia;
//		boost::python::list ja;
//		int i, NNZ;
//		int *IA, *JA;
//		
//		daeIDASolver::GetSparseMatrixData(NNZ, &IA, &JA);
//
//		if(NNZ == 0)
//			return boost::python::make_tuple(0, 0, ia, ja);
//		
//		for(i = 0; i < m_nNumberOfEquations+1; i++)
//			ia.append(IA[i]);
//
//		for(i = 0; i < NNZ; i++)
//			ja.append(JA[i]);
//
//		return boost::python::make_tuple(m_nNumberOfEquations, NNZ, ia, ja);
//	}
	

};

}

#endif
