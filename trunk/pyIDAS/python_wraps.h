#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "../dae_develop.h"
#include "../IDAS_DAESolver/ida_solver.h"

namespace daepython
{
template<typename ITEM>
boost::python::list getListFromVectorByValue(std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(arrItems[i]);

    return l;
}

real_t daeArray_GetItem(daeArray<real_t>& self, size_t index);
boost::python::list daeArray_GetValues(daeArray<real_t>& self);

real_t daeDenseMatrix_GetItem(daeDenseMatrix& self, size_t i, size_t j);

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
    
    void OnCalculateResiduals() 
    {
        this->get_override("OnCalculateResiduals")();
    }
    
    void OnCalculateConditions() 
    {
        this->get_override("OnCalculateConditions")();
	}
    
    void OnCalculateJacobian() 
    {
        this->get_override("OnCalculateJacobian")();
	}
    
    void OnCalculateSensitivityResiduals()
    {
        this->get_override("OnCalculateSensitivityResiduals")();
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

    daeIDALASolver_t* GetLASolver(void)
    {
        return daeIDASolver::GetLASolver();
    }

    void SetLASolver1(daeeIDALASolverType eLASolverType)
	{
		daeIDASolver::SetLASolver(eLASolverType);
	}

	void SetLASolver2(daeIDALASolver_t* pLASolver)
	{
		daeIDASolver::SetLASolver(pLASolver);
	}
	
    void OnCalculateResiduals()
    {
        if(boost::python::override f = this->get_override("OnCalculateResiduals"))
            f();
		else
			this->daeIDASolver::OnCalculateResiduals();
	}
	void def_OnCalculateResiduals()
	{
        this->daeIDASolver::OnCalculateResiduals();
	}

    void OnCalculateConditions() 
    {
        if(boost::python::override f = this->get_override("OnCalculateConditions"))
            f();
		else
			this->daeIDASolver::OnCalculateConditions();
	}
	void def_OnCalculateConditions()
	{
        this->daeIDASolver::OnCalculateConditions();
	}
    
    void OnCalculateJacobian() 
    {
        if(boost::python::override f = this->get_override("OnCalculateJacobian"))
            f();
		else
			this->daeIDASolver::OnCalculateJacobian();
	}
	void def_OnCalculateJacobian()
	{
        this->daeIDASolver::OnCalculateJacobian();
	}
    
    void OnCalculateSensitivityResiduals()
    {
        if(boost::python::override f = this->get_override("OnCalculateSensitivityResiduals"))
            f();
		else
			this->daeIDASolver::OnCalculateSensitivityResiduals();
	}
	void def_OnCalculateSensitivityResiduals()
	{
        this->daeIDASolver::OnCalculateSensitivityResiduals();
	}

    boost::python::list GetEstLocalErrors_()
    {
        std::vector<real_t> arr = daeIDASolver::GetEstLocalErrors();
        return getListFromVectorByValue(arr);
    }

    boost::python::list GetErrWeights_()
    {
        std::vector<real_t> arr = daeIDASolver::GetErrWeights();
        return getListFromVectorByValue(arr);
    }
};

}

#endif
