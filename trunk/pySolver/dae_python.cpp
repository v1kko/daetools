#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pySolver)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  
/**************************************************************
    Enums
***************************************************************/
//	enum_<daeeIDALASolverType>("daeeIDALASolverType")
//		.value("eSundialsLU",		dae::solver::eSundialsLU)
//		.value("eSundialsGMRES",	dae::solver::eSundialsGMRES)
//		.value("eThirdParty",		dae::solver::eThirdParty)
//		.export_values()
//	;

/**************************************************************
	daeSolver
***************************************************************/
	class_<daepython::daeDAESolverWrapper, boost::noncopyable>("daeDAESolver_t", no_init)
		.add_property("Log",	make_function(&daeDAESolver_t::GetLog,   return_internal_reference<>()))
		//.add_property("Block",	make_function(&daeDAESolver_t::GetBlock, return_internal_reference<>()))
		.add_property("RelativeTolerance",	  &daeDAESolver_t::GetRelativeTolerance,     &daeDAESolver_t::SetRelativeTolerance)
		.add_property("InitialConditionMode", &daeDAESolver_t::GetInitialConditionMode,  &daeDAESolver_t::SetInitialConditionMode)
 
		.def("Initialize",	pure_virtual(&daeDAESolver_t::Initialize))
		.def("Solve",		pure_virtual(&daeDAESolver_t::Solve))
		;
 
	class_<daepython::daeIDASolverWrapper, bases<daeDAESolver_t>, boost::noncopyable>("daeIDASolver")
		.def("Initialize",					&daeDAESolver_t::Initialize, &daepython::daeIDASolverWrapper::def_Initialize)
		.def("Solve",						&daeDAESolver_t::Solve,		 &daepython::daeIDASolverWrapper::def_Solve)
		.def("SetLASolver",					&daeIDASolver::SetLASolver) 
		.def("SaveMatrixAsXPM",				&daeIDASolver::SaveMatrixAsXPM)
		.def("SaveMatrixAsPBM",				&daeIDASolver::SaveMatrixAsPBM) 
		;

}
