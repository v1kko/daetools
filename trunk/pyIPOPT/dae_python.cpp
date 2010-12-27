#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyIPOPT)
{
    import_array(); 
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
 
/**************************************************************
	daeSimulation_t
***************************************************************/
//	class_<daepython::daeSimulationWrapper, boost::noncopyable>("daeSimulation_t", no_init)
//		.def("GetModel",                    pure_virtual(&daeSimulation_t::GetModel), return_internal_reference<>())
//		.def("SetModel",                    pure_virtual(&daeSimulation_t::SetModel))
//		.def("SetUpParametersAndDomains",   pure_virtual(&daeSimulation_t::SetUpParametersAndDomains))
//		.def("SetUpVariables",              pure_virtual(&daeSimulation_t::SetUpVariables))
//		.def("Run",                         pure_virtual(&daeSimulation_t::Run))
//		.def("Finalize",                    pure_virtual(&daeSimulation_t::Finalize))
//		.def("ReRun",                       pure_virtual(&daeSimulation_t::ReRun))
//		.def("ReportData",                  pure_virtual(&daeSimulation_t::ReportData))
//		.def("StoreInitializationValues",   pure_virtual(&daeSimulation_t::StoreInitializationValues))
//		.def("LoadInitializationValues",    pure_virtual(&daeSimulation_t::LoadInitializationValues))

//		.def("Pause",                       pure_virtual(&daeSimulation_t::Pause))
//		.def("Resume",                      pure_virtual(&daeSimulation_t::Resume))
		
//		.def("Initialize",                  pure_virtual(&daeSimulation_t::Initialize))
//		.def("InitializeOptimization",      pure_virtual(&daeSimulation_t::InitializeOptimization))
//		.def("Reinitialize",                pure_virtual(&daeSimulation_t::Reinitialize))
//		.def("SolveInitial",                pure_virtual(&daeSimulation_t::SolveInitial))
//		.def("Integrate",                   pure_virtual(&daeSimulation_t::Integrate))
//		.def("IntegrateForTimeInterval",    pure_virtual(&daeSimulation_t::IntegrateForTimeInterval))
//		.def("IntegrateUntilTime",          pure_virtual(&daeSimulation_t::IntegrateUntilTime))
//	   ;

		      
	class_<daepython::daeNLPSolverWrapper, boost::noncopyable>("daeNLPSolver_t", no_init)
        .def("Initialize",               pure_virtual(&daeNLPSolver_t::Initialize))
        .def("Solve",                    pure_virtual(&daeNLPSolver_t::Solve))
		;
	
    class_<daepython::daeIPOPTWrapper, bases<daeNLPSolver_t>, boost::noncopyable>("daeIPOPT")
		.def("Initialize",	&daeIPOPTSolver::Initialize)
		.def("Solve",	    &daeIPOPTSolver::Solve)
        .def("SetOption",	&daepython::daeIPOPTWrapper::SetOptionS)
        .def("SetOption",	&daepython::daeIPOPTWrapper::SetOptionN)
        .def("SetOption",	&daepython::daeIPOPTWrapper::SetOptionI) 
        ;
     
}
