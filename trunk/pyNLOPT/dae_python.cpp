#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyNLOPT)
{
    import_array();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<daepython::daeNLPSolverWrapper, boost::noncopyable>("daeNLPSolver_t", no_init)
        .def("Initialize",               pure_virtual(&daeNLPSolver_t::Initialize))
        .def("Solve",                    pure_virtual(&daeNLPSolver_t::Solve))
		;
	
    class_<daepython::daeNLOPTWrapper, bases<daeNLPSolver_t>, boost::noncopyable>("daeNLOPT")
		.def("Initialize",	 &daeNLOPTSolver::Initialize)
		.def("Solve",	     &daeNLOPTSolver::Solve)
		
		.def("SetAlgorithm", &daepython::daeNLOPTWrapper::SetAlgorithm1) 
		.def("SetAlgorithm", &daepython::daeNLOPTWrapper::SetAlgorithm2) 
		.def("GetAlgorithm", &daeNLOPTSolver::GetAlgorithm) 
		
//        .def("SetOption",	&daepython::daeNLOPTWrapper::SetOptionS)
//        .def("SetOption",	&daepython::daeNLOPTWrapper::SetOptionN)
//        .def("SetOption",	&daepython::daeNLOPTWrapper::SetOptionI) 
//		
//        .def("ClearOptions",		&daeNLOPTSolver::ClearOptions) 
//        .def("PrintOptions",		&daeNLOPTSolver::PrintOptions) 
//        .def("PrintUserOptions",	&daeNLOPTSolver::PrintUserOptions) 
//        .def("LoadOptionsFile",	&daeNLOPTSolver::LoadOptionsFile) 
        ; 
}


