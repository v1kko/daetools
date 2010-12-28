#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyIPOPT)
{
    import_array(); 
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

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
