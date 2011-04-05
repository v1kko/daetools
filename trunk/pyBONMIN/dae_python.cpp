#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

#ifdef daeBONMIN	
BOOST_PYTHON_MODULE(pyBONMIN)
#endif
#ifdef daeIPOPT
BOOST_PYTHON_MODULE(pyIPOPT)
#endif
{
    import_array();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

	class_<daepython::daeNLPSolverWrapper, boost::noncopyable>("daeNLPSolver_t", no_init)
        .def("Initialize",               pure_virtual(&daeNLPSolver_t::Initialize))
        .def("Solve",                    pure_virtual(&daeNLPSolver_t::Solve))
		;
	
#ifdef daeBONMIN	
    class_<daepython::daeBONMINWrapper, bases<daeNLPSolver_t>, boost::noncopyable>("daeBONMIN")
#endif
#ifdef daeIPOPT
	class_<daepython::daeBONMINWrapper, bases<daeNLPSolver_t>, boost::noncopyable>("daeIPOPT")
#endif
		.add_property("Name",		&daeBONMINSolver::GetName)
		.def("Initialize",			&daeBONMINSolver::Initialize)
		.def("Solve",				&daeBONMINSolver::Solve)
		
        .def("SetOption",			&daepython::daeBONMINWrapper::SetOptionS)
        .def("SetOption",			&daepython::daeBONMINWrapper::SetOptionN)
        .def("SetOption",			&daepython::daeBONMINWrapper::SetOptionI) 
		
        .def("ClearOptions",		&daeBONMINSolver::ClearOptions) 
        .def("PrintOptions",		&daeBONMINSolver::PrintOptions) 
        .def("PrintUserOptions",	&daeBONMINSolver::PrintUserOptions) 
        .def("LoadOptionsFile",		&daeBONMINSolver::LoadOptionsFile) 
        ; 
}


