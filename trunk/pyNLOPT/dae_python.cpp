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
	
    class_<daepython::daeNLOPTWrapper, bases<daeNLPSolver_t>, boost::noncopyable>("daeNLOPT", init<string>())
		.add_property("Name",		&daeNLOPTSolver::GetName)
		.add_property("xtol_rel",	&daeNLOPTSolver::get_xtol_rel, &daeNLOPTSolver::set_xtol_rel)
		.add_property("xtol_abs",	&daeNLOPTSolver::get_xtol_abs, &daeNLOPTSolver::set_xtol_abs)
		.add_property("ftol_rel",	&daeNLOPTSolver::get_ftol_rel, &daeNLOPTSolver::set_ftol_rel)
		.add_property("ftol_abs",	&daeNLOPTSolver::get_ftol_abs, &daeNLOPTSolver::set_ftol_abs)
		.def("Initialize",			&daeNLOPTSolver::Initialize)
		.def("Solve",				&daeNLOPTSolver::Solve)
        .def("PrintOptions",		&daeNLOPTSolver::PrintOptions) 
        ; 
}

