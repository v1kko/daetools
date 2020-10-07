#include "stdafx.h"
#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python/docstring_options.hpp>
#include "docstrings.h"
//#include <noprefix.h>
using namespace boost::python;

#ifdef daeBONMIN	
BOOST_PYTHON_MODULE(pyBONMIN)
#endif
#ifdef daeIPOPT
BOOST_PYTHON_MODULE(pyIPOPT)
#endif
{
    //import_array();
    //boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    
    docstring_options doc_options(true, true, false);

	class_<daepython::daeNLPSolverWrapper, boost::noncopyable>("daeNLPSolver_t", no_init)
        .def("Initialize",               pure_virtual(&daeNLPSolver_t::Initialize))
        .def("Solve",                    pure_virtual(&daeNLPSolver_t::Solve))
		;
	
#ifdef daeBONMIN	
    class_<daeBONMINSolver, bases<daeNLPSolver_t>, boost::noncopyable>("daeBONMIN", DOCSTR_daeBONMIN)
#endif
#ifdef daeIPOPT
    class_<daeBONMINSolver, bases<daeNLPSolver_t>, boost::noncopyable>("daeIPOPT", DOCSTR_daeBONMIN)
#endif
        .def(init<>(( boost::python::arg("self") ), DOCSTR_daeBONMIN_init))
        
        .add_property("Name",		&daeBONMINSolver::GetName, DOCSTR_daeBONMIN_Name)
        .def("Initialize",			&daeBONMINSolver::Initialize, ( boost::python::arg("self"),
                                                                   boost::python::arg("simulation"),
                                                                   boost::python::arg("daeSolver"),
                                                                   boost::python::arg("dataReporter"),
                                                                   boost::python::arg("log")
                                                                 ), DOCSTR_daeBONMIN_Initialize)
        .def("Solve",				&daeBONMINSolver::Solve, ( boost::python::arg("self") ), DOCSTR_daeBONMIN_Solve)

        .def("SetOption",			&daepython::SetOptionS, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_daeBONMIN_SetOptionS)
        .def("SetOption",			&daepython::SetOptionF, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_daeBONMIN_SetOptionN)
        .def("SetOption",			&daepython::SetOptionI, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_daeBONMIN_SetOptionI)
		
        .def("ClearOptions",		&daeBONMINSolver::ClearOptions,         ( boost::python::arg("self") ), DOCSTR_daeBONMIN_ClearOptions)
        .def("PrintOptions",		&daeBONMINSolver::PrintOptions,         ( boost::python::arg("self") ), DOCSTR_daeBONMIN_PrintOptions)
        .def("PrintUserOptions",	&daeBONMINSolver::PrintUserOptions,     ( boost::python::arg("self") ), DOCSTR_daeBONMIN_PrintUserOptions)
        .def("LoadOptionsFile",		&daeBONMINSolver::LoadOptionsFile,      ( boost::python::arg("self"), boost::python::arg("optionsFilename") ), DOCSTR_daeBONMIN_LoadOptionsFile)
        .def("SetOpenBLASNoThreads",&daeBONMINSolver::SetOpenBLASNoThreads, ( boost::python::arg("self"), boost::python::arg("noThreads") ))
        ;
}


