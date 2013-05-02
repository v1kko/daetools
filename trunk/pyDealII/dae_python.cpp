#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyDealII)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");  
    
    docstring_options doc_options(true, true, false);

/**************************************************************
    Enums
***************************************************************/
	enum_<RefinementMode>("RefinementMode")
		.value("global_refinement",		global_refinement)
        .value("adaptive_refinement",	adaptive_refinement) 
		.export_values()
	;
    
/**************************************************************
	daeDEAL_II_t
***************************************************************/
//	class_<daeDEAL_II_t, boost::noncopyable>("daeDEAL_II_t")
//        .def("setup_system",		pure_virtual(&daeDEAL_II_t::setup_system))
//        .def("refine_grid",			pure_virtual(&daeDEAL_II_t::refine_grid))
//        .def("assemble_system",		pure_virtual(&daeDEAL_II_t::assemble_system))
//        .def("solve",               pure_virtual(&daeDEAL_II_t::solve))
//        .def("process_solution",	pure_virtual(&daeDEAL_II_t::process_solution))
//		;

    class_<daeDEAL_II_1D, boost::noncopyable>("daeDEAL_II_1D", no_init)
        .def(init<unsigned int, RefinementMode>())
        .def("setup_system",		&daeDEAL_II_1D::setup_system,       &daeDEAL_II_1D::def_setup_system)
        .def("refine_grid",			&daeDEAL_II_1D::refine_grid,        &daeDEAL_II_1D::def_refine_grid)
        .def("assemble_system",		&daeDEAL_II_1D::assemble_system,    &daeDEAL_II_1D::def_assemble_system)
        .def("solve",               &daeDEAL_II_1D::solve,              &daeDEAL_II_1D::def_solve)
        .def("process_solution",	&daeDEAL_II_1D::process_solution,   &daeDEAL_II_1D::def_process_solution)
        .def("run",             	&daeDEAL_II_1D::run)
        ;
    
    class_<daeDEAL_II_2D, boost::noncopyable>("daeDEAL_II_2D", no_init)
        .def(init<unsigned int, RefinementMode>())
        .def("setup_system",		&daeDEAL_II_2D::setup_system,       &daeDEAL_II_2D::def_setup_system)
        .def("refine_grid",			&daeDEAL_II_2D::refine_grid,        &daeDEAL_II_2D::def_refine_grid)
        .def("assemble_system",		&daeDEAL_II_2D::assemble_system,    &daeDEAL_II_2D::def_assemble_system)
        .def("solve",               &daeDEAL_II_2D::solve,              &daeDEAL_II_2D::def_solve)
        .def("process_solution",	&daeDEAL_II_2D::process_solution,   &daeDEAL_II_2D::def_process_solution)
        .def("run",             	&daeDEAL_II_2D::run)
        ;
    
    class_<daeDEAL_II_3D, boost::noncopyable>("daeDEAL_II_3D", no_init)
        .def(init<unsigned int, RefinementMode>())
        .def("setup_system",		&daeDEAL_II_3D::setup_system,       &daeDEAL_II_3D::def_setup_system)
        .def("refine_grid",			&daeDEAL_II_3D::refine_grid,        &daeDEAL_II_3D::def_refine_grid)
        .def("assemble_system",		&daeDEAL_II_3D::assemble_system,    &daeDEAL_II_3D::def_assemble_system)
        .def("solve",               &daeDEAL_II_3D::solve,              &daeDEAL_II_3D::def_solve)
        .def("process_solution",	&daeDEAL_II_3D::process_solution,   &daeDEAL_II_3D::def_process_solution)
        .def("run",             	&daeDEAL_II_3D::run)
        ;
}
