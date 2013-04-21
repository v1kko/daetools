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
    
    
    
    enum_<UpdateFlags>("UpdateFlags")
        .value("update_default",		                update_default)
        .value("update_values",                         update_values) 
        .value("update_gradients",                      update_gradients) 
        .value("update_hessians",                       update_hessians) 
        .value("update_boundary_forms",                 update_boundary_forms) 
        .value("update_quadrature_points",            	update_quadrature_points) 
        .value("update_JxW_values",                     update_JxW_values) 
        .value("update_normal_vectors",                 update_normal_vectors) 
        .value("update_face_normal_vectors",        	update_face_normal_vectors) 
        .value("update_cell_normal_vectors",	        update_cell_normal_vectors) 
        .value("update_jacobians",                      update_jacobians) 
        .value("update_jacobian_grads",                 update_jacobian_grads) 
        .value("update_inverse_jacobians",              update_inverse_jacobians) 
        .value("update_covariant_transformation",       update_covariant_transformation) 
        .value("update_contravariant_transformation",	update_contravariant_transformation) 
        .value("update_transformation_values",          update_transformation_values) 
        .value("update_transformation_gradients",       update_transformation_gradients) 
        .value("update_volume_elements",                update_volume_elements) 
        .value("update_support_points",                 update_support_points) 
        .value("update_support_jacobians",              update_support_jacobians) 
        .value("update_support_inverse_jacobians",      update_support_inverse_jacobians) 
        .value("update_q_points",                       update_q_points) 
        .value("update_second_derivatives",             update_second_derivatives) 
        .value("update_piola",                          update_piola) 
        .export_values()
    ;
/**************************************************************
	daeDEAL_II_t
***************************************************************/
	class_<Triangulation_1D, boost::noncopyable>("Triangulation_1D", no_init)
		;
    class_<Triangulation_2D, boost::noncopyable>("Triangulation_2D", no_init)
		;
    class_<Triangulation_3D, boost::noncopyable>("Triangulation_3D", no_init)
		;
   
    /*
	class_<daeDEAL_II_t, boost::noncopyable>("daeDEAL_II_t")
        .def("setup_system",		pure_virtual(&daeDEAL_II_t::setup_system))
        .def("refine_grid",			pure_virtual(&daeDEAL_II_t::refine_grid))
        .def("assemble_system",		pure_virtual(&daeDEAL_II_t::assemble_system))
        .def("solve",               pure_virtual(&daeDEAL_II_t::solve))
        .def("process_solution",	pure_virtual(&daeDEAL_II_t::process_solution))
		;
    */

    class_<std::vector< Point<1> > >("Point_1D")
        .def(vector_indexing_suite<std::vector< Point<1> > >())
    ;
    class_<std::vector< Point<2> > >("Point_2D")
        .def(vector_indexing_suite<std::vector< Point<2> > >())
    ;
    class_<std::vector< Point<3> > >("Point_3D")
        .def(vector_indexing_suite<std::vector< Point<3 > >())
    ;
    
    class_<Quadrature<1>, boost::noncopyable>("Quadrature_1D", no_init)
        .def("size",		&Quadrature<1>::size)
//        .def("point",		&Quadrature<1>::point)
//        .def("weight",		&Quadrature<1>::weight)
//        .def("get_points",	&Quadrature<1>::get_points)
    ;
    class_<Quadrature<2>, boost::noncopyable>("Quadrature_2D", no_init)
        .def("size",		&Quadrature<2>::size)
    ;
    class_<Quadrature<3>, boost::noncopyable>("Quadrature_3D", no_init)
        .def("size",		&Quadrature<3>::size)
    ;
    
    class_<QGauss<1>, bases< Quadrature<1> >, boost::noncopyable>("QGauss_1D", no_init)
        .def(init<unsigned int>())
    ;
    class_<QGauss<2>, bases< Quadrature<2> >, boost::noncopyable>("QGauss_2D", no_init)
        .def(init<unsigned int>())
    ;
    class_<QGauss<2>, bases< Quadrature<3> >, boost::noncopyable>("QGauss_3D", no_init)
        .def(init<unsigned int>())
    ;
    
    class_<QGaussLobatto<1>, bases< Quadrature<1> >, boost::noncopyable>("QGaussLobatto_1D", no_init)
        .def(init<unsigned int>())
    ;
    class_<QGaussLobatto<2>, bases< Quadrature<2> >, boost::noncopyable>("QGaussLobatto_2D", no_init)
        .def(init<unsigned int>())
    ;
    class_<QGaussLobatto<2>, bases< Quadrature<3> >, boost::noncopyable>("QGaussLobatto_3D", no_init)
        .def(init<unsigned int>())
    ;
    
    class_<FullMatrix<double>, boost::noncopyable>("FullMatrix", no_init)
        .def(init<const unsigned int, const unsigned int>())
        .def("clear",       &clearMatrix)
        .def("__call__",    &FullMatrix<double>::operator()) 
    ;
    
    class_<Vector<double>, boost::noncopyable>("Vector", no_init)
        .def(init<const unsigned int>())
        .def("__call__",    &Vector<double>::operator[])
    ;
    
    class_<FE_Q<1>, boost::noncopyable>("FE_Q_1D", no_init)
        .def(init<unsigned int>())
        //.def(init<const Quadrature<1>&>())
        .def("get_name",                    &FE_Q<1>::get_name)
        .def_readonly("dofs_per_cell",		&FE_Q<1>::dofs_per_cell)
    ;
    class_<FE_Q<2>, boost::noncopyable>("FE_Q_2D", no_init)
        .def(init<unsigned int>())
        .def("size",		&FE_Q<2>::size)
    ;
    class_<FE_Q<3>, boost::noncopyable>("FE_Q_3D", no_init)
        .def(init<unsigned int>())
        .def("size",		&FE_Q<3>::size)
    ;
    
    
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
