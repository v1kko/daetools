#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/return_value_policy.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyDealII)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");  
    
    docstring_options doc_options(true, true, false);

    class_< Tensor<1,1,double>, boost::noncopyable>("Tensor_1_1D")
        .add_property("dimension",                  &daepython::Tensor_1_1D_dimension)
        .add_property("rank",                       &daepython::Tensor_1_1D_rank)
        .add_property("n_independent_components",   &daepython::Tensor_1_1D_n_independent_components)

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)

        .def("__getitem__",         &daepython::Tensor_1_1D_getitem)
        .def("__setitem__",         &daepython::Tensor_1_1D_setitem)
        .def("__str__",             &daepython::Tensor_1_1D_str)
        .def("__repr__",            &daepython::Tensor_1_1D_repr)

        .def("norm",                &Tensor_1_1D::norm)
        .def("norm_square",         &Tensor_1_1D::norm_square)
        .def("clear",               &Tensor_1_1D::clear)
        .def("memory_consumption",  &Tensor_1_1D::memory_consumption)
    ;

    class_< Tensor<1,2,double>, boost::noncopyable>("Tensor_1_2D")
        .add_property("dimension",                  &daepython::Tensor_1_2D_dimension)
        .add_property("rank",                       &daepython::Tensor_1_2D_rank)
        .add_property("n_independent_components",   &daepython::Tensor_1_2D_n_independent_components)

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)

        .def("__getitem__",         &daepython::Tensor_1_2D_getitem)
        .def("__setitem__",         &daepython::Tensor_1_2D_setitem)
        .def("__str__",             &daepython::Tensor_1_2D_str)
        .def("__repr__",            &daepython::Tensor_1_2D_repr)

        .def("norm",                &Tensor_1_2D::norm)
        .def("norm_square",         &Tensor_1_2D::norm_square)
        .def("clear",               &Tensor_1_2D::clear)
        .def("memory_consumption",  &Tensor_1_2D::memory_consumption)
    ;

    class_< Tensor<1,3,double>, boost::noncopyable>("Tensor_1_3D")
        .add_property("dimension",                  &daepython::Tensor_1_3D_dimension)
        .add_property("rank",                       &daepython::Tensor_1_3D_rank)
        .add_property("n_independent_components",   &daepython::Tensor_1_3D_n_independent_components)

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)

        .def("__getitem__",         &daepython::Tensor_1_3D_getitem)
        .def("__setitem__",         &daepython::Tensor_1_3D_setitem)
        .def("__str__",             &daepython::Tensor_1_3D_str)
        .def("__repr__",            &daepython::Tensor_1_3D_repr)

        .def("norm",                &Tensor_1_3D::norm)
        .def("norm_square",         &Tensor_1_3D::norm_square)
        .def("clear",               &Tensor_1_3D::clear)
        .def("memory_consumption",  &Tensor_1_3D::memory_consumption)
    ;

    class_<Point<1,double>, bases<Tensor_1_1D>, boost::noncopyable>("Point_1D")
        .def(init<double>())

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())

        .def("__repr__",   &daepython::Point_1D_repr)

        .def("distance",   &Point_1D::distance)
        .def("square",     &Point_1D::square)

        .add_property("x", &daepython::Point_1D_x)
    ;

    class_<Point<2,double>, bases<Tensor_1_2D>, boost::noncopyable>("Point_2D")
        .def(init<double, double>())

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())

        .def("__repr__",   &daepython::Point_2D_repr)

        .def("distance",   &Point_2D::distance)
        .def("square",     &Point_2D::square)

        .add_property("x", &daepython::Point_2D_x)
        .add_property("y", &daepython::Point_2D_y)
    ;

    class_<Point<3,double>, bases<Tensor_1_3D>, boost::noncopyable>("Point_3D")
        .def(init<double, double, double>())

        .def(self == self)
        .def(self != self)
        .def(self += self)
        .def(self -= self)
        .def(self *= double())
        .def(self /= double())
        .def(- self)
        .def(self + self)
        .def(self - self)
        .def(self * self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())

        .def("__repr__",   &daepython::Point_3D_repr)

        .def("distance",   &Point_3D::distance)
        .def("square",     &Point_3D::square)

        .add_property("x", &daepython::Point_3D_x)
        .add_property("y", &daepython::Point_3D_y)
        .add_property("z", &daepython::Point_3D_z)
    ;

//    class_< std::map< unsigned int, dealiiFunction<1> > >("map_Uint_Function_1D")
//        .def(map_indexing_suite< std::map< unsigned int, dealiiFunction<1> > >())
//    ;
//    class_< std::map< unsigned int, dealiiFunction<2> > >("map_Uint_Function_2D")
//        .def(map_indexing_suite< std::map< unsigned int, dealiiFunction<2> > >())
//    ;
//    class_< std::map< unsigned int, dealiiFunction<3> > >("map_Uint_Function_3D")
//        .def(map_indexing_suite< std::map< unsigned int, dealiiFunction<3> > >())
//    ;

    class_<daepython::Function_wrapper<1>, boost::noncopyable>("Function_1D", no_init)
        .def(init<unsigned int>((arg("self"), arg("n_components") = 1)))

        .add_property("dimension",      &daepython::Function_wrapper_1D::dimension)
        .add_property("n_components",   &daepython::Function_wrapper_1D::n_components)

        .def("value",           pure_virtual(&daepython::Function_wrapper_1D::value),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_value",    pure_virtual(&daepython::Function_wrapper_1D::vector_value),
                                ( arg("self"), arg("point") ) )

        .def("gradient",        pure_virtual(&daepython::Function_wrapper_1D::gradient),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_gradient",	pure_virtual(&daepython::Function_wrapper_1D::vector_gradient),
                                ( arg("self"), arg("point") ) )
    ;

    class_<daepython::Function_wrapper<2>, boost::noncopyable>("Function_2D", no_init)
        .def(init<unsigned int>((arg("self"), arg("n_components") = 1)))

        .add_property("dimension",      &daepython::Function_wrapper_2D::dimension)
        .add_property("n_components",   &daepython::Function_wrapper_2D::n_components)

        .def("value",           pure_virtual(&daepython::Function_wrapper_2D::value),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_value",    pure_virtual(&daepython::Function_wrapper_2D::vector_value),
                                ( arg("self"), arg("point") ) )

        .def("gradient",        pure_virtual(&daepython::Function_wrapper_2D::gradient),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_gradient",	pure_virtual(&daepython::Function_wrapper_2D::vector_gradient),
                                ( arg("self"), arg("point") ) )
    ;

    class_<daepython::Function_wrapper<3>, boost::noncopyable>("Function_3D", no_init)
        .def(init<unsigned int>((arg("self"), arg("n_components") = 1)))

        .add_property("dimension",      &daepython::Function_wrapper_3D::dimension)
        .add_property("n_components",   &daepython::Function_wrapper_3D::n_components)

        .def("value",           pure_virtual(&daepython::Function_wrapper_3D::value),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_value",    pure_virtual(&daepython::Function_wrapper_3D::vector_value),
                                ( arg("self"), arg("point") ) )

        .def("gradient",        pure_virtual(&daepython::Function_wrapper_3D::gradient),
                                ( arg("self"), arg("point"), arg("component") = 0 ) )
        .def("vector_gradient",	pure_virtual(&daepython::Function_wrapper_3D::vector_gradient),
                                ( arg("self"), arg("point") ) )
    ;

    class_<daeDealIIDataReporter, boost::noncopyable>("daeDealIIDataReporter", no_init)
        .def("Connect",				&daeDealIIDataReporter::Connect)
        .def("Disconnect",			&daeDealIIDataReporter::Disconnect)
        .def("IsConnected",			&daeDealIIDataReporter::IsConnected)
        .def("StartRegistration",	&daeDealIIDataReporter::StartRegistration)
        .def("RegisterDomain",		&daeDealIIDataReporter::RegisterDomain)
        .def("RegisterVariable",	&daeDealIIDataReporter::RegisterVariable)
        .def("EndRegistration",		&daeDealIIDataReporter::EndRegistration)
        .def("StartNewResultSet",	&daeDealIIDataReporter::StartNewResultSet)
        .def("EndOfData",	    	&daeDealIIDataReporter::EndOfData)
        .def("SendVariable",	  	&daeDealIIDataReporter::SendVariable)
    ;



    class_<dealiiCellIterator<1>, boost::noncopyable>("dealiiCellIterator_1D", no_init)
    ;
    class_<dealiiCellIterator<2>, boost::noncopyable>("dealiiCellIterator_2D", no_init)
    ;
    class_<dealiiCellIterator<3>, boost::noncopyable>("dealiiCellIterator_3D", no_init)
    ;

    class_< std::vector<unsigned int> >("vector_uint")
        .def(vector_indexing_suite< std::vector<unsigned int> >())
    ;
    class_< std::vector<double> >("vector_double")
        .def(vector_indexing_suite< std::vector<double> >())
    ;

    class_< std::vector< Point<1,double> > >("vector_Point_1D")
        .def(vector_indexing_suite< std::vector< Point<1,double> > >())
    ;
    class_< std::vector< Point<2,double> > >("vector_Point_2D")
        .def(vector_indexing_suite< std::vector< Point<2,double> > >())
    ;
    class_< std::vector< Point<3,double> > >("vector_Point_3D")
        .def(vector_indexing_suite< std::vector< Point<3,double> > >())
    ;

//    class_< std::vector< const Point<1,double> > >("vector_constPoint_1D")
//        .def(vector_indexing_suite< std::vector< const Point<1,double> > >())
//    ;
//    class_< std::vector< const Point<2,double> > >("vector_constPoint_2D")
//        .def(vector_indexing_suite< std::vector< const Point<2,double> > >())
//    ;
//    class_< std::vector< const Point<3,double> > >("vector_constPoint_3D")
//        .def(vector_indexing_suite< std::vector< const Point<3,double> > >())
//    ;

    class_<FEValuesBase<1>, boost::noncopyable>("FEValuesBase_1D", no_init)
        .def("shape",                   &FEValuesBase<1>::shape_value,              return_value_policy<copy_const_reference>())
        .def("shape_value_component",	&FEValuesBase<1>::shape_value_component)
        .def("shape_grad",              &FEValuesBase<1>::shape_grad,               return_internal_reference<>())
        .def("shape_grad_component",	&FEValuesBase<1>::shape_grad_component)
        .def("shape_hessian",           &FEValuesBase<1>::shape_hessian,            return_internal_reference<>())
        .def("shape_hessian_component",	&FEValuesBase<1>::shape_hessian_component)
        .def("get_quadrature_points",	&FEValuesBase<1>::get_quadrature_points,    return_internal_reference<>())
        .def("get_JxW_values",          &FEValuesBase<1>::get_JxW_values,           return_internal_reference<>())
        .def("quadrature_point",        &FEValuesBase<1>::quadrature_point,         return_internal_reference<>())
        .def("JxW",                     &FEValuesBase<1>::JxW)
        .def("normal_vector",           &FEValuesBase<1>::normal_vector,            return_internal_reference<>())
        .def("cell_normal_vector",      &FEValuesBase<1>::cell_normal_vector,       return_internal_reference<>())
    ;

    class_<FEValuesBase<2>, boost::noncopyable>("FEValuesBase_2D", no_init)
        .def("shape",                   &FEValuesBase<2>::shape_value,              return_value_policy<copy_const_reference>())
        .def("shape_value_component",	&FEValuesBase<2>::shape_value_component)
        .def("shape_grad",              &FEValuesBase<2>::shape_grad,               return_internal_reference<>())
        .def("shape_grad_component",	&FEValuesBase<2>::shape_grad_component)
        .def("shape_hessian",           &FEValuesBase<2>::shape_hessian,            return_internal_reference<>())
        .def("shape_hessian_component",	&FEValuesBase<2>::shape_hessian_component)
        .def("get_quadrature_points",	&FEValuesBase<2>::get_quadrature_points,    return_internal_reference<>())
        .def("get_JxW_values",          &FEValuesBase<2>::get_JxW_values,           return_internal_reference<>())
        .def("quadrature_point",        &FEValuesBase<2>::quadrature_point,         return_internal_reference<>())
        .def("JxW",                     &FEValuesBase<2>::JxW)
        .def("normal_vector",           &FEValuesBase<2>::normal_vector,            return_internal_reference<>())
        .def("cell_normal_vector",      &FEValuesBase<2>::cell_normal_vector,       return_internal_reference<>())
    ;

    class_<FEValuesBase<3>, boost::noncopyable>("FEValuesBase_3D", no_init)
        .def("shape",                   &FEValuesBase<3>::shape_value,              return_value_policy<copy_const_reference>())
        .def("shape_value_component",	&FEValuesBase<3>::shape_value_component)
        .def("shape_grad",              &FEValuesBase<3>::shape_grad,               return_internal_reference<>())
        .def("shape_grad_component",	&FEValuesBase<3>::shape_grad_component)
        .def("shape_hessian",           &FEValuesBase<3>::shape_hessian,            return_internal_reference<>())
        .def("shape_hessian_component",	&FEValuesBase<3>::shape_hessian_component)
        .def("get_quadrature_points",	&FEValuesBase<3>::get_quadrature_points,    return_internal_reference<>())
        .def("get_JxW_values",          &FEValuesBase<3>::get_JxW_values,           return_internal_reference<>())
        .def("quadrature_point",        &FEValuesBase<3>::quadrature_point,         return_internal_reference<>())
        .def("JxW",                     &FEValuesBase<3>::JxW)
        .def("normal_vector",           &FEValuesBase<3>::normal_vector,            return_internal_reference<>())
        .def("cell_normal_vector",      &FEValuesBase<3>::cell_normal_vector,       return_internal_reference<>())
    ;


    class_<FEValues<1>, bases< FEValuesBase<1> >, boost::noncopyable>("FEValues_1D", no_init)
    ;
    class_<FEValues<2>, bases< FEValuesBase<2> >, boost::noncopyable>("FEValues_2D", no_init)
    ;
    class_<FEValues<3>, bases< FEValuesBase<3> >, boost::noncopyable>("FEValues_3D", no_init)
    ;

    class_<FEFaceValues<1>, bases< FEValuesBase<1> >, boost::noncopyable>("FEFaceValues_1D", no_init)
    ;
    class_<FEFaceValues<2>, bases< FEValuesBase<2> >, boost::noncopyable>("FEFaceValues_2D", no_init)
    ;
    class_<FEFaceValues<3>, bases< FEValuesBase<3> >, boost::noncopyable>("FEFaceValues_3D", no_init)
    ;

    class_<dealiiCell<1>, boost::noncopyable>("dealiiCell_1D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_1D::dofs_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_1D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_1D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_1D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_1D::fe_values)
        .def_readonly("fe_face_values",     &dealiiCell_1D::fe_face_values)
    ;

    class_<dealiiCell<2>, boost::noncopyable>("dealiiCell_2D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_2D::dofs_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_2D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_2D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_2D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_2D::fe_values)
        .def_readonly("fe_face_values",     &dealiiCell_2D::fe_face_values)
    ;

    class_<dealiiCell<3>, boost::noncopyable>("dealiiCell_3D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_3D::dofs_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_3D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_3D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_3D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_3D::fe_values)
        .def_readonly("fe_face_values",     &dealiiCell_3D::fe_face_values)
    ;

    class_<daeConvectionDiffusion_1D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_1D", no_init)
        .def("__init__",         make_constructor(daepython::daeConvectionDiffusion__init__<1>,
                                                  default_call_policies(),
                                                  (  arg("name"),
                                                     arg("parentModel"),
                                                     arg("description"),
                                                     arg("meshFilename"),
                                                     arg("quadratureFormula"),
                                                     arg("polynomialOrder"),
                                                     arg("outputDirectory"),
                                                     arg("diffusivity"),
                                                     arg("velocity"),
                                                     arg("generation"),
                                                     arg("dirichletBC"),
                                                     arg("neumannBC")
                                                  )))

        .add_property("DataOut",        make_function(&daeConvectionDiffusion_1D::GetDataOut, return_internal_reference<>()) )

        .def("__iter__",                boost::python::iterator<daeConvectionDiffusion_1D, return_value_policy<reference_existing_object> >())
        .def("AssembleSystem",          &daeConvectionDiffusion_1D::AssembleSystem)
        .def("GenerateEquations",       &daeConvectionDiffusion_1D::GenerateEquations)

//        .add_property("DirichletBC",    &daeConvectionDiffusion_1D::GetDirichletBC)
//        .add_property("NeumannBC",      &daeConvectionDiffusion_1D::GetNeumannBC)

//        .def_readonly("Diffusivity",    &daeConvectionDiffusion_1D::m_Diffusivity)
//        .def_readonly("Velocity",       &daeConvectionDiffusion_1D::m_Velocity)
//        .def_readonly("Generation",     &daeConvectionDiffusion_1D::m_Generation)

//        .add_property("Diffusivity",    &daeConvectionDiffusion_1D::GetDiffusivity)
//        .add_property("Velocity",       &daeConvectionDiffusion_1D::GetVelocity)
//        .add_property("Generation",     &daeConvectionDiffusion_1D::GetGeneration)
    ;

    class_<daeConvectionDiffusion_2D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_2D", no_init)
        .def("__init__",         make_constructor(daepython::daeConvectionDiffusion__init__<2>,
                                                  default_call_policies(),
                                                  (  arg("name"),
                                                     arg("parentModel"),
                                                     arg("description"),
                                                     arg("meshFilename"),
                                                     arg("quadratureFormula"),
                                                     arg("polynomialOrder"),
                                                     arg("outputDirectory"),
                                                     arg("diffusivity"),
                                                     arg("velocity"),
                                                     arg("generation"),
                                                     arg("dirichletBC"),
                                                     arg("neumannBC")
                                                  )))

        .add_property("DataOut", make_function(&daeConvectionDiffusion_2D::GetDataOut, return_internal_reference<>()) )

        .def("__iter__",            boost::python::iterator<daeConvectionDiffusion_2D, return_value_policy<reference_existing_object> >())
        .def("AssembleSystem",      &daeConvectionDiffusion_2D::AssembleSystem)
        .def("GenerateEquations",   &daeConvectionDiffusion_2D::GenerateEquations)
    ;
    
    class_<daeConvectionDiffusion_3D, bases<daeModel>, boost::noncopyable>("daeConvectionDiffusion_3D", no_init)
        .def("__init__",         make_constructor(daepython::daeConvectionDiffusion__init__<3>,
                                                  default_call_policies(),
                                                  (  arg("name"),
                                                     arg("parentModel"),
                                                     arg("description"),
                                                     arg("meshFilename"),
                                                     arg("quadratureFormula"),
                                                     arg("polynomialOrder"),
                                                     arg("outputDirectory"),
                                                     arg("diffusivity"),
                                                     arg("velocity"),
                                                     arg("generation"),
                                                     arg("dirichletBC"),
                                                     arg("neumannBC")
                                                  )))

        .add_property("DataOut", make_function(&daeConvectionDiffusion_3D::GetDataOut, return_internal_reference<>()) )

        .def("__iter__",            boost::python::iterator<daeConvectionDiffusion_3D, return_value_policy<reference_existing_object> >())
        .def("AssembleSystem",      &daeConvectionDiffusion_3D::AssembleSystem)
        .def("GenerateEquations",   &daeConvectionDiffusion_3D::GenerateEquations)
        //.def("elements",          range< return_value_policy<reference_existing_object> >(&daeConvectionDiffusion_3D::begin, &daeConvectionDiffusion_3D::end))
    ;
}
