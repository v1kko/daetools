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

    class_<Point<1,double>, bases<Tensor_1_1D> >("Point_1D")
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

    class_<Point<2,double>, bases<Tensor_1_2D> >("Point_2D")
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

    class_<Point<3,double>, bases<Tensor_1_3D> >("Point_3D")
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

    class_< std::vector< Point<1,double> > >("vector_Point_1D")
        .def(vector_indexing_suite< std::vector< Point<1,double> > >())
    ;
    class_< std::vector< Point<2,double> > >("vector_Point_2D")
        .def(vector_indexing_suite< std::vector< Point<2,double> > >())
    ;
    class_< std::vector< Point<3,double> > >("vector_Point_3D")
        .def(vector_indexing_suite< std::vector< Point<3,double> > >())
    ;

    class_<Vector<double>, boost::noncopyable>("Vector", no_init)
        .def("__call__",    &daepython::Vector_getitem)
        .def("__getitem__", &daepython::Vector_getitem)
        .def("__setitem__", &daepython::Vector_set)
        .def("add",         &daepython::Vector_add)
    ;
    class_<FullMatrix<double>, boost::noncopyable>("FullMatrix", no_init)
        .def("__call__",    &daepython::FullMatrix_getitem)
        .def("set",         &daepython::FullMatrix_set)
        .def("add",         &daepython::FullMatrix_add)
    ;
    class_<SparseMatrix<double>, boost::noncopyable>("SparseMatrix", no_init)
        .add_property("n",                  &SparseMatrix<double>::n)
        .add_property("m",                  &SparseMatrix<double>::m)
        .add_property("n_nonzero_elements", &SparseMatrix<double>::n_nonzero_elements)
        .def("__call__",                    &daepython::SparseMatrix_getitem)
        .def("el",                          &daepython::SparseMatrix_el)
        .def("set",                         &daepython::SparseMatrix_set)
        .def("add",                         &daepython::SparseMatrix_add)
    ;

    class_<Quadrature<0>, boost::noncopyable>("Quadrature_0D", no_init)
    ;
    class_<Quadrature<1>, boost::noncopyable>("Quadrature_1D", no_init)
    ;
    class_<Quadrature<2>, boost::noncopyable>("Quadrature_2D", no_init)
    ;
    class_<Quadrature<3>, boost::noncopyable>("Quadrature_3D", no_init)
    ;

    class_<QGauss<0>, bases< Quadrature<0> >, boost::noncopyable>("QGauss_0D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGauss<1>, bases< Quadrature<1> >, boost::noncopyable>("QGauss_1D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGauss<2>, bases< Quadrature<2> >, boost::noncopyable>("QGauss_2D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGauss<3>, bases< Quadrature<3> >, boost::noncopyable>("QGauss_3D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;

    class_<QGaussLobatto<0>, bases< Quadrature<0> >, boost::noncopyable>("QGaussLobatto_0D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGaussLobatto<1>, bases< Quadrature<1> >, boost::noncopyable>("QGaussLobatto_1D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGaussLobatto<2>, bases< Quadrature<2> >, boost::noncopyable>("QGaussLobatto_2D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    class_<QGaussLobatto<3>, bases< Quadrature<3> >, boost::noncopyable>("QGaussLobatto_3D", no_init)
        .def(init<const unsigned int>((arg("self"), arg("n_quadrature_points"))))
    ;
    /*
    class_<QMidpoint<0>, bases< Quadrature<0> >, boost::noncopyable>("QMidpoint_0D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMidpoint<1>, bases< Quadrature<1> >, boost::noncopyable>("QMidpoint_1D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMidpoint<2>, bases< Quadrature<2> >, boost::noncopyable>("QMidpoint_2D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMidpoint<3>, bases< Quadrature<3> >, boost::noncopyable>("QMidpoint_3D", no_init)
        .def(init<>((arg("self"))))
    ;

    class_<QSimpson<0>, bases< Quadrature<0> >, boost::noncopyable>("QSimpson_0D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QSimpson<1>, bases< Quadrature<1> >, boost::noncopyable>("QSimpson_1D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QSimpson<2>, bases< Quadrature<2> >, boost::noncopyable>("QSimpson_2D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QSimpson<3>, bases< Quadrature<3> >, boost::noncopyable>("QSimpson_3D", no_init)
        .def(init<>((arg("self"))))
    ;

    class_<QTrapez<0>, bases< Quadrature<0> >, boost::noncopyable>("QTrapez_0D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QTrapez<1>, bases< Quadrature<1> >, boost::noncopyable>("QTrapez_1D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QTrapez<2>, bases< Quadrature<2> >, boost::noncopyable>("QTrapez_2D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QTrapez<3>, bases< Quadrature<3> >, boost::noncopyable>("QTrapez_3D", no_init)
        .def(init<>((arg("self"))))
    ;

    class_<QMilne<0>, bases< Quadrature<0> >, boost::noncopyable>("QMilne_0D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMilne<1>, bases< Quadrature<1> >, boost::noncopyable>("QMilne_1D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMilne<2>, bases< Quadrature<2> >, boost::noncopyable>("QMilne_2D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QMilne<3>, bases< Quadrature<3> >, boost::noncopyable>("QMilne_3D", no_init)
        .def(init<>((arg("self"))))
    ;

    class_<QWeddle<0>, bases< Quadrature<0> >, boost::noncopyable>("QWeddle_0D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QWeddle<1>, bases< Quadrature<1> >, boost::noncopyable>("QWeddle_1D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QWeddle<2>, bases< Quadrature<2> >, boost::noncopyable>("QWeddle_2D", no_init)
        .def(init<>((arg("self"))))
    ;
    class_<QWeddle<3>, bases< Quadrature<3> >, boost::noncopyable>("QWeddle_3D", no_init)
        .def(init<>((arg("self"))))
    ;

    class_<QGaussLog<0>, bases< Quadrature<0> >, boost::noncopyable>("QGaussLog_0D", no_init)
        .def(init<const unsigned int, optional<bool> >((arg("self"), arg("n_quadrature_points"), arg("revert") = false)))
    ;
    class_<QGaussLog<1>, bases< Quadrature<1> >, boost::noncopyable>("QGaussLog_1D", no_init)
        .def(init<const unsigned int, bool>((arg("self"), arg("n_quadrature_points"), arg("revert"))))
    ;
    class_<QGaussLog<2>, bases< Quadrature<2> >, boost::noncopyable>("QGaussLog_2D", no_init)
        .def(init<const unsigned int, optional<bool> >((arg("self"), arg("n_quadrature_points"), arg("revert") = false)))
    ;
    class_<QGaussLog<3>, bases< Quadrature<3> >, boost::noncopyable>("QGaussLog_3D", no_init)
        .def(init<const unsigned int, optional<bool> >((arg("self"), arg("n_quadrature_points"), arg("revert") = false)))
    ;

    class_<QGaussLogR<0>, bases< Quadrature<0> >, boost::noncopyable>("QGaussLogR_0D", no_init)
        .def(init<const unsigned int, optional<const Point<0>, const double, const bool> >((arg("self"), arg("n_quadrature_points"), arg("x0") = Point<0>(), arg("alpha") = 1.0, arg("factor_out_singular_weight") = false)))
    ;
    class_<QGaussLogR<1>, bases< Quadrature<1> >, boost::noncopyable>("QGaussLogR_1D", no_init)
        .def(init<const unsigned int, const Point<1>, const double, const bool>((arg("self"), arg("n_quadrature_points"), arg("x0"), arg("alpha"), arg("factor_out_singular_weight"))))
    ;
    class_<QGaussLogR<2>, bases< Quadrature<2> >, boost::noncopyable>("QGaussLogR_2D", no_init)
        .def(init<const unsigned int, optional<const Point<2>, const double, const bool> >((arg("self"), arg("n_quadrature_points"), arg("x0") = Point<2>(), arg("alpha") = 1.0, arg("factor_out_singular_weight") = false)))
    ;
    class_<QGaussLogR<3>, bases< Quadrature<3> >, boost::noncopyable>("QGaussLogR_3D", no_init)
        .def(init<const unsigned int, optional<const Point<3>, const double, const bool> >((arg("self"), arg("n_quadrature_points"), arg("x0") = Point<3>(), arg("alpha") = 1.0, arg("factor_out_singular_weight") = false)))
    ;

    class_<QGaussOneOverR<0>, bases< Quadrature<0> >, boost::noncopyable>("QGaussOneOverR_0D", no_init)
        .def(init<const unsigned int, const Point<0>, optional<const bool> >((arg("self"), arg("n_quadrature_points"), arg("singularity"), arg("factor_out_singular_weight") = false)))
    ;
    class_<QGaussOneOverR<1>, bases< Quadrature<1> >, boost::noncopyable>("QGaussOneOverR_1D", no_init)
        .def(init<const unsigned int, const Point<1>, optional<const bool> >((arg("self"), arg("n_quadrature_points"), arg("singularity"), arg("factor_out_singular_weight") = false)))
    ;
    class_<QGaussOneOverR<2>, bases< Quadrature<2> >, boost::noncopyable>("QGaussOneOverR_2D", no_init)
        .def(init<const unsigned int, const Point<2>, const bool>((arg("self"), arg("n_quadrature_points"), arg("singularity"), arg("factor_out_singular_weight"))))
    ;
    class_<QGaussOneOverR<3>, bases< Quadrature<3> >, boost::noncopyable>("QGaussOneOverR_3D", no_init)
        .def(init<const unsigned int, const Point<3>, optional<const bool> >((arg("self"), arg("n_quadrature_points"), arg("singularity"), arg("factor_out_singular_weight") = false)))
    ;
    */

    class_<FEValuesBase<1>, boost::noncopyable>("FEValuesBase_1D", no_init)
        .def("shape_value",             &FEValuesBase<1>::shape_value,              return_value_policy<copy_const_reference>())
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
        .def("shape_value",             &FEValuesBase<2>::shape_value,              return_value_policy<copy_const_reference>())
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
        .def("shape_value",             &FEValuesBase<3>::shape_value,              return_value_policy<copy_const_reference>())
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

    class_<dealiiFace<1>, boost::noncopyable>("dealiiFace_1D", no_init)
        .add_property("fe_values",     make_function(&dealiiFace_1D::get_fe_values, return_internal_reference<>()))
        .add_property("n_q_points",    &dealiiFace_1D::get_n_q_points)
        .add_property("at_boundary",   &dealiiFace_1D::get_at_boundary)
        .add_property("boundary_id",   &dealiiFace_1D::get_boundary_id)
    ;

    class_<dealiiFace<2>, boost::noncopyable>("dealiiFace_2D", no_init)
        .add_property("fe_values",     make_function(&dealiiFace_2D::get_fe_values, return_internal_reference<>()))
        .add_property("n_q_points",    &dealiiFace_2D::get_n_q_points)
        .add_property("at_boundary",   &dealiiFace_2D::get_at_boundary)
        .add_property("boundary_id",   &dealiiFace_2D::get_boundary_id)
    ;

    class_<dealiiFace<3>, boost::noncopyable>("dealiiFace_3D", no_init)
        .add_property("fe_values",     make_function(&dealiiFace_3D::get_fe_values, return_internal_reference<>()))
        .add_property("n_q_points",    &dealiiFace_3D::get_n_q_points)
        .add_property("at_boundary",   &dealiiFace_3D::get_at_boundary)
        .add_property("boundary_id",   &dealiiFace_3D::get_boundary_id)
    ;

    class_<dealiiCell<1>, boost::noncopyable>("dealiiCell_1D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_1D::dofs_per_cell)
        .def_readonly("faces_per_cell",     &dealiiCell_1D::faces_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_1D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_1D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_1D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_1D::fe_values)
        .def_readonly("cell_matrix",        &dealiiCell_1D::cell_matrix)
        .def_readonly("cell_matrix_dt",     &dealiiCell_1D::cell_matrix_dt)
        .def_readonly("cell_rhs",           &dealiiCell_1D::cell_rhs)
        .add_property("system_matrix",      make_function(&dealiiCell_1D::get_system_matrix,    return_internal_reference<>()))
        .add_property("system_matrix_dt",   make_function(&dealiiCell_1D::get_system_matrix_dt, return_internal_reference<>()))
        .add_property("system_rhs",         make_function(&dealiiCell_1D::get_system_rhs,       return_internal_reference<>()))
        .add_property("faces",              range< return_value_policy<reference_existing_object> >(&dealiiCell_1D::begin_faces, &dealiiCell_1D::end_faces))
    ;

    class_<dealiiCell<2>, boost::noncopyable>("dealiiCell_2D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_2D::dofs_per_cell)
        .def_readonly("faces_per_cell",     &dealiiCell_2D::faces_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_2D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_2D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_2D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_2D::fe_values)
        .def_readonly("cell_matrix",        &dealiiCell_2D::cell_matrix)
        .def_readonly("cell_matrix_dt",     &dealiiCell_2D::cell_matrix_dt)
        .def_readonly("cell_rhs",           &dealiiCell_2D::cell_rhs)
        .add_property("system_matrix",      make_function(&dealiiCell_2D::get_system_matrix,    return_internal_reference<>()))
        .add_property("system_matrix_dt",   make_function(&dealiiCell_2D::get_system_matrix_dt, return_internal_reference<>()))
        .add_property("system_rhs",         make_function(&dealiiCell_2D::get_system_rhs,       return_internal_reference<>()))
        .add_property("faces",              range< return_value_policy<reference_existing_object> >(&dealiiCell_2D::begin_faces, &dealiiCell_2D::end_faces))
    ;

    class_<dealiiCell<3>, boost::noncopyable>("dealiiCell_3D", no_init)
        .def_readonly("dofs_per_cell",      &dealiiCell_3D::dofs_per_cell)
        .def_readonly("faces_per_cell",     &dealiiCell_3D::faces_per_cell)
        .def_readonly("n_q_points",         &dealiiCell_3D::n_q_points)
        .def_readonly("n_face_q_points",    &dealiiCell_3D::n_face_q_points)
        .def_readonly("local_dof_indices",  &dealiiCell_3D::local_dof_indices)
        .def_readonly("fe_values",          &dealiiCell_3D::fe_values)
        .def_readonly("cell_matrix",        &dealiiCell_3D::cell_matrix)
        .def_readonly("cell_matrix_dt",     &dealiiCell_3D::cell_matrix_dt)
        .def_readonly("cell_rhs",           &dealiiCell_3D::cell_rhs)
        .add_property("system_matrix",      make_function(&dealiiCell_3D::get_system_matrix,    return_internal_reference<>()))
        .add_property("system_matrix_dt",   make_function(&dealiiCell_3D::get_system_matrix_dt, return_internal_reference<>()))
        .add_property("system_rhs",         make_function(&dealiiCell_3D::get_system_rhs,       return_internal_reference<>()))
        .add_property("faces",              range< return_value_policy<reference_existing_object> >(&dealiiCell_3D::begin_faces, &dealiiCell_3D::end_faces))
    ;

    class_<daeDealIIDataReporter, bases<daeDataReporter_t>, boost::noncopyable>("daeDealIIDataReporter", no_init)
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

    class_<daepython::dealiiFiniteElementObjectWrapper<1>, bases<daeFiniteElementObject>, boost::noncopyable>("dealiiFiniteElementObject_1D", no_init)
        .def(init<string,
                  unsigned int,
                  const Quadrature<1>&,
                  const Quadrature<0>&,
                  boost::python::dict,
                  boost::python::dict,
                  boost::python::dict>(( arg("meshFilename"),
                                         arg("polynomialOrder"),
                                         arg("quadrature"),
                                         arg("faceQuadrature"),
                                         arg("functions"),
                                         arg("dirichletBC"),
                                         arg("neumannBC")
                                      )))
        .def("AssembleSystem",      &daepython::dealiiFiniteElementObjectWrapper<1>::AssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<1>::def_AssembleSystem, ( arg("self") ))
        .def("ReAssembleSystem",    &daepython::dealiiFiniteElementObjectWrapper<1>::ReAssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<1>::def_ReAssembleSystem, ( arg("self") ))
        .def("NeedsReAssembling",   &daepython::dealiiFiniteElementObjectWrapper<1>::NeedsReAssembling,
                                    &daepython::dealiiFiniteElementObjectWrapper<1>::def_NeedsReAssembling, ( arg("self") ))
        .def("GetDOFtoBoundaryMap", &daepython::dealiiFiniteElementObjectWrapper<1>::GetDOFtoBoundaryMap,
                                    &daepython::dealiiFiniteElementObjectWrapper<1>::def_GetDOFtoBoundaryMap, ( arg("self") ))
        .def("CreateDataReporter",  &dealiiFiniteElementObject<1>::CreateDataReporter, ( arg("self") ), return_value_policy<manage_new_object>())

//        .def("__iter__",                            boost::python::iterator<daeConvectionDiffusion_2D, return_value_policy<reference_existing_object> >())
//        .def("AssembleSystem",                      &daeConvectionDiffusion_2D::AssembleSystem)
//        .def("GenerateEquations",                   &daeConvectionDiffusion_2D::GenerateEquations)
//        .def("CondenseHangingNodeConstraints",      &daeConvectionDiffusion_2D::CondenseHangingNodeConstraints)
//        .def("InterpolateAndApplyBoundaryValues",   &daeConvectionDiffusion_2D::InterpolateAndApplyBoundaryValues)
    ;
    
    class_<daepython::dealiiFiniteElementObjectWrapper<2>, bases<daeFiniteElementObject>, boost::noncopyable>("dealiiFiniteElementObject_2D", no_init)
        .def(init<string,
                  unsigned int,
                  const Quadrature<2>&,
                  const Quadrature<1>&,
                  boost::python::dict,
                  boost::python::dict,
                  boost::python::dict>(( arg("meshFilename"),
                                         arg("polynomialOrder"),
                                         arg("quadratureFormula"),
                                         arg("numberOfQuadraturePoints"),
                                         arg("functions"),
                                         arg("dirichletBC"),
                                         arg("neumannBC")
                                      )))

        .def("AssembleSystem",      &daepython::dealiiFiniteElementObjectWrapper<2>::AssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<2>::def_AssembleSystem, ( arg("self") ))
        .def("ReAssembleSystem",    &daepython::dealiiFiniteElementObjectWrapper<2>::ReAssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<2>::def_ReAssembleSystem, ( arg("self") ))
        .def("NeedsReAssembling",   &daepython::dealiiFiniteElementObjectWrapper<2>::NeedsReAssembling,
                                    &daepython::dealiiFiniteElementObjectWrapper<2>::def_NeedsReAssembling, ( arg("self") ))
        .def("GetDOFtoBoundaryMap", &daepython::dealiiFiniteElementObjectWrapper<2>::GetDOFtoBoundaryMap,
                                    &daepython::dealiiFiniteElementObjectWrapper<2>::def_GetDOFtoBoundaryMap, ( arg("self") ))
        .def("CreateDataReporter",  &dealiiFiniteElementObject<2>::CreateDataReporter, ( arg("self") ), return_value_policy<manage_new_object>())
    ;

    class_<daepython::dealiiFiniteElementObjectWrapper<3>, bases<daeFiniteElementObject>, boost::noncopyable>("dealiiFiniteElementObject_3D", no_init)
        .def(init<string,
                  unsigned int,
                  const Quadrature<3>&,
                  const Quadrature<2>&,
                  boost::python::dict,
                  boost::python::dict,
                  boost::python::dict>(( arg("meshFilename"),
                                         arg("polynomialOrder"),
                                         arg("quadratureFormula"),
                                         arg("numberOfQuadraturePoints"),
                                         arg("functions"),
                                         arg("dirichletBC"),
                                         arg("neumannBC")
                                      )))
        .def("AssembleSystem",      &daepython::dealiiFiniteElementObjectWrapper<3>::AssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<3>::def_AssembleSystem, ( arg("self") ))
        .def("ReAssembleSystem",    &daepython::dealiiFiniteElementObjectWrapper<3>::ReAssembleSystem,
                                    &daepython::dealiiFiniteElementObjectWrapper<3>::def_ReAssembleSystem, ( arg("self") ))
        .def("NeedsReAssembling",   &daepython::dealiiFiniteElementObjectWrapper<3>::NeedsReAssembling,
                                    &daepython::dealiiFiniteElementObjectWrapper<3>::def_NeedsReAssembling, ( arg("self") ))
        .def("GetDOFtoBoundaryMap", &daepython::dealiiFiniteElementObjectWrapper<3>::GetDOFtoBoundaryMap,
                                    &daepython::dealiiFiniteElementObjectWrapper<3>::def_GetDOFtoBoundaryMap, ( arg("self") ))
        .def("CreateDataReporter",  &dealiiFiniteElementObject<3>::CreateDataReporter, ( arg("self") ), return_value_policy<manage_new_object>())
    ;

}
