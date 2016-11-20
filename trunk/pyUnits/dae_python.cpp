#include "stdafx.h"
#include "python_wraps.h"
#include "docstrings.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyUnits)
{
	//import_array();
	//boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    
    docstring_options doc_options(true, true, false);
  
/**************************************************************
	base_unit
***************************************************************/
    class_<base_unit>("base_unit", DOCSTR_base_unit)
        .def("__init__",	make_constructor(daepython::__init__base_unit))

        .def_readonly("L", &base_unit::L, DOCSTR_base_unit_L)
        .def_readonly("M", &base_unit::M, DOCSTR_base_unit_M)
        .def_readonly("T", &base_unit::T, DOCSTR_base_unit_T)
        .def_readonly("C", &base_unit::C, DOCSTR_base_unit_C)
        .def_readonly("I", &base_unit::I, DOCSTR_base_unit_I)
        .def_readonly("O", &base_unit::O, DOCSTR_base_unit_O)
        .def_readonly("N", &base_unit::N, DOCSTR_base_unit_N)
        .def_readonly("multiplier", &base_unit::multiplier, DOCSTR_base_unit_multiplier)

        .def("__str__",		&daepython::base_unit__str__)
		.def("__repr__",	&daepython::base_unit__repr__) 
		
		.def(self * self)
		.def(self / self)
		
		.def(self == self)
		.def(self != self)

		.def(self * double())
		.def(self / double())
		.def(pow(self, double()))

		.def(double() * self)
		.def(double() / self)
		;
	
/**************************************************************
	unit
***************************************************************/
    class_<unit>("unit", DOCSTR_unit)
        .def("__init__",	make_constructor(daepython::__init__unit), DOCSTR_unit_init)

        .def("__str__",		&daepython::unit__str__)  // represents a valid python expression (using operators *, /, **)
		.def("__repr__",	&daepython::unit__repr__)

        .add_property("baseUnit",	    &unit::getBaseUnit,   DOCSTR_unit_baseUnit)

        .def("toDict",  &daepython::unit_toDict, DOCSTR_unit_toDictionary)
        .def("toJSON",  &unit::toJSON,           DOCSTR_unit_toJSON)

		.def(self * self)
		.def(self / self)
		
		.def(self == self)
		.def(self != self)

		.def(self * double())
		.def(self / double())
		.def(pow(self, double()))

		.def(double() * self)
		.def(double() / self)

        .def("__truediv__",  &daepython::unit_true_divide1)   // unit / unit
        .def("__truediv__",  &daepython::unit_true_divide2)   // unit / real_t
        .def("__truediv__",  &daepython::unit_true_divide3)   // real_t  / unit
        ;

    class_<quantity>("quantity", DOCSTR_quantity)
        .def(init<double, const unit&>(( arg("self"), arg("value"), arg("unit") ), DOCSTR_quantity_init))
		
        .add_property("valueInSIUnits",	&quantity::getValueInSIUnits,                       DOCSTR_quantity_valueInSIUnits)
        .add_property("value",			&quantity::getValue, &daepython::quantity_setValue, DOCSTR_quantity_value)
        .add_property("units",			&quantity::getUnits, &quantity::setUnits,           DOCSTR_quantity_units)
        
		.def("__str__",		&daepython::quantity__str__)
		.def("__repr__",	&daepython::quantity__repr__)
        .def("scaleTo",		&daepython::quantity_scaleTo, ( arg("self"), arg("referrer") ), DOCSTR_quantity_scaleTo)

        .def("toDict",  &daepython::quantity_toDict, DOCSTR_quantity_toDictionary)
        .def("toJSON",  &quantity::toJSON,           DOCSTR_quantity_toJSON)

		.def(- self)
		.def(+ self)

		.def(self + self)
		.def(self - self)
		.def(self * self)
		.def(self / self)
		.def(pow(self, self))
		.def(self == self)
		.def(self != self)
		.def(self <  self)
		.def(self <= self)
		.def(self >  self)
		.def(self >= self)

		.def(self * unit())
		.def(self / unit())

		.def(self + double())
		.def(self - double())
		.def(self * double())
		.def(self / double())
		.def(pow(self, double()))
		.def(self == double())
		.def(self != double())
		.def(self <  double())
		.def(self <= double())
		.def(self >  double())
		.def(self >= double())

		.def(double() + self)
		.def(double() - self)
		.def(double() * self)
		.def(double() / self)
		.def(pow(double(), self))
		.def(double() == self)
		.def(double() != self)
		.def(double() <  self)
		.def(double() <= self)
		.def(double() >  self)
		.def(double() >= self)

        .def("__truediv__",  &daepython::quantity_true_divide1)   // quantity / quantity
        .def("__truediv__",  &daepython::quantity_true_divide2)   // quantity / real_t
        .def("__truediv__",  &daepython::quantity_true_divide3)   // real_t  / quantity
        ;
	

    boost::python::dict all_si_and_derived_units;

    std::map<std::string, base_unit>& bu = unit::get_base_units();
	for(std::map<std::string, base_unit>::iterator iter = bu.begin(); iter != bu.end(); iter++)
    {
        all_si_and_derived_units[(*iter).first] = unit((*iter).first, 1.0);
        scope().attr((*iter).first.c_str()) = unit((*iter).first, 1.0);
    }

    scope().attr("all_si_and_derived_units") = all_si_and_derived_units;
}

