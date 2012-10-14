#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyUnits)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray"); 
  
/**************************************************************
	base_unit
***************************************************************/
	class_<base_unit>("base_unit")
        .def("__init__",	make_constructor(daepython::__init__base_unit))
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
	class_<unit>("unit")
        .def("__init__",	make_constructor(daepython::__init__unit))
		.def("__str__",		&daepython::unit__str__)
		.def("__repr__",	&daepython::unit__repr__)

		.add_property("baseUnit",	    &unit::getBaseUnit)
        .add_property("unitDictionary",	&daepython::unit_get_units_dict)
        
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

	class_<quantity>("quantity")
		.def(init<double, const unit&>())
		
		.add_property("valueInSIUnits",	&quantity::getValueInSIUnits)
		.add_property("value",			&quantity::getValue, &daepython::quantity_setValue)
		.add_property("units",			&quantity::getUnits, &quantity::setUnits)
        
		.def("__str__",		&daepython::quantity__str__)
		.def("__repr__",	&daepython::quantity__repr__)
		.def("scaleTo",		&daepython::quantity_scaleTo)
        
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
		;
	
	std::map<std::string, base_unit>& bu = unit::get_base_units();
	for(std::map<std::string, base_unit>::iterator iter = bu.begin(); iter != bu.end(); iter++)
		scope().attr((*iter).first.c_str()) = unit((*iter).first, 1.0);
}

