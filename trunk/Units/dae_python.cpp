#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
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
        .def("__str__",		&base_unit::toString)
		;
	
/**************************************************************
	unit
***************************************************************/
	class_<unit>("unit")
        .def("__init__",			make_constructor(daepython::__init__unit))
		.add_property("baseUnit",	&unit::getBaseUnit)
		.def("__str__",				&unit::toString)
		;
   

}

