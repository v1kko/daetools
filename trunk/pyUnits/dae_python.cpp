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

		.add_property("baseUnit",	&unit::getBaseUnit)
        
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
	
//	scope().attr("m")   = units_pool::m;
//	scope().attr("kg")  = units_pool::kg;
//	scope().attr("s")   = units_pool::s;
//	scope().attr("cd")  = units_pool::cd;
//	scope().attr("A")   = units_pool::A;
//	scope().attr("K")   = units_pool::K;
//	scope().attr("mol") = units_pool::mol;
//	
//	scope().attr("rad") = units_pool::rad;
//	scope().attr("sr")  = units_pool::sr;
//
//	scope().attr("ms")   = units_pool::ms;
//	scope().attr("us")   = units_pool::us;
//	scope().attr("min")  = units_pool::min;
//	scope().attr("hour") = units_pool::hour;
//	scope().attr("day")  = units_pool::day;
//	scope().attr("Hz")   = units_pool::Hz;
//	scope().attr("kHz")  = units_pool::kHz;
//	scope().attr("MHz")  = units_pool::MHz;
//	
//	scope().attr("km") = units_pool::km;
//	scope().attr("dm") = units_pool::dm;
//	scope().attr("cm") = units_pool::cm;
//	scope().attr("mm") = units_pool::mm;
//	scope().attr("um") = units_pool::um;
//	scope().attr("nm") = units_pool::nm;
//	
//	scope().attr("lit") = units_pool::lit;
//	scope().attr("dl")  = units_pool::dl;
//	
//	scope().attr("N")  = units_pool::N;
//	scope().attr("J")  = units_pool::J;
//	scope().attr("kJ") = units_pool::kJ;
//	scope().attr("W")  = units_pool::W;
//	scope().attr("kW") = units_pool::kW;
//	
//	scope().attr("V")   = units_pool::V;
//	scope().attr("C")   = units_pool::C;
//	scope().attr("F")   = units_pool::F;
//	scope().attr("Ohm") = units_pool::Ohm;
//	scope().attr("T")   = units_pool::T;
//	scope().attr("H")   = units_pool::H;
//	scope().attr("S")   = units_pool::S; 
//	scope().attr("Wb")  = units_pool::Wb;
//	scope().attr("mV")  = units_pool::mV;
//	scope().attr("mA")  = units_pool::mA;
//	
//	// Pressure:
//	scope().attr("Pa")  = units_pool::Pa;
//	scope().attr("kPa") = units_pool::kPa;
//	scope().attr("MPa") = units_pool::MPa;
//	
//	// Viscosity
//	scope().attr("P")  = units_pool::P;
//	scope().attr("St") = units_pool::St;
//	
//	// Radioactivity
//	scope().attr("Bq")  = units_pool::Bq;
//	scope().attr("Gy")  = units_pool::Gy;
//	scope().attr("Sv")  = units_pool::Sv;
//	scope().attr("mSv") = units_pool::mSv;
//	scope().attr("uSv") = units_pool::uSv;
//	
//	// Optics
//	scope().attr("lx") = units_pool::lx;
//	scope().attr("lm") = units_pool::lm;
//	
//	scope().attr("kat") = units_pool::kat;
//	
//	scope().attr("knot") = units_pool::knot;
//	scope().attr("bar ") = units_pool::bar;
//	scope().attr("b   ") = units_pool::b;
//	scope().attr("Ci  ") = units_pool::Ci;
//	scope().attr("R   ") = units_pool::R;
//	scope().attr("rd  ") = units_pool::rd;
//	scope().attr("rem ") = units_pool::rem;
	
	std::map<std::string, base_unit>& bu = unit::get_base_units();  
	for(std::map<std::string, base_unit>::iterator iter = bu.begin(); iter != bu.end(); iter++)
		scope().attr((*iter).first.c_str()) = unit((*iter).first, 1);
}

