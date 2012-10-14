#ifndef PYTHON_WRAPS_H
#define PYTHON_WRAPS_H

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <map>
#include <vector>
#include <boost/python.hpp>
#include "../Units/units.h"
#include "../dae_develop.h"
using namespace units;

namespace daepython
{
template<typename KEY, typename VALUE>
boost::python::dict getDictFromMapByValue(std::map<KEY,VALUE>& mapItems)
{
    typedef typename std::map<KEY,VALUE>::const_iterator c_iterator;

    boost::python::dict d;
    for(c_iterator iter = mapItems.begin(); iter != mapItems.end(); iter++)
        d[iter->first] = iter->second;

    return d;
}

/*******************************************************
	base_unit
*******************************************************/
base_unit* __init__base_unit(double multi, boost::python::dict kwargs)
{
	double L = boost::python::extract<double>(kwargs.get("L", 0.0));
	double M = boost::python::extract<double>(kwargs.get("M", 0.0));
	double T = boost::python::extract<double>(kwargs.get("T", 0.0));
	double C = boost::python::extract<double>(kwargs.get("C", 0.0));
	double I = boost::python::extract<double>(kwargs.get("I", 0.0));
	double O = boost::python::extract<double>(kwargs.get("O", 0.0));
	double N = boost::python::extract<double>(kwargs.get("N", 0.0));
	
	return new base_unit(multi, L, M, T, C, I, O, N);
}

string base_unit__str__(base_unit& self)
{
	return self.toString(false);
}

string base_unit__repr__(base_unit& self)
{
	return (boost::format("base_unit(multiplier=%17.10e, L=%f, M=%f, T=%f, C=%f, I=%f, O=%f, N=%d)") 
   		    % self.multiplier % self.L % self.M % self.T % self.C % self.I % self.O % self.N).str();
}

/*******************************************************
	unit
*******************************************************/
unit* __init__unit(boost::python::dict kwargs)
{
	string name;
	double exp;
    boost::python::ssize_t i, n;
    boost::python::tuple t;
	std::map<std::string, double> mapUnits;
	
	boost::python::list items = kwargs.items();
    n = boost::python::len(items);
    
	for(i = 0; i < n; i++)
    {
        t = boost::python::extract<boost::python::tuple>(items[i]);
        name  = boost::python::extract<string>(t[0]);
        exp   = boost::python::extract<double>(t[1]);
		mapUnits[name] = exp;
	}
	return new unit(mapUnits);
}

string unit__str__(unit& self)
{
	return self.toString();
}

string unit__repr__(unit& self)
{
	return (boost::format("unit(%s)") % self).str();
}

boost::python::dict unit_get_units_dict(unit& self)
{
    return getDictFromMapByValue(self.units);
}

/*******************************************************
	quantity
*******************************************************/
quantity quantity_scaleTo(quantity& self, boost::python::object o)
{
	boost::python::extract<quantity> qValue(o);
	boost::python::extract<unit>     uValue(o);
	unit referrer;
	if(qValue.check())
	{
		referrer = qValue().getUnits();
	}
	else if(uValue.check())
	{
		referrer = uValue();
	}
	else
	{
		throw units_error("Invalid argument for quantity.scaleTo() function");
	}

	return self.scaleTo(referrer);
}

string quantity__str__(quantity& self)
{
	return self.toString();
}

string quantity__repr__(quantity& self)
{
	return (boost::format("quantity(%17.10e, %s)") % self.getValue() % self.getUnits()).str();
}

void quantity_setValue(quantity& self, boost::python::object o)
{
	boost::python::extract<quantity> qValue(o);
	boost::python::extract<double>   dValue(o);
	if(qValue.check())
	{
		self.setValue(qValue());
	}
	else if(dValue.check())
	{
		self.setValue(dValue());
	}
	else
	{
		throw units_error("Invalid argument for quantity.scaleTo() function");
	}
}

}

#endif // PYTHON_WRAPS_H
