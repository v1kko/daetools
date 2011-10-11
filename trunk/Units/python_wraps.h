#ifndef PYTHON_WRAPS_H
#define PYTHON_WRAPS_H

#include <string>
#include <map>
#include <vector>
#include <boost/python.hpp>
#include "units.h"
#include "../dae_develop.h"
using namespace units;

namespace daepython
{
/*******************************************************
	base_unitWrapper
*******************************************************/
//class base_unitWrapper : public base_unit,
//                         public boost::python::wrapper<base_unit>
//{
//public:
//	base_unitWrapper(double multi, boost::python::dict kwargs)
//	{
//        multiplier = multi;
//		L = boost::python::extract<double>(kwargs.get("L", 0.0));
//		M = boost::python::extract<double>(kwargs.get("M", 0.0));
//		T = boost::python::extract<double>(kwargs.get("T", 0.0));
//		C = boost::python::extract<double>(kwargs.get("C", 0.0));
//		I = boost::python::extract<double>(kwargs.get("I", 0.0));
//		O = boost::python::extract<double>(kwargs.get("O", 0.0));
//		N = boost::python::extract<double>(kwargs.get("N", 0.0));
//	}

//};

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


/*******************************************************
	unitWrapper
*******************************************************/
//class unitWrapper : public unit,
//                    public boost::python::wrapper<unit>
//{
//public:
//	unitWrapper(boost::python::dict kwargs)
//	{
//		string name;
//		double exp;
//        boost::python::ssize_t i, n;
//        boost::python::tuple t;
//		std::map<std::string, double> mapUnits;
		
//		boost::python::list items = kwargs.items();
//        n = boost::python::len(items);
        
//		for(i = 0; i < n; i++)
//        {
//            t = boost::python::extract<boost::python::tuple>(items[i]);
//            name  = boost::python::extract<string>(t[0]);
//            exp   = boost::python::extract<double>(t[1]);
//			mapUnits[name] = exp;
//		}
//		unit::setUnits(mapUnits);
//	}	
//};
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

}

#endif // PYTHON_WRAPS_H
