#include "python_wraps.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#define NO_IMPORT_ARRAY
//#include <noprefix.h>
using namespace std;
using namespace boost;
using namespace boost::python;

namespace daepython
{
boost::python::list daeArray_GetValues(daeArray<real_t>& self)
{
    boost::python::list l;

    for(size_t i = 0; i < self.GetSize(); i++)
        l.append(self.GetItem(i));

    return l;
}

boost::python::object daeDenseMatrix_ndarray(daeDenseMatrix& self)
{
    // Import numpy
    boost::python::object main_module = import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    exec("import numpy", main_namespace);
    boost::python::object numpy = main_namespace["numpy"];

    // Create shape
    int N = self.GetNrows();
    int M = self.GetNcols();
    boost::python::tuple shape = boost::python::make_tuple(N, M);

    // Create a 2d list of values
    boost::python::list lrows;
    for(size_t row = 0; row < N; row++)
    {
        boost::python::list lcols;
        for(size_t col = 0; col < M; col++)
            lcols.append(self.GetItem(row,col));
        lrows.append(lcols);
    }

    // Create a flat ndarray
    boost::python::dict kwargs;
    if(typeid(real_t) == typeid(double))
        kwargs["dtype"] = numpy.attr("float64");
    else
        kwargs["dtype"] = numpy.attr("float32");
    boost::python::tuple args = boost::python::make_tuple(lrows);
    boost::python::object ndarray = numpy.attr("array")(*args, **kwargs);

    return ndarray;
}

boost::python::dict GetCallStats(daeIDASolver& self)
{
    std::map<std::string, call_stats::TimeAndCount> stats = self.GetCallStats();
    return getDictFromMapByValue(stats);
}

}
