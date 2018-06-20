#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "python_wraps.h"
using namespace boost::python;
using namespace dae::solver;

// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_VER == 1900
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(daeLASolver_t)
POINTER_CONVERSION(daeIntelPardisoSolver)
}
#endif
#endif

BOOST_PYTHON_MODULE(pyIntelPardiso)
{
/**************************************************************
    LA Solver
***************************************************************/
    class_<daeLASolver_t, boost::noncopyable>("daeLASolver_t", no_init)
        .def("SaveAsXPM",	pure_virtual(&daeLASolver_t::SaveAsXPM))

        .def("GetOption_bool",   pure_virtual(&daeLASolver_t::GetOption_bool))
        .def("GetOption_int",    pure_virtual(&daeLASolver_t::GetOption_int))
        .def("GetOption_float",	 pure_virtual(&daeLASolver_t::GetOption_float))
        .def("GetOption_string", pure_virtual(&daeLASolver_t::GetOption_string))

        .def("SetOption_bool",   pure_virtual(&daeLASolver_t::SetOption_bool))
        .def("SetOption_int",    pure_virtual(&daeLASolver_t::SetOption_int))
        .def("SetOption_float",  pure_virtual(&daeLASolver_t::SetOption_float))
        .def("SetOption_string", pure_virtual(&daeLASolver_t::SetOption_string))
    ;

    class_<daeIntelPardisoSolver, bases<daeLASolver_t>, boost::noncopyable>("daeIntelPardisoSolver")
        .add_property("Name",             &daeIntelPardisoSolver::GetName)
        .add_property("CallStats",        &daepython::GetCallStats)
        .def("get_iparm",                 &daepython::get_iparm)
        .def("set_iparm",                 &daepython::set_iparm)
        .def("SaveAsXPM",                 &daeIntelPardisoSolver::SaveAsXPM)
        .def("SaveAsMatrixMarketFile",    &daeIntelPardisoSolver::SaveAsMatrixMarketFile)

        .def("GetOption_bool",   &daeIntelPardisoSolver::GetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_int",    &daeIntelPardisoSolver::GetOption_int,    ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_float",	 &daeIntelPardisoSolver::GetOption_float,  ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_string", &daeIntelPardisoSolver::GetOption_string, ( boost::python::arg("self"), boost::python::arg("name") ))

        .def("SetOption_bool",   &daeIntelPardisoSolver::SetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_int",    &daeIntelPardisoSolver::SetOption_int,    ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_float",  &daeIntelPardisoSolver::SetOption_float,  ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_string", &daeIntelPardisoSolver::SetOption_string, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        ;

    def("daeCreateIntelPardisoSolver", daeCreateIntelPardisoSolver, return_value_policy<manage_new_object>());

}
