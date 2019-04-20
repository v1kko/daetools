#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "python_wraps.h"
using namespace boost::python;
using namespace daetools::solver;

BOOST_PYTHON_MODULE(pyPardiso)
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

    class_<daePardisoSolver, bases<daeLASolver_t>, boost::noncopyable>("daePardisoSolver")
        .add_property("Name",             &daePardisoSolver::GetName)
        .add_property("CallStats",        &daepython::GetCallStats)
        .def("get_iparm",                 &daepython::get_iparm)
        .def("set_iparm",                 &daepython::set_iparm)
        .def("SaveAsXPM",                 &daePardisoSolver::SaveAsXPM)
        .def("SaveAsMatrixMarketFile",    &daePardisoSolver::SaveAsMatrixMarketFile)

        .def("GetOption_bool",   &daePardisoSolver::GetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_int",    &daePardisoSolver::GetOption_int,    ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_float",	 &daePardisoSolver::GetOption_float,  ( boost::python::arg("self"), boost::python::arg("name") ))
        .def("GetOption_string", &daePardisoSolver::GetOption_string, ( boost::python::arg("self"), boost::python::arg("name") ))

        .def("SetOption_bool",   &daePardisoSolver::SetOption_bool,   ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_int",    &daePardisoSolver::SetOption_int,    ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_float",  &daePardisoSolver::SetOption_float,  ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        .def("SetOption_string", &daePardisoSolver::SetOption_string, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ))
        ;

    def("daeCreatePardisoSolver", daeCreatePardisoSolver, return_value_policy<manage_new_object>());

}
