#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "python_wraps.h"
using namespace boost::python;
using namespace dae::solver;

BOOST_PYTHON_MODULE(pyPardiso)
{
/**************************************************************
    LA Solver
***************************************************************/
    class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
        .def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
        ;

    class_<daePardisoSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daePardisoSolver")
        .add_property("Name",                   &daePardisoSolver::GetName)
        .add_property("EvaluationCallsStats",   &daepython::GetEvaluationCallsStats_)
        .def("get_iparm",                       &daepython::daePardisoSolver_get_iparm)
        .def("set_iparm",                       &daepython::daePardisoSolver_set_iparm)
        .def("SaveAsXPM",                       &daePardisoSolver::SaveAsXPM)
        .def("SaveAsMatrixMarketFile",          &daePardisoSolver::SaveAsMatrixMarketFile)
        ;

    def("daeCreatePardisoSolver", daeCreatePardisoSolver, return_value_policy<manage_new_object>());

}
