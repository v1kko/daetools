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

POINTER_CONVERSION(daeIDALASolver_t)
POINTER_CONVERSION(daeIntelPardisoSolver)
}
#endif
#endif

BOOST_PYTHON_MODULE(pyIntelPardiso)
{
/**************************************************************
    LA Solver
***************************************************************/
    class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
        .def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
        ;

    class_<daeIntelPardisoSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeIntelPardisoSolver")
        .add_property("Name",                   &daeIntelPardisoSolver::GetName)
        .add_property("EvaluationCallsStats",   &daepython::daeIntelPardisoSolver_GetEvaluationCallsStats_)
        .def("get_iparm",                       &daepython::daeIntelPardisoSolver_get_iparm)
        .def("set_iparm",                       &daepython::daeIntelPardisoSolver_set_iparm)
        .def("SaveAsXPM",                       &daeIntelPardisoSolver::SaveAsXPM)
        .def("SaveAsMatrixMarketFile",          &daeIntelPardisoSolver::SaveAsMatrixMarketFile)
        ;

    def("daeCreateIntelPardisoSolver", daeCreateIntelPardisoSolver, return_value_policy<manage_new_object>());

}
