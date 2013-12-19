#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "python_wraps.h"
using namespace boost::python;
using namespace dae::solver;

BOOST_PYTHON_MODULE(pyIntelPardiso)
{
/**************************************************************
	LA Solver
***************************************************************/
	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		;

	class_<daeIntelPardisoSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeIntelPardisoSolver")
		.add_property("Name",			&daeIntelPardisoSolver::GetName)
        .def("get_iparm",               &daepython::daeIntelPardisoSolver_get_iparm)
        .def("set_iparm",               &daepython::daeIntelPardisoSolver_set_iparm)
        .def("SaveAsXPM",				&daeIntelPardisoSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeIntelPardisoSolver::SaveAsMatrixMarketFile)
		;

    def("daeCreateIntelPardisoSolver", daeCreateIntelPardisoSolver, return_value_policy<manage_new_object>());

}
