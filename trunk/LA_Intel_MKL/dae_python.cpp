#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "mkl_pardiso_sparse_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

BOOST_PYTHON_MODULE(pyIntelPardiso)
{
/**************************************************************
	LA Solver
***************************************************************/
	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
		.def("SaveAsPBM",	pure_virtual(&daeIDALASolver_t::SaveAsPBM))
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		;

	class_<daeIntelPardisoSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeIntelPardisoSolver")
		.def("Create",		&daeIntelPardisoSolver::Create)
		.def("Reinitialize",&daeIntelPardisoSolver::Reinitialize)
		.def("SaveAsPBM",	&daeIntelPardisoSolver::SaveAsPBM)
		.def("SaveAsXPM",	&daeIntelPardisoSolver::SaveAsXPM)
		;

	def("daeCreateIntelPardisoSolver", daeCreateIntelPardisoSolver, return_value_policy<reference_existing_object>());

}
