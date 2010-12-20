#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "lapack_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

#ifdef daeHasIntelMKL
BOOST_PYTHON_MODULE(pyIntelMKL)
#elif daeHasLapack
BOOST_PYTHON_MODULE(pyLapack)
#elif daeHasAtlas
BOOST_PYTHON_MODULE(pyAtlas)
#elif daeHasAmdACML
BOOST_PYTHON_MODULE(pyAmdACML)
#endif
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

	class_<daeLapackSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeLapackSolver")
		.def("Create",		&daeLapackSolver::Create)
		.def("Reinitialize",&daeLapackSolver::Reinitialize)
		.def("SaveAsPBM",	&daeLapackSolver::SaveAsPBM)
		.def("SaveAsXPM",	&daeLapackSolver::SaveAsXPM)
		;

	def("daeCreateLapackSolver",  daeCreateLapackSolver,  return_value_policy<reference_existing_object>());
}
