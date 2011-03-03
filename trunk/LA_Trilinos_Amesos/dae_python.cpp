#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "trilinos_amesos_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

boost::python::list pydaeTrilinosAmesosSupportedSolvers(void);

boost::python::list pydaeTrilinosAmesosSupportedSolvers(void)
{
	std::vector<string> strarrSolvers = daeTrilinosAmesosSupportedSolvers();
	boost::python::list l;
	
	for(size_t i = 0; i < strarrSolvers.size(); i++)
		l.append(strarrSolvers[i]);
	return l;
}

BOOST_PYTHON_MODULE(pyTrilinosAmesos)
{
/**************************************************************
	TrilinosAmesos LA Solver
***************************************************************/
	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		//.def("SaveAsMatrixMarketFile",	pure_virtual(&daeIDALASolver_t::SaveAsMatrixMarketFile))
		;

	class_<daeTrilinosAmesosSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeTrilinosAmesosSolver", init<string>())
		.def_readwrite("NumIters",		&daeTrilinosAmesosSolver::m_nNumIters)
		.def_readwrite("Tolerance",		&daeTrilinosAmesosSolver::m_dTolerance)
		.def("Create",					&daeTrilinosAmesosSolver::Create)
		.def("Reinitialize",			&daeTrilinosAmesosSolver::Reinitialize)
		.def("SaveAsXPM",				&daeTrilinosAmesosSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeTrilinosAmesosSolver::SaveAsMatrixMarketFile)
		.def("SetAztecOption",			&daeTrilinosAmesosSolver::SetAztecOption)
		.def("SetAztecParameter",		&daeTrilinosAmesosSolver::SetAztecParameter)
		;

	def("daeCreateTrilinosAmesosSolver",      daeCreateTrilinosAmesosSolver,  return_value_policy<reference_existing_object>());
	def("daeTrilinosAmesosSupportedSolvers",  pydaeTrilinosAmesosSupportedSolvers);
}
