#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "trilinos_amesos_la_solver.h"
using namespace boost::python;
using namespace dae::solver;

boost::python::list pydaeTrilinosSupportedSolvers(void);

boost::python::list pydaeTrilinosSupportedSolvers(void)
{
	std::vector<string> strarrSolvers = daeTrilinosSupportedSolvers();
	boost::python::list l;
	
	for(size_t i = 0; i < strarrSolvers.size(); i++)
		l.append(strarrSolvers[i]);
	return l;
}

void pydaeParameterListSet1(Teuchos::ParameterList& list, const string& strName, const string& Value);
void pydaeParameterListSet2(Teuchos::ParameterList& list, const string& strName, double Value);
void pydaeParameterListSet3(Teuchos::ParameterList& list, const string& strName, int Value);
void pydaeParameterListSet4(Teuchos::ParameterList& list, const string& strName, bool Value);

void pydaeParameterListSet1(Teuchos::ParameterList& list, const string& strName, const string& Value)
{
	list.set(strName, Value);
}

void pydaeParameterListSet2(Teuchos::ParameterList& list, const string& strName, double Value)
{
	list.set(strName, Value);
}

void pydaeParameterListSet3(Teuchos::ParameterList& list, const string& strName, int Value)
{
	list.set(strName, Value);
}

void pydaeParameterListSet4(Teuchos::ParameterList& list, const string& strName, bool Value)
{
	list.set(strName, Value);
}


BOOST_PYTHON_MODULE(pyTrilinos)
{
/**************************************************************
	TrilinosAmesos LA Solver
***************************************************************/
	class_<Teuchos::ParameterList, boost::noncopyable>("TeuchosParameterList")
		.def("set",		&pydaeParameterListSet1)
		.def("set",		&pydaeParameterListSet2)
		.def("set",		&pydaeParameterListSet3)
		.def("set",		&pydaeParameterListSet4)
		;

	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		//.def("SaveAsMatrixMarketFile",	pure_virtual(&daeIDALASolver_t::SaveAsMatrixMarketFile))
		;

	class_<daeTrilinosSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeTrilinosSolver", init<string>())
		.def_readwrite("NumIters",		&daeTrilinosSolver::m_nNumIters)
		.def_readwrite("Tolerance",		&daeTrilinosSolver::m_dTolerance)
		.def("Create",					&daeTrilinosSolver::Create)
		.def("Reinitialize",			&daeTrilinosSolver::Reinitialize)
		.def("SaveAsXPM",				&daeTrilinosSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeTrilinosSolver::SaveAsMatrixMarketFile)
		.def("SetAztecOptions",			&daeTrilinosSolver::SetAztecOptions)
		.def("SetIfpackOptions",		&daeTrilinosSolver::SetIfpackOptions)
		.def("SetAmesosOptions",		&daeTrilinosSolver::SetAmesosOptions)
		//.def("SetAztecOption",		&daeTrilinosSolver::SetAztecOption)
		//.def("SetAztecParameter",		&daeTrilinosSolver::SetAztecParameter)
		;

	def("daeCreateTrilinosSolver",      daeCreateTrilinosSolver,  return_value_policy<reference_existing_object>());
	def("daeTrilinosSupportedSolvers",  pydaeTrilinosSupportedSolvers);
}
