#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "docstrings.h"
#include "../LA_Trilinos_Amesos/trilinos_amesos_la_solver.h"
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

void pydaeParameterListPrint(Teuchos::ParameterList& list);
void pydaeParameterListSet_string(Teuchos::ParameterList& list, const string& strName, const string& Value);
void pydaeParameterListSet_float(Teuchos::ParameterList& list, const string& strName, double Value);
void pydaeParameterListSet_int(Teuchos::ParameterList& list, const string& strName, int Value);
void pydaeParameterListSet_bool(Teuchos::ParameterList& list, const string& strName, bool Value);
string pydaeParameterListGet_string(Teuchos::ParameterList& list, const string& strName);
double pydaeParameterListGet_float(Teuchos::ParameterList& list, const string& strName);
int    pydaeParameterListGet_int(Teuchos::ParameterList& list, const string& strName);
bool   pydaeParameterListGet_bool(Teuchos::ParameterList& list, const string& strName);


void pydaeParameterListPrint(Teuchos::ParameterList& list)
{
	list.print(std::cout, 0, true, true);
}

void pydaeParameterListSet_string(Teuchos::ParameterList& list, const string& strName, const string& Value)
{
	list.set<string>(strName, Value);
}

void pydaeParameterListSet_float(Teuchos::ParameterList& list, const string& strName, double Value)
{
	list.set<double>(strName, Value);
}

void pydaeParameterListSet_int(Teuchos::ParameterList& list, const string& strName, int Value)
{
	list.set<int>(strName, Value);
}

void pydaeParameterListSet_bool(Teuchos::ParameterList& list, const string& strName, bool Value)
{
	list.set<bool>(strName, Value);
}


string pydaeParameterListGet_string(Teuchos::ParameterList& list, const string& strName)
{
	return list.get<string>(strName);
}

double pydaeParameterListGet_float(Teuchos::ParameterList& list, const string& strName)
{
	return list.get<double>(strName);
}

int pydaeParameterListGet_int(Teuchos::ParameterList& list, const string& strName)
{
	return list.get<int>(strName);
}

bool pydaeParameterListGet_bool(Teuchos::ParameterList& list, const string& strName)
{
	return list.get<bool>(strName);
}

BOOST_PYTHON_MODULE(pyTrilinos)
{
    docstring_options doc_options(true, true, false);
    
/**************************************************************
	TrilinosAmesos LA Solver
***************************************************************/
	class_<Teuchos::ParameterList>("TeuchosParameterList")
		.def("Print",	&pydaeParameterListPrint)

		.def("get_string",	&pydaeParameterListGet_string)
		.def("get_float",	&pydaeParameterListGet_float)
		.def("get_int",		&pydaeParameterListGet_int)
		.def("get_bool",	&pydaeParameterListGet_bool)
		
		.def("set_string",	&pydaeParameterListSet_string)
		.def("set_float",	&pydaeParameterListSet_float)
		.def("set_int",		&pydaeParameterListSet_int)
		.def("set_bool",	&pydaeParameterListSet_bool)
		;

	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("Create",		pure_virtual(&daeIDALASolver_t::Create))
		.def("Reinitialize",pure_virtual(&daeIDALASolver_t::Reinitialize))
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		//.def("SaveAsMatrixMarketFile",	pure_virtual(&daeIDALASolver_t::SaveAsMatrixMarketFile))
		;

	class_<daeTrilinosSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeTrilinosSolver", init<string, string>())
		.add_property("Name",				&daeTrilinosSolver::GetName)
		.add_property("PreconditionerName",	&daeTrilinosSolver::GetPreconditionerName)
		.def_readwrite("NumIters",			&daeTrilinosSolver::m_nNumIters)
		.def_readwrite("Tolerance",			&daeTrilinosSolver::m_dTolerance)
		
		.def("Create",					&daeTrilinosSolver::Create)
		.def("Reinitialize",			&daeTrilinosSolver::Reinitialize)		
		.def("SaveAsXPM",				&daeTrilinosSolver::SaveAsXPM)
		.def("SaveAsMatrixMarketFile",	&daeTrilinosSolver::SaveAsMatrixMarketFile)
		.def("PrintPreconditionerInfo",	&daeTrilinosSolver::PrintPreconditionerInfo)

		.def("GetAztecOOOptions",		&daeTrilinosSolver::GetAztecOOOptions, return_value_policy<reference_existing_object>())
		.def("GetIfpackOptions",		&daeTrilinosSolver::GetIfpackOptions,  return_value_policy<reference_existing_object>())
		.def("GetMLOptions",			&daeTrilinosSolver::GetMLOptions,      return_value_policy<reference_existing_object>())
		.def("GetAmesosOptions",		&daeTrilinosSolver::GetAmesosOptions,  return_value_policy<reference_existing_object>())
		;

	def("daeCreateTrilinosSolver",      daeCreateTrilinosSolver, return_value_policy<manage_new_object>());
	def("daeTrilinosSupportedSolvers",  pydaeTrilinosSupportedSolvers);
}
