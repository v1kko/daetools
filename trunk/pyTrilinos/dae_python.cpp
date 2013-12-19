#include "stdafx.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
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
    class_<Teuchos::ParameterList, boost::noncopyable>("TeuchosParameterList", DOCSTR_TeuchosParameterList, no_init)
        .def("Print",	&pydaeParameterListPrint, ( boost::python::arg("self") ), DOCSTR_TeuchosParameterList_Print)

        .def("get_string",	&pydaeParameterListGet_string, ( boost::python::arg("self"), boost::python::arg("name") ), DOCSTR_TeuchosParameterList_get_string)
        .def("get_float",	&pydaeParameterListGet_float,  ( boost::python::arg("self"), boost::python::arg("name") ), DOCSTR_TeuchosParameterList_get_float)
        .def("get_int",		&pydaeParameterListGet_int,    ( boost::python::arg("self"), boost::python::arg("name") ), DOCSTR_TeuchosParameterList_get_int)
        .def("get_bool",	&pydaeParameterListGet_bool,   ( boost::python::arg("self"), boost::python::arg("name") ), DOCSTR_TeuchosParameterList_get_bool)
		
        .def("set_string",	&pydaeParameterListSet_string, ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_TeuchosParameterList_set_string)
        .def("set_float",	&pydaeParameterListSet_float,  ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_TeuchosParameterList_set_float)
        .def("set_int",		&pydaeParameterListSet_int,    ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_TeuchosParameterList_set_int)
        .def("set_bool",	&pydaeParameterListSet_bool,   ( boost::python::arg("self"), boost::python::arg("name"), boost::python::arg("value") ), DOCSTR_TeuchosParameterList_set_bool)
		;

	class_<daeIDALASolver_t, boost::noncopyable>("daeIDALASolver_t", no_init)
		.def("SaveAsXPM",	pure_virtual(&daeIDALASolver_t::SaveAsXPM))
		;

    class_<daeTrilinosSolver, bases<daeIDALASolver_t>, boost::noncopyable>("daeTrilinosSolver", DOCSTR_daeTrilinosSolver, no_init)
        .add_property("Name",				&daeTrilinosSolver::GetName,                DOCSTR_daeTrilinosSolver_Name)
        .add_property("PreconditionerName",	&daeTrilinosSolver::GetPreconditionerName,  DOCSTR_daeTrilinosSolver_PreconditionerName)
        .def_readwrite("NumIters",			&daeTrilinosSolver::m_nNumIters,            DOCSTR_daeTrilinosSolver_NumIters)
        .def_readwrite("Tolerance",			&daeTrilinosSolver::m_dTolerance,           DOCSTR_daeTrilinosSolver_Tolerance)

        .def("SaveAsXPM",				&daeTrilinosSolver::SaveAsXPM,               ( boost::python::arg("self"), boost::python::arg("xpmFilename") ), DOCSTR_daeTrilinosSolver_SaveAsXPM)
        .def("SaveAsMatrixMarketFile",	&daeTrilinosSolver::SaveAsMatrixMarketFile,  ( boost::python::arg("self"), boost::python::arg("filename"), boost::python::arg("matrixName"), boost::python::arg("description") ), DOCSTR_daeTrilinosSolver_SaveAsMatrixMarketFile)
        .def("PrintPreconditionerInfo",	&daeTrilinosSolver::PrintPreconditionerInfo, ( boost::python::arg("self") ), DOCSTR_daeTrilinosSolver_PrintPreconditionerInfo)
        .def("SetOpenBLASNoThreads",	&daeTrilinosSolver::SetOpenBLASNoThreads,    ( boost::python::arg("self"), boost::python::arg("noThreads") ))

        .add_property("AztecOOOptions",		make_function(&daeTrilinosSolver::GetAztecOOOptions, return_internal_reference<>()), DOCSTR_daeTrilinosSolver_AztecOOOptions)
        .add_property("IfpackOptions",		make_function(&daeTrilinosSolver::GetIfpackOptions,  return_internal_reference<>()), DOCSTR_daeTrilinosSolver_IfpackOptions)
        .add_property("MLOptions",			make_function(&daeTrilinosSolver::GetMLOptions,      return_internal_reference<>()), DOCSTR_daeTrilinosSolver_MLOptions)
        .add_property("AmesosOptions",		make_function(&daeTrilinosSolver::GetAmesosOptions,  return_internal_reference<>()), DOCSTR_daeTrilinosSolver_AmesosOptions)
		;

    def("daeCreateTrilinosSolver",      daeCreateTrilinosSolver, return_value_policy<manage_new_object>(), ( boost::python::arg("solverName"), boost::python::arg("preconditionerName") ), DOCSTR_daeCreateTrilinosSolver);
    def("daeTrilinosSupportedSolvers",  pydaeTrilinosSupportedSolvers, DOCSTR_daeTrilinosSupportedSolvers);
}
