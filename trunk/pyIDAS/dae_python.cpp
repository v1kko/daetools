#include "stdafx.h"
#include "python_wraps.h"
#include "docstrings.h"
//#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
//#include <noprefix.h>
using namespace boost::python;

// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_FULL_VER  == 190024210
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(daeLog_t)
POINTER_CONVERSION(daeLASolver_t)
POINTER_CONVERSION(daeIDALASolver_t)
POINTER_CONVERSION(daeMatrix<real_t>)
POINTER_CONVERSION(daeDenseMatrix)
POINTER_CONVERSION(daeRawDataArray<real_t>)
}
#endif
#endif

BOOST_PYTHON_MODULE(pyIDAS)
{
    //import_array();
    //boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

    docstring_options doc_options(true, true, false);

/**************************************************************
    Enums
***************************************************************/
    enum_<daeeIDALASolverType>("daeeIDALASolverType")
        .value("eSundialsLU",		dae::solver::eSundialsLU)
        .value("eSundialsLapack",	dae::solver::eSundialsLapack)
        .value("eSundialsGMRES",	dae::solver::eSundialsGMRES)
        .value("eThirdParty",		dae::solver::eThirdParty)
        .export_values()
    ;

    class_<daeArray<real_t>, boost::noncopyable>("daeArray_real", no_init)
        .add_property("n",	     &daeArray<real_t>::GetSize)
        .add_property("Values",	 &daepython::daeArray_GetValues)
        .def("__getitem__",  &daeArray<real_t>::GetItem,   ( arg("self"), arg("item") ))
        .def("__setitem__",  &daeArray<real_t>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
        .def("GetItem",      &daeArray<real_t>::GetItem,   ( arg("self"), arg("item") ))
        .def("SetItem",      &daeArray<real_t>::SetItem,   ( arg("self"), arg("item"), arg("value") ))
    ;

    class_<daeRawDataArray<real_t>, bases< daeArray<real_t> >, boost::noncopyable>("daeRawDataArray_real", no_init)
        .def("Print", &daeRawDataArray<real_t>::Print)
    ;

    class_<daeMatrix<real_t>, boost::noncopyable>("daeMatrix_real", no_init)
        .add_property("n",	 &daeMatrix<real_t>::GetNrows)
        .add_property("m",	 &daeMatrix<real_t>::GetNcols)
        .def("__call__",     &daeMatrix<real_t>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("GetItem",      &daeMatrix<real_t>::GetItem,  ( arg("self"), arg("row"), arg("column") ))
        .def("SetItem",      &daeMatrix<real_t>::SetItem,  ( arg("self"), arg("row"), arg("column"), arg("value") ))
    ;

    class_<daeDenseMatrix, bases< daeMatrix<real_t> >, boost::noncopyable>("daeDenseMatrix", no_init)
        .add_property("npyValues",  &daepython::daeDenseMatrix_ndarray)
    ;

/**************************************************************
    daeSolver
***************************************************************/
    class_<daepython::daeDAESolverWrapper, boost::noncopyable>("daeDAESolver_t", DOCSTR_daeDAESolver_t, no_init)
        .add_property("NumberOfVariables",		&daeDAESolver_t::GetNumberOfVariables, DOCSTR_daeDAESolver_t_NumberOfVariables)
        .add_property("Log",					make_function(&daeDAESolver_t::GetLog, return_internal_reference<>()),  DOCSTR_daeDAESolver_t_Log)
        .add_property("RelativeTolerance",		&daeDAESolver_t::GetRelativeTolerance,
                                                &daeDAESolver_t::SetRelativeTolerance, DOCSTR_daeDAESolver_t_RelativeTolerance)
        .add_property("InitialConditionMode",	&daeDAESolver_t::GetInitialConditionMode,
                                                &daeDAESolver_t::SetInitialConditionMode, DOCSTR_daeDAESolver_t_InitialConditionMode)
        .add_property("Name",					&daeDAESolver_t::GetName, DOCSTR_daeDAESolver_t_Name)
        .add_property("SensitivityMatrix",      make_function(&daeDAESolver_t::GetSensitivities, return_internal_reference<>()))

        .def("OnCalculateResiduals",		    pure_virtual(&daeDAESolver_t::OnCalculateResiduals),
                                                ( arg("self") ), DOCSTR_daeDAESolver_t_OnCalculateResiduals)
        .def("OnCalculateConditions",		    pure_virtual(&daeDAESolver_t::OnCalculateConditions),
                                                ( arg("self") ), DOCSTR_daeDAESolver_t_OnCalculateConditions)
        .def("OnCalculateJacobian",		        pure_virtual(&daeDAESolver_t::OnCalculateJacobian),
                                                ( arg("self") ), DOCSTR_daeDAESolver_t_OnCalculateJacobian)
        .def("OnCalculateSensitivityResiduals",	pure_virtual(&daeDAESolver_t::OnCalculateSensitivityResiduals),
                                                ( arg("self") ), DOCSTR_daeDAESolver_t_OnCalculateSensitivityResiduals)
        ;

    class_<daepython::daeIDASolverWrapper, bases<daeDAESolver_t>, boost::noncopyable>("daeIDAS", DOCSTR_daeIDAS, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeIDAS_init))

        .def_readonly("Values",                 &daepython::daeIDASolverWrapper::m_arrValues)
        .def_readonly("TimeDerivatives",        &daepython::daeIDASolverWrapper::m_arrTimeDerivatives)
        .def_readonly("Residuals",              &daepython::daeIDASolverWrapper::m_arrResiduals)
        .def_readonly("Jacobian",               &daepython::daeIDASolverWrapper::m_matJacobian)
        .def_readonly("SensitivityResiduals",	&daepython::daeIDASolverWrapper::m_matSResiduals)

        .add_property("LASolver",               &daepython::daeIDASolverWrapper::GetLASolver,         DOCSTR_daeIDAS_LASolver)
        .add_property("EstLocalErrors",		    &daepython::daeIDASolverWrapper::GetEstLocalErrors_,  DOCSTR_daeIDAS_EstLocalErrors)
        .add_property("ErrWeights",		   	    &daepython::daeIDASolverWrapper::GetErrWeights_,      DOCSTR_daeIDAS_ErrWeights)
        .add_property("IntegratorStats",		&daepython::daeIDASolverWrapper::GetIntegratorStats_, DOCSTR_daeIDAS_IntegratorStats)

        .def("SetLASolver",		&daepython::daeIDASolverWrapper::SetLASolver1, ( arg("self"), arg("laSolverType") ), DOCSTR_daeIDAS_SetLASolver1)
        .def("SetLASolver",		&daepython::daeIDASolverWrapper::SetLASolver2, ( arg("self"), arg("laSolver") ),     DOCSTR_daeIDAS_SetLASolver2)
        .def("SaveMatrixAsXPM",	&daeIDASolver::SaveMatrixAsXPM,                ( arg("self"), arg("xpmFilename") ),  DOCSTR_daeIDAS_SaveMatrixAsXPM)

        .def("OnCalculateResiduals",		    &daeDAESolver_t::OnCalculateResiduals, &daepython::daeIDASolverWrapper::def_OnCalculateResiduals,
                                                ( arg("self") ), DOCSTR_daeIDAS_OnCalculateResiduals)
        .def("OnCalculateConditions",		    &daeDAESolver_t::OnCalculateConditions, &daepython::daeIDASolverWrapper::def_OnCalculateConditions,
                                                ( arg("self") ), DOCSTR_daeIDAS_OnCalculateConditions)
        .def("OnCalculateJacobian",		        &daeDAESolver_t::OnCalculateJacobian, &daepython::daeIDASolverWrapper::def_OnCalculateJacobian,
                                                ( arg("self") ), DOCSTR_daeIDAS_OnCalculateJacobian)
        .def("OnCalculateSensitivityResiduals",	&daeDAESolver_t::OnCalculateSensitivityResiduals, &daepython::daeIDASolverWrapper::def_OnCalculateSensitivityResiduals,
                                                ( arg("self") ), DOCSTR_daeIDAS_OnCalculateSensitivityResiduals)
        ;

}
