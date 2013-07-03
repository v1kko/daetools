#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include "docstrings.h"
#include <noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyIDAS)
{
	import_array(); 
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");  
    
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
    
    class_<daeArray<real_t>, boost::noncopyable>("daeArray", no_init)  
        .add_property("N",	     &daeArray<real_t>::GetSize)
        .add_property("Values",	 &daepython::daeArray_GetValues)
        .def("__getitem__",	     &daepython::daeArray_GetItem)
    ;
    
    class_<daeDenseMatrix, boost::noncopyable>("daeDenseMatrix", no_init)
        .add_property("N",	 &daeDenseMatrix::GetNrows)
        .add_property("M",	 &daeDenseMatrix::GetNcols)
        .def("__getitem__",	 &daeDenseMatrix::GetItem)
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
