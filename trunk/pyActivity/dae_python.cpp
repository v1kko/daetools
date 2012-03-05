#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <numpy/core/include/numpy/noprefix.h>
using namespace boost::python;

BOOST_PYTHON_MODULE(pyActivity)
{
    import_array(); 
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
 
/**************************************************************
    Enums
***************************************************************/
    enum_<daeeStopCriterion>("daeeStopCriterion")
        .value("eStopAtGlobalDiscontinuity",    dae::core::eStopAtGlobalDiscontinuity)
        .value("eStopAtModelDiscontinuity",     dae::core::eStopAtModelDiscontinuity)
        .value("eDoNotStopAtDiscontinuity",     dae::core::eDoNotStopAtDiscontinuity)
        .export_values()
    ;

    enum_<daeeActivityAction>("daeeActivityAction")
        .value("eAAUnknown",        dae::activity::eAAUnknown)
        .value("eRunActivity",      dae::activity::eRunActivity)
        .value("ePauseActivity",    dae::activity::ePauseActivity)
        .export_values()
    ;
    enum_<daeeSimulationMode>("daeeSimulationMode")
        .value("eSimulation",			dae::activity::eSimulation)
        .value("eOptimization",			dae::activity::eOptimization)
        .value("eParameterEstimation",	dae::activity::eParameterEstimation)
        .export_values()
    ;
	
/**************************************************************
    daeSimulation_t
***************************************************************/
    class_<daeSimulation_t, boost::noncopyable>("daeSimulation_t", no_init)
        .def("GetModel",                    pure_virtual(&daeSimulation_t::GetModel), return_internal_reference<>())
        .def("SetModel",                    pure_virtual(&daeSimulation_t::SetModel))
        .def("SetUpParametersAndDomains",   pure_virtual(&daeSimulation_t::SetUpParametersAndDomains))
        .def("SetUpVariables",              pure_virtual(&daeSimulation_t::SetUpVariables))
        .def("SetUpOptimization",           pure_virtual(&daeSimulation_t::SetUpOptimization))
        .def("SetUpParameterEstimation",    pure_virtual(&daeSimulation_t::SetUpParameterEstimation))
        .def("SetUpSensitivityAnalysis",    pure_virtual(&daeSimulation_t::SetUpSensitivityAnalysis))
        .def("Run",                         pure_virtual(&daeSimulation_t::Run))
	    .def("CleanUpSetupData",            pure_virtual(&daeSimulation_t::CleanUpSetupData))
        .def("Finalize",                    pure_virtual(&daeSimulation_t::Finalize))
        .def("Reset",                       pure_virtual(&daeSimulation_t::Reset))
        .def("ReRun",                       pure_virtual(&daeSimulation_t::ReRun))
        .def("ReportData",                  pure_virtual(&daeSimulation_t::ReportData))
        .def("StoreInitializationValues",   pure_virtual(&daeSimulation_t::StoreInitializationValues))
        .def("LoadInitializationValues",    pure_virtual(&daeSimulation_t::LoadInitializationValues))

        .def("Pause",                       pure_virtual(&daeSimulation_t::Pause))
        .def("Resume",                      pure_virtual(&daeSimulation_t::Resume)) 
        
        .def("Initialize",                  pure_virtual(&daeSimulation_t::Initialize), ( boost::python::arg("CalculateSensitivities") = false ) )
        .def("Reinitialize",                pure_virtual(&daeSimulation_t::Reinitialize))
        .def("SolveInitial",                pure_virtual(&daeSimulation_t::SolveInitial))
        .def("Integrate",                   pure_virtual(&daeSimulation_t::Integrate))
        .def("IntegrateForTimeInterval",    pure_virtual(&daeSimulation_t::IntegrateForTimeInterval))
        .def("IntegrateUntilTime",          pure_virtual(&daeSimulation_t::IntegrateUntilTime))
       ;
       
    class_<daepython::daeDefaultSimulationWrapper, bases<daeSimulation_t>, boost::noncopyable>("daeSimulation")
		.add_property("Model",                  &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_)
		.add_property("model",                  &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_)
		.add_property("m",                      &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_)  
        
		.add_property("DataReporter",			&daepython::daeDefaultSimulationWrapper::GetDataReporter_)
		.add_property("Log",					&daepython::daeDefaultSimulationWrapper::GetLog_)
		.add_property("DAESolver",				&daepython::daeDefaultSimulationWrapper::GetDAESolver_)  

        .add_property("CurrentTime",			&daeSimulation::GetCurrentTime)  
        .add_property("TimeHorizon",			&daeSimulation::GetTimeHorizon,			&daeSimulation::SetTimeHorizon)
        .add_property("ReportingInterval",		&daeSimulation::GetReportingInterval,	&daeSimulation::SetReportingInterval)
        .add_property("NextReportingTime",		&daeSimulation::GetNextReportingTime)       
        .add_property("ReportingTimes",			&daepython::daeDefaultSimulationWrapper::GetReportingTimes,    &daepython::daeDefaultSimulationWrapper::SetReportingTimes)
        
		.add_property("ActivityAction",			&daeSimulation::GetActivityAction)
        .add_property("InitialConditionMode",   &daeSimulation::GetInitialConditionMode,	&daeSimulation::SetInitialConditionMode)
        .add_property("SimulationMode",			&daeSimulation::GetSimulationMode,			&daeSimulation::SetSimulationMode)

        .add_property("NumberOfObjectiveFunctions",		&daeSimulation::GetNumberOfObjectiveFunctions, &daeSimulation::SetNumberOfObjectiveFunctions)
		.add_property("ObjectiveFunction",				&daepython::daeDefaultSimulationWrapper::GetObjectiveFunction_)
        .add_property("ObjectiveFunctions",				&daepython::daeDefaultSimulationWrapper::GetObjectiveFunctions)
        .add_property("OptimizationVariables",			&daepython::daeDefaultSimulationWrapper::GetOptimizationVariables)
        .add_property("Constraints",					&daepython::daeDefaultSimulationWrapper::GetConstraints)
        
		.add_property("MeasuredVariables",				&daepython::daeDefaultSimulationWrapper::GetMeasuredVariables)
        .add_property("InputVariables",					&daepython::daeDefaultSimulationWrapper::GetInputVariables)
        .add_property("ModelParameters",				&daepython::daeDefaultSimulationWrapper::GetModelParameters)

        //.def("GetModel",                    &daeSimulation::GetModel, return_internal_reference<>())
        //.def("SetModel",                    &daeSimulation::SetModel)  
 
        .def("SetUpParametersAndDomains",   &daeSimulation::SetUpParametersAndDomains, &daepython::daeDefaultSimulationWrapper::def_SetUpParametersAndDomains)
        .def("SetUpVariables",              &daeSimulation::SetUpVariables,            &daepython::daeDefaultSimulationWrapper::def_SetUpVariables)
        .def("SetUpOptimization",			&daeSimulation::SetUpOptimization,		   &daepython::daeDefaultSimulationWrapper::def_SetUpOptimization)
        .def("SetUpParameterEstimation",    &daeSimulation::SetUpParameterEstimation,  &daepython::daeDefaultSimulationWrapper::def_SetUpParameterEstimation)
        .def("SetUpSensitivityAnalysis",    &daeSimulation::SetUpSensitivityAnalysis,  &daepython::daeDefaultSimulationWrapper::def_SetUpSensitivityAnalysis)
        .def("Run",                         &daeSimulation::Run,                       &daepython::daeDefaultSimulationWrapper::def_Run)
	    .def("CleanUpSetupData",            &daeSimulation::CleanUpSetupData,		   &daepython::daeDefaultSimulationWrapper::def_CleanUpSetupData)

        .def("Reset",                       &daeSimulation::Reset)  
        .def("ReRun",                       &daeSimulation::ReRun)
        .def("Finalize",                    &daeSimulation::Finalize)
        .def("ReportData",                  &daeSimulation::ReportData)
        .def("StoreInitializationValues",   &daeSimulation::StoreInitializationValues)
        .def("LoadInitializationValues",    &daeSimulation::LoadInitializationValues)

        .def("Pause",                       &daeSimulation::Pause)
        .def("Resume",                      &daeSimulation::Resume)
   
        .def("Initialize",					&daepython::daeDefaultSimulationWrapper::Initialize, ( boost::python::arg("CalculateSensitivities") = false ) )
        .def("Reinitialize",                &daeSimulation::Reinitialize)  
        .def("SolveInitial",                &daeSimulation::SolveInitial)
        .def("Integrate",                   &daeSimulation::Integrate, ( boost::python::arg("ReportDataAroundDiscontinuities") = true ) )
        .def("IntegrateForTimeInterval",    &daeSimulation::IntegrateForTimeInterval, ( boost::python::arg("ReportDataAroundDiscontinuities") = true ) )
        .def("IntegrateUntilTime",          &daeSimulation::IntegrateUntilTime, ( boost::python::arg("ReportDataAroundDiscontinuities") = true ) )
     
        .def("CreateEqualityConstraint",    &daeSimulation::CreateEqualityConstraint, return_internal_reference<>())
        .def("CreateInequalityConstraint",  &daeSimulation::CreateInequalityConstraint, return_internal_reference<>())

        .def("SetContinuousOptimizationVariable",	&daepython::daeDefaultSimulationWrapper::SetContinuousOptimizationVariable1, return_internal_reference<>())
        .def("SetIntegerOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetIntegerOptimizationVariable1, return_internal_reference<>())
        .def("SetBinaryOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetBinaryOptimizationVariable1, return_internal_reference<>())

        .def("SetContinuousOptimizationVariable",	&daepython::daeDefaultSimulationWrapper::SetContinuousOptimizationVariable2, return_internal_reference<>())
        .def("SetIntegerOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetIntegerOptimizationVariable2, return_internal_reference<>())
        .def("SetBinaryOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetBinaryOptimizationVariable2, return_internal_reference<>())
		
        .def("SetMeasuredVariable",   &daepython::daeDefaultSimulationWrapper::SetMeasuredVariable1, return_internal_reference<>())
        .def("SetInputVariable",      &daepython::daeDefaultSimulationWrapper::SetInputVariable1, return_internal_reference<>())
        .def("SetModelParameter",     &daepython::daeDefaultSimulationWrapper::SetModelParameter1, return_internal_reference<>())

        .def("SetMeasuredVariable",   &daepython::daeDefaultSimulationWrapper::SetMeasuredVariable2, return_internal_reference<>())
        .def("SetInputVariable",      &daepython::daeDefaultSimulationWrapper::SetInputVariable2, return_internal_reference<>())
        .def("SetModelParameter",     &daepython::daeDefaultSimulationWrapper::SetModelParameter2, return_internal_reference<>())

        //.def("EnterConditionalIntegrationMode",   &daeSimulation::EnterConditionalIntegrationMode)
        //.def("IntegrateUntilConditionSatisfied",  &daeSimulation::IntegrateUntilConditionSatisfied)
        ; 
      
/**************************************************************
	daeOptimization_t
***************************************************************/
	class_<daepython::daeOptimizationWrapper, boost::noncopyable>("daeOptimization_t", no_init)
		.def("Initialize",             pure_virtual(&daeOptimization_t::Initialize))
		.def("Run",                    pure_virtual(&daeOptimization_t::Run))
		.def("Finalize",               pure_virtual(&daeOptimization_t::Finalize))
	;
	
	class_<daeOptimization, bases<daeOptimization_t>, boost::noncopyable>("daeOptimization")
		.def("Initialize",             &daeOptimization::Initialize)
		.def("Run",                    &daeOptimization::Run)
		.def("Finalize",               &daeOptimization::Finalize)
	 ;

    
    
}
