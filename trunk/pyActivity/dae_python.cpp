#include "stdafx.h"
#include "python_wraps.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <noprefix.h>
#include "docstrings.h"
using namespace boost::python;

BOOST_PYTHON_MODULE(pyActivity)
{
    import_array();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    
    docstring_options doc_options(true, true, false);
 
/**************************************************************
    Enums
***************************************************************/
    enum_<daeeStopCriterion>("daeeStopCriterion")
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
        .def("RegisterData",                pure_virtual(&daeSimulation_t::RegisterData))
        .def("StoreInitializationValues",   pure_virtual(&daeSimulation_t::StoreInitializationValues))
        .def("LoadInitializationValues",    pure_virtual(&daeSimulation_t::LoadInitializationValues))

        .def("Pause",                       pure_virtual(&daeSimulation_t::Pause))
        .def("Resume",                      pure_virtual(&daeSimulation_t::Resume))
        
        .def("Initialize",                  pure_virtual(&daeSimulation_t::Initialize))
        .def("Reinitialize",                pure_virtual(&daeSimulation_t::Reinitialize))
        .def("SolveInitial",                pure_virtual(&daeSimulation_t::SolveInitial))
        .def("Integrate",                   pure_virtual(&daeSimulation_t::Integrate))
        .def("IntegrateForTimeInterval",    pure_virtual(&daeSimulation_t::IntegrateForTimeInterval))
        .def("IntegrateUntilTime",          pure_virtual(&daeSimulation_t::IntegrateUntilTime))
       ;
       
    class_<daepython::daeDefaultSimulationWrapper, bases<daeSimulation_t>, boost::noncopyable>("daeSimulation", DOCSTR_daeSimulation, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeSimulation_init))
		
        .add_property("Model",                  &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_, DOCSTR_daeSimulation_Model)
		.add_property("model",                  &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_, DOCSTR_daeSimulation_model)
		.add_property("m",                      &daepython::daeDefaultSimulationWrapper::GetModel_,
												&daepython::daeDefaultSimulationWrapper::SetModel_, DOCSTR_daeSimulation_m) 

        .add_property("EquationExecutionInfos", &daepython::daeDefaultSimulationWrapper::GetEqExecutionInfos,       DOCSTR_daeSimulation_EquationExecutionInfos)
        .add_property("Values",                 &daepython::daeDefaultSimulationWrapper::GetValues,                 DOCSTR_daeSimulation_Values)
        .add_property("TimeDerivatives",        &daepython::daeDefaultSimulationWrapper::GetTimeDerivatives,        DOCSTR_daeSimulation_TimeDerivatives)
        .add_property("VariableTypes",          &daepython::daeDefaultSimulationWrapper::GetVariableTypes,          DOCSTR_daeSimulation_VariableTypes)
        .add_property("IndexMappings",          &daepython::daeDefaultSimulationWrapper::GetIndexMappings,          DOCSTR_daeSimulation_IndexMappings)
        .add_property("NumberOfEquations",      &daepython::daeDefaultSimulationWrapper::GetNumberOfEquations,      DOCSTR_daeSimulation_NumberOfEquations)
        .add_property("TotalNumberOfVariables", &daepython::daeDefaultSimulationWrapper::GetTotalNumberOfVariables, DOCSTR_daeSimulation_TotalNumberOfVariables)
        .add_property("RelativeTolerance",      &daepython::daeDefaultSimulationWrapper::GetRelativeTolerance,      DOCSTR_daeSimulation_RelativeTolerance)
        .add_property("AbsoluteTolerances",     &daepython::daeDefaultSimulationWrapper::GetAbsoluteTolerances,     DOCSTR_daeSimulation_AbsoluteTolerances)
            
        .add_property("LastSatisfiedCondition", make_function(&daeSimulation::GetLastSatisfiedCondition, return_internal_reference<>()),
                                                                                                            DOCSTR_daeSimulation_LastSatisfiedCondition)


        .add_property("JSONRuntimeSettings",    &daeSimulation::GetJSONRuntimeSettings,
                                                &daeSimulation::SetJSONRuntimeSettings,                     DOCSTR_daeSimulation_JSONRuntimeSettings)


        .add_property("DataReporter",			&daepython::daeDefaultSimulationWrapper::GetDataReporter_,  DOCSTR_daeSimulation_DataReporter)
        .add_property("Log",					&daepython::daeDefaultSimulationWrapper::GetLog_,           DOCSTR_daeSimulation_Log)
        .add_property("DAESolver",				&daepython::daeDefaultSimulationWrapper::GetDAESolver_,     DOCSTR_daeSimulation_DAESolver)

        .add_property("CurrentTime",			&daeSimulation::GetCurrentTime,                             DOCSTR_daeSimulation_CurrentTime)  
        .add_property("TimeHorizon",			&daeSimulation::GetTimeHorizon,			
                                                &daeSimulation::SetTimeHorizon,                             DOCSTR_daeSimulation_TimeHorizon)
        .add_property("ReportingInterval",		&daeSimulation::GetReportingInterval,	
                                                &daeSimulation::SetReportingInterval,                       DOCSTR_daeSimulation_ReportingInterval)
        .add_property("NextReportingTime",		&daeSimulation::GetNextReportingTime,                       DOCSTR_daeSimulation_NextReportingTime)       
        .add_property("ReportingTimes",			&daepython::daeDefaultSimulationWrapper::GetReportingTimes,   
                                                &daepython::daeDefaultSimulationWrapper::SetReportingTimes, DOCSTR_daeSimulation_ReportingTimes)
        
		.add_property("ActivityAction",			&daeSimulation::GetActivityAction,          DOCSTR_daeSimulation_ActivityAction)
        .add_property("InitialConditionMode",   &daeSimulation::GetInitialConditionMode,
                                                &daeSimulation::SetInitialConditionMode,    DOCSTR_daeSimulation_InitialConditionMode)
        .add_property("SimulationMode",			&daeSimulation::GetSimulationMode,			
                                                &daeSimulation::SetSimulationMode,          DOCSTR_daeSimulation_SimulationMode)

        .add_property("NumberOfObjectiveFunctions",		&daeSimulation::GetNumberOfObjectiveFunctions, 
                                                        &daeSimulation::SetNumberOfObjectiveFunctions,                      DOCSTR_daeSimulation_NumberOfObjectiveFunctions)
		.add_property("ObjectiveFunction",				&daepython::daeDefaultSimulationWrapper::GetObjectiveFunction_,     DOCSTR_daeSimulation_ObjectiveFunction)
        .add_property("ObjectiveFunctions",				&daepython::daeDefaultSimulationWrapper::GetObjectiveFunctions,     DOCSTR_daeSimulation_ObjectiveFunctions)
        .add_property("OptimizationVariables",			&daepython::daeDefaultSimulationWrapper::GetOptimizationVariables,  DOCSTR_daeSimulation_OptimizationVariables)
        .add_property("Constraints",					&daepython::daeDefaultSimulationWrapper::GetConstraints,            DOCSTR_daeSimulation_Constraints)
        
		.add_property("MeasuredVariables",				&daepython::daeDefaultSimulationWrapper::GetMeasuredVariables,      DOCSTR_daeSimulation_MeasuredVariables)
        .add_property("InputVariables",					&daepython::daeDefaultSimulationWrapper::GetInputVariables,         DOCSTR_daeSimulation_InputVariables)
        .add_property("ModelParameters",				&daepython::daeDefaultSimulationWrapper::GetModelParameters,        DOCSTR_daeSimulation_ModelParameters)

        //.def("GetModel",                    &daeSimulation::GetModel, return_internal_reference<>())
        //.def("SetModel",                    &daeSimulation::SetModel)  
 
        .def("SetUpParametersAndDomains",   &daeSimulation::SetUpParametersAndDomains, &daepython::daeDefaultSimulationWrapper::def_SetUpParametersAndDomains, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SetUpParametersAndDomains)
        .def("SetUpVariables",              &daeSimulation::SetUpVariables,            &daepython::daeDefaultSimulationWrapper::def_SetUpVariables, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SetUpVariables)
        .def("SetUpOptimization",			&daeSimulation::SetUpOptimization,		   &daepython::daeDefaultSimulationWrapper::def_SetUpOptimization, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SetUpOptimization)
        .def("SetUpParameterEstimation",    &daeSimulation::SetUpParameterEstimation,  &daepython::daeDefaultSimulationWrapper::def_SetUpParameterEstimation, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SetUpParameterEstimation)
        .def("SetUpSensitivityAnalysis",    &daeSimulation::SetUpSensitivityAnalysis,  &daepython::daeDefaultSimulationWrapper::def_SetUpSensitivityAnalysis, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SetUpSensitivityAnalysis)
        .def("Run",                         &daeSimulation::Run,                       &daepython::daeDefaultSimulationWrapper::def_Run, 
                                            ( arg("self") ), DOCSTR_daeSimulation_Run)
	    .def("CleanUpSetupData",            &daeSimulation::CleanUpSetupData,		   &daepython::daeDefaultSimulationWrapper::def_CleanUpSetupData, 
                                            ( arg("self") ), DOCSTR_daeSimulation_CleanUpSetupData)

        .def("Reset",                       &daeSimulation::Reset,                      ( arg("self") ), DOCSTR_daeSimulation_Reset) 
        .def("ReRun",                       &daeSimulation::ReRun,                      ( arg("self") ), DOCSTR_daeSimulation_ReRun)
        .def("Finalize",                    &daeSimulation::Finalize,                   ( arg("self") ), DOCSTR_daeSimulation_Finalize)
        .def("ReportData",                  &daeSimulation::ReportData,                 ( arg("self"), arg("currentTime") ), DOCSTR_daeSimulation_ReportData)
        .def("RegisterData",                &daeSimulation::RegisterData,               ( arg("self"), arg("iteration") ), DOCSTR_daeSimulation_RegisterData)
        .def("StoreInitializationValues",   &daeSimulation::StoreInitializationValues,  ( arg("self"), arg("filename") ), DOCSTR_daeSimulation_StoreInitializationValues)
        .def("LoadInitializationValues",    &daeSimulation::LoadInitializationValues,   ( arg("self"), arg("filename") ), DOCSTR_daeSimulation_LoadInitializationValues)

        .def("Pause",                       &daeSimulation::Pause,  ( arg("self") ), DOCSTR_daeSimulation_Pause)
        .def("Resume",                      &daeSimulation::Resume, ( arg("self") ), DOCSTR_daeSimulation_Resume)
   
        .def("Initialize",					&daeSimulation::Initialize,
                                            ( arg("self"), arg("daeSolver"), arg("dataReporter"), arg("log"), arg("calculateSensitivities") = false, arg("jsonRuntimeSettings") = "" ), DOCSTR_daeSimulation_Initialize)
        .def("Reinitialize",                &daeSimulation::Reinitialize, 
                                            ( arg("self") ), DOCSTR_daeSimulation_Reinitialize)  
        .def("SolveInitial",                &daeSimulation::SolveInitial, 
                                            ( arg("self") ), DOCSTR_daeSimulation_SolveInitial)
        .def("Integrate",                   &daeSimulation::Integrate, 
                                            ( arg("self"), arg("stopCriterion"), arg("reportDataAroundDiscontinuities") = true ), DOCSTR_daeSimulation_Integrate)
        .def("IntegrateForTimeInterval",    &daeSimulation::IntegrateForTimeInterval, 
                                            ( arg("self"), arg("timeInterval"), arg("reportDataAroundDiscontinuities") = true ), DOCSTR_daeSimulation_IntegrateForTimeInterval)
        .def("IntegrateUntilTime",          &daeSimulation::IntegrateUntilTime, 
                                            ( arg("self"), arg("time"), arg("stopCriterion"), arg("reportDataAroundDiscontinuities") = true ), DOCSTR_daeSimulation_IntegrateUntilTime)
     
        .def("CreateEqualityConstraint",    &daeSimulation::CreateEqualityConstraint, return_internal_reference<>(), 
                                            ( arg("self"), arg("description") ), DOCSTR_daeSimulation_CreateEqualityConstraint)
        .def("CreateInequalityConstraint",  &daeSimulation::CreateInequalityConstraint, return_internal_reference<>(), 
                                            ( arg("self"), arg("description") ), DOCSTR_daeSimulation_CreateInequalityConstraint)

        .def("SetContinuousOptimizationVariable",	&daepython::daeDefaultSimulationWrapper::SetContinuousOptimizationVariable1, 
                                                    return_internal_reference<>(), ( arg("self"), arg("variable"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetContinuousOptimizationVariable)
        .def("SetIntegerOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetIntegerOptimizationVariable1, 
                                                    return_internal_reference<>(), ( arg("self"), arg("variable"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetIntegerOptimizationVariable)
        .def("SetBinaryOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetBinaryOptimizationVariable1, 
                                                    return_internal_reference<>(), ( arg("self"), arg("variable"), arg("defaultValue") ), DOCSTR_daeSimulation_SetBinaryOptimizationVariable)

        .def("SetContinuousOptimizationVariable",	&daepython::daeDefaultSimulationWrapper::SetContinuousOptimizationVariable2, 
                                                    return_internal_reference<>(), ( arg("self"), arg("ad"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetContinuousOptimizationVariable)
        .def("SetIntegerOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetIntegerOptimizationVariable2, 
                                                    return_internal_reference<>(), ( arg("self"), arg("ad"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetIntegerOptimizationVariable)
        .def("SetBinaryOptimizationVariable",		&daepython::daeDefaultSimulationWrapper::SetBinaryOptimizationVariable2,
                                                    return_internal_reference<>(), ( arg("self"), arg("ad"), arg("defaultValue") ), DOCSTR_daeSimulation_SetBinaryOptimizationVariable)
		
        .def("SetMeasuredVariable",     &daepython::daeDefaultSimulationWrapper::SetMeasuredVariable1, 
                                        return_internal_reference<>(), ( arg("self"), arg("variable") ), DOCSTR_daeSimulation_SetMeasuredVariable)
        .def("SetInputVariable",        &daepython::daeDefaultSimulationWrapper::SetInputVariable1, 
                                        return_internal_reference<>(), ( arg("self"), arg("variable") ), DOCSTR_daeSimulation_SetInputVariable)
        .def("SetModelParameter",       &daepython::daeDefaultSimulationWrapper::SetModelParameter1, 
                                        return_internal_reference<>(), ( arg("self"), arg("variable"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetModelParameter)

        .def("SetMeasuredVariable",     &daepython::daeDefaultSimulationWrapper::SetMeasuredVariable2, 
                                        return_internal_reference<>(), ( arg("self"), arg("ad") ), DOCSTR_daeSimulation_SetMeasuredVariable)
        .def("SetInputVariable",        &daepython::daeDefaultSimulationWrapper::SetInputVariable2, 
                                        return_internal_reference<>(), ( arg("self"), arg("ad") ), DOCSTR_daeSimulation_SetInputVariable)
        .def("SetModelParameter",       &daepython::daeDefaultSimulationWrapper::SetModelParameter2, 
                                        return_internal_reference<>(), ( arg("self"), arg("ad"), arg("lowerBound"), arg("upperBound"), arg("defaultValue") ), DOCSTR_daeSimulation_SetModelParameter)

        //.def("EnterConditionalIntegrationMode",   &daeSimulation::EnterConditionalIntegrationMode)
        //.def("IntegrateUntilConditionSatisfied",  &daeSimulation::IntegrateUntilConditionSatisfied)
        ; 
      
/**************************************************************
	daeOptimization_t
***************************************************************/
	class_<daepython::daeOptimization_tWrapper, boost::noncopyable>("daeOptimization_t", no_init)
		.def("Initialize",             pure_virtual(&daeOptimization_t::Initialize))
		.def("Run",                    pure_virtual(&daeOptimization_t::Run))
		.def("Finalize",               pure_virtual(&daeOptimization_t::Finalize))
        .def("StartIterationRun",      pure_virtual(&daeOptimization_t::StartIterationRun))
        .def("EndIterationRun",        pure_virtual(&daeOptimization_t::EndIterationRun))
	;
	
	class_<daepython::daeOptimizationWrapper, bases<daeOptimization_t>, boost::noncopyable>("daeOptimization", DOCSTR_daeOptimization, no_init)
        .def(init<>(( arg("self") ), DOCSTR_daeOptimization_init))

        .add_property("Simulation",	   make_function(&daepython::daeOptimizationWrapper::GetSimulation_, return_internal_reference<>()))
            
        .def("Initialize",             &daeOptimization::Initialize, ( arg("self"),
                                                                       arg("simulation"),
                                                                       arg("nlpSolver"), 
                                                                       arg("daeSolver"),
                                                                       arg("dataReporter"),
                                                                       arg("log"),
                                                                       arg("initializationFile") = std::string("")
                                                                     ), DOCSTR_daeOptimization_Initialize)
        .def("Run",                    &daeOptimization::Run,        ( arg("self") ), DOCSTR_daeOptimization_Run)
		.def("Finalize",               &daeOptimization::Finalize,   ( arg("self") ), DOCSTR_daeOptimization_Finalize)
            
        .def("StartIterationRun",      &daeOptimization_t::StartIterationRun, &daepython::daeOptimizationWrapper::def_StartIterationRun,
                                       ( arg("self"), arg("iteration") ), DOCSTR_daeOptimization_StartIteration)
            
        .def("EndIterationRun",        &daeOptimization_t::EndIterationRun, &daepython::daeOptimizationWrapper::def_EndIterationRun,
                                       ( arg("self"), arg("iteration") ), DOCSTR_daeOptimization_EndIteration)
	 ;
    
    
}
