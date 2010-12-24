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
     
/**************************************************************
    daeSimulation_t
***************************************************************/
    class_<daepython::daeSimulationWrapper, boost::noncopyable>("daeSimulation_t", no_init)
        .def("GetModel",                    pure_virtual(&daeSimulation_t::GetModel), return_internal_reference<>())
        .def("SetModel",                    pure_virtual(&daeSimulation_t::SetModel))
        .def("SetUpParametersAndDomains",   pure_virtual(&daeSimulation_t::SetUpParametersAndDomains))
        .def("SetUpVariables",              pure_virtual(&daeSimulation_t::SetUpVariables))
        .def("Run",                         pure_virtual(&daeSimulation_t::Run))
        .def("Finalize",                    pure_virtual(&daeSimulation_t::Finalize))
        .def("ReRun",                       pure_virtual(&daeSimulation_t::ReRun))
        .def("ReportData",                  pure_virtual(&daeSimulation_t::ReportData))
        .def("StoreInitializationValues",   pure_virtual(&daeSimulation_t::StoreInitializationValues))
        .def("LoadInitializationValues",    pure_virtual(&daeSimulation_t::LoadInitializationValues))

        .def("Pause",                       pure_virtual(&daeSimulation_t::Pause))
        .def("Resume",                      pure_virtual(&daeSimulation_t::Resume))
        
        .def("InitSimulation",              pure_virtual(&daeSimulation_t::InitSimulation))
        .def("InitOptimization",            pure_virtual(&daeSimulation_t::InitOptimization))
        .def("Reinitialize",                pure_virtual(&daeSimulation_t::Reinitialize))
        .def("SolveInitial",                pure_virtual(&daeSimulation_t::SolveInitial))
        .def("Integrate",                   pure_virtual(&daeSimulation_t::Integrate))
        .def("IntegrateForTimeInterval",    pure_virtual(&daeSimulation_t::IntegrateForTimeInterval))
        .def("IntegrateUntilTime",          pure_virtual(&daeSimulation_t::IntegrateUntilTime))
       ;
       
    class_<daepython::daeDefaultSimulationWrapper, bases<daeSimulation_t>, boost::noncopyable>("daeSimulation")
        .add_property("Model",                  make_function(&daepython::daeDefaultSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultSimulationWrapper::SetModel_))
        .add_property("model",                  make_function(&daepython::daeDefaultSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultSimulationWrapper::SetModel_))
        .add_property("m",                      make_function(&daepython::daeDefaultSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultSimulationWrapper::SetModel_))
        
        .add_property("DataReporter",       make_function(&daeSimulation::GetDataReporter, return_internal_reference<>()))
        .add_property("Log",                make_function(&daeSimulation::GetLog,          return_internal_reference<>()))
        .add_property("DAESolver",          make_function(&daeSimulation::GetDAESolver,    return_internal_reference<>()))

        .add_property("TimeHorizon",        &daeSimulation::GetTimeHorizon,          &daeSimulation::SetTimeHorizon)
        .add_property("ReportingInterval",  &daeSimulation::GetReportingInterval,    &daeSimulation::SetReportingInterval)
        .add_property("ActivityAction",     &daeSimulation::GetActivityAction)
        
        .add_property("CurrentTime",            make_function(&daeSimulation::GetCurrentTime))       
        .add_property("InitialConditionMode",   &daeSimulation::GetInitialConditionMode,  &daeSimulation::SetInitialConditionMode)

        .add_property("ObjectiveFunction",       make_function(&daeSimulation::GetObjectiveFunction, return_internal_reference<>()))

        .def("GetModel",                    &daeSimulation::GetModel, return_internal_reference<>())
        .def("SetModel",                    &daeSimulation::SetModel)
 
        .def("SetUpParametersAndDomains",   &daeSimulation::SetUpParametersAndDomains, &daepython::daeDefaultSimulationWrapper::def_SetUpParametersAndDomains)
        .def("SetUpVariables",              &daeSimulation::SetUpVariables,            &daepython::daeDefaultSimulationWrapper::def_SetUpVariables)
        .def("SetUpOptimization",			&daeSimulation::SetUpOptimization,		   &daepython::daeDefaultSimulationWrapper::def_SetUpOptimization)
        .def("Run",                         &daeSimulation::Run,                       &daepython::daeDefaultSimulationWrapper::def_Run)

        .def("ReRun",                       &daeSimulation::ReRun)
        .def("Finalize",                    &daeSimulation::Finalize)
        .def("ReportData",                  &daeSimulation::ReportData)
        .def("StoreInitializationValues",   &daeSimulation::StoreInitializationValues)
        .def("LoadInitializationValues",    &daeSimulation::LoadInitializationValues)

        .def("Pause",                       &daeSimulation::Pause)
        .def("Resume",                      &daeSimulation::Resume)
    
        .def("InitSimulation",              &daeSimulation::InitSimulation)
        .def("InitOptimization",            &daeSimulation::InitOptimization)
        .def("Reinitialize",                &daeSimulation::Reinitialize)
        .def("SolveInitial",                &daeSimulation::SolveInitial)
        .def("Integrate",                   &daeSimulation::Integrate)
        .def("IntegrateForTimeInterval",    &daeSimulation::IntegrateForTimeInterval)
        .def("IntegrateUntilTime",          &daeSimulation::IntegrateUntilTime)
  
        .def("CreateEqualityConstraint",    &daeSimulation::CreateEqualityConstraint, return_internal_reference<>())
        .def("CreateInequalityConstraint",  &daeSimulation::CreateInequalityConstraint, return_internal_reference<>())
        .def("SetOptimizationVariable",     &daeSimulation::SetOptimizationVariable)

        //.def("EnterConditionalIntegrationMode",   &daeSimulation::EnterConditionalIntegrationMode)
        //.def("IntegrateUntilConditionSatisfied",  &daeSimulation::IntegrateUntilConditionSatisfied)
        ; 
    
    class_<daepython::daeIPOPTWrapper, boost::noncopyable>("daeIPOPT")
        .def("Initialize",	&daeIPOPT::Initialize)
        .def("Run",			&daeIPOPT::Run)
        .def("Finalize",	&daeIPOPT::Finalize)
        ;

    
    
}
