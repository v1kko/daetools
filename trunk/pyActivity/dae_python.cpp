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
    class_<daepython::daeDefaultSimulationWrapper, boost::noncopyable>("daeSimulation")
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

        .def("GetModel",                    &daeSimulation::GetModel, return_internal_reference<>())
        .def("SetModel",                    &daeSimulation::SetModel)
 
        .def("SetUpParametersAndDomains",   &daeSimulation::SetUpParametersAndDomains, &daepython::daeDefaultSimulationWrapper::def_SetUpParametersAndDomains)
        .def("SetUpVariables",              &daeSimulation::SetUpVariables,            &daepython::daeDefaultSimulationWrapper::def_SetUpVariables)
        .def("SetUpOptimization",			&daeSimulation::SetUpOptimization,		   &daepython::daeDefaultSimulationWrapper::def_SetUpOptimization)
        .def("Run",                         &daeSimulation::Run,                       &daepython::daeDefaultSimulationWrapper::def_Run)

        .def("Reset",                       &daeSimulation::Reset)
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
  
        //.def("EnterConditionalIntegrationMode",   &daeSimulation::EnterConditionalIntegrationMode)
        //.def("IntegrateUntilConditionSatisfied",  &daeSimulation::IntegrateUntilConditionSatisfied)
        ; 
    
    class_<daepython::daeIPOPTWrapper, boost::noncopyable>("daeIPOPT")
        .def("Initialize", &daeIPOPT::Initialize)
        ;

    
    
}
