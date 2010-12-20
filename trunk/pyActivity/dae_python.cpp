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
    daeActivity
***************************************************************/
    class_<daepython::daeActivityWrapper, boost::noncopyable>("daeActivity_t", no_init)
        .add_property("DataReporter",       make_function(&daeActivity_t::GetDataReporter, return_internal_reference<>()))
        .add_property("Log",                make_function(&daeActivity_t::GetLog, return_internal_reference<>()))

        .def("GetModel",                    pure_virtual(&daeActivity_t::GetModel), return_internal_reference<>())
        .def("SetModel",                    pure_virtual(&daeActivity_t::SetModel))
        .def("SetUpParametersAndDomains",   pure_virtual(&daeActivity_t::SetUpParametersAndDomains))
        .def("SetUpVariables",              pure_virtual(&daeActivity_t::SetUpVariables))
        .def("Run",                         pure_virtual(&daeActivity_t::Run))
        .def("Finalize",                    pure_virtual(&daeActivity_t::Finalize))
        .def("Reset",                       pure_virtual(&daeActivity_t::Reset))
        .def("ReportData",                  pure_virtual(&daeActivity_t::ReportData))
        .def("StoreInitializationValues",   pure_virtual(&daeActivity_t::StoreInitializationValues))
        .def("LoadInitializationValues",    pure_virtual(&daeActivity_t::LoadInitializationValues))
        ;

    class_<daepython::daeDynamicActivityWrapper, bases<daeActivity_t>, boost::noncopyable>("daeDynamicActivity_t", no_init)
        .add_property("TimeHorizon",        &daeDynamicActivity_t::GetTimeHorizon,          &daeDynamicActivity_t::SetTimeHorizon)
        .add_property("ReportingInterval",  &daeDynamicActivity_t::GetReportingInterval,    &daeDynamicActivity_t::SetReportingInterval)
        .add_property("ActivityAction",     &daeDynamicActivity_t::GetActivityAction)
        
        .def("Pause",                       pure_virtual(&daeDynamicActivity_t::Pause))
        .def("Resume",                      pure_virtual(&daeDynamicActivity_t::Resume))
        ;
  
    class_<daepython::daeDynamicSimulationWrapper, bases<daeDynamicActivity_t>, boost::noncopyable>("daeDynamicSimulation_t", no_init)
        .add_property("DAESolver",          make_function(&daeDynamicSimulation_t::GetDAESolver, return_internal_reference<>()))
        
        .def("Initialize",                  pure_virtual(&daeDynamicSimulation_t::Initialize))
        .def("Reinitialize",                pure_virtual(&daeDynamicSimulation_t::Reinitialize))
        .def("SolveInitial",                pure_virtual(&daeDynamicSimulation_t::SolveInitial))
        .def("Integrate",                   pure_virtual(&daeDynamicSimulation_t::Integrate))
        .def("IntegrateForTimeInterval",    pure_virtual(&daeDynamicSimulation_t::IntegrateForTimeInterval))
        .def("IntegrateUntilTime",          pure_virtual(&daeDynamicSimulation_t::IntegrateUntilTime))
        ;  

    class_<daepython::daeDefaultDynamicSimulationWrapper, bases<daeDynamicSimulation_t>, boost::noncopyable>("daeDynamicSimulation")
        .add_property("Model",                  make_function(&daepython::daeDefaultDynamicSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultDynamicSimulationWrapper::SetModel_))
        .add_property("model",                  make_function(&daepython::daeDefaultDynamicSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultDynamicSimulationWrapper::SetModel_))
        .add_property("m",                      make_function(&daepython::daeDefaultDynamicSimulationWrapper::GetModel_),
                                                make_function(&daepython::daeDefaultDynamicSimulationWrapper::SetModel_))
        .add_property("CurrentTime",            make_function(&daeDynamicSimulation::GetCurrentTime))       
        .add_property("InitialConditionMode",   &daeDynamicSimulation::GetInitialConditionMode,  &daeDynamicSimulation::SetInitialConditionMode)
 
        .def("SetUpParametersAndDomains",   &daeDynamicSimulation_t::SetUpParametersAndDomains, &daepython::daeDefaultDynamicSimulationWrapper::def_SetUpParametersAndDomains)
        .def("SetUpVariables",              &daeDynamicSimulation_t::SetUpVariables,            &daepython::daeDefaultDynamicSimulationWrapper::def_SetUpVariables)
        .def("Run",                         &daeActivity_t::Run,   &daepython::daeDefaultDynamicSimulationWrapper::def_Run)
        .def("Reset",                       &daeDynamicSimulation::Reset)
        .def("Finalize",                    &daeDynamicSimulation::Finalize)
        .def("ReportData",                  &daeDynamicSimulation::ReportData)
        .def("StoreInitializationValues",   &daeDynamicSimulation::StoreInitializationValues)
        .def("LoadInitializationValues",    &daeDynamicSimulation::LoadInitializationValues)

        .def("Pause",                       &daeDynamicSimulation::Pause)
        .def("Resume",                      &daeDynamicSimulation::Resume)

        .def("Initialize",                  &daeDynamicSimulation::Initialize)
        .def("Reinitialize",                &daeDynamicSimulation::Reinitialize)
        .def("SolveInitial",                &daeDynamicSimulation::SolveInitial)
        .def("Integrate",                   &daeDynamicSimulation::Integrate)
        .def("IntegrateForTimeInterval",    &daeDynamicSimulation::IntegrateForTimeInterval)
        .def("IntegrateUntilTime",          &daeDynamicSimulation::IntegrateUntilTime)
  
        //.def("EnterConditionalIntegrationMode",   &daeDynamicSimulation::EnterConditionalIntegrationMode)
        //.def("IntegrateUntilConditionSatisfied",  &daeDynamicSimulation::IntegrateUntilConditionSatisfied)
        ;
}
