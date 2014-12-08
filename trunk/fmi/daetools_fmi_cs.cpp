#include "daetools_fmi_cs.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/detail/file_parser_error.hpp>

/***************************************************
  Types for Common Functions
****************************************************/
const char* fmi2GetTypesPlatform(void)
{
    return fmi2TypesPlatform;
}

const char* fmi2GetVersion(void)
{
    return fmi2Version;
}

fmi2Status fmi2SetDebugLogging(fmi2Component comp, fmi2Boolean, size_t, const fmi2String[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

/* Creation and destruction of FMU instances and setting debug status */
fmi2Component fmi2Instantiate(fmi2String                    instanceName,
                              fmi2Type                      fmuType,
                              fmi2String                    fmuGUID,
                              fmi2String                    fmuResourceLocation,
                              const fmi2CallbackFunctions*  functions,
                              fmi2Boolean                   visible,
                              fmi2Boolean                   loggingOn)
{
    if(fmuType != fmi2CoSimulation)
        return NULL;

    try
    {
        daeFMIComponent_t* c = new daeFMIComponent_t();

        c->instanceName         = instanceName;
        c->fmuGUID              = fmuGUID;
        c->fmuResourceLocation  = fmuResourceLocation;
        c->functions            = functions;
        c->visible              = visible;
        c->loggingOn            = loggingOn;

        std::string strPythonFile, strSimulationClass,
                    strDAESolver, strLASolver, strDataReporter,
                    strDataReporterConnectionString, strLog;

        strDAESolver    = "Sundials IDAS";
        strDataReporter = "TCPIPDataReporter"; // BlackHoleDataReporter
        strLog          = "BaseLog";

        boost::property_tree::ptree pt;
        std::string					json_settings = std::string(fmuResourceLocation) + "/resources/settings.json";
        std::string					json_ini_file = std::string(fmuResourceLocation) + "/resources/init.json";

        boost::property_tree::json_parser::read_json(json_settings, pt);
        strPythonFile      = pt.get<std::string>("simulationClass");
        strSimulationClass = pt.get<std::string>("simulationName");
        strLASolver        = pt.get<std::string>("LASolver");

        c->simulationLoader.LoadSimulation(strPythonFile, strSimulationClass);
        c->simulationLoader.Initialize(strDAESolver, strLASolver, strDataReporter, strDataReporterConnectionString, strLog, false, "");

        return c;
    }
    catch(std::exception& e)
    {
    }
    return NULL;
}

void fmi2FreeInstance(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return;

    try
    {
        delete c;
    }
    catch(std::exception& e)
    {
    }
}

/* Enter and exit initialization mode, terminate and reset */
fmi2Status fmi2SetupExperiment(fmi2Component comp,
                               fmi2Boolean   toleranceDefined,
                               fmi2Real      tolerance,
                               fmi2Real      startTime,
                               fmi2Boolean   stopTimeDefined,
                               fmi2Real      stopTime)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        if(stopTimeDefined)
        {
            c->simulationLoader.SetTimeHorizon(stopTime);
            c->simulationLoader.SetReportingInterval(stopTime/2);
        }
        c->simulationLoader.SolveInitial();
        c->simulationLoader.ReportData();
    }
    catch(std::exception& e)
    {
        return fmi2Error;
    }

    return fmi2OK;
}

fmi2Status fmi2EnterInitializationMode(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2ExitInitializationMode(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2Terminate(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        c->simulationLoader.Finalize();
    }
    catch(std::exception& e)
    {
        return fmi2Error;
    }

    return fmi2OK;
}

fmi2Status fmi2Reset(fmi2Component comp)
{
    // Not implemented
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

/* Getting and setting variable values */
fmi2Status fmi2GetReal(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        for(size_t i = 0; i < nvr; i++)
        {
            unsigned int reference = static_cast<unsigned int>(vr[i]);
            value[i] = c->simulationLoader.GetFMIValue(reference);
        }
    }
    catch(std::exception& e)
    {
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2GetInteger(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, fmi2Integer value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetBoolean(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, fmi2Boolean value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetString(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, fmi2String value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Fatal;
/*
    try
    {
        for(size_t i = 0; i < nvr; i++)
        {
            unsigned int reference = static_cast<unsigned int>(vr[i]);
            strncpy(value[i], c->simulationLoader.GetFMIActiveState(reference).c_str(), 255);
        }
    }
    catch(std::exception& e)
    {
        return fmi2Fatal;
    }
*/
    return fmi2OK;
}

fmi2Status fmi2SetReal(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        for(size_t i = 0; i < nvr; i++)
        {
            unsigned int reference = static_cast<unsigned int>(vr[i]);
            c->simulationLoader.SetFMIValue(reference, value[i]);
        }
    }
    catch(std::exception& e)
    {
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2SetInteger(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Integer value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2SetBoolean(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Boolean value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2SetString(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2String value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        for(size_t i = 0; i < nvr; i++)
        {
            unsigned int reference = static_cast<unsigned int>(vr[i]);
            c->simulationLoader.SetFMIActiveState(reference, std::string(value[i]));
        }
    }
    catch(std::exception& e)
    {
        return fmi2Fatal;
    }

    return fmi2OK;
}

/***************************************************
  Types for Functions for FMI2 for Co-Simulation
****************************************************/
/* Simulating the slave */
fmi2Status fmi2DoStep(fmi2Component comp,
                      fmi2Real      currentCommunicationPoint,
                      fmi2Real      communicationStepSize,
                      fmi2Boolean   noSetFMUStatePriorToCurrentPoint)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        double stepTimeHorizon = currentCommunicationPoint + communicationStepSize;

        /* Integrate until the specified step final time. */
        //if(c->debugMode)
        //    c->functions->logger("Integrating from %.6f to %.6f ...\n", stepStartTime, stepTimeHorizon);

        /* Set the time horizon. */
        c->simulationLoader.SetTimeHorizon(stepTimeHorizon);

        /* Integrate until specified time and report data. */
        c->simulationLoader.Reinitialize();
        c->simulationLoader.IntegrateUntilTime(stepTimeHorizon, false, true);
        c->simulationLoader.ReportData();
    }
    catch(std::exception& e)
    {
        return fmi2Error;
    }

    return fmi2OK;
}
/*
fmi2Status fmi2CancelStep(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}
*/
/* Inquire slave status */
/*
fmi2Status fmi2GetStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Status* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetRealStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Real* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetIntegerStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Integer*)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetBooleanStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Boolean*)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetStringStatus(fmi2Component comp, const fmi2StatusKind, fmi2String* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}
*/

/***************************************************
 *  daeFMIComponent_t
***************************************************/
daeFMIComponent_t::daeFMIComponent_t()
{
    debugMode = false;
}

daeFMIComponent_t::~daeFMIComponent_t()
{

}
