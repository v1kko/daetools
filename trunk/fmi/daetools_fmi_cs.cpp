#include "daetools_fmi_cs.h"

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
fmi2Status fmi2GetReal(fmi2Component comp, const fmi2ValueReference[], size_t, fmi2Real   [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;
/*
        int i;
        unsigned int noPoints;
        char name[512];
        int nOutletPorts = c->simulationLoader.GetNumberOfOutputs();
        for(i = 0; i < nOutletPorts; i++)
        {
            GetOutputInfo(simulation, i, name, &noPoints);
            if(debugMode)
                ssPrintf("Output %d name: %s, no.points: %d\n", i, name, noPoints);
            if(noPoints != ssGetOutputPortWidth(S, i))
            {
                sprintf(msg, "Invalid width of outlet port %s: %d (expected %d)", name, ssGetOutputPortWidth(S, i), noPoints);
                ssSetErrorStatus(S, msg);
                return;
            }

            double* data = (double*)malloc(noPoints * sizeof(double));
            GetOutputValue(simulation, i, data, noPoints);
            if(debugMode)
                ssPrintf("Get output %d value: %f\n", i, data[0]);

            double* y = ssGetOutputPortRealSignal(S, i);
            memcpy(y, data, noPoints * sizeof(double));

            free(data);
        }
*/
    return fmi2OK;
}

fmi2Status fmi2GetInteger(fmi2Component comp, const fmi2ValueReference[], size_t, fmi2Integer[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetBoolean(fmi2Component comp, const fmi2ValueReference[], size_t, fmi2Boolean[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetString(fmi2Component comp, const fmi2ValueReference[], size_t, fmi2String [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2SetReal(fmi2Component comp, const fmi2ValueReference[], size_t, const fmi2Real   [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;
/*
        int i;
        unsigned int noPoints;
        char name[512];
        int nInletPorts  = c->simulationLoader.GetNumberOfInputs();
        for(i = 0; i < nInletPorts; i++)
        {
            c->simulationLoader.GetInputInfo(i, name, &noPoints);
            if(debugMode)
                ssPrintf("Input %d name: %s, no.points: %d\n", i, name, noPoints);
            if(noPoints != ssGetInputPortWidth(S, i))
            {
                sprintf(msg, "Invalid width of port %s: %d (expected %d)", name, ssGetInputPortWidth(S, i), noPoints);
                ssSetErrorStatus(S, msg);
                return;
            }
            const double* data = ssGetInputPortRealSignal(S, i);
            if(debugMode)
                ssPrintf("Set input %d value to: %f\n", i, data[0]);
            SetInputValue(simulation, i, data, noPoints);
        }
*/
    return fmi2OK;
}

fmi2Status fmi2SetInteger(fmi2Component comp, const fmi2ValueReference[], size_t, const fmi2Integer[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2SetBoolean(fmi2Component comp, const fmi2ValueReference[], size_t, const fmi2Boolean[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2SetString(fmi2Component comp, const fmi2ValueReference[], size_t, const fmi2String [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

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

fmi2Status fmi2CancelStep(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

/* Inquire slave status */
fmi2Status fmi2GetStatus(fmi2Component comp, const fmi2StatusKind, fmi2Status* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetRealStatus(fmi2Component comp, const fmi2StatusKind, fmi2Real* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetIntegerStatus(fmi2Component comp, const fmi2StatusKind, fmi2Integer*)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetBooleanStatus(fmi2Component comp, const fmi2StatusKind, fmi2Boolean*)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetStringStatus (fmi2Component comp, const fmi2StatusKind, fmi2String* )
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

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
