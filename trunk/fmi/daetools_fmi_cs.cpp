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
fmi2Component fmi2Instantiate(fmi2String instanceName,
                              fmi2Type fmuType,
                              fmi2String fmuGUID,
                              fmi2String fmuResourceLocation,
                              const fmi2CallbackFunctions* functions,
                              fmi2Boolean visible,
                              fmi2Boolean loggingOn)
{
    if(fmuType != fmi2CoSimulation)
        return NULL;

    daeFMIComponent_t* c = new daeFMIComponent_t();

    c->fmuGUID              = fmuGUID;
    c->fmuResourceLocation  = fmuResourceLocation;
    c->visible              = visible;
    c->loggingOn            = loggingOn;

    return c;
}

void fmi2FreeInstance(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return;

    delete c;
}

/* Enter and exit initialization mode, terminate and reset */
fmi2Status fmi2SetupExperiment(fmi2Component comp, fmi2Boolean, fmi2Real, fmi2Real, fmi2Boolean, fmi2Real)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

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

    return fmi2OK;
}

fmi2Status fmi2Reset(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

/* Getting and setting variable values */
fmi2Status fmi2GetReal(fmi2Component comp, const fmi2ValueReference[], size_t, fmi2Real   [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

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
fmi2Status fmi2SetRealInputDerivatives(fmi2Component comp, const fmi2ValueReference [], size_t, const fmi2Integer [], const fmi2Real [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2GetRealOutputDerivatives(fmi2Component comp, const fmi2ValueReference [], size_t, const fmi2Integer [], fmi2Real [])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2OK;
}

fmi2Status fmi2DoStep(fmi2Component comp, fmi2Real, fmi2Real, fmi2Boolean)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

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

fmi2Status fmi2GetRealStatus(fmi2Component comp, const fmi2StatusKind, fmi2Real*   )
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

}

daeFMIComponent_t::~daeFMIComponent_t()
{

}
