#include "fmi_component.h"
#include "include/rapidjson/document.h"
#include <boost/process/spawn.hpp>

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

    daeFMIComponent_t* c = NULL;

    try
    {
        c = new daeFMIComponent_t();

        c->instanceName         = instanceName;
        c->fmuGUID              = fmuGUID;
        c->fmuResourceLocation  = fmuResourceLocation;
        c->functions            = functions;
        c->visible              = visible;
        c->loggingOn            = loggingOn;

        // Start the daetools fmi web service.
        boost::process::spawn("python -m daetools.dae_simulator.daetools_fmi_ws");

        std::string function = "fmi2Instantiate";
        boost::format queryFmt("function=%s&instanceName=%s&guid=%s&resourceLocation=%s");
        std::string queryParameters = (queryFmt % function
                                                % instanceName
                                                % fmuGUID
                                                % fmuResourceLocation).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());

        c->simulationID = document["Result"]["simulationID"].GetString();
        c->startTime    = document["Result"]["startTime"].GetDouble();
        c->stopTime     = document["Result"]["stopTime"].GetDouble();
        c->step         = document["Result"]["step"].GetDouble();
        c->tolerance    = document["Result"]["tolerance"].GetDouble();

        rapidjson::Value fmiInterface = document["Result"]["FMI_Interface"].GetObject();
        for (rapidjson::Value::ConstMemberIterator iter = fmiInterface.MemberBegin(); iter != fmiInterface.MemberEnd(); ++iter)
        {
            // iter->name is the key (equal to reference)
            fmiObject obj;
            obj.name        = iter->value["name"].GetString();
            obj.description = iter->value["description"].GetString();
            obj.reference   = iter->value["reference"].GetInt();
            obj.type        = iter->value["type"].GetString();
            obj.units       = iter->value["units"].GetString();

            c->m_FMI_Interface[obj.reference] = obj;
            //printf("name = %s, description = %s, reference = %d, type = %s, units = %s\n", obj.name.c_str(), obj.description.c_str(), obj.reference, obj.type.c_str(), obj.units.c_str());
        }

        return c;
    }
    catch(std::exception& e)
    {
        std::string error = "Cannot instantiate FMU: ";
        error += std::string(instanceName) + " {" + std::string(fmuGUID) + "}\n";
        error += std::string(e.what()) + "\n\n";

        if(c)
            c->logFatal(error);
    }

    return NULL;
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
        std::string function = "fmi2SetupExperiment";
        boost::format queryFmt("function=%s&simulationID=%s&toleranceDefined=%s&tolerance=%f&startTime=%f&stopTimeDefined=%s&stopTime=%f");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % (toleranceDefined ? "true" : "false")
                                                % tolerance
                                                % startTime
                                                % (stopTimeDefined ? "true" : "false")
                                                % stopTime).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2EnterInitializationMode(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2EnterInitializationMode";
        boost::format queryFmt("function=%s&simulationID=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2ExitInitializationMode(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2ExitInitializationMode";
        boost::format queryFmt("function=%s&simulationID=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] == "Error")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2Terminate(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2Terminate";
        boost::format queryFmt("function=%s&simulationID=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

void fmi2FreeInstance(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return;

    try
    {
        std::string function = "fmi2FreeInstance";
        boost::format queryFmt("function=%s&simulationID=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());

        if(c->m_ptcpipSocket && c->m_ptcpipSocket->is_open())
            c->m_ptcpipSocket->close();
        c->m_ptcpipSocket.reset();
        delete c;
    }
    catch(std::exception& e)
    {
        c->logError(e.what());
    }
}

fmi2Status fmi2Reset(fmi2Component comp)
{
    // Not implemented
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2Reset";
        boost::format queryFmt("function=%s&simulationID=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logError(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

/* Getting and setting variable values */
fmi2Status fmi2GetReal(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2GetReal";
        boost::format queryFmt("function=%s&simulationID=%s&valReferences=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % array_to_json_str(vr, nvr)).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());

        rapidjson::Value values_json = document["Result"]["Values"].GetArray();
        if(!values_json.IsArray())
            throw std::runtime_error("Response is not an array");
        if(values_json.Size() != nvr)
            throw std::runtime_error((boost::format("Array size is wrong: %d (should be %d)") % values_json.Size() % nvr).str());

        for(size_t i = 0; i < nvr; i++)
            value[i] = values_json[i].GetDouble();
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
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

    try
    {
        std::string function = "fmi2GetString";
        boost::format queryFmt("function=%s&simulationID=%s&valReferences=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % array_to_json_str(vr, nvr)).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());

        rapidjson::Value values_json = document["Result"]["Values"].GetArray();
        if(!values_json.IsArray())
            throw std::runtime_error("Response is not an array");
        if(values_json.Size() != nvr)
            throw std::runtime_error((boost::format("Array size is wrong: %d (should be %d)") % values_json.Size() % nvr).str());

        for(size_t i = 0; i < nvr; i++)
            value[i] = values_json[i].GetString();
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
        return fmi2Fatal;
    }

    return fmi2OK;
}

fmi2Status fmi2SetReal(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    try
    {
        std::string function = "fmi2SetReal";
        boost::format queryFmt("function=%s&simulationID=%s&valReferences=%s&values=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % array_to_json_str(vr, nvr)
                                                % array_to_json_str(value, nvr)).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
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
        std::string function = "fmi2SetString";
        boost::format queryFmt("function=%s&simulationID=%s&valReferences=%s&values=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % array_to_json_str(vr, nvr)
                                                % array_to_json_str(value, nvr)).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logFatal(e.what());
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
        std::string function = "fmi2DoStep";
        boost::format queryFmt("function=%s&simulationID=%s&currentCommunicationPoint=%f&communicationStepSize=%f&noSetFMUStatePriorToCurrentPoint=%s");
        std::string queryParameters = (queryFmt % function
                                                % c->simulationID
                                                % currentCommunicationPoint
                                                % communicationStepSize
                                                % (noSetFMUStatePriorToCurrentPoint ? "true" : "false")).str();
        std::string queryResponseHeaders, queryResponseContent;
        c->executeQuery(queryParameters, queryResponseHeaders, queryResponseContent);

        rapidjson::Document document;
        document.Parse(queryResponseContent.c_str(), queryResponseContent.size());

        if(document["Status"] != "Success")
            throw std::runtime_error((boost::format("Function %s failed:\n%s") % function % document["Reason"].GetString()).str());
    }
    catch(std::exception& e)
    {
        c->logError(e.what());
        return fmi2Error;
    }

    return fmi2OK;
}

fmi2Status fmi2CancelStep(fmi2Component comp)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

/* Inquire slave status */
fmi2Status fmi2GetStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Status* status)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetRealStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Real* status)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetIntegerStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Integer* status)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetBooleanStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2Boolean* status)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetStringStatus(fmi2Component comp, const fmi2StatusKind kind, fmi2String* status)
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2SetRealInputDerivatives(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Integer order[], const fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}

fmi2Status fmi2GetRealOutputDerivatives(fmi2Component comp, const fmi2ValueReference vr[], size_t nvr, const fmi2Integer order[], fmi2Real value[])
{
    daeFMIComponent_t* c = (daeFMIComponent_t*)comp;
    if(c == NULL)
        return fmi2Fatal;

    return fmi2Error;
}
