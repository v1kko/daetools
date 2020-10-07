#ifndef DAE_FMI_COMPONENT_H
#define DAE_FMI_COMPONENT_H

#include <iostream>
#include "include/fmi2Functions.h"
#include <boost/format.hpp>
#include <boost/asio.hpp>
using boost::asio::ip::tcp;

const std::string webServiceName    = "daetools_fmi_ws";
const std::string webServiceAddress = "127.0.0.1";
const int         webServicePort    = 8002;

struct fmiObject
{
    std::string name;
    std::string type;
    std::string description;
    std::string units;
    fmi2ValueReference reference;
};

class daeFMIComponent_t
{
public:
    daeFMIComponent_t();
    virtual ~daeFMIComponent_t();

    void executeQuery(const std::string& queryParameters, std::string& queryResponseHeaders, std::string& queryResponseContent);

    void logFatal(const std::string& error);
    void logError(const std::string& error);

public:
    std::shared_ptr<tcp::socket>  m_ptcpipSocket;
    boost::asio::io_service         m_ioService;
    std::map<int, fmiObject>        m_FMI_Interface;

    double                       startTime;
    double                       stopTime;
    double                       step;
    double                       tolerance;
    std::string                  simulationID;
    std::string                  instanceName;
    std::string                  fmuGUID;
    std::string                  fmuResourceLocation;
    bool                         visible;
    const fmi2CallbackFunctions* functions;
    bool                         loggingOn;
    bool                         debugMode;
};

/*
double hexfloat_to_double(const char* val_hex_str);
std::string double_to_hexfloat(double value);
double str_to_double(const char* val_hex_str);
*/
std::string double_to_str(double value);
std::string array_to_json_str(const fmi2ValueReference vr[], size_t nvr);
std::string array_to_json_str(const fmi2Real value[], size_t nvr);
std::string array_to_json_str(const fmi2String value[], size_t nvr);

#endif
