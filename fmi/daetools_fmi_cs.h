#ifndef DAE_FMI_H
#define DAE_FMI_H

#include "include/fmi2Functions.h"
#include "../daetools-core.h"
#include "../simulation_loader/simulation_loader.h"
#include "../simulation_loader/simulation_loader_c.h"

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef FMI_CS_EXPORTS
#define DAE_FMI_CS_API __declspec(dllexport)
#else
#define DAE_FMI_CS_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_FMI_CS_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_FMI_CS_API
#endif // WIN32


class DAE_FMI_CS_API daeFMIComponent_t
{
public:
    daeFMIComponent_t();
    virtual ~daeFMIComponent_t();

    void logFatal(const std::string& error);
    void logError(const std::string& error);

public:
    daeSimulationLoader          simulationLoader;
    std::string                  instanceName;
    std::string                  fmuGUID;
    std::string                  fmuResourceLocation;
    bool                         visible;
    const fmi2CallbackFunctions* functions;
    bool                         loggingOn;
    bool                         debugMode;
};

#endif
