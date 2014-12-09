#ifndef DAE_FMI_H
#define DAE_FMI_H

#include "include/fmi2Functions.h"
#include "../dae.h"
#include "../simulation_loader/simulation_loader.h"
#include "../simulation_loader/simulation_loader_c.h"

class daeFMIComponent_t
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
