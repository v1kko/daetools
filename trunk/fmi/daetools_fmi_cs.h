#ifndef DAE_FMI_H
#define DAE_FMI_H

#include "include/fmi2Functions.h"
#include "../dae.h"
#include "../simulation_loader/simulation_loader.h"

class daeFMIComponent_t
{
public:
    daeFMIComponent_t();
    virtual ~daeFMIComponent_t();

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
