#ifndef DAE_FMI_H
#define DAE_FMI_H

#include "include/fmi2Functions.h"
#include "../dae.h"

class daeFMIComponent_t
{
public:
    daeFMIComponent_t();
    virtual ~daeFMIComponent_t();

public:
    std::string instanceName;
    std::string fmuGUID;
    std::string fmuResourceLocation;
    bool visible;
    bool loggingOn;
};

#endif
