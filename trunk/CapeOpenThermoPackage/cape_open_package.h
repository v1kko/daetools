#pragma once

#include "../Core/thermo_package.h"

#ifdef DAE_CAPE_OPEN_EXPORTS
#define DAE_CAPE_OPEN_API __declspec(dllexport)
#else // MODEL_EXPORTS
#define DAE_CAPE_OPEN_API __declspec(dllimport)
#endif // MODEL_EXPORTS

DAE_CAPE_OPEN_API dae::tpp::daeThermoPhysicalPropertyPackage_t* daeCreateCapeOpenPropertyPackage();
DAE_CAPE_OPEN_API void daeDeleteCapeOpenPropertyPackage(dae::tpp::daeThermoPhysicalPropertyPackage_t* package);
