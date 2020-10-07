#ifndef IDA_LA_SOLVER_INTERFACE_H
#define IDA_LA_SOLVER_INTERFACE_H

#include "solver_class_factory.h"
#include <string>

namespace daetools
{
namespace solver
{
enum daeeIDALASolverType
{
    eSundialsLU = 0,
    eSundialsLapack,
    eSundialsGMRES,
    eThirdParty
};
}
}

#endif
