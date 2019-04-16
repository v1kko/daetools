#ifndef DAE_NLOPT_SOLVERS_H
#define DAE_NLOPT_SOLVERS_H

#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"

namespace dae
{
namespace nlpsolver
{
DAE_NLOPT_API daeNLPSolver_t* daeCreateNLOPTSolver(const std::string& algorithm);
}
}

#endif
