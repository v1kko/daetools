#ifndef DAE_NLOPT_SOLVERS_H
#define DAE_NLOPT_SOLVERS_H

//#include "stdafx.h"
#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"

namespace daetools
{
namespace nlpsolver
{
DAE_NLOPT_API daetools::solver::daeNLPSolver_t* daeCreateNLOPTSolver(const std::string& algorithm);
}
}

#endif
