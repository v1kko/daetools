#ifndef DAE_BONMIN_BASESOLVERS_H
#define DAE_BONMIN_BASESOLVERS_H

#include "nlpsolver_class_factory.h"
#include "../Core/optimization.h"

using namespace dae::solver;

namespace dae
{
namespace nlpsolver
{
DAE_BONMIN_API daeNLPSolver_t* daeCreateIPOPTSolver(void);
DAE_BONMIN_API daeNLPSolver_t* daeCreateBONMINSolver(void);
}
}
#endif
