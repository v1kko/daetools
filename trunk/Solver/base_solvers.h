#ifndef DAE_BASE_SOLVERS_H
#define DAE_BASE_SOLVERS_H
#include "../Core/solver.h"

namespace dae 
{
namespace solver 
{
daeDAESolver_t*          daeCreateIDASolver(void);
daeSolverClassFactory_t* daeCreateSolverClassFactory(void);

}
}

#endif
