#ifndef DAE_TRILINOS_BASESOLVERS_H
#define DAE_TRILINOS_BASESOLVERS_H

#include "../Core/solver.h"
//#include "../IDAS_DAESolver/ida_la_solver_interface.h"

namespace dae
{
namespace solver
{
daeIDALASolver_t*   daeCreateTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName);
std::vector<string> daeTrilinosSupportedSolvers();
std::vector<string> daeIfpackSupportedPreconditioners();
}
}

#endif
