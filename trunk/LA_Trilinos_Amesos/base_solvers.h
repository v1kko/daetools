#ifndef DAE_TRILINOS_BASESOLVERS_H
#define DAE_TRILINOS_BASESOLVERS_H

#include "../Core/solver.h"
//#include "../IDAS_DAESolver/ida_la_solver_interface.h"

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef TRILINOS_EXPORTS
#define DAE_TRILINOS_API __declspec(dllexport)
#else
#define DAE_TRILINOS_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_TRILINOS_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_TRILINOS_API
#endif // WIN32

namespace dae
{
namespace solver
{
DAE_TRILINOS_API daeLASolver_t*   daeCreateTrilinosSolver(const std::string& strSolverName, const std::string& strPreconditionerName);
DAE_TRILINOS_API std::vector<string> daeTrilinosSupportedSolvers();
DAE_TRILINOS_API std::vector<string> daeIfpackSupportedPreconditioners();
}
}

#endif
