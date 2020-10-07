#ifndef DAE_TRILINOS_BASESOLVERS_H
#define DAE_TRILINOS_BASESOLVERS_H

#include "../Core/solver.h"

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef INTEL_PARDISO_EXPORTS
#define DAE_INTEL_PARDISO_API __declspec(dllexport)
#else
#define DAE_INTEL_PARDISO_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_INTEL_PARDISO_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_INTEL_PARDISO_API
#endif // WIN32

namespace daetools
{
namespace solver
{
DAE_INTEL_PARDISO_API daeLASolver_t* daeCreateIntelPardisoSolver();
}
}

#endif
