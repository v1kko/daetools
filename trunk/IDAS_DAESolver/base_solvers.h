#ifndef DAE_BASE_SOLVERS_H
#define DAE_BASE_SOLVERS_H
#include "../Core/solver.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef IDAS_EXPORTS
#define DAE_IDAS_API __declspec(dllexport)
#else
#define DAE_IDAS_API __declspec(dllimport)
#endif
#else
#define DAE_IDAS_API
#endif

#else // WIN32
#define DAE_IDAS_API
#endif // WIN32

namespace dae 
{
namespace solver 
{
DAE_IDAS_API daeDAESolver_t*          daeCreateIDASolver(void);
DAE_IDAS_API daeSolverClassFactory_t* daeCreateSolverClassFactory(void);

}
}

#endif
