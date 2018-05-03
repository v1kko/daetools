#ifndef DAE_PYTHON_WRAPS_H
#define DAE_PYTHON_WRAPS_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include "../LA_IntelPardiso/mkl_pardiso_sparse_la_solver.h"
#include <boost/python.hpp>
//#include <boost/python/numeric.hpp>
#include <boost/python/slice.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/reference_existing_object.hpp>
using namespace dae::solver;

namespace daepython
{
boost::python::list daeIntelPardisoSolver_get_iparm(daeIntelPardisoSolver& self);
void daeIntelPardisoSolver_set_iparm(daeIntelPardisoSolver& self, boost::python::list l_iparm);
boost::python::dict daeIntelPardisoSolver_GetEvaluationCallsStats_(daeIntelPardisoSolver& self);
}

#endif
