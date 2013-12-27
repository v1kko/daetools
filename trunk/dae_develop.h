#ifndef DAE_DEVELOP_INCLUDE_H
#define DAE_DEVELOP_INCLUDE_H

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#pragma warning(disable: 4250)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)

#ifdef AddPort
#undef AddPort
#endif

#endif

#include "config.h"
#include "Core/definitions.h"
#include "Core/xmlfile.h"
using namespace dae::xml;
#include "Core/coreimpl.h"
using namespace dae::core;
#include "Activity/base_activities.h"
#include "Activity/simulation.h"
using namespace dae::activity;
#include "DataReporting/datareporters.h"
#include "DataReporting/base_data_reporters_receivers.h"
using namespace dae::datareporting;
#include "Core/helpers.h"
using namespace dae;
#include "IDAS_DAESolver/base_solvers.h"
#include "IDAS_DAESolver/ida_solver.h"
using namespace dae::solver;
#include "Core/base_logging.h"
using namespace dae::logging;
//#include "Core/macros.h"
#include "Core/class_factory.h"

#include "BONMIN_MINLPSolver/base_solvers.h"
#include "LA_SuperLU/superlu_solvers.h"
using namespace dae::nlpsolver;

#endif
