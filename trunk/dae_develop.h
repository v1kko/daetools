#ifndef DAE_DEVELOP_INCLUDE_H
#define DAE_DEVELOP_INCLUDE_H

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

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
#include "Solver/base_solvers.h"
#include "Solver/ida_solver.h"
using namespace dae::solver;
#include "Core/base_logging.h"
using namespace dae::logging;
#include "Core/macros.h"
#include "Core/class_factory.h"

#endif
