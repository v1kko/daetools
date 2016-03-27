/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2016
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_SIMULATION_H
#define DAE_SIMULATION_H

#include "auxiliary.h"
#include "model.h"
#include "daesolver.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

/* daeSimulation related functions*/
void simStoreInitializationValues(daeSimulation_t* s, const char* strFileName);
void simLoadInitializationValues(daeSimulation_t* s, const char* strFileName);

void simInitialize(daeSimulation_t* s, daeModel_t* model, daeIDASolver_t* dae_solver, bool bCalculateSensitivities);
void simSolveInitial(daeSimulation_t* s);
void simRun(daeSimulation_t* s);
void simReinitialize(daeSimulation_t* s);
void simFinalize(daeSimulation_t* s);
void simReportData(daeSimulation_t* s);
real_t simIntegrate(daeSimulation_t* s, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);
real_t simIntegrateForTimeInterval(daeSimulation_t* s, real_t time_interval, bool bReportDataAroundDiscontinuities);
real_t simIntegrateUntilTime(daeSimulation_t* s, real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);

// #ifdef __cplusplus
// }
// #endif

#endif

