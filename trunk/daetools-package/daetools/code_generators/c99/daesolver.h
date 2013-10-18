/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2013
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_SOLVER_H
#define DAE_SOLVER_H

#include <idas/idas.h>
#include <idas/idas_dense.h>
#include <sundials/sundials_math.h>
#include <nvector/nvector_serial.h>
#include "auxiliary.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sundials IDAS related functions */
int solInitialize(daeIDASolver_t* s, void* model, void* simulation, long Neqns, const real_t*  initValues, const real_t* initDerivatives,
                  const real_t*  absTolerances, const int* IDs, real_t  relativeTolerance);
int solReinitialize(daeIDASolver_t* s, bool bCopyDataFromBlock, bool bResetSensitivities);
int solDestroy(daeIDASolver_t* s);

int solRefreshRootFunctions(daeIDASolver_t* s, int noRoots);
int solResetIDASolver(daeIDASolver_t* s, bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities);
real_t solSolve(daeIDASolver_t* s, real_t dTime, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities);
int solSolveInitial(daeIDASolver_t* s);

#ifdef __cplusplus
}
#endif

#endif

