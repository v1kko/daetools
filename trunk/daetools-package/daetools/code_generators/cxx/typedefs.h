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
#ifndef DAE_TYPEDEFS_H
#define DAE_TYPEDEFS_H

#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// #ifdef __cplusplus
// extern "C" {
// #endif

#ifndef real_t
#define real_t double
#endif

typedef enum
{
    eDCTUnknown = 0,
    eGlobalDiscontinuity,
    eModelDiscontinuity,
    eModelDiscontinuityWithDataChange,
    eNoDiscontinuity
} daeeDiscontinuityType;

typedef enum
{
    eStopAtModelDiscontinuity = 0,
    eDoNotStopAtDiscontinuity
} daeeStopCriterion;

typedef enum
{
    eIMUnknown = 0,
    eContinueFor,
    eContinueUntil
} daeeIntegrationMode;

typedef struct
{
    real_t  m_dCurrentTime;
    real_t  m_dTargetTime;
    real_t  rtol;
    real_t* yval;
    real_t* ypval;
    real_t* atval;
    real_t* idsval;
    long    Nequations;

    /* Opaque pointers */
    void*   model;
    void*   simulation;
    void*   mem;
    void*   yy;
    void*   yp;

} daeIDASolver_t;

typedef struct
{
    void*                    m_pModel;
    daeIDASolver_t*          m_pDAESolver;
    bool                     m_bCalculateSensitivities;
    bool                     m_bIsInitialized;
    real_t                   m_dCurrentTime;
    real_t                   m_dTimeHorizon;
    real_t                   m_dReportingInterval;

} daeSimulation_t;

// #ifdef __cplusplus
// }
// #endif

#endif
