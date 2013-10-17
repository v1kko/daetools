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
#include "auxiliary.h"
#include "simulation.h"
#include "dae_solver.h"
#include "daetools_model.h"


void simInitialize(daeSimulation_t* s, daeModel_t* model, daeIDASolver_t* dae_solver, bool bCalculateSensitivities)
{
    s->m_bCalculateSensitivities = bCalculateSensitivities;
    s->m_pModel                  = model;
    s->m_pDAESolver              = dae_solver;
    s->m_bIsInitialized          = true;
    s->m_dCurrentTime            = 0.0;
    s->m_dTimeHorizon            = _end_time_;
    s->m_dReportingInterval      = _reporting_interval_;

    int no_roots = number_of_roots(model);
    solInitialize(dae_solver, model, s, _Neqns_, _initValues_, _initDerivatives_, _absolute_tolerances_, _IDs_, no_roots, _relative_tolerance_);
}

void simFinalize(daeSimulation_t* s)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    solDestroy(dae_solver);
}

void simReinitialize(daeSimulation_t* s)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    solReinitialize(dae_solver, true, false);
}

void simSolveInitial(daeSimulation_t* s)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    s->m_dCurrentTime = 0.0;
    solSolveInitial(dae_solver);
    simReportData(s);
}

void simRun(daeSimulation_t* s)
{
    real_t t;
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;

    while(s->m_dCurrentTime < s->m_dTimeHorizon)
    {
        t = s->m_dCurrentTime + s->m_dReportingInterval;
        if(t > s->m_dTimeHorizon)
            t = s->m_dTimeHorizon;

        /* If discontinuity is found, loop until the end of the integration period
         * The data will be reported around discontinuities! */
        while(t > s->m_dCurrentTime)
        {
            printf("Integrating from [%f] to [%f]...\n", s->m_dCurrentTime, t);
            simIntegrateUntilTime(s, t, eStopAtModelDiscontinuity, true);
        }

        simReportData(s);
    }

}

real_t simIntegrate(daeSimulation_t* s, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    s->m_dCurrentTime = solSolve(dae_solver, s->m_dTimeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

real_t simIntegrateForTimeInterval(daeSimulation_t* s, real_t time_interval, bool bReportDataAroundDiscontinuities)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    s->m_dCurrentTime = solSolve(dae_solver, s->m_dCurrentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

real_t simIntegrateUntilTime(daeSimulation_t* s, real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    s->m_dCurrentTime = solSolve(dae_solver, time, eStopCriterion, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

void simReportData(daeSimulation_t* s)
{
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)s->m_pDAESolver;
    printf("Results at time: %12.5f\n", dae_solver->m_dCurrentTime);
    for(int i = 0; i < dae_solver->Neqns; i++)
        printf("%s = %20.14e\n", _variable_names_[i], dae_solver->yval[i]);
    printf("\n");
}

void simStoreInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}

void simLoadInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}
