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

void simInitialize(daeSimulation_t* s, daeModel_t* model, daeIDASolver_t* dae_solver, bool bCalculateSensitivities)
{
    s->m_bCalculateSensitivities = bCalculateSensitivities;
    s->m_pModel                  = model;
    s->m_pDAESolver              = dae_solver;
    s->m_bIsInitialized          = true;
    s->m_dCurrentTime            = 0.0;
    s->m_dTimeHorizon            = _end_time_;
    s->m_dReportingInterval      = _reporting_interval_;

    solInitialize(dae_solver, model, s, _Neqns_, _initValues_, _initDerivatives_, _absolute_tolerances_, _IDs_, _relative_tolerance_);
    
    model->values          = dae_solver->yval;
    model->timeDerivatives = dae_solver->ypval;
    modSetInitialConditions(model, dae_solver->yval);
    modInitializeValuesReferences(model);
}

void simFinalize(daeSimulation_t* s)
{
    solDestroy(s->m_pDAESolver);
}

void simReinitialize(daeSimulation_t* s)
{
    solReinitialize(s->m_pDAESolver, true, false);
}

void simSolveInitial(daeSimulation_t* s)
{
    s->m_dCurrentTime = 0.0;
    solSolveInitial(s->m_pDAESolver);
    simReportData(s);
}

void simRun(daeSimulation_t* s)
{
    real_t t;

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
    s->m_dCurrentTime = solSolve(s->m_pDAESolver, s->m_dTimeHorizon, eStopCriterion, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

real_t simIntegrateForTimeInterval(daeSimulation_t* s, real_t time_interval, bool bReportDataAroundDiscontinuities)
{
    s->m_dCurrentTime = solSolve(s->m_pDAESolver, s->m_dCurrentTime + time_interval, eDoNotStopAtDiscontinuity, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

real_t simIntegrateUntilTime(daeSimulation_t* s, real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities)
{
    s->m_dCurrentTime = solSolve(s->m_pDAESolver, time, eStopCriterion, bReportDataAroundDiscontinuities);
    return s->m_dCurrentTime;
}

void simReportData(daeSimulation_t* s)
{
    printf("Results at time: %12.5f\n", s->m_pDAESolver->m_dCurrentTime);
    for(int i = 0; i < s->m_pDAESolver->Neqns; i++)
        printf("%s = %20.14e\n", _variable_names_[i], s->m_pDAESolver->yval[i]);
    printf("\n");
}

void simStoreInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}

void simLoadInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}
