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
#include "auxiliary.h"
#include "simulation.h"

void simInitialize(daeSimulation_t* s, daeModel_t* model, daeIDASolver_t* dae_solver, bool bCalculateSensitivities)
{
    s->m_bCalculateSensitivities = bCalculateSensitivities;
    s->m_pModel                  = model;
    s->m_pDAESolver              = dae_solver;
    s->m_bIsInitialized          = true;
    s->m_dCurrentTime            = model->startTime;
    s->m_dTimeHorizon            = model->timeHorizon;
    s->m_dReportingInterval      = model->reportingInterval;

    solInitialize(dae_solver, model, s, model->Nequations,
                                        model->Nequations_local,
                                        model->initValues,
                                        model->initDerivatives,
                                        model->absoluteTolerances,
                                        model->IDs,
                                        model->relativeTolerance);
    
    modInitializeValuesReferences(model, dae_solver->yval, dae_solver->ypval);
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
    int res = solSolveInitial(s->m_pDAESolver);
    if(res < 0)
        exit(res);
    simReportData(s);
    printf("System successfuly initialised\n");
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
    daeModel_t* model = (daeModel_t*)s->m_pModel;

    printf("Results at time: %12.5f\n", s->m_pDAESolver->m_dCurrentTime);
    for(int i = 0; i < s->m_pDAESolver->Nequations; i++)
    {
        //printf("%s = %20.14e\n", model->variableNames[i], s->m_pDAESolver->yval[i]);
        printf("(%d=%20.14e),", i, s->m_pDAESolver->yval[i]);
    }
    printf("\n");
/*
    int i;
    char* out;
    daeModel_t* model = (daeModel_t*)s->m_pModel;
    int* lengths = malloc(s->m_pDAESolver->Nequations * sizeof(int));

    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
        lengths[i] = MAX(strlen(model->variableNames[i]) + 1, 21);

    printf("\n");
    printf("Results at time: %.7f\n", s->m_pDAESolver->m_dCurrentTime);
    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");
    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
        printf("| %-*s ", lengths[i], model->variableNames[i]);
    printf("|\n");
    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");

    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
        printf("| %-*.14e ", lengths[i], s->m_pDAESolver->yval[i]);
    printf("|\n");
    for(i = 0; i < s->m_pDAESolver->Nequations; i++)
    {
        out = calloc(lengths[i]+1, sizeof(char));
        memset(out, '-', lengths[i]);
        printf("+-%s-", out);
    }
    printf("+\n");

    free(out);
    free(lengths);
*/
}

void simStoreInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}

void simLoadInitializationValues(daeSimulation_t* s, const char* strFileName)
{
}
