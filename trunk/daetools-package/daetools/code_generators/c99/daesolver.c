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
#include "daesolver.h"
#include "simulation.h"

/* Sundials IDAS related functions */
#define JACOBIAN(A) (A->cols)

/* Private functions */
static void print_final_stats(void *mem);
static int check_flag(void *flagvalue, char *funcname, int opt);

int resid(realtype tres,
          N_Vector yy,
          N_Vector yp,
          N_Vector resval,
          void *user_data);

int root(realtype t,
         N_Vector yy,
         N_Vector yp,
         realtype *gout,
         void *user_data);

int jacob(long int Neq,
          realtype tt,
          realtype cj,
          N_Vector yy,
          N_Vector yp,
          N_Vector resvec,
          DlsMat JJ,
          void *user_data,
          N_Vector tempv1,
          N_Vector tempv2,
          N_Vector tempv3);


int solInitialize(daeIDASolver_t* s, void* model, void* simulation, long Neqns, const real_t*  initValues, const real_t* initDerivatives,
                  const real_t*  absTolerances, const int* IDs, real_t  relativeTolerance)
{
    int i, retval;
    N_Vector yy, yp, avtol, ids;
    realtype t0;

    s->model      = model;
    s->simulation = simulation;
    s->Neqns      = Neqns;
    s->rtol       = relativeTolerance;

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(Neqns);
    if(check_flag((void *)yy, "N_VNew_Serial", 0))
        return(-1);
    s->yy = yy;

    yp = N_VNew_Serial(Neqns);
    if(check_flag((void *)yp, "N_VNew_Serial", 0))
        return(-1);
    s->yp = yp;

    avtol = N_VNew_Serial(Neqns);
    if(check_flag((void *)avtol, "N_VNew_Serial", 0))
        return(-1);

    ids = N_VNew_Serial(Neqns);
    if(check_flag((void *)ids, "N_VNew_Serial", 0))
        return(-1);

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    s->yval = NV_DATA_S(yy);
    /* First copy default values and then set initial conditions */
    for(i = 0; i < Neqns; i++)
        s->yval[i] = initValues[i];

    s->ypval = NV_DATA_S(yp);
    for(i = 0; i < Neqns; i++)
        s->ypval[i] = initDerivatives[i];

    s->atval = NV_DATA_S(avtol);
    for(i = 0; i < Neqns; i++)
        s->atval[i] = absTolerances[i];

    s->idsval = NV_DATA_S(ids);
    for(i = 0; i < Neqns; i++)
        s->idsval[i] = (real_t)IDs[i];

    /* Integration limits */
    t0 = 0;

    /* Call IDACreate and IDAMalloc to initialize IDA memory */
    s->mem = IDACreate();
    if(check_flag((void *)s->mem, "IDACreate", 0))
        return(-1);

    retval = IDASetId(s->mem, ids);
    if(check_flag(&retval, "IDASetId", 1))
        return(-1);

    retval = IDAInit(s->mem, resid, t0, yy, yp);
    if(check_flag(&retval, "IDAInit", 1))
        return(-1);

    retval = IDASVtolerances(s->mem, s->rtol, avtol);
    if(check_flag(&retval, "IDASVtolerances", 1))
        return(-1);

    /* Free avtol */
    N_VDestroy_Serial(avtol);
    N_VDestroy_Serial(ids);

    retval = IDASetUserData(s->mem, s);
    if(check_flag(&retval, "IDASetUserData", 1))
        return(-1);

    /* Call IDARootInit to set the root function and number of roots. */
    int noRoots = modNumberOfRoots(model);
    retval = IDARootInit(s->mem, noRoots, root);
    if (check_flag(&retval, "IDARootInit", 1))
         return(-1);

    /* Call IDADense and set up the linear solver. */
    retval = IDADense(s->mem, Neqns);
    if(check_flag(&retval, "IDADense", 1))
        return(-1);

    retval = IDADlsSetDenseJacFn(s->mem, jacob);
    if(check_flag(&retval, "IDADlsSetDenseJacFn", 1))
        return(-1);

    return IDA_SUCCESS;
}

int solDestroy(daeIDASolver_t* s)
{
    print_final_stats(s->mem);

    /* Free memory */
    IDAFree(&s->mem);
    N_VDestroy_Serial((N_Vector)s->yy);
    N_VDestroy_Serial((N_Vector)s->yp);

    return IDA_SUCCESS;
}

int solRefreshRootFunctions(daeIDASolver_t* s, int noRoots)
{
    printf("    Resetting root functions...\n");
    return IDARootInit(s->mem, noRoots, root);
}

int solResetIDASolver(daeIDASolver_t* s, bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities)
{
    printf("    Resetting IDA solver...\n");
    s->m_dCurrentTime = dCurrentTime;
    return IDAReInit(s->mem, dCurrentTime, (N_Vector)s->yy, (N_Vector)s->yp);
}

real_t solSolve(daeIDASolver_t* s, real_t dTime, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities)
{
    int retval;
    bool copyValuesToSolver;
    daeeDiscontinuityType eDiscontinuityType;

    daeSimulation_t* simulation = (daeSimulation_t*)s->simulation;
    s->m_dTargetTime = dTime;

    for(;;)
    {
        retval = IDASolve(s->mem, s->m_dTargetTime, &s->m_dCurrentTime, (N_Vector)s->yy, (N_Vector)s->yp, IDA_NORMAL);

        if(retval == IDA_TOO_MUCH_WORK)
        {
            printf("Warning: IDAS solver error at TIME = %f [IDA_TOO_MUCH_WORK]\n", s->m_dCurrentTime);
            printf("  Try to increase MaxNumSteps option\n");
            realtype tolsfac = 0;
            retval = IDAGetTolScaleFactor(s->mem, &tolsfac);
            printf("  Suggested factor by which the user’s tolerances should be scaled is %f\n", tolsfac);
            continue;
        }
        else if(retval < 0)
        {
            printf("Sundials IDAS solver cowardly failed to solve the system at TIME = %f; time horizon [%f]; %s\n", s->m_dCurrentTime, s->m_dTargetTime, IDAGetReturnFlagName(retval));
            return -1;
        }

        /* If a root has been found, check if any of conditions are satisfied and do what is necessary */
        if(retval == IDA_ROOT_RETURN)
        {
            bool discontinuity_found = modCheckForDiscontinuities((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);
            if(discontinuity_found)
            {
                /* Data will be reported only if there is a discontinuity */
                if(bReportDataAroundDiscontinuities)
                    simReportData(simulation);

                eDiscontinuityType = modExecuteActions((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);

                if(eDiscontinuityType == eModelDiscontinuity)
                {
                    solReinitialize(s, false, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simReportData(simulation);

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return s->m_dCurrentTime;
                }
                else if(eDiscontinuityType == eModelDiscontinuityWithDataChange)
                {
                    solReinitialize(s, true, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simReportData(simulation);

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return s->m_dCurrentTime;
                }
                else if(eDiscontinuityType == eGlobalDiscontinuity)
                {
                    printf("Not supported discontinuity type: eGlobalDiscontinuity\n");
                    return -1;
                }
            }
            else
            {
            }
        }

        if(s->m_dCurrentTime == s->m_dTargetTime)
            break;
    }

    return s->m_dCurrentTime;
}

int solSolveInitial(daeIDASolver_t* s)
{
    int retval, iCounter, noRoots;

    daeModel_t* model = (daeModel_t*)s->model;

    for(iCounter = 0; iCounter < 100; iCounter++)
    {
        if(model->quasySteadyState)
            retval = IDACalcIC(s->mem, IDA_YA_YDP_INIT, 0.001);
        else
            retval = IDACalcIC(s->mem, IDA_Y_INIT, 0.001);

        if(retval < 0)
        {
            printf("Sundials IDAS solver cowardly failed re-initialize the system at TIME = %f; %s\n",  s->m_dCurrentTime, IDAGetReturnFlagName(retval));
            return -1;
        }

        bool discontinuity_found = modCheckForDiscontinuities((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);
        if(discontinuity_found)
        {
            modExecuteActions((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);
            noRoots = modNumberOfRoots((daeModel_t*)s->model);
            solRefreshRootFunctions(s, noRoots);
            solResetIDASolver(s, true, s->m_dCurrentTime, false);
        }
        else
        {
            break;
        }
    }

    if(iCounter >= 100)
    {
        printf("Sundials IDAS solver cowardly failed initialize the system at TIME = 0: Max number of STN rebuilds reached; %s\n", IDAGetReturnFlagName(retval));
        return -1;
    }

    /* Get the corrected IC and send them to the block */
    retval = IDAGetConsistentIC(s->mem, (N_Vector)s->yy, (N_Vector)s->yp);

    /* Get the corrected sensitivity IC */
    /*
    if(s->m_bCalculateSensitivities)
        retval = IDAGetSensConsistentIC(s->mem, (N_Vector)s->yysens, (N_Vector)s->ypsens);
    */
    return IDA_SUCCESS;
}

int solReinitialize(daeIDASolver_t* s, bool bCopyDataFromBlock, bool bResetSensitivities)
{
    int retval, iCounter;

    printf("    Reinitializing at time: %f\n", s->m_dCurrentTime);

    int noRoots = modNumberOfRoots((daeModel_t*)s->model);
    solRefreshRootFunctions(s, noRoots);
    solResetIDASolver(s, bCopyDataFromBlock, s->m_dCurrentTime, bResetSensitivities);

    for(iCounter = 0; iCounter < 100; iCounter++)
    {
        /* Here we always use the IDA_YA_YDP_INIT flag (and discard InitialConditionMode).
         * The reason is that in this phase we may have been reinitialized the diff. variables
         * with the new values and using the eQuasySteadyState flag would be meaningless.
         */
        retval = IDACalcIC(s->mem, IDA_YA_YDP_INIT, s->m_dCurrentTime + 0.001);
        if(retval < 0)
        {
            printf("Sundials IDAS solver cowardly failed re-initialize the system at TIME = %f; %s\n",  s->m_dCurrentTime, IDAGetReturnFlagName(retval));
            return -1;
        }

        bool discontinuity_found = modCheckForDiscontinuities((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);
        if(discontinuity_found)
        {
            modExecuteActions((daeModel_t*)s->model, s->m_dCurrentTime, s->yval, s->ypval);
            noRoots = modNumberOfRoots((daeModel_t*)s->model);
            solRefreshRootFunctions(s, noRoots);
            solResetIDASolver(s, true, s->m_dCurrentTime, false);
        }
        else
        {
            break;
        }
    }

    if(iCounter >= 100)
    {
        printf("Sundials IDAS solver dastardly failed re-initialize the system at TIME = %f: Max number of STN rebuilds reached; %s\n", s->m_dCurrentTime, IDAGetReturnFlagName(retval));
        return -1;
    }

    /* Get the corrected IC and send them to the block */
    retval = IDAGetConsistentIC(s->mem, (N_Vector)s->yy, (N_Vector)s->yp);

   /* Get the corrected sensitivity IC */
    /*
    if(s->m_bCalculateSensitivities)
        retval = IDAGetSensConsistentIC(s->mem, (N_Vector)s->yysens, (N_Vector)s->ypsens);
    */
    return IDA_SUCCESS;
}

/* Private functions */
int resid(realtype tres,
          N_Vector yy,
          N_Vector yp,
          N_Vector resval,
          void *user_data)
{
    realtype* yval   = NV_DATA_S(yy);
    realtype* ypval  = NV_DATA_S(yp);
    realtype* res    = NV_DATA_S(resval);
    
    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t* model    = (daeModel_t*)dae_solver->model;

    return modResiduals(model, tres, yval, ypval, res);
}

int root(realtype t,
         N_Vector yy,
         N_Vector yp,
         realtype *gout,
         void *user_data)
{
    realtype* yval   = NV_DATA_S(yy);
    realtype* ypval  = NV_DATA_S(yp);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t* model    = (daeModel_t*)dae_solver->model;

    return modRoots(model, t, yval, ypval, gout);
}

int jacob(long int Neq,
          realtype tt,
          realtype cj,
          N_Vector yy,
          N_Vector yp,
          N_Vector resvec,
          DlsMat JJ,
          void *user_data,
          N_Vector tempv1,
          N_Vector tempv2,
          N_Vector tempv3)
{
    realtype* yval   = NV_DATA_S(yy);
    realtype* ypval  = NV_DATA_S(yp);
    realtype* res    = NV_DATA_S(resvec);
    realtype** jacob = JACOBIAN(JJ);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t* model    = (daeModel_t*)dae_solver->model;

    return modJacobian(model, Neq, tt,  cj, yval, ypval, res, jacob);
}

static void print_final_stats(void *mem)
{
    int retval;
    long int nst, nni, nje, nre, nreLS, netf, ncfn, nge;

    retval = IDAGetNumSteps(mem, &nst);
    check_flag(&retval, "IDAGetNumSteps", 1);
    retval = IDAGetNumResEvals(mem, &nre);
    check_flag(&retval, "IDAGetNumResEvals", 1);
    retval = IDADlsGetNumJacEvals(mem, &nje);
    check_flag(&retval, "IDADlsGetNumJacEvals", 1);
    retval = IDAGetNumNonlinSolvIters(mem, &nni);
    check_flag(&retval, "IDAGetNumNonlinSolvIters", 1);
    retval = IDAGetNumErrTestFails(mem, &netf);
    check_flag(&retval, "IDAGetNumErrTestFails", 1);
    retval = IDAGetNumNonlinSolvConvFails(mem, &ncfn);
    check_flag(&retval, "IDAGetNumNonlinSolvConvFails", 1);
    retval = IDADlsGetNumResEvals(mem, &nreLS);
    check_flag(&retval, "IDADlsGetNumResEvals", 1);
    retval = IDAGetNumGEvals(mem, &nge);
    check_flag(&retval, "IDAGetNumGEvals", 1);

    printf("\nFinal Run Statistics: \n\n");
    printf("Number of steps                    = %ld\n", nst);
    printf("Number of residual evaluations     = %ld\n", nre+nreLS);
    printf("Number of Jacobian evaluations     = %ld\n", nje);
    printf("Number of nonlinear iterations     = %ld\n", nni);
    printf("Number of error test failures      = %ld\n", netf);
    printf("Number of nonlinear conv. failures = %ld\n", ncfn);
    printf("Number of root fn. evaluations     = %ld\n", nge);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
    int *errflag;
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL)
    {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        return(1);
    }
    else if (opt == 1)
    {
        /* Check if flag < 0 */
        errflag = (int *) flagvalue;
        if (*errflag < 0)
        {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag);
            return(1);
        }
    }
    else if (opt == 2 && flagvalue == NULL)
    {
        /* Check if function returned NULL pointer - no memory allocated */
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
        return(1);
    }

    return(0);
}

