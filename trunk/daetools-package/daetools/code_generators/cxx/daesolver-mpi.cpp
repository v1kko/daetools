/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <mpi.h> // this is important to include first (otherwise we get "error: template with C linkage")
#include <idas/idas.h>
#include <idas/idas_spgmr.h>
#include <idas/idas_bbdpre.h>
#include <nvector/nvector_parallel.h>
#include <sundials/sundials_math.h>
#include "auxiliary.h"
#include "daesolver.h"
#include "simulation.h"

/* Sundials IDAS related functions */
//#define JACOBIAN(A) (A->cols)

/* Private functions */
static void print_final_stats(void *mem, int mpi_rank);
static int check_flag(void *flagvalue, const char *funcname, int opt);

static int bbd_local_fn(long int Nlocal,
                        realtype tres,
                        N_Vector yy,
                        N_Vector yp,
                        N_Vector resval,
                        void *user_data);
static int setup_preconditioner(realtype tt,
                                N_Vector yy,
                                N_Vector yp,
                                N_Vector rr,
                                realtype c_j,
                                void *user_data,
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int solve_preconditioner(realtype tt,
                                N_Vector uu,
                                N_Vector up,
                                N_Vector rr,
                                N_Vector rvec,
                                N_Vector zvec,
                                realtype c_j,
                                realtype delta,
                                void *user_data,
                                N_Vector tmp);

static int resid(realtype tres,
                 N_Vector yy,
                 N_Vector yp,
                 N_Vector resval,
                 void *user_data);

static int root(realtype t,
                N_Vector yy,
                N_Vector yp,
                realtype *gout,
                void *user_data);

int solInitialize(daeIDASolver_t* s, void* model, void* simulation, long Nequations, long Nequations_local,
                  const real_t*  initValues, const real_t* initDerivatives,
                  const real_t*  absTolerances, const int* IDs, real_t  relativeTolerance)
{
    int i, retval;
    N_Vector yy, yp, avtol, ids;
    realtype t0;

    s->model      = model;
    s->simulation = simulation;
    s->Nequations = Nequations_local;
    s->rtol       = relativeTolerance;

    daeModel_t* pmodel = (daeModel_t*)model;
    MPI_Comm comm = (MPI_Comm)pmodel->mpi_comm;;

    /* Allocate N-vectors. */
    yy = N_VNew_Parallel(comm, pmodel->Nequations_local, pmodel->Nequations);
    if(check_flag((void *)yy, "N_VNew_Parallel", 0))
        return(-1);
    s->yy = yy;

    yp = N_VNew_Parallel(comm, pmodel->Nequations_local, pmodel->Nequations);
    if(check_flag((void *)yp, "N_VNew_Parallel", 0))
        return(-1);
    s->yp = yp;

    avtol = N_VNew_Parallel(comm, pmodel->Nequations_local, pmodel->Nequations);
    if(check_flag((void *)avtol, "N_VNew_Parallel", 0))
        return(-1);

    ids = N_VNew_Parallel(comm, pmodel->Nequations_local, pmodel->Nequations);
    if(check_flag((void *)ids, "N_VNew_Parallel", 0))
        return(-1);

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    s->yval = NV_DATA_P(yy);
    /* First copy default values and then set initial conditions */
    for(i = 0; i < pmodel->Nequations_local; i++)
        s->yval[i] = initValues[i];

    s->ypval = NV_DATA_P(yp);
    for(i = 0; i < pmodel->Nequations_local; i++)
        s->ypval[i] = initDerivatives[i];

    s->atval = NV_DATA_P(avtol);
    for(i = 0; i < pmodel->Nequations_local; i++)
        s->atval[i] = absTolerances[i];

    s->idsval = NV_DATA_P(ids);
    for(i = 0; i < pmodel->Nequations_local; i++)
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
    N_VDestroy_Parallel(avtol);
    N_VDestroy_Parallel(ids);

    retval = IDASetUserData(s->mem, s);
    if(check_flag(&retval, "IDASetUserData", 1))
        return(-1);

    /* Call IDARootInit to set the root function and number of roots. */
    int noRoots = modNumberOfRoots(pmodel);
    retval = IDARootInit(s->mem, noRoots, root);
    if (check_flag(&retval, "IDARootInit", 1))
         return(-1);

    /* Set up the linear solver. */
    int maxl = 0; // Maximum dimension of the Krylov subspace to be used.
                  // Pass 0 to use the default value IDA SPILS MAXL = 5
    retval = IDASpgmr(s->mem, maxl);

    /* Set up the BBD preconditioner. */
/*
    long int mudq      = 5; // Upper half-bandwidth to be used in the difference-quotient Jacobian approximation.
    long int mldq      = 5; // Lower half-bandwidth to be used in the difference-quotient Jacobian approximation.
    long int mukeep    = 2; // Upper half-bandwidth of the retained banded approximate Jacobian block
    long int mlkeep    = 2; // Lower half-bandwidth of the retained banded approximate Jacobian block
    realtype dq_rel_yy = 0.0; // The relative increment in components of y used in the difference quotient approximations.
                              // The default is dq_rel_yy = sqrt(unit roundoff), which can be specified by passing dq_rel_yy = 0.0
    retval = IDABBDPrecInit(s->mem, pmodel->Nequations_local, mudq, mldq, mukeep, mlkeep, dq_rel_yy, bbd_local_fn, NULL);
    if(check_flag(&retval, "IDABBDPrecInit", 1))
        return(-1);
*/
    s->pp.resize(pmodel->Nequations_local, false);
    s->jacob.resize(pmodel->Nequations_local, pmodel->Nequations_local, false);

    retval = IDASpilsSetPreconditioner(s->mem, setup_preconditioner, solve_preconditioner);
    if(check_flag(&retval, "IDASpilsSetPreconditioner", 1))
        return(-1);

    return IDA_SUCCESS;
}

int solDestroy(daeIDASolver_t* s)
{
    daeModel_t* pmodel = (daeModel_t*)s->model;

    if(pmodel->mpi_rank == 0)
        print_final_stats(s->mem, pmodel->mpi_rank);

    /* Free memory */
    s->pp.clear();
    s->jacob.clear();
    IDAFree(&s->mem);
    N_VDestroy_Parallel((N_Vector)s->yy);
    N_VDestroy_Parallel((N_Vector)s->yp);

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
    daeeDiscontinuityType eDiscontinuityType;

    daeModel_t* model = (daeModel_t*)s->model;
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
            bool discontinuity_found = modCheckForDiscontinuities(model, s->m_dCurrentTime, s->yval, s->ypval);
            if(discontinuity_found)
            {
                /* The data will be reported only if there is a discontinuity */
                if(bReportDataAroundDiscontinuities)
                    simReportData(simulation);

                eDiscontinuityType = modExecuteActions(model, s->m_dCurrentTime, s->yval, s->ypval);

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
            retval = IDACalcIC(s->mem, IDA_Y_INIT, 0.001);
        else
            retval = IDACalcIC(s->mem, IDA_YA_YDP_INIT, 0.001);

        if(retval < 0)
        {
            printf("Sundials IDAS solver cowardly failed to solve the system at TIME = %f; %s\n",  s->m_dCurrentTime, IDAGetReturnFlagName(retval));
            return -1;
        }

        bool discontinuity_found = modCheckForDiscontinuities(model, s->m_dCurrentTime, s->yval, s->ypval);
        if(discontinuity_found)
        {
            modExecuteActions(model, s->m_dCurrentTime, s->yval, s->ypval);
            noRoots = modNumberOfRoots(model);
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

    daeModel_t* model = (daeModel_t*)s->model;
    int noRoots = modNumberOfRoots(model);
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

        bool discontinuity_found = modCheckForDiscontinuities(model, s->m_dCurrentTime, s->yval, s->ypval);
        if(discontinuity_found)
        {
            modExecuteActions(model, s->m_dCurrentTime, s->yval, s->ypval);
            noRoots = modNumberOfRoots(model);
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
int bbd_local_fn(long int Nlocal,
                 realtype tres,
                 N_Vector yy,
                 N_Vector yp,
                 N_Vector resval,
                 void *user_data)
{
    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);
    realtype* res    = NV_DATA_P(resval);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t*     model      = (daeModel_t*)dae_solver->model;

    for(int i = 0; i < dae_solver->Nequations; i++)
    {
        dae_solver->yval[i]  = yval[i];
        dae_solver->ypval[i] = ypval[i];
    }

    return modResiduals(model, tres, yval, ypval, res);
}

int setup_preconditioner(realtype tt,
                         N_Vector yy, N_Vector yp, N_Vector rr,
                         realtype cj, void *user_data,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t*     model      = (daeModel_t*)dae_solver->model;

    for(int i = 0; i < dae_solver->Nequations; i++)
    {
        dae_solver->yval[i]  = yval[i];
        dae_solver->ypval[i] = ypval[i];
    }

    int res = modJacobian(model, dae_solver->Nequations, tt,  cj, NULL, NULL, NULL, dae_solver->jacob);

    for(int i = 0; i < dae_solver->Nequations; i++)
        dae_solver->pp[i] = 1.0 / (dae_solver->jacob(i,i) + 1e-20);

    return res;
}

int solve_preconditioner(realtype tt,
                         N_Vector uu, N_Vector up, N_Vector rr,
                         N_Vector rvec, N_Vector zvec,
                         realtype c_j, realtype delta, void *user_data,
                         N_Vector tmp)
{
    realtype* r = NV_DATA_P(rvec);
    realtype* z = NV_DATA_P(zvec);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t*     model      = (daeModel_t*)dae_solver->model;

    for(int i = 0; i < dae_solver->Nequations; i++)
        z[i] = dae_solver->pp[i] * r[i];

    return(0);
}

int resid(realtype tres,
          N_Vector yy,
          N_Vector yp,
          N_Vector resval,
          void *user_data)
{
    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);
    realtype* res    = NV_DATA_P(resval);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t*     model      = (daeModel_t*)dae_solver->model;

    for(int i = 0; i < dae_solver->Nequations; i++)
    {
        dae_solver->yval[i]  = yval[i];
        dae_solver->ypval[i] = ypval[i];
    }

    // Call MPI synchronise data every time before calculating residuals
    modSynchroniseData(model);

    return modResiduals(model, tres, yval, ypval, res);
}

int root(realtype t,
         N_Vector yy,
         N_Vector yp,
         realtype *gout,
         void *user_data)
{
    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    daeModel_t* model    = (daeModel_t*)dae_solver->model;

    return modRoots(model, t, yval, ypval, gout);
}
/*
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
*/
static void print_final_stats(void *mem, int mpi_rank)
{
    int retval;
    long int nst, nni, nre, nli, nreLS, nge, npe, nps;

    if(mpi_rank == 0)
    {
        retval = IDAGetNumSteps(mem, &nst);
        check_flag(&retval, "IDAGetNumSteps", 1);
        retval = IDAGetNumResEvals(mem, &nre);
        check_flag(&retval, "IDAGetNumResEvals", 1);
        retval = IDAGetNumNonlinSolvIters(mem, &nni);
        check_flag(&retval, "IDAGetNumNonlinSolvIters", 1);
        retval = IDASpilsGetNumLinIters(mem, &nli);
        check_flag(&retval, "IDASpilsGetNumLinIters", 1);
        retval = IDASpilsGetNumResEvals(mem, &nreLS);
        check_flag(&retval, "IDASpilsGetNumResEvals", 1);
        retval = IDABBDPrecGetNumGfnEvals(mem, &nge);
        check_flag(&retval, "IDABBDPrecGetNumGfnEvals", 1);
        retval = IDASpilsGetNumPrecEvals(mem, &npe);
        check_flag(&retval, "IDASpilsGetPrecEvals", 1);
        retval = IDASpilsGetNumPrecSolves(mem, &nps);
        check_flag(&retval, "IDASpilsGetNumPrecSolves", 1);

        printf("\nFinal Run Statistics: \n\n");
        printf("Number of steps                    = %ld\n", nst);
        printf("Number of residual evaluations     = %ld\n", nre);
        printf("Number of nonlinear iterations     = %ld\n", nni);
        printf("Number of linear iterations        = %ld\n", nli);
        printf("Number of sp residual evaluations  = %ld\n", nreLS);
        printf("Number of Gfn evals                = %ld\n", nge);
        printf("Number of preconditioner evals     = %ld\n", npe);
        printf("Number of preconditioner solves    = %ld\n", nps);
    }
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

static int check_flag(void *flagvalue, const char *funcname, int opt)
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

