/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2010
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
#include "daetools_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <idas/idas.h>
#include <idas/idas_dense.h>
#include <sundials/sundials_math.h>
#include <nvector/nvector_serial.h>

#define JACOBIAN(A) (A->cols)

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

/* Prototypes of private functions */
static void print_final_stats(void *mem);
static int check_flag(void *flagvalue, char *funcname, int opt);

int main(int argc, char *argv[])
{
    void *mem;
    int i, retval, no_roots;
    bool copy_to_solver, discontinuity_found;
    N_Vector yy, yp, avtol, ids;
    realtype rtol, *yval, *ypval, *atval, *idsval;
    realtype t, t0, tend, tret, treport;

    mem = NULL;
    yy = yp = avtol = NULL;
    yval = ypval = atval = NULL;

    initial_conditions();

    /* Allocate N-vectors. */
    yy = N_VNew_Serial(_Neqns_);
    if(check_flag((void *)yy, "N_VNew_Serial", 0))
        return(1);

    yp = N_VNew_Serial(_Neqns_);
    if(check_flag((void *)yp, "N_VNew_Serial", 0))
        return(1);

    avtol = N_VNew_Serial(_Neqns_);
    if(check_flag((void *)avtol, "N_VNew_Serial", 0))
        return(1);

    ids = N_VNew_Serial(_Neqns_);
    if(check_flag((void *)avtol, "N_VNew_Serial", 0))
        return(1);

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    yval  = NV_DATA_S(yy);
    for(i = 0; i < _Neqns_; i++)
        yval[i] = _initValues_[i];

    ypval = NV_DATA_S(yp);
    for(i = 0; i < _Neqns_; i++)
        ypval[i] = _initDerivatives_[i];

    rtol = _relative_tolerance_;

    atval = NV_DATA_S(avtol);
    for(i = 0; i < _Neqns_; i++)
        atval[i] = _absolute_tolerances_[i];

    idsval  = NV_DATA_S(ids);
    for(i = 0; i < _Neqns_; i++)
        idsval[i] = (realtype)_IDs_[i];

    /* Integration limits */
    t0      = _start_time_;
    tend    = _end_time_;
    treport = _reporting_interval_;

    /* Call IDACreate and IDAMalloc to initialize IDA memory */
    mem = IDACreate();
    if(check_flag((void *)mem, "IDACreate", 0))
        return(1);

    retval = IDASetId(mem, ids);
    if(check_flag(&retval, "IDASetId", 1))
        return(1);

    retval = IDAInit(mem, resid, t0, yy, yp);
    if(check_flag(&retval, "IDAInit", 1))
        return(1);

    retval = IDASVtolerances(mem, rtol, avtol);
    if(check_flag(&retval, "IDASVtolerances", 1))
        return(1);

    /* Free avtol */
    N_VDestroy_Serial(avtol);
    N_VDestroy_Serial(ids);

    /* For IF state transition network we do not know which state is initially active.
     * Therefore, we have to call check_for_discontinuities() which will check if
     * there should be any change in active states. If there is, we have to call
     * function execute_actions() which will activate correct states. */
    discontinuity_found = check_for_discontinuities(t0, yval, ypval);
    if(discontinuity_found)
        copy_to_solver = execute_actions(tret, yval, ypval);
    
    /* Now when we obtained correct active states we have to get the number of roots
     * and call specify IDARootInit to set the root function and number of roots. */
    no_roots = number_of_roots();
    retval = IDARootInit(mem, no_roots, root);
    if (check_flag(&retval, "IDARootInit", 1))
         return(1);

    /* Call IDADense and set up the linear solver. */
    retval = IDADense(mem, _Neqns_);
    if(check_flag(&retval, "IDADense", 1))
        return(1);

    retval = IDADlsSetDenseJacFn(mem, jacob);
    if(check_flag(&retval, "IDADlsSetDenseJacFn", 1))
        return(1);

    retval = IDACalcIC(mem, IDA_YA_YDP_INIT, 0.1);
    if(check_flag(&retval, "IDACalcIC", 1))
        return(1);

    /* Print the results at time = 0 */
    _print_results_(t0, yval, _variable_names_, _Neqns_);

    /* In loop, call IDASolve, print results, and test for error.
     * Break out of loop when NOUT preset output times have been reached. */
    t = t0 + treport;
    tret = t0;
    while(tret < tend)
    {
        retval = IDASolve(mem, t, &tret, yy, yp, IDA_NORMAL);
        if(check_flag(&retval, "IDASolve", 1))
            return(1);

        yval  = NV_DATA_S(yy);
        ypval = NV_DATA_S(yp);

        if(retval == IDA_ROOT_RETURN)
        {
            discontinuity_found = check_for_discontinuities(tret, yval, ypval);
            if(discontinuity_found)
            {
                printf("**************************************\n");
                printf("DISCONTINUITY FOUND: %f \n", tret);
                printf("**************************************\n");

                /* Print results before the discontinuity */
                _print_results_(tret, yval, _variable_names_, _Neqns_);

                /* Execute ON_CONDITION actions
                 * Here some actions write directly into the yval array and we do not need
                 * to copy values to the solver (it will be done during the IDAReInit() call) */
                copy_to_solver = execute_actions(tret, yval, ypval);

                /* Find the number of root functions and inform ida */
                no_roots = number_of_roots();
                retval = IDARootInit(mem, no_roots, root);
                if (check_flag(&retval, "IDARootInit", 1))
                     return(1);

                /* Reinitialize the system */
                retval = IDAReInit(mem, tret, yy, yp);
                if(check_flag(&retval, "IDAReInit", 1))
                    return(1);

                retval = IDACalcIC(mem, IDA_YA_YDP_INIT, t + 1E-05);
                if(check_flag(&retval, "IDACalcIC", 1))
                    return(1);

                /* Print results again after the discontinuity */
                _print_results_(tret, yval, _variable_names_, _Neqns_);
            }
            
            /* This is important if the root is exactly at the end of the reporting interval */
            if(tret < t)
                continue;
        }
        else if(retval == IDA_SUCCESS)
        {
            _print_results_(tret, yval, _variable_names_, _Neqns_);
        }
        else
        {
            printf("IDA return value: %d", retval);
        }

        t += treport;
        if(t > tend)
            t = tend;
    }

    print_final_stats(mem);

    /* Free memory */

    IDAFree(&mem);
    N_VDestroy_Serial(yy);
    N_VDestroy_Serial(yp);

    return 0;
}

int resid(realtype tres,
          N_Vector yy,
          N_Vector yp,
          N_Vector resval,
          void *user_data)
{
    realtype* yval   = NV_DATA_S(yy);
    realtype* ypval  = NV_DATA_S(yp);
    realtype* res    = NV_DATA_S(resval);

    return residuals(tres, yval, ypval, res);
}

int root(realtype t,
         N_Vector yy,
         N_Vector yp,
         realtype *gout,
         void *user_data)
{
    realtype* yval   = NV_DATA_S(yy);
    realtype* ypval  = NV_DATA_S(yp);

    return roots(t, yval, ypval, gout);
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

    return jacobian(Neq, tt,  cj, yval, ypval, res, jacob);
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
