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
#include <fstream>
#include <math.h>
#include <iomanip>
#include <idas/idas.h>
#include <idas/idas_spgmr.h>
#include <idas/idas_bbdpre.h>
#include <nvector/nvector_parallel.h>
#include <sundials/sundials_math.h>
#include "auxiliary.h"
#include "cs_simulator.h"
#include "idas_la_functions.h"
#include <boost/format.hpp>
#include <idas/idas_impl.h>
#include "../idas/src/idas/idas_spils_impl.h"

namespace cs_dae_simulator
{
/* Private functions */
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
                                N_Vector tmp1,
                                N_Vector tmp2,
                                N_Vector tmp3);
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
static int jacobian_vector_multiply(realtype tt,
                                    N_Vector yy,
                                    N_Vector yp,
                                    N_Vector rr,
                                    N_Vector v,
                                    N_Vector Jv,
                                    realtype c_j,
                                    void *user_data,
                                    N_Vector tmp1,
                                    N_Vector tmp2);
static int jacobian_vector_dq(realtype tt,
                              N_Vector yy,
                              N_Vector yp,
                              N_Vector rr,
                              N_Vector vvec,
                              N_Vector Jvvec,
                              realtype c_j,
                              void *user_data,
                              N_Vector tmp1,
                              N_Vector tmp2);

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

daeIDASolver_t::daeIDASolver_t()
{
    mem         = NULL;
    yy          = NULL;
    yp          = NULL;
    model       = NULL;
    simulation  = NULL;
    integrationMode = IDA_NORMAL;
}

daeIDASolver_t::~daeIDASolver_t()
{
    Free();
}

void daeIDASolver_t::Initialize(csDAEModel_t*    pmodel,
                                daeSimulation_t* psimulation,
                                long             Neq,
                                long             Neq_local,
                                const real_t*    initValues,
                                const real_t*    initDerivatives,
                                const real_t*    absTolerances,
                                const int*       IDs)
{
    int i, retval;
    N_Vector yy_nv, yp_nv, avtol_nv, ids_nv;
    realtype t0;
    real_t* atval;
    real_t* idsval;

    model          = pmodel;
    simulation     = psimulation;
    preconditioner = NULL;
    lasolver       = NULL;
    Nequations     = Neq_local;

    MPI_Comm comm = (MPI_Comm)model->mpi_comm;
    daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();

    printInfo = cfg.GetBoolean("DAESolver.PrintInfo", false);
    rtol      = cfg.GetFloat("DAESolver.Parameters.RelativeTolerance", 1e-5);
    printf("rtol = %f\n", rtol);

    std::string integrationMode_s = cfg.GetString("DAESolver.Parameters.IntegrationMode", "Normal");
    if(integrationMode_s == "Normal")
        integrationMode = IDA_NORMAL;
    else if(integrationMode_s == "OneStep")
        integrationMode = IDA_ONE_STEP;
    else
        daeThrowException("Invalid integration mode specified: " + integrationMode_s);
    //printf("IntegrationMode = %s\n", integrationMode_s.c_str());
    //printf("IntegrationMode = %d\n", integrationMode);

    /* Allocate N-vectors. */
    yy_nv = N_VNew_Parallel(comm, model->Nequations_local, model->Nequations);
    if(yy_nv == NULL)
        daeThrowException("Allocation of yy array failed");
    yy = yy_nv;

    yp_nv = N_VNew_Parallel(comm, model->Nequations_local, model->Nequations);
    if(yp_nv == NULL)
        daeThrowException("Allocation of yp array failed");
    yp = yp_nv;

    avtol_nv = N_VNew_Parallel(comm, model->Nequations_local, model->Nequations);
    if(avtol_nv == NULL)
        daeThrowException("Allocation of avtol array failed");

    ids_nv = N_VNew_Parallel(comm, model->Nequations_local, model->Nequations);
    if(ids_nv == NULL)
        daeThrowException("Allocation of ids array failed");

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    yval = NV_DATA_P(yy_nv);
    for(i = 0; i < model->Nequations_local; i++)
        yval[i] = initValues[i];

    ypval = NV_DATA_P(yp_nv);
    for(i = 0; i < model->Nequations_local; i++)
        ypval[i] = initDerivatives[i];

    atval = NV_DATA_P(avtol_nv);
    for(i = 0; i < model->Nequations_local; i++)
        atval[i] = absTolerances[i];

    idsval = NV_DATA_P(ids_nv);
    for(i = 0; i < model->Nequations_local; i++)
        idsval[i] = (real_t)IDs[i];

    /* Integration limits */
    t0 = 0;

    /* Call IDACreate and IDAMalloc to initialize IDA memory */
    mem = IDACreate();
    if(mem == NULL)
        daeThrowException("IDACreate failed");

    retval = IDASetId(mem, ids_nv);
    if(retval < 0)
        daeThrowException( (boost::format("IDASetId failed: %s") % IDAGetReturnFlagName(retval)).str() );

    retval = IDAInit(mem, resid, t0, yy_nv, yp_nv);
    if(retval < 0)
        daeThrowException( (boost::format("IDAInit failed: %s") % IDAGetReturnFlagName(retval)).str() );

    retval = IDASVtolerances(mem, rtol, avtol_nv);
    if(retval < 0)
        daeThrowException( (boost::format("IDASVtolerances failed: %s") % IDAGetReturnFlagName(retval)).str() );

    /* Free avtol */
    N_VDestroy_Parallel(avtol_nv);
    N_VDestroy_Parallel(ids_nv);

    /* Set maximum time. */
    IDASetStopTime(mem, simulation->timeHorizon);

    /* Solver options. */
    IDASetMaxOrd(mem,      cfg.GetInteger("DAESolver.Parameters.MaxOrd",      5));
    IDASetMaxNumSteps(mem, cfg.GetInteger("DAESolver.Parameters.MaxNumSteps", 500));

    realtype fval = cfg.GetFloat("DAESolver.Parameters.InitStep", 0.0);
    if(fval > 0.0)
        IDASetInitStep(mem, fval);

    fval = cfg.GetFloat("DAESolver.Parameters.MaxStep", 0.0);
    if(fval > 0.0)
        IDASetMaxStep(mem, fval);

    IDASetMaxErrTestFails(mem,   cfg.GetInteger("DAESolver.Parameters.MaxErrTestFails", 10));
    IDASetMaxNonlinIters(mem,    cfg.GetInteger("DAESolver.Parameters.MaxNonlinIters",  4));
    IDASetMaxConvFails(mem,      cfg.GetInteger("DAESolver.Parameters.MaxConvFails",    10));
    IDASetNonlinConvCoef(mem,    cfg.GetFloat  ("DAESolver.Parameters.NonlinConvCoef",  0.33));
    IDASetSuppressAlg(mem,       cfg.GetBoolean("DAESolver.Parameters.SuppressAlg",     false));
    bool bval = cfg.GetBoolean("DAESolver.Parameters.NoInactiveRootWarn", false);
    if(bval)
        IDASetNoInactiveRootWarn(mem);

    /* Inital condition options. */
    IDASetNonlinConvCoefIC(mem,  cfg.GetFloat  ("DAESolver.Parameters.NonlinConvCoefIC", 0.0033));
    IDASetMaxNumStepsIC(mem,     cfg.GetInteger("DAESolver.Parameters.MaxNumStepsIC",    5));
    IDASetMaxNumJacsIC(mem,      cfg.GetInteger("DAESolver.Parameters.MaxNumJacsIC",     4));
    IDASetMaxNumItersIC(mem,     cfg.GetInteger("DAESolver.Parameters.MaxNumItersIC",    10));
    IDASetLineSearchOffIC(mem,   cfg.GetBoolean("DAESolver.Parameters.LineSearchOffIC",  false));

    /* Set user data. */
    retval = IDASetUserData(mem, this);
    if(retval < 0)
        daeThrowException( (boost::format("IDASetUserData failed: %s") % IDAGetReturnFlagName(retval)).str() );

    /* Call IDARootInit to set the root function and number of roots. */
    int noRoots = model->NumberOfRoots();
    retval = IDARootInit(mem, noRoots, root);
    if(retval < 0)
        daeThrowException( (boost::format("IDARootInit failed: %s") % IDAGetReturnFlagName(retval)).str() );

    /* Set up the linear solver. */
    std::string lasolverLibrary = cfg.GetString("LinearSolver.Library");
    if(lasolverLibrary == "Sundials")
    {
        std::string preconditionerName = cfg.GetString("LinearSolver.Preconditioner.Library", "Not specified");

        if(preconditionerName == "Ifpack")
            preconditioner.reset(new daePreconditioner_Ifpack);
        else if(preconditionerName == "ML")
            preconditioner.reset(new daePreconditioner_ML);
        else if(preconditionerName == "Jacobi")
            preconditioner.reset(new daePreconditioner_Jacobi);
        else
            daeThrowException("Invalid preconditioner library specified: " + preconditionerName);

        retval = preconditioner->Initialize(pmodel, model->Nequations_local);
        if(retval < 0)
            daeThrowException( (boost::format("Preconditioner initialize failed: %s") % IDAGetReturnFlagName(retval)).str() );

        // Maximum dimension of the Krylov subspace to be used (if 0 use the default value MAXL = 5)
        int kspace = cfg.GetInteger("LinearSolver.Parameters.kspace", 0);
        retval = IDASpgmr(mem, kspace);
        if(retval < 0)
            daeThrowException( (boost::format("IDASpgmr failed: %s") % IDAGetReturnFlagName(retval)).str() );

        IDASpilsSetEpsLin(mem,      cfg.GetFloat  ("LinearSolver.Parameters.EpsLin",      0.05));
        IDASpilsSetMaxRestarts(mem, cfg.GetInteger("LinearSolver.Parameters.MaxRestarts",    5));
        std::string gstype = cfg.GetString("LinearSolver.Parameters.GSType", "MODIFIED_GS");
        if(gstype == "MODIFIED_GS")
            IDASpilsSetGSType(mem, MODIFIED_GS);
        else if(gstype == "CLASSICAL_GS")
            IDASpilsSetGSType(mem, CLASSICAL_GS);

        retval = IDASpilsSetPreconditioner(mem, setup_preconditioner, solve_preconditioner);
        if(retval < 0)
            daeThrowException( (boost::format("IDASpilsSetPreconditioner failed: %s") % IDAGetReturnFlagName(retval)).str() );

        // This is very important for overall performance!!
        // For some reasons works very slow if jacobian_vector_multiply is used for Npe > 1.
        std::string jtimes = cfg.GetString("LinearSolver.Parameters.JacTimesVecFn", "DifferenceQuotient");
        if(jtimes == "DifferenceQuotient")
        {
            IDASpilsSetIncrementFactor(mem, cfg.GetFloat("LinearSolver.Parameters.DQIncrementFactor",   1.0));
        }
        else if(jtimes == "JacobianVectorMultiply")
        {
            IDASpilsSetJacTimesVecFn(mem, jacobian_vector_multiply);
        }
        else if(jtimes == "DifferenceQuotient_timed")
        {
            // jacobian_vector_dq calls default function: IDASpilsDQJtimes (but measures the time required).
            IDASpilsSetIncrementFactor(mem, cfg.GetFloat("LinearSolver.Parameters.DQIncrementFactor",   1.0));
            IDASpilsSetJacTimesVecFn(mem, jacobian_vector_dq);
        }
        else
        {
            daeThrowException( (boost::format("Not supported LinearSolver.Parameters.JacTimesVecFn option: %s") % jtimes).str() );
        }
    }
    else if(lasolverLibrary == "Trilinos")
    {
        lasolver.reset(new daeLASolver_t);
        retval = lasolver->Initialize(pmodel, model->Nequations_local);
        if(retval < 0)
            daeThrowException( (boost::format("LA Solver initialize failed: %s") % IDAGetReturnFlagName(retval)).str() );

        IDAMem ida_mem = (IDAMem)mem;

        ida_mem->ida_linit	      = init_la;
        ida_mem->ida_lsetup       = setup_la;
        ida_mem->ida_lsolve       = solve_la;
        ida_mem->ida_lperf	      = NULL;
        ida_mem->ida_lfree        = free_la;
        ida_mem->ida_lmem         = lasolver.get();
        ida_mem->ida_setupNonNull = TRUE;
    }
    else
    {
        // Maximum dimension of the Krylov subspace to be used (if 0 use the default value MAXL = 5)
        int kspace = cfg.GetInteger("LinearSolver.Parameters.kspace", 0);
        retval = IDASpgmr(mem, kspace);
        if(retval < 0)
            daeThrowException( (boost::format("IDASpgmr failed: %s") % IDAGetReturnFlagName(retval)).str() );

        /* Set up the BBD preconditioner. */
        long int mudq      = 5; // Upper half-bandwidth to be used in the difference-quotient Jacobian approximation.
        long int mldq      = 5; // Lower half-bandwidth to be used in the difference-quotient Jacobian approximation.
        long int mukeep    = 2; // Upper half-bandwidth of the retained banded approximate Jacobian block
        long int mlkeep    = 2; // Lower half-bandwidth of the retained banded approximate Jacobian block
        realtype dq_rel_yy = 0.0; // The relative increment in components of y used in the difference quotient approximations.
                                  // The default is dq_rel_yy = sqrt(unit roundoff), which can be specified by passing dq_rel_yy = 0.0
        retval = IDABBDPrecInit(mem, model->Nequations_local, mudq, mldq, mukeep, mlkeep, dq_rel_yy, bbd_local_fn, NULL);
        if(retval < 0)
            daeThrowException( (boost::format("IDABBDPrecInit failed: %s") % IDAGetReturnFlagName(retval)).str() );
    }
}

void daeIDASolver_t::Free()
{
    if(mem)
    {
        CollectSolverStats();

        IDAFree(&mem);
        N_VDestroy_Parallel((N_Vector)yy);
        N_VDestroy_Parallel((N_Vector)yp);
    }

    mem         = NULL;
    yy          = NULL;
    yp          = NULL;
    model       = NULL;
    simulation  = NULL;
    lasolver.reset();
    preconditioner.reset();
}

void daeIDASolver_t::RefreshRootFunctions(int noRoots)
{
    int retval = IDARootInit(mem, noRoots, root);
    if(retval < 0)
        daeThrowException( (boost::format("IDARootInit failed: %s") % IDAGetReturnFlagName(retval)).str() );
}

void daeIDASolver_t::ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities)
{
    currentTime = dCurrentTime;
    int retval = IDAReInit(mem, dCurrentTime, (N_Vector)yy, (N_Vector)yp);
    if(retval < 0)
        daeThrowException( (boost::format("IDAReInit failed: %s") % IDAGetReturnFlagName(retval)).str() );
}

real_t daeIDASolver_t::Solve(real_t dTime, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities)
{
    int retval;
    csDiscontinuityType eDiscontinuityType;

    targetTime = dTime;

    for(;;)
    {
        retval = IDASolve(mem, targetTime, &currentTime, (N_Vector)yy, (N_Vector)yp, integrationMode);

        if(retval == IDA_TOO_MUCH_WORK)
        {
            printf("Warning: IDAS solver error at TIME = %f [IDA_TOO_MUCH_WORK]\n", currentTime);
            printf("  Try to increase MaxNumSteps option\n");
            realtype tolsfac = 0;
            retval = IDAGetTolScaleFactor(mem, &tolsfac);
            printf("  Suggested factor by which the user’s tolerances should be scaled is %f\n", tolsfac);
            continue;
        }
        else if(retval < 0)
        {
            std::string msg = (boost::format("Sundials IDAS solver cowardly failed to solve the system at time = %.15f; "
                                             "time horizon [%.15f]; %s\n") % currentTime % targetTime % IDAGetReturnFlagName(retval)).str();
            daeThrowException(msg);
        }

        /* If a root has been found, check if any of conditions are satisfied and do what is necessary */
        if(retval == IDA_ROOT_RETURN)
        {
            bool discontinuity_found = model->CheckForDiscontinuities(currentTime, yval, ypval);
            if(discontinuity_found)
            {
                /* The data will be reported only if there is a discontinuity */
                if(bReportDataAroundDiscontinuities)
                    simulation->ReportData();

                eDiscontinuityType = model->ExecuteActions(currentTime, yval, ypval);

                if(eDiscontinuityType == eModelDiscontinuity)
                {
                    Reinitialize(false, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simulation->ReportData();

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return currentTime;
                }
                else if(eDiscontinuityType == eModelDiscontinuityWithDataChange)
                {
                    Reinitialize(true, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simulation->ReportData();

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return currentTime;
                }
                else if(eDiscontinuityType == eGlobalDiscontinuity)
                {
                    daeThrowException("Not supported discontinuity type: eGlobalDiscontinuity");
                }
            }
            else
            {
            }
        }

        if(printInfo && integrationMode == IDA_ONE_STEP)
        {
            static long cum_psolves = 0;
            int kcur;
            realtype hcur;
            long psolves;

            IDAGetCurrentOrder(mem, &kcur);
            IDAGetCurrentStep(mem, &hcur);
            IDASpilsGetNumPrecSolves(mem, &psolves);

            printf("    t = %.15f, k = %d, h = %.15f, psolves = %ld\n", currentTime, kcur, hcur, psolves-cum_psolves);
            cum_psolves = psolves;
        }

        if(currentTime >= targetTime)
            break;
    }

    return currentTime;
}

void daeIDASolver_t::SolveInitial()
{
    int retval, iCounter, noRoots;

    for(iCounter = 0; iCounter < 100; iCounter++)
    {
        if(model->quasiSteadyState)
            retval = IDACalcIC(mem, IDA_Y_INIT, 1e-5);
        else
            retval = IDACalcIC(mem, IDA_YA_YDP_INIT, 1e-5);
        if(retval < 0)
            daeThrowException( (boost::format("IDACalcIC failed: %s") % IDAGetReturnFlagName(retval)).str() );

        if(retval < 0)
        {
            std::string msg = (boost::format("Sundials IDAS solver cowardly failed to solve initial; %s\n")  % IDAGetReturnFlagName(retval)).str();
            daeThrowException(msg);
        }

        bool discontinuity_found = model->CheckForDiscontinuities(currentTime, yval, ypval);
        if(discontinuity_found)
        {
            model->ExecuteActions(currentTime, yval, ypval);
            noRoots = model->NumberOfRoots();
            RefreshRootFunctions(noRoots);
            ResetIDASolver(true, currentTime, false);
        }
        else
        {
            break;
        }
    }

    if(iCounter >= 100)
    {
        std::string msg = (boost::format("Sundials IDAS solver cowardly failed to solve initial: Max number of STN rebuilds reached; "
                                         "%s\n")  % IDAGetReturnFlagName(retval)).str();
        daeThrowException(msg);
    }

    /* Get the corrected IC and send them to the block */
    retval = IDAGetConsistentIC(mem, (N_Vector)yy, (N_Vector)yp);
    if(retval < 0)
        daeThrowException( (boost::format("IDAGetConsistentIC failed: %s") % IDAGetReturnFlagName(retval)).str() );

    /* Get the corrected sensitivity IC */
    /*
    if(m_bCalculateSensitivities)
        retval = IDAGetSensConsistentIC(mem, (N_Vector)yysens, (N_Vector)ypsens);
    */
}

void daeIDASolver_t::Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities)
{
    int retval, iCounter;

    printf("    Reinitializing at time: %f\n", currentTime);

    int noRoots = model->NumberOfRoots();
    RefreshRootFunctions(noRoots);
    ResetIDASolver(bCopyDataFromBlock, currentTime, bResetSensitivities);

    for(iCounter = 0; iCounter < 100; iCounter++)
    {
        /* Here we always use the IDA_YA_YDP_INIT flag (and discard InitialConditionMode).
         * The reason is that in this phase we may have been reinitialized the diff. variables
         * with the new values and using the eQuasiSteadyState flag would be meaningless.
         */
        retval = IDACalcIC(mem, IDA_YA_YDP_INIT, currentTime + 1e-5);
        if(retval < 0)
        {
            std::string msg = (boost::format("Sundials IDAS solver cowardly failed to re-initialize the system at time = %.15f; "
                                             "%s\n")  % currentTime % IDAGetReturnFlagName(retval)).str();
            daeThrowException(msg);
        }

        bool discontinuity_found = model->CheckForDiscontinuities(currentTime, yval, ypval);
        if(discontinuity_found)
        {
            model->ExecuteActions(currentTime, yval, ypval);
            noRoots = model->NumberOfRoots();
            RefreshRootFunctions(noRoots);
            ResetIDASolver(true, currentTime, false);
        }
        else
        {
            break;
        }
    }

    if(iCounter >= 100)
    {
        std::string msg = (boost::format("Sundials IDAS solver cowardly failed to re-initialize the system at time = %.15f: "
                                         "Max number of STN rebuilds reached; %s\n")  % currentTime % IDAGetReturnFlagName(retval)).str();
        daeThrowException(msg);
    }

    /* Get the corrected IC and send them to the block */
    retval = IDAGetConsistentIC(mem, (N_Vector)yy, (N_Vector)yp);
    if(retval < 0)
        daeThrowException("IDAGetConsistentIC failed");

   /* Get the corrected sensitivity IC */
    /*
    if(m_bCalculateSensitivities)
        retval = IDAGetSensConsistentIC(mem, (N_Vector)yysens, (N_Vector)ypsens);
    */
}

void daeIDASolver_t::CollectSolverStats()
{
    if(!mem)
        return;

    stats.clear();

    long int nst, nni, nre, nli, nreLS, nge, npe, nps, njvtimes;
    nst = nni = nre = nli = nreLS = nge = npe = nps = njvtimes = 0;

    //IDAGetNumSteps(mem, &nst);
    //IDAGetNumResEvals(mem, &nre);
    //IDAGetNumNonlinSolvIters(mem, &nni);
    IDASpilsGetNumLinIters(mem, &nli);
    IDASpilsGetNumResEvals(mem, &nreLS);
    IDASpilsGetNumPrecEvals(mem, &npe);
    IDASpilsGetNumPrecSolves(mem, &nps);
    IDASpilsGetNumJtimesEvals(mem, &njvtimes);
    stats["NumLinIters"]    = nli;
    stats["NumResEvals"]    = nreLS;
    stats["NumPrecEvals"]   = npe;
    stats["NumPrecSolves"]  = nps;
    stats["NumJtimesEvals"] = njvtimes;

    long int nsteps, nrevals, nlinsetups, netfails;
    int klast, kcur;
    realtype hinused, hlast, hcur, tcur;
    nsteps = nrevals = nlinsetups = netfails = 0;
    klast = kcur = 0;
    hinused = hlast = hcur = tcur = 0.0;

    IDAGetIntegratorStats(mem, &nsteps, &nrevals, &nlinsetups,
                               &netfails, &klast, &kcur, &hinused,
                               &hlast, &hcur, &tcur);
    stats["NumSteps"]         = nsteps;
    stats["NumResEvals"]      = nrevals;
    stats["NumLinSolvSetups"] = nlinsetups;
    stats["NumErrTestFails"]  = netfails;
    stats["LastOrder"]        = klast;
    stats["CurrentOrder"]     = kcur;
    stats["ActualInitStep"]   = hinused;
    stats["LastStep"]         = hlast;
    stats["CurrentStep"]      = hcur;
    stats["CurrentTime"]      = tcur;

    long int nniters, nncfails;
    nniters = nncfails = 0;
    IDAGetNonlinSolvStats(mem, &nniters, &nncfails);
    stats["NumNonlinSolvIters"]     = nniters;
    stats["NumNonlinSolvConvFails"] = nncfails;
}

void daeIDASolver_t::PrintSolverStats()
{
    if(stats.empty())
        CollectSolverStats();

    printf("DAE solver stats:\n");
    printf("    NumSteps                    = %15d\n",    (int)stats["NumSteps"]);
    printf("    NumResEvals                 = %15d\n",    (int)stats["NumResEvals"]);
    printf("    NumErrTestFails             = %15d\n",    (int)stats["NumErrTestFails"]);
    printf("    LastOrder                   = %15d\n",    (int)stats["LastOrder"]);
    printf("    CurrentOrder                = %15d\n",    (int)stats["CurrentOrder"]);
    printf("    LastStep                    = %15.12f\n", stats["LastStep"]);
    printf("    CurrentStep                 = %15.12f\n", stats["CurrentStep"]);

    printf("Nonlinear solver stats:\n");
    printf("    NumNonlinSolvIters          = %15d\n", (int)stats["NumNonlinSolvIters"]);
    printf("    NumNonlinSolvConvFails      = %15d\n", (int)stats["NumNonlinSolvConvFails"]);

    printf("Linear solver stats (Sundials spils):\n");
    printf("    NumLinIters                 = %15d\n", (int)stats["NumLinIters"]);
    printf("    NumResEvals                 = %15d\n", (int)stats["NumResEvals"]);
    printf("    NumPrecEvals                = %15d\n", (int)stats["NumPrecEvals"]);
    printf("    NumPrecSolves               = %15d\n", (int)stats["NumPrecSolves"]);
    printf("    NumJtimesEvals              = %15d\n", (int)stats["NumJtimesEvals"]);
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
    csDAEModel_t*     model      = (csDAEModel_t*)dae_solver->model;

    // Call MPI synchronise data every time before calculating residuals
    model->SetAndSynchroniseData(tres, yval, ypval);

    model->EvaluateResiduals(tres, res);

    return 0;
}

int setup_preconditioner(realtype tt,
                         N_Vector yy, N_Vector yp, N_Vector rr,
                         realtype cj, void *user_data,
                         N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.PSetup);

    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);
    realtype* res    = NV_DATA_P(rr);

    daeIDASolver_t*       dae_solver     = (daeIDASolver_t*)user_data;
    daePreconditioner_t*  preconditioner = dae_solver->preconditioner.get();

    //printf("    setup_preconditioner (time = %.15f)\n", time);
    return preconditioner->Setup(tt, cj, yval, ypval, res);
}

int solve_preconditioner(realtype tt,
                         N_Vector uu, N_Vector up, N_Vector rr,
                         N_Vector rvec, N_Vector zvec,
                         realtype c_j, realtype delta, void *user_data,
                         N_Vector tmp)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.PSolve);

    realtype* yval  = NV_DATA_P(uu);
    realtype* ypval = NV_DATA_P(up);
    realtype* res   = NV_DATA_P(rr);
    realtype* r     = NV_DATA_P(rvec);
    realtype* z     = NV_DATA_P(zvec);

    daeIDASolver_t*       dae_solver     = (daeIDASolver_t*)user_data;
    daePreconditioner_t*  preconditioner = dae_solver->preconditioner.get();

    //printf("    solve_preconditioner (time = %.15f)\n", tt);
    return preconditioner->Solve(tt, r, z);
}

int jacobian_vector_dq(realtype tt,
                       N_Vector yy,
                       N_Vector yp,
                       N_Vector rr,
                       N_Vector vvec,
                       N_Vector Jvvec,
                       realtype c_j,
                       void *user_data,
                       N_Vector tmp1,
                       N_Vector tmp2)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.JvtimesDQ);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    int ret = IDASpilsDQJtimes(tt,
                               yy,
                               yp,
                               rr,
                               vvec,
                               Jvvec,
                               c_j,
                               dae_solver->mem,
                               tmp1,
                               tmp2);
    return ret;
}

int jacobian_vector_multiply(realtype tt,
                             N_Vector yy,
                             N_Vector yp,
                             N_Vector rr,
                             N_Vector vvec,
                             N_Vector Jvvec,
                             realtype c_j,
                             void *user_data,
                             N_Vector tmp1,
                             N_Vector tmp2)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.Jvtimes);

    daeIDASolver_t*       dae_solver     = (daeIDASolver_t*)user_data;
    daePreconditioner_t*  preconditioner = dae_solver->preconditioner.get();

    realtype* v  = NV_DATA_P(vvec);
    realtype* Jv = NV_DATA_P(Jvvec);

    //printf("    jacobian_x_vector    (time = %.15f)\n", tt);
    return preconditioner->JacobianVectorMultiply(tt, v, Jv);
}

void save_step(FILE* f, double time, double* yval, double* ypval, double* res, int n)
{
    fprintf(f, "x    = [");
    for(size_t i = 0; i < n; i++)
        fprintf(f, "%.15f, ", yval[i]);
    fprintf(f, "]\n");
    fprintf(f, "dxdt = [");
    for(size_t i = 0; i < n; i++)
        fprintf(f, "%.15f, ", ypval[i]);
    fprintf(f, "]\n");
    fprintf(f, "res  = [");
    for(size_t i = 0; i < 30; i++)
        fprintf(f, "%.15f, ", res[i]);
    fprintf(f, "]\n");
    fprintf(f, "results[%.15f] = (x,xdot)\n\n", time);
    fflush(f);
}

int resid(realtype tres,
          N_Vector yy,
          N_Vector yp,
          N_Vector resval,
          void *user_data)
{
    realtype* yval  = NV_DATA_P(yy);
    realtype* ypval = NV_DATA_P(yp);
    realtype* res   = NV_DATA_P(resval);

    daeIDASolver_t* dae_solver = (daeIDASolver_t*)user_data;
    csDAEModel_t*     model      = (csDAEModel_t*)dae_solver->model;

/*
    IDAMem ida_mem       = (IDAMem)dae_solver->mem;
    IDASpilsMem ida_lmem = (IDASpilsMem)ida_mem->ida_lmem;

    printf("s_sqrtN     = %.15f\n", ida_lmem->s_sqrtN);
    printf("ida_epsNewt = %.15f\n", ida_mem->ida_epsNewt);
    printf("s_eplifac   = %.15f\n", ida_lmem->s_eplifac);
    printf("s_epslin    = %.15f\n", ida_lmem->s_epslin);
*/
    /* The values and timeDerivatives must be copied in mpiSynchroniseData function.
     * Call MPI synchronise data every time before calculating residuals. */
    model->SetAndSynchroniseData(tres, yval, ypval);

    /* Evaluate residuals. */
    model->EvaluateResiduals(tres, res);

    //printf("    EvaluateResiduals (time = %.15f)\n", tres);
    //for(size_t i = 0; i < 10/*model->Nequations_local*/; i++)
    //    printf("%.15f, ", res[i]);
    //printf("\n");

/*
    int currentOrder;
    realtype currentStep;
    IDAGetCurrentOrder(dae_solver->mem, &currentOrder);
    IDAGetCurrentStep(dae_solver->mem,  &currentStep);

    std::string filename = model->inputDirectory + (boost::format("/out-%05d.txt") % model->mpi_rank).str();
    FILE* f = fopen(filename.c_str(), "a");
    save_step(f, tres, yval, ypval, res, 30);
    fclose(f);

    printf("  residuals at t = %.14f\n", tres);
    printf("      currentOrder = %d\n", currentOrder);
    printf("      currentStep  = %.14f\n", currentStep);
    printf("      x    = [");
    for(size_t i = 0; i < 30; i++)
        printf("%.15f, ", yval[i]);
    printf("]\n");
    printf("      dxdt = [");
    for(size_t i = 0; i < 30; i++)
        printf("%.15f, ", ypval[i]);
    printf("]\n");
    printf("      res  = [");
    for(size_t i = 0; i < 30; i++)
        printf("%.15f, ", res[i]);
    printf("]\n");
*/

    return 0;
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
    csDAEModel_t* model    = (csDAEModel_t*)dae_solver->model;

    model->Roots(t, yval, ypval, gout);
    return 0;
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
    csDAEModel_t* model    = (csDAEModel_t*)dae_solver->model;

    return modJacobian(model, Neq, tt,  cj, yval, ypval, res, jacob);
}
*/

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

int check_flag(void *flagvalue, const char *funcname, int opt)
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

}
