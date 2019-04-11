/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include <mpi.h> // this is important to include first ("error: template with C linkage")
#include <fstream>
#include <math.h>
#include <iomanip>
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_spgmr.h>
#include <cvodes/cvodes_bbdpre.h>
#include <nvector/nvector_parallel.h>
#include <sundials/sundials_math.h>
#include "auxiliary.h"
#include "daesimulator.h"
#include "cvodes_la_functions.h"
#include <boost/format.hpp>
#include <cvodes/cvodes_impl.h>
#include <src/cvodes/cvodes_spils_impl.h>
using namespace cs;

namespace cs_dae_simulator
{
/* Private functions */
static int check_flag(void *flagvalue, const char *funcname, int opt);

static int setup_preconditioner(realtype tt,
                                N_Vector yy,
                                N_Vector rrhs,
                                booleantype jok,
                                booleantype *jcurPtr,
                                realtype gamma,
                                void *user_data,
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int solve_preconditioner(realtype tt,
                                N_Vector ry,
                                N_Vector rrhs,
                                N_Vector rvec,
                                N_Vector zvec,
                                realtype gamma,
                                realtype delta,
                                int lr,
                                void *user_data,
                                N_Vector tmp);

static int jacobian_vector_multiply(N_Vector v,
                                    N_Vector Jv,
                                    realtype tt,
                                    N_Vector yy,
                                    N_Vector fy,
                                    void *user_data,
                                    N_Vector tmp2);
static int jacobian_vector_dq(N_Vector vvec,
                              N_Vector Jvvec,
                              realtype tt,
                              N_Vector yy,
                              N_Vector fy,
                              void *user_data,
                              N_Vector tmp2);

static int rhs(realtype time,
                 N_Vector yy,
                 N_Vector yp,
                 void *user_data);

static int root(realtype t,
                N_Vector yy,
                realtype *gout,
                void *user_data);

// Timed functions.
static int cvode_spils_setup_la(CVodeMem cvode_mem, int convfail,
                                N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                                N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);
static int cvode_spils_solve_la(CVodeMem cvode_mem, N_Vector b, N_Vector weight,
                                N_Vector ycur, N_Vector fcur);
// Pointers to original functions.
int (*cv_lsetup)(CVodeMem   CVode_mem,int convfail,
                  N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                  N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3);
int (*cv_lsolve)(CVodeMem CVode_mem,N_Vector b, N_Vector weight,
                  N_Vector ycur, N_Vector fcur);


odeiSolver_t::odeiSolver_t()
{
    currentTime = 0.0;
    mem         = NULL;
    yy          = NULL;
    yp          = NULL;
    model       = NULL;
    simulation  = NULL;
    integrationMode          = CV_NORMAL;
    linearMultistepMethod    = CV_BDF;
    nonlinearSolverIteration = CV_NEWTON;
    linearSolverType         = eUnknownLinearSolver;
}

odeiSolver_t::~odeiSolver_t()
{
    Free();
}

void odeiSolver_t::Initialize(csDifferentialEquationModel_t* pmodel,
                              daeSimulation_t*               psimulation,
                              long                           Neq,
                              long                           Neq_local,
                              const real_t*                  initValues,
                              const real_t*                  initDerivatives,
                              const real_t*                  absTolerances,
                              const int*                     variableTypes)
{
    int i, retval;
    N_Vector yy_nv, yp_nv, avtol_nv;
    realtype t0;
    real_t* atval;

    model          = pmodel;
    simulation     = psimulation;
    preconditioner = NULL;
    lasolver       = NULL;
    Nequations     = Neq_local;

    MPI_Comm comm = MPI_COMM_WORLD;
    daeSimulationOptions& cfg = daeSimulationOptions::GetConfig();

    printInfo = cfg.GetBoolean("Solver.PrintInfo", false);
    rtol      = cfg.GetFloat("Solver.Parameters.RelativeTolerance", 1e-5);

    std::string integrationMode_s = cfg.GetString("Solver.Parameters.IntegrationMode", "Normal");
    if(integrationMode_s == "Normal")
        integrationMode = CV_NORMAL;
    else if(integrationMode_s == "OneStep")
        integrationMode = CV_ONE_STEP;
    else
        csThrowException("Invalid integration mode specified: " + integrationMode_s);

    std::string linearMultistepMethod_s = cfg.GetString("Solver.Parameters.LinearMultistepMethod", "BDF");
    if(linearMultistepMethod_s == "BDF")
        linearMultistepMethod = CV_BDF;
    else if(linearMultistepMethod_s == "Adams")
        linearMultistepMethod = CV_ADAMS;
    else
        csThrowException("Invalid linear multistep method specified: " + linearMultistepMethod_s);

    std::string nonlinearSolverIteration_s = cfg.GetString("Solver.Parameters.IterationType", "Newton");
    if(nonlinearSolverIteration_s == "Newton")
        nonlinearSolverIteration = CV_NEWTON;
    else if(nonlinearSolverIteration_s == "Functional")
        nonlinearSolverIteration = CV_FUNCTIONAL;
    else
        csThrowException("Invalid nonlinear solver iteration mode specified: " + nonlinearSolverIteration_s);

    /* Allocate N-vectors. */
    yy_nv = N_VNew_Parallel(comm, Neq_local, Neq);
    if(yy_nv == NULL)
        csThrowException("Allocation of yy array failed");
    yy = yy_nv;

    yp_nv = N_VNew_Parallel(comm, Neq_local, Neq);
    if(yp_nv == NULL)
        csThrowException("Allocation of yp array failed");
    yp = yp_nv;

    avtol_nv = N_VNew_Parallel(comm, Neq_local, Neq);
    if(avtol_nv == NULL)
        csThrowException("Allocation of avtol array failed");

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    yval = NV_DATA_P(yy_nv);
    for(i = 0; i < Neq_local; i++)
        yval[i] = initValues[i];

    /* Initial derivatives are not relevant for ODE systems - set them to zero. */
    ypval = NV_DATA_P(yp_nv);
    for(i = 0; i < Neq_local; i++)
        ypval[i] = 0.0;

    atval = NV_DATA_P(avtol_nv);
    for(i = 0; i < Neq_local; i++)
        atval[i] = absTolerances[i];

    /* Integration limits */
    t0 = 0;

    /* Call CVodeCreate to initialize IDA memory */
    mem = CVodeCreate(linearMultistepMethod, nonlinearSolverIteration);
    if(mem == NULL)
        csThrowException("CVodeCreate failed");

    retval = CVodeInit(mem, rhs, t0, yy_nv);
    if(retval < 0)
        csThrowException( (boost::format("CVodeInit failed: %s") % CVodeGetReturnFlagName(retval)).str() );

    retval = CVodeSVtolerances(mem, rtol, avtol_nv);
    if(retval < 0)
        csThrowException( (boost::format("CVodeSVtolerances failed: %s") % CVodeGetReturnFlagName(retval)).str() );

    /* Free avtol */
    N_VDestroy_Parallel(avtol_nv);

    /* Set maximum time. */
    CVodeSetStopTime(mem, simulation->timeHorizon);

    /* Solver options. */
    bool stldet = cfg.GetBoolean("Solver.Parameters.StabLimDet", false);
    CVodeSetStabLimDet(mem, (stldet ? TRUE : FALSE));

    realtype hin = cfg.GetFloat("Solver.Parameters.InitStep", 0.0);
    if(hin > 0.0)
        CVodeSetInitStep(mem, hin);

    realtype hmin = cfg.GetFloat("Solver.Parameters.MinStep", 0.0);
    if(hmin > 0.0)
        CVodeSetMinStep(mem, hmin);

    realtype hmax = cfg.GetFloat("Solver.Parameters.MaxStep", 0.0);
    if(hmax > 0.0)
        CVodeSetMaxStep(mem, hmax);

    CVodeSetMaxOrd(mem,            cfg.GetInteger("Solver.Parameters.MaxOrd",           5));
    CVodeSetMaxNumSteps(mem,       cfg.GetInteger("Solver.Parameters.MaxNumSteps",      500));
    CVodeSetMaxHnilWarns(mem,      cfg.GetFloat  ("Solver.Parameters.MaxHnilWarns",     10));
    CVodeSetMaxErrTestFails(mem,   cfg.GetInteger("Solver.Parameters.MaxErrTestFails",  7));
    CVodeSetMaxNonlinIters(mem,    cfg.GetInteger("Solver.Parameters.MaxNonlinIters",   3));
    CVodeSetMaxConvFails(mem,      cfg.GetInteger("Solver.Parameters.MaxConvFails",     10));
    CVodeSetNonlinConvCoef(mem,    cfg.GetFloat  ("Solver.Parameters.NonlinConvCoef",   0.1));
    bool bval = cfg.GetBoolean("Solver.Parameters.NoInactiveRootWarn", false);
    if(bval)
        CVodeSetNoInactiveRootWarn(mem);

    /* Not existing in CVode: Inital condition options. */
    //CVodeSetNonlinConvCoefIC(mem,  cfg.GetFloat  ("Solver.Parameters.NonlinConvCoefIC", 0.0033));
    //CVodeSetMaxNumStepsIC(mem,     cfg.GetInteger("Solver.Parameters.MaxNumStepsIC",    5));
    //CVodeSetMaxNumJacsIC(mem,      cfg.GetInteger("Solver.Parameters.MaxNumJacsIC",     4));
    //CVodeSetMaxNumItersIC(mem,     cfg.GetInteger("Solver.Parameters.MaxNumItersIC",    10));
    //CVodeSetLineSearchOffIC(mem,   cfg.GetBoolean("Solver.Parameters.LineSearchOffIC",  false));

    /* Set user data. */
    retval = CVodeSetUserData(mem, this);
    if(retval < 0)
        csThrowException( (boost::format("IDASetUserData failed: %s") % CVodeGetReturnFlagName(retval)).str() );

    /* Call IDARootInit to set the root function and number of roots. */
    int noRoots = model->NumberOfRoots();
    retval = CVodeRootInit(mem, noRoots, root);
    if(retval < 0)
        csThrowException( (boost::format("IDARootInit failed: %s") % CVodeGetReturnFlagName(retval)).str() );

    /* Set up the linear solver. */
    std::string lasolverLibrary = cfg.GetString("LinearSolver.Library");
    if(lasolverLibrary == "Sundials")
    {
        linearSolverType = eSundialsSpils;

        // Maximum dimension of the Krylov subspace to be used (if 0 use the default value MAXL = 5)
        int kspace = cfg.GetInteger("LinearSolver.Parameters.kspace", 0);
        int preconditioningType = PREC_RIGHT;
        std::string preconditioningType_s = cfg.GetString("LinearSolver.Parameters.preconditioningType", "left");
        if(preconditioningType_s == "left")
            preconditioningType = PREC_LEFT;
        else if(preconditioningType_s == "right")
            preconditioningType = PREC_RIGHT;
        else if(preconditioningType_s == "both")
            preconditioningType = PREC_BOTH;
        else if(preconditioningType_s == "none")
            preconditioningType = PREC_NONE;
        else
            csThrowException("Invalid preconditioning type specified: " + preconditioningType_s);

        retval = CVSpgmr(mem, preconditioningType, kspace);
        if(retval < 0)
            csThrowException( (boost::format("CVSpgmr failed: %s") % CVodeGetReturnFlagName(retval)).str() );

        std::string preconditionerName = cfg.GetString("LinearSolver.Preconditioner.Library", "Not specified");
        if(preconditionerName == "Ifpack")
            preconditioner.reset(new daePreconditioner_Ifpack);
        else if(preconditionerName == "ML")
            preconditioner.reset(new daePreconditioner_ML);
        else if(preconditionerName == "Jacobi")
            preconditioner.reset(new daePreconditioner_Jacobi);
        else
            csThrowException("Invalid preconditioner library specified: " + preconditionerName);

        bool isODESystem = true;
        retval = preconditioner->Initialize(pmodel, Neq_local, isODESystem);
        if(retval < 0)
            csThrowException( (boost::format("Preconditioner initialize failed: %s") % CVodeGetReturnFlagName(retval)).str() );

        CVSpilsSetEpsLin(mem, cfg.GetFloat("LinearSolver.Parameters.EpsLin", 0.05));
        std::string gstype = cfg.GetString("LinearSolver.Parameters.GSType", "MODIFIED_GS");
        if(gstype == "MODIFIED_GS")
            CVSpilsSetGSType(mem, MODIFIED_GS);
        else if(gstype == "CLASSICAL_GS")
            CVSpilsSetGSType(mem, CLASSICAL_GS);

        retval = CVSpilsSetPreconditioner(mem, setup_preconditioner, solve_preconditioner);
        if(retval < 0)
            csThrowException( (boost::format("IDASpilsSetPreconditioner failed: %s") % CVodeGetReturnFlagName(retval)).str() );

        // This is very important for overall performance!!
        // For some reasons works very slow if jacobian_vector_multiply is used for Npe > 1.
        std::string jtimes = cfg.GetString("LinearSolver.Parameters.JacTimesVecFn", "DifferenceQuotient");
        if(jtimes == "DifferenceQuotient")
        {
            // Do nothing here (it is the default).
        }
        else if(jtimes == "JacobianVectorMultiply")
        {
            CVSpilsSetJacTimesVecFn(mem, jacobian_vector_multiply);
        }
        else if(jtimes == "DifferenceQuotient_timed")
        {
            // jacobian_vector_dq calls default function: IDASpilsDQJtimes (but measures the time required).
            CVSpilsSetJacTimesVecFn(mem, jacobian_vector_dq);
        }
        else
        {
            csThrowException( (boost::format("Not supported LinearSolver.Parameters.JacTimesVecFn option: %s") % jtimes).str() );
        }

        // Replace lsolve/lsetup function with the timed versions.
        CVodeMem cv_mem = (CVodeMem)mem;
        // Save original CVodes function pointers.
        cv_lsetup = cv_mem->cv_lsetup;
        cv_lsolve = cv_mem->cv_lsolve;
        // Replace the function pointers with the timed ones.
        // They will call the original function with the same arguments.
        cv_mem->cv_lsetup = cvode_spils_setup_la;
        cv_mem->cv_lsolve = cvode_spils_solve_la;
    }
    else if(lasolverLibrary == "Trilinos")
    {
        linearSolverType = eThirdPartyLinearSolver;

        lasolver.reset(new daeLinearSolver_Trilinos);
        bool isODESystem = true;
        retval = lasolver->Initialize(pmodel, Neq_local, isODESystem);
        if(retval < 0)
            csThrowException( (boost::format("LA Solver initialize failed: %s") % CVodeGetReturnFlagName(retval)).str() );

        CVodeMem cv_mem = (CVodeMem)mem;

        cv_mem->cv_lmem       = lasolver.get();
        cv_mem->cv_linit	  = init_la_ode;
        cv_mem->cv_lsetup     = setup_la_ode;
        cv_mem->cv_lsolve     = solve_la_ode;
        cv_mem->cv_lfree      = free_la_ode;
        cv_mem->cv_forceSetup = TRUE;
    }
    else
    {
        csThrowException("Invalid LinearSolver.Library specified: " + lasolverLibrary);

        /*
        // Maximum dimension of the Krylov subspace to be used (if 0 use the default value MAXL = 5)
        int kspace = cfg.GetInteger("LinearSolver.Parameters.kspace", 0);
        retval = IDASpgmr(mem, kspace);
        if(retval < 0)
            csThrowException( (boost::format("IDASpgmr failed: %s") % CVodeGetReturnFlagName(retval)).str() );

        // Set up the BBD preconditioner.
        long int mudq      = 5; // Upper half-bandwidth to be used in the difference-quotient Jacobian approximation.
        long int mldq      = 5; // Lower half-bandwidth to be used in the difference-quotient Jacobian approximation.
        long int mukeep    = 2; // Upper half-bandwidth of the retained banded approximate Jacobian block
        long int mlkeep    = 2; // Lower half-bandwidth of the retained banded approximate Jacobian block
        realtype dq_rel_yy = 0.0; // The relative increment in components of y used in the difference quotient approximations.
                                  // The default is dq_rel_yy = sqrt(unit roundoff), which can be specified by passing dq_rel_yy = 0.0
        retval = IDABBDPrecInit(mem, Neq_local, mudq, mldq, mukeep, mlkeep, dq_rel_yy, bbd_local_fn, NULL);
        if(retval < 0)
            csThrowException( (boost::format("IDABBDPrecInit failed: %s") % CVodeGetReturnFlagName(retval)).str() );
        */
    }
}

void odeiSolver_t::Free()
{
    if(mem)
    {
        CollectSolverStats();

        CVodeFree(&mem);
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

void odeiSolver_t::RefreshRootFunctions(int noRoots)
{
    int retval = CVodeRootInit(mem, noRoots, root);
    if(retval < 0)
        csThrowException( (boost::format("IDARootInit failed: %s") % CVodeGetReturnFlagName(retval)).str() );
}

void odeiSolver_t::ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities)
{
    currentTime = dCurrentTime;
    int retval = CVodeReInit(mem, dCurrentTime, (N_Vector)yy);
    if(retval < 0)
        csThrowException( (boost::format("IDAReInit failed: %s") % CVodeGetReturnFlagName(retval)).str() );
}

real_t odeiSolver_t::Solve(real_t dTime, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities)
{
    int retval;
    csDiscontinuityType eDiscontinuityType;

    targetTime = dTime;

    for(;;)
    {
        retval = CVode(mem, targetTime, (N_Vector)yy, &currentTime, integrationMode);

        if(retval == CV_TOO_MUCH_WORK)
        {
            std::snprintf(msgBuffer, msgBufferSize, "Warning: IDAS solver error at TIME = %f [IDA_TOO_MUCH_WORK]\n", currentTime);
            simulation->log->Message(msgBuffer);
            std::snprintf(msgBuffer, msgBufferSize, "  Try to increase MaxNumSteps option\n");
            simulation->log->Message(msgBuffer);
            realtype tolsfac = 0;
            retval = CVodeGetTolScaleFactor(mem, &tolsfac);
            std::snprintf(msgBuffer, msgBufferSize, "  Suggested factor by which the user’s tolerances should be scaled is %f\n", tolsfac);
            simulation->log->Message(msgBuffer);
            continue;
        }
        else if(retval < 0)
        {
            std::string msg = (boost::format("Sundials IDAS solver cowardly failed to solve the system at time = %.15f; "
                                             "time horizon [%.15f]; %s\n") % currentTime % targetTime % CVodeGetReturnFlagName(retval)).str();
            csThrowException(msg);
        }

        /* If a root has been found, check if any of conditions are satisfied and do what is necessary */
        if(retval == CV_ROOT_RETURN)
        {
            bool discontinuity_found = model->CheckForDiscontinuities(currentTime, yval, ypval);
            if(discontinuity_found)
            {
                /* The data will be reported only if there is a discontinuity */
                if(bReportDataAroundDiscontinuities)
                    simulation->ReportData(currentTime);

                eDiscontinuityType = model->ExecuteActions(currentTime, yval, ypval);

                if(eDiscontinuityType == eModelDiscontinuity)
                {
                    Reinitialize(false, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simulation->ReportData(currentTime);

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return currentTime;
                }
                else if(eDiscontinuityType == eModelDiscontinuityWithDataChange)
                {
                    Reinitialize(true, false);

                    /* The data will be reported again ONLY if there was a discontinuity */
                    if(bReportDataAroundDiscontinuities)
                        simulation->ReportData(currentTime);

                    if(eCriterion == eStopAtModelDiscontinuity)
                        return currentTime;
                }
                else if(eDiscontinuityType == eGlobalDiscontinuity)
                {
                    csThrowException("Not supported discontinuity type: eGlobalDiscontinuity");
                }
            }
            else
            {
            }
        }

        /* Get dy/dt at the current time (used for data reporting only). */
        retval = CVodeGetDky(mem, currentTime, 1, (N_Vector)yp);
        if(retval != CV_SUCCESS)
            csThrowException("Erorr calling CVodeGetDky");

        if(printInfo && integrationMode == CV_ONE_STEP)
        {
            static long cum_psolves = 0;
            int kcur;
            realtype hcur;
            long psolves;

            CVodeGetCurrentOrder(mem, &kcur);
            CVodeGetCurrentStep(mem, &hcur);
            CVSpilsGetNumPrecSolves(mem, &psolves);

            printf("    t = %.15f, k = %d, h = %.15f, psolves = %ld\n", currentTime, kcur, hcur, psolves-cum_psolves);
            cum_psolves = psolves;
        }

        if(currentTime >= targetTime)
            break;
    }

    return currentTime;
}

void odeiSolver_t::SolveInitial()
{
}

void odeiSolver_t::Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities)
{
//    int retval, iCounter;

//    printf("    Reinitializing at time: %f\n", currentTime);

//    int noRoots = model->NumberOfRoots();
//    RefreshRootFunctions(noRoots);
//    ResetIDASolver(bCopyDataFromBlock, currentTime, bResetSensitivities);

//    for(iCounter = 0; iCounter < 100; iCounter++)
//    {
//        /* Here we always use the IDA_YA_YDP_INIT flag (and discard InitialConditionMode).
//         * The reason is that in this phase we may have been reinitialized the diff. variables
//         * with the new values and using the eQuasiSteadyState flag would be meaningless.
//         */
//        retval = IDACalcIC(mem, IDA_YA_YDP_INIT, currentTime + 1e-5);
//        if(retval < 0)
//        {
//            std::string msg = (boost::format("Sundials IDAS solver cowardly failed to re-initialize the system at time = %.15f; "
//                                             "%s\n")  % currentTime % CVodeGetReturnFlagName(retval)).str();
//            csThrowException(msg);
//        }

//        bool discontinuity_found = model->CheckForDiscontinuities(currentTime, yval, ypval);
//        if(discontinuity_found)
//        {
//            model->ExecuteActions(currentTime, yval, ypval);
//            noRoots = model->NumberOfRoots();
//            RefreshRootFunctions(noRoots);
//            ResetIDASolver(true, currentTime, false);
//        }
//        else
//        {
//            break;
//        }
//    }

//    if(iCounter >= 100)
//    {
//        std::string msg = (boost::format("Sundials IDAS solver cowardly failed to re-initialize the system at time = %.15f: "
//                                         "Max number of STN rebuilds reached; %s\n")  % currentTime % CVodeGetReturnFlagName(retval)).str();
//        csThrowException(msg);
//    }

//    /* Get the corrected IC and send them to the block */
//    retval = IDAGetConsistentIC(mem, (N_Vector)yy, (N_Vector)yp);
//    if(retval < 0)
//        csThrowException("IDAGetConsistentIC failed");

//   /* Get the corrected sensitivity IC */
//    /*
//    if(m_bCalculateSensitivities)
//        retval = IDAGetSensConsistentIC(mem, (N_Vector)yysens, (N_Vector)ypsens);
//    */
}

void odeiSolver_t::CollectSolverStats()
{
    if(!mem)
        return;

    stats.clear();

    long int nst, nni, nre, nli, nreLS, nge, npe, nps, njvtimes;
    nst = nni = nre = nli = nreLS = nge = npe = nps = njvtimes = 0;

    if(linearSolverType == eSundialsSpils)
    {
        CVSpilsGetNumLinIters(mem, &nli);
        CVSpilsGetNumRhsEvals(mem, &nreLS);
        CVSpilsGetNumPrecEvals(mem, &npe);
        CVSpilsGetNumPrecSolves(mem, &nps);
        CVSpilsGetNumJtimesEvals(mem, &njvtimes);
    }
    stats["NumLinIters"]      = nli;
    stats["NumEquationEvals"] = nreLS;
    stats["NumPrecEvals"]     = npe;
    stats["NumPrecSolves"]    = nps;
    stats["NumJtimesEvals"]   = njvtimes;

    long int nsteps, nrevals, nlinsetups, netfails;
    int klast, kcur;
    realtype hinused, hlast, hcur, tcur;
    nsteps = nrevals = nlinsetups = netfails = 0;
    klast = kcur = 0;
    hinused = hlast = hcur = tcur = 0.0;

    CVodeGetIntegratorStats(mem, &nsteps, &nrevals, &nlinsetups,
                                 &netfails, &klast, &kcur, &hinused,
                                 &hlast, &hcur, &tcur);
    stats["NumSteps"]         = nsteps;
    stats["NumEquationEvals"] = nrevals;
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
    CVodeGetNonlinSolvStats(mem, &nniters, &nncfails);
    stats["NumNonlinSolvIters"]     = nniters;
    stats["NumNonlinSolvConvFails"] = nncfails;
}

void odeiSolver_t::PrintSolverStats()
{
    if(stats.empty())
        CollectSolverStats();

    std::string message;

    std::snprintf(msgBuffer, msgBufferSize, "ODE solver stats:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumSteps                    = %15d\n",    (int)stats["NumSteps"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumRHSEvals                 = %15d\n",    (int)stats["NumEquationEvals"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumErrTestFails             = %15d\n",    (int)stats["NumErrTestFails"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    LastOrder                   = %15d\n",    (int)stats["LastOrder"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    CurrentOrder                = %15d\n",    (int)stats["CurrentOrder"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    LastStep                    = %15.12f\n", stats["LastStep"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    CurrentStep                 = %15.12f\n", stats["CurrentStep"]);
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "Nonlinear solver stats:\n");
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumNonlinSolvIters          = %15d\n", (int)stats["NumNonlinSolvIters"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumNonlinSolvConvFails      = %15d\n", (int)stats["NumNonlinSolvConvFails"]);
    message += msgBuffer;

    std::snprintf(msgBuffer, msgBufferSize, "Linear solver stats (Sundials spils):\n");
    message += msgBuffer;
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumLinIters                 = %15d\n", (int)stats["NumLinIters"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumRHSEvals                 = %15d\n", (int)stats["NumEquationEvals"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumPrecEvals                = %15d\n", (int)stats["NumPrecEvals"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumPrecSolves               = %15d\n", (int)stats["NumPrecSolves"]);
    message += msgBuffer;
    std::snprintf(msgBuffer, msgBufferSize, "    NumJtimesEvals              = %15d\n", (int)stats["NumJtimesEvals"]);
    message += msgBuffer;

    simulation->log->Message(message);
}

static int cvode_spils_setup_la(CVodeMem cvode_mem,int convfail,
                                N_Vector ypred, N_Vector fpred, booleantype *jcurPtr,
                                N_Vector vtemp1, N_Vector vtemp2, N_Vector vtemp3)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASetup);

    return cv_lsetup(cvode_mem, convfail,
                     ypred, fpred, jcurPtr,
                     vtemp1, vtemp2, vtemp3);
}

static int cvode_spils_solve_la(CVodeMem cvode_mem, N_Vector b, N_Vector weight,
                                N_Vector ycur, N_Vector fcur)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASolve);

    return cv_lsolve(cvode_mem, b, weight, ycur, fcur);
}

/* Private functions */
static int rhs(realtype time,
               N_Vector yy,
               N_Vector fy,
               void *user_data)
{
    realtype* yval  = NV_DATA_P(yy);
    realtype* ypval = NV_DATA_P(fy);

    odeiSolver_t*                   ode_solver = (odeiSolver_t*)user_data;
    csDifferentialEquationModel_t*  model      = (csDifferentialEquationModel_t*)ode_solver->model;

    /* The values and timeDerivatives must be copied in mpiSynchroniseData function.
     * Call MPI synchronise data every time before calculating residuals.
     * Important:
     *   timeDerivatives are not used by the RHS function.
     *   Should we communicate it among the processing elements?
     */
    model->SetAndSynchroniseData(time, yval, ypval);

    /* Evaluate RHS. */
    model->EvaluateEquations(time, ypval);

    return 0;
}

static int setup_preconditioner(realtype tt,
                                N_Vector yy,
                                N_Vector yp,
                                booleantype jok,
                                booleantype *jcurPtr,
                                realtype gamma,
                                void *user_data,
                                N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.PSetup);

    odeiSolver_t*        dae_solver     = (odeiSolver_t*)user_data;
    daePreconditioner_t* preconditioner = dae_solver->preconditioner.get();

    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NV_DATA_P(yp);
    realtype* res    = NULL;
    realtype  hcur;
    CVodeGetCurrentStep(dae_solver->mem, &hcur);
    realtype inverseTimeStep = 1.0/hcur;

    // Regardless of the value of 'jok' argument we always re-evaluate Jacobian
    // (more details in the preconditioner::Setup function).
    // Thus, always set jcurPtr to TRUE.
    *jcurPtr = TRUE;

    //printf("    setup_preconditioner (time = %.15f)\n", time);
    return preconditioner->Setup(tt, inverseTimeStep, yval, ypval, true, gamma);
}

static int solve_preconditioner(realtype tt,
                                N_Vector ry,
                                N_Vector ryp,
                                N_Vector rvec,
                                N_Vector zvec,
                                realtype gamma,
                                realtype delta,
                                int lr,
                                void *user_data,
                                N_Vector tmp)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.PSolve);

    // if lr == 1 -> left preconditioning
    // if lr == 2 -> right preconditioning
    realtype* yval  = NV_DATA_P(ry);
    realtype* ypval = NV_DATA_P(ryp);
    realtype* rhs   = NULL;
    realtype* r     = NV_DATA_P(rvec);
    realtype* z     = NV_DATA_P(zvec);

    odeiSolver_t*        dae_solver     = (odeiSolver_t*)user_data;
    daePreconditioner_t* preconditioner = dae_solver->preconditioner.get();

    //printf("    solve_preconditioner (time = %.15f)\n", tt);
    return preconditioner->Solve(tt, r, z);
}

static int jacobian_vector_dq(N_Vector vvec,
                              N_Vector Jvvec,
                              realtype tt,
                              N_Vector yy,
                              N_Vector fy,
                              void *user_data,
                              N_Vector tmp2)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.JvtimesDQ);

    odeiSolver_t* dae_solver = (odeiSolver_t*)user_data;
    int ret = CVSpilsDQJtimes(vvec,
                              Jvvec,
                              tt,
                              yy,
                              fy,
                              dae_solver->mem,
                              tmp2);
    return ret;
}

int jacobian_vector_multiply(N_Vector vvec,
                             N_Vector Jvvec,
                             realtype tt,
                             N_Vector yy,
                             N_Vector fy,
                             void *user_data,
                             N_Vector tmp2)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.Jvtimes);

    odeiSolver_t*         dae_solver     = (odeiSolver_t*)user_data;
    daePreconditioner_t*  preconditioner = dae_solver->preconditioner.get();

    realtype* v  = NV_DATA_P(vvec);
    realtype* Jv = NV_DATA_P(Jvvec);

    //printf("    jacobian_x_vector    (time = %.15f)\n", tt);
    return preconditioner->JacobianVectorMultiply(tt, v, Jv);
}

static void save_step(FILE* f, double time, double* yval, double* ypval, double* res, int n)
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

static int root(realtype t,
                N_Vector yy,
                realtype *gout,
                void *user_data)
{
    realtype* yval   = NV_DATA_P(yy);
    realtype* ypval  = NULL;

    odeiSolver_t*                   ode_solver = (odeiSolver_t*)user_data;
    csDifferentialEquationModel_t*  model      = (csDifferentialEquationModel_t*)ode_solver->model;

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

    odeiSolver_t*                   ode_solver = (odeiSolver_t*)user_data;
    csDifferentialEquationModel_t*  model      = (csDifferentialEquationModel_t*)ode_solver->model;

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



int init_la_ode(CVodeMem cv_mem)
{
    cs_dae_simulator::odeiSolver_t* pODESolver = (cs_dae_simulator::odeiSolver_t*)cv_mem->cv_user_data;
    cs_dae_simulator::daeLinearSolver_t* pLASolver = pODESolver->lasolver.get();
    if(!pLASolver)
        return CV_MEM_NULL;
    return CV_SUCCESS;
}

int setup_la_ode(CVodeMem cv_mem,
                 int convfail,
                 N_Vector ypred,
                 N_Vector fpred,
                 booleantype *jcurPtr,
                 N_Vector vtemp1,
                 N_Vector vtemp2,
                 N_Vector vtemp3)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASetup);

    cs_dae_simulator::odeiSolver_t* pODESolver = (cs_dae_simulator::odeiSolver_t*)cv_mem->cv_user_data;
    cs_dae_simulator::daeLinearSolver_t* pLASolver = pODESolver->lasolver.get();
    if(!pLASolver)
        return CV_MEM_NULL;

    realtype *pdValues, *pdTimeDerivatives;

    realtype tcur = 0;
    CVodeGetCurrentTime(cv_mem, &tcur);

    realtype hcur = 0;
    CVodeGetCurrentStep(cv_mem, &hcur);
    realtype inverseTimeStep = 1.0/hcur;

    realtype gamma = cv_mem->cv_gamma;

    pdValues			= NV_DATA_P(ypred);
    pdTimeDerivatives	= NV_DATA_P(fpred);

    // We always re-evaluate Jacobian, thus, always set jcurPtr to TRUE.
    *jcurPtr = TRUE;

    int ret = pLASolver->Setup(tcur,
                               inverseTimeStep,
                               gamma,
                               pdValues,
                               pdTimeDerivatives);

    if(ret < 0)
        return CV_LSETUP_FAIL;
    return CV_SUCCESS;
}

int solve_la_ode(CVodeMem cv_mem,
                 N_Vector b,
                 N_Vector weight,
                 N_Vector ycur,
                 N_Vector fcur)
{
    auxiliary::daeTimesAndCounters& tcs = auxiliary::daeTimesAndCounters::GetTimesAndCounters();
    auxiliary::TimerCounter tc(tcs.LASolve);

    cs_dae_simulator::odeiSolver_t* pODESolver = (cs_dae_simulator::odeiSolver_t*)cv_mem->cv_user_data;
    cs_dae_simulator::daeLinearSolver_t* pLASolver = pODESolver->lasolver.get();
    if(!pLASolver)
        return CV_MEM_NULL;

    realtype *pdValues, *pdTimeDerivatives, *pdWeight, *pdB;

    realtype tcur = 0;
    CVodeGetCurrentTime(cv_mem, &tcur);

    realtype hcur = 0;
    CVodeGetCurrentStep(cv_mem, &hcur);
    realtype inverseTimeStep = 1.0/hcur;

    realtype cjratio = 1.0; // applicable to DAE systems (must set it to 1)

    pdWeight			= NV_DATA_P(weight);
    pdB      			= NV_DATA_P(b);
    pdValues			= NV_DATA_P(ycur);
    pdTimeDerivatives	= NV_DATA_P(fcur);

    int ret = pLASolver->Solve(tcur,
                               inverseTimeStep,
                               cjratio,
                               pdB,
                               pdWeight,
                               pdValues,
                               pdTimeDerivatives);
    if(ret < 0)
        return CV_LSOLVE_FAIL;
    return CV_SUCCESS;
}

void free_la_ode(CVodeMem cv_mem)
{
    cs_dae_simulator::odeiSolver_t* pODESolver = (cs_dae_simulator::odeiSolver_t*)cv_mem->cv_user_data;
    cs_dae_simulator::daeLinearSolver_t* pLASolver = pODESolver->lasolver.get();
    if(!pLASolver)
        return;

    // It is the responsibility of the user to delete LA solver pointer!!
    cv_mem->cv_lmem = NULL;

    pLASolver->Free();
}



}
