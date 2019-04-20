#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "pardiso_sparse_la_solver.h"
#include <omp.h>

namespace daetools
{
namespace solver
{
daeLASolver_t* daeCreatePardisoSolver()
{
    return new daePardisoSolver;
}

daePardisoSolver::daePardisoSolver(void)
{
    int res = 0;
    m_pBlock = NULL;
    m_vecB   = NULL;

    for(size_t i = 0; i < 64; i++)
    {
        pt[i]    = 0;
        iparm[i] = 0;
        dparm[i] = 0;
    }

    maxfct      = 1; /* Maximum number of numerical factorizations. */
    mnum        = 1; /* Which factorization to use. */
    msglvl      = 0; /* Print statistical information in file */
    mtype       = 11;/* Real unsymmetric matrix */
    nrhs        = 1; /* Number of right hand sides. */
    solver_type = 0; /* Sparse direct solver */

    iparm[0] = 0; /* 0: Use defaults for all options */
    pardisoinit (pt,  &mtype, &solver_type, iparm, dparm, &res);

    if(res != 0)
    {
        daeDeclareException(exInvalidCall);
        if(res == -10 )
            e << "No license file found";
        else if(res == -11 )
            e << "License is expired";
        else if(res == -12 )
            e << "Wrong username or hostname";
        else
            e << "Unknown Pardiso error";
        throw e;
    }

    /* Numbers of processors, value of OMP_NUM_THREADS */
    char* omp_no_threads_s = getenv("OMP_NUM_THREADS");
    if(omp_no_threads_s != NULL)
    {
        int omp_no_threads = atoi(omp_no_threads_s);
        if(omp_no_threads > 0)
            omp_set_num_threads(omp_no_threads);
    }
}

daePardisoSolver::~daePardisoSolver(void)
{
    FreeMemory();
}

//int daePardisoSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
int daePardisoSolver::Create(size_t n,
                             size_t nnz,
                             daeBlockOfEquations_t* block)
{
    call_stats::TimerCounter tc(m_stats["Create"]);

    m_pBlock       = block;
    m_nNoEquations = n;

    InitializePardiso(nnz);

    if(!CheckData())
        return -1;

// Initialize sparse matrix
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();

    return 0;
}

int daePardisoSolver::Reinitialize(size_t nnz)
{
    call_stats::TimerCounter tc(m_stats["Reinitialize"]);

    int res, idum;
    real_t ddum;

// Memory release
    phase = -1;
    pardiso (pt,
             &maxfct,
             &mnum,
             &mtype,
             &phase,
             &m_nNoEquations,
             m_matJacobian.A,
             m_matJacobian.IA,
             m_matJacobian.JA,
             &idum,
             &nrhs,
             iparm,
             &msglvl,
             &ddum,
             &ddum,
             &res,
             dparm);

    ResetMatrix(nnz);
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();

    return 0;
}

int daePardisoSolver::SaveAsXPM(const std::string& strFileName)
{
    m_matJacobian.SaveMatrixAsXPM(strFileName);
    return 0;
}

int daePardisoSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
    m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
    return 0;
}

std::string daePardisoSolver::GetName(void) const
{
    return string("Pardiso");
}

void daePardisoSolver::ResetMatrix(size_t nnz)
{
    m_vecB = (real_t*)realloc(m_vecB, m_nNoEquations * sizeof(real_t));
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_FORTRAN_STYLE);
}

bool daePardisoSolver::CheckData() const
{
    if(m_nNoEquations && m_vecB)
        return true;
    else
        return false;
}

void daePardisoSolver::FreeMemory(void)
{
    m_matJacobian.Free();
    if(m_vecB)
        free(m_vecB);
    m_vecB = NULL;
}

void daePardisoSolver::InitializePardiso(size_t nnz)
{
    m_vecB = (real_t*)malloc(m_nNoEquations * sizeof(real_t));
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_FORTRAN_STYLE);
}

int daePardisoSolver::Init()
{
    return 0;
}

int daePardisoSolver::Setup(real_t  time,
                            real_t  inverseTimeStep,
                            real_t* pdValues,
                            real_t* pdTimeDerivatives,
                            real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Setup"]);

    int res, idum;
    real_t ddum;

    size_t Neq = m_nNoEquations;

    {
        call_stats::TimerCounter tcje(m_stats["Jacobian"]);

        m_arrValues.InitArray(Neq, pdValues);
        m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
        m_arrResiduals.InitArray(Neq, pdResiduals);

        m_matJacobian.ClearValues();

        m_pBlock->CalculateJacobian(time,
                                    inverseTimeStep,
                                    m_arrValues,
                                    m_arrResiduals,
                                    m_arrTimeDerivatives,
                                    m_matJacobian);
    }

    pardiso_chkmatrix(&mtype, &m_nNoEquations, m_matJacobian.A, m_matJacobian.IA, m_matJacobian.JA, &res);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Error in consistency of matrix (error code: " << res << ")\n";
        throw e;
    }

// Reordering and Symbolic Factorization.
// This step also allocates all memory
// that is necessary for the factorization.
    phase = 11;
    pardiso (pt,
             &maxfct,
             &mnum,
             &mtype,
             &phase,
             &m_nNoEquations,
             m_matJacobian.A,
             m_matJacobian.IA,
             m_matJacobian.JA,
             &idum,
             &nrhs,
             iparm,
             &msglvl,
             &ddum,
             &ddum,
             &res,
             dparm);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Analyse/Reorder PARDISO LA solver";
        throw e;
    }
    //std::cout << "Reordering completed"                           << std::endl;
    //std::cout << "Number of nonzeros in factors  = " << iparm[17] << std::endl;
    //std::cout << "Number of factorization GFLOPS = " << iparm[18] << std::endl;

// Numerical factorization
    phase = 22;
    pardiso (pt,
             &maxfct,
             &mnum,
             &mtype,
             &phase,
             &m_nNoEquations,
             m_matJacobian.A,
             m_matJacobian.IA,
             m_matJacobian.JA,
             &idum,
             &nrhs,
             iparm,
             &msglvl,
             &ddum,
             &ddum,
             &res,
             dparm);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Factor PARDISO LA solver";
        throw e;
    }

    return 0;
}

int daePardisoSolver::Solve(real_t  time,
                            real_t  inverseTimeStep,
                            real_t  cjratio,
                            real_t* pdB,
                            real_t* weight,
                            real_t* pdValues,
                            real_t* pdTimeDerivatives,
                            real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Solve"]);

    int res, idum;

    size_t Neq = m_nNoEquations;
    memcpy(m_vecB, pdB, Neq*sizeof(real_t));

// Solve
    phase = 33;
    iparm[7] = 2; /* Max numbers of iterative refinement steps. */
    pardiso (pt,
             &maxfct,
             &mnum,
             &mtype,
             &phase,
             &m_nNoEquations,
             m_matJacobian.A,
             m_matJacobian.IA,
             m_matJacobian.JA,
             &idum,
             &nrhs,
             iparm,
             &msglvl,
             m_vecB,
             pdB,
             &res,
             dparm);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Factor PARDISO LA solver";
        throw e;
    }

    if(cjratio != 1.0)
    {
        for(size_t i = 0; i < Neq; i++)
            pdB[i] *= 2.0 / (1.0 + cjratio);
    }

    return 0;
}

int daePardisoSolver::Free()
{
    real_t ddum;
    int res, idum;

// Memory release
// Here I am getting an error
// It seems I call Pardiso  AFTER  the linear solver object has been deleted!!!
    phase = -1;
    pardiso (pt,
             &maxfct,
             &mnum,
             &mtype,
             &phase,
             &m_nNoEquations,
             m_matJacobian.A,
             m_matJacobian.IA,
             m_matJacobian.JA,
             &idum,
             &nrhs,
             iparm,
             &msglvl,
             &ddum,
             &ddum,
             &res,
             dparm);

    return 0;
}

std::map<std::string, call_stats::TimeAndCount> daePardisoSolver::GetCallStats() const
{
    return m_stats;
}

void daePardisoSolver::SetOption_string(const std::string& strName, const std::string& Value)
{
}

void daePardisoSolver::SetOption_float(const std::string& strName, double Value)
{
}

void daePardisoSolver::SetOption_int(const std::string& strName, int Value)
{
}

void daePardisoSolver::SetOption_bool(const std::string& strName, bool Value)
{
}

std::string daePardisoSolver::GetOption_string(const std::string& strName)
{
    return "";
}

double daePardisoSolver::GetOption_float(const std::string& strName)
{
    return 0.0;
}

int daePardisoSolver::GetOption_int(const std::string& strName)
{
    return 0;
}

bool daePardisoSolver::GetOption_bool(const std::string& strName)
{
    return false;
}

}
}

