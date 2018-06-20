#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include "mkl_pardiso_sparse_la_solver.h"
#include "mkl.h"
#include <omp.h>
#include "../config.h"

namespace dae
{
namespace solver
{
daeLASolver_t* daeCreateIntelPardisoSolver(void)
{
    return new daeIntelPardisoSolver;
}

daeIntelPardisoSolver::daeIntelPardisoSolver(void)
{
    m_pBlock = NULL;
    m_vecB   = NULL;

    for(size_t i = 0; i < 64; i++)
    {
        pt[i]    = 0;
        iparm[i] = 0;
    }

    maxfct    = 1; /* Maximum number of numerical factorizations. */
    mnum      = 1; /* Which factorization to use. */
    msglvl    = 0; /* Print statistical information in file. */
    mtype     = 11;/* Real unsymmetric matrix. */
    nrhs      = 1; /* Number of right hand sides. */

    iparm[0] = 0; /* 0: Use defaults for all options. */
    pardisoinit(pt, &mtype, iparm);

    mkl_set_dynamic(0);

    /* Numbers of processors, value of MKL_NUM_THREADS or OMP_NUM_THREADS */
    daeConfig& cfg = daeConfig::GetConfig();
    char* mkl_no_threads = getenv("MKL_NUM_THREADS");
    char* omp_no_threads = getenv("OMP_NUM_THREADS");
    m_no_threads = 0;
    if(mkl_no_threads != NULL)
        m_no_threads = atoi(mkl_no_threads);
    else if(omp_no_threads != NULL)
        m_no_threads = atoi(omp_no_threads);
    else
        m_no_threads = cfg.GetInteger("daetools.intel_pardiso.numThreads", 0);
    if(m_no_threads <= 0)
        m_no_threads = omp_get_num_procs();
    printf("IntelPardiso numThreads = %d\n", m_no_threads);

    iparm[26] = 1; /* check the sparse matrix representation */

    if(typeid(real_t) == typeid(double))
        iparm[27] = 0; /* Double precision */
    else
        iparm[27] = 1; /* Single precision */

    iparm[34] = 1; /* Zero-based indexing: columns and rows indexing in arrays ia, ja, and perm starts from 0.  */
}

daeIntelPardisoSolver::~daeIntelPardisoSolver(void)
{
}

int daeIntelPardisoSolver::Create(size_t n,
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

int daeIntelPardisoSolver::Reinitialize(size_t nnz)
{
    call_stats::TimerCounter tc(m_stats["Reinitialize"]);

    _INTEGER_t res, idum;
    real_t ddum;

    if(!m_pBlock)
        return -1;

// Memory release
    phase = -1;
    PARDISO (pt,
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
             &res);

    ResetMatrix(nnz);
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();

    return 0;
}

int daeIntelPardisoSolver::SaveAsXPM(const std::string& strFileName)
{
    m_matJacobian.SaveMatrixAsXPM(strFileName);
    return 0;
}

int daeIntelPardisoSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
    m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
    return 0;
}

std::string daeIntelPardisoSolver::GetName(void) const
{
    return string("Intel Pardiso");
}

void daeIntelPardisoSolver::ResetMatrix(size_t nnz)
{
    m_vecB = (real_t*)realloc(m_vecB, m_nNoEquations * sizeof(real_t));
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
}

bool daeIntelPardisoSolver::CheckData() const
{
    if(m_nNoEquations && m_vecB)
        return true;
    else
        return false;
}

void daeIntelPardisoSolver::InitializePardiso(size_t nnz)
{
    m_vecB = (real_t*)malloc(m_nNoEquations * sizeof(real_t));
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
}

int daeIntelPardisoSolver::Init()
{
    return 0;
}

int daeIntelPardisoSolver::Setup(real_t  time,
                                 real_t  inverseTimeStep,
                                 real_t* pdValues,
                                 real_t* pdTimeDerivatives,
                                 real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Setup"]);

    _INTEGER_t res, idum;
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

// Reordering and Symbolic Factorization.
// This step also allocates all memory
// that is necessary for the factorization.
    mkl_set_num_threads(m_no_threads);

    phase = 11;
    PARDISO (pt,
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
             &res);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Analyse/Reorder MKL PARDISO LA solver";
        throw e;
    }
//	std::cout << "Reordering completed"                                  << std::endl;
//	std::cout << "Number of nonzeros in factors  = " << iparm[17] << std::endl;
//	std::cout << "Number of factorization MFLOPS = " << iparm[18] << std::endl;

// Numerical factorization
    phase = 22;
    PARDISO (pt,
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
             &res);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Factor MKL PARDISO LA solver";
        throw e;
    }

    return 0;
}

int daeIntelPardisoSolver::Solve(real_t  time,
                                 real_t  inverseTimeStep,
                                 real_t  cjratio,
                                 real_t* pdB,
                                 real_t* weight,
                                 real_t* pdValues,
                                 real_t* pdTimeDerivatives,
                                 real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Solve"]);

    _INTEGER_t res, idum;

    size_t Neq = m_nNoEquations;
    memcpy(m_vecB, pdB, Neq*sizeof(real_t));

// Solve
    mkl_set_num_threads(m_no_threads);

    phase = 33;
    iparm[7] = 2; /* Max numbers of iterative refinement steps. */
    PARDISO (pt,
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
             &res);
    if(res != 0)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to Factor MKL PARDISO LA solver";
        throw e;
    }

    if(cjratio != 1.0)
    {
        for(size_t i = 0; i < Neq; i++)
            pdB[i] *= 2.0 / (1.0 + cjratio);
    }

    return 0;
}

int daeIntelPardisoSolver::Free()
{
    real_t ddum;
    _INTEGER_t res, idum;

// Memory release
// Here I am getting an error
// It seems I call Pardiso  AFTER  the linear solver object has been deleted!!!
    phase = -1;
    PARDISO (pt,
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
             &res);

    m_matJacobian.Free();
    if(m_vecB)
        free(m_vecB);
    m_vecB = NULL;

    return 0;
}

std::map<std::string, call_stats::TimeAndCount> daeIntelPardisoSolver::GetCallStats() const
{
    return m_stats;
}

void daeIntelPardisoSolver::SetOption_string(const std::string& strName, const std::string& Value)
{
}

void daeIntelPardisoSolver::SetOption_float(const std::string& strName, double Value)
{
}

void daeIntelPardisoSolver::SetOption_int(const std::string& strName, int Value)
{
}

void daeIntelPardisoSolver::SetOption_bool(const std::string& strName, bool Value)
{
}

std::string daeIntelPardisoSolver::GetOption_string(const std::string& strName)
{
    return "";
}

double daeIntelPardisoSolver::GetOption_float(const std::string& strName)
{
    return 0.0;
}

int daeIntelPardisoSolver::GetOption_int(const std::string& strName)
{
    return 0;
}

bool daeIntelPardisoSolver::GetOption_bool(const std::string& strName)
{
    return false;
}

}
}

