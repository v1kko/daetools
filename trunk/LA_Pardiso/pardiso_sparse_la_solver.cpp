#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "pardiso_sparse_la_solver.h"

namespace dae
{
namespace solver
{
int init_la(IDAMem ida_mem);
int setup_la(IDAMem ida_mem,
              N_Vector	vectorVariables,
              N_Vector	vectorTimeDerivatives,
              N_Vector	vectorResiduals,
              N_Vector	vectorTemp1,
              N_Vector	vectorTemp2,
              N_Vector	vectorTemp3);
int solve_la(IDAMem ida_mem,
              N_Vector	b,
              N_Vector	weight,
              N_Vector	vectorVariables,
              N_Vector	vectorTimeDerivatives,
              N_Vector	vectorResiduals);
int free_la(IDAMem ida_mem);

daeIDALASolver_t* daeCreatePardisoSolver(void)
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
    char* omp_no_threads = getenv("OMP_NUM_THREADS");
    if(omp_no_threads != NULL)
        iparm[2] = atoi(omp_no_threads);
    else
        iparm[2] = 1;
}

daePardisoSolver::~daePardisoSolver(void)
{
    FreeMemory();
}

int daePardisoSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
{
    IDAMem ida_mem = (IDAMem)ida;
    if(!ida_mem)
        return IDA_MEM_NULL;
    if(!ida_mem || !pDAESolver)
        return IDA_ILL_INPUT;

    m_pBlock = pDAESolver->GetBlock();
    if(!m_pBlock)
        return IDA_MEM_NULL;

    int nnz = 0;
    m_pBlock->CalcNonZeroElements(nnz);
    if(nnz == 0)
        return IDA_ILL_INPUT;

    m_nNoEquations	= n;
    m_pDAESolver	= pDAESolver;

    InitializePardiso(nnz);

    if(!CheckData())
        return IDA_MEM_NULL;

// Initialize sparse matrix
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();

    ida_mem->ida_linit	= init_la;
    ida_mem->ida_lsetup = setup_la;
    ida_mem->ida_lsolve = solve_la;
    ida_mem->ida_lperf	= NULL;
    ida_mem->ida_lfree	= free_la;

    ida_mem->ida_lmem         = this;
    ida_mem->ida_setupNonNull = TRUE;

    return IDA_SUCCESS;
}

int daePardisoSolver::Reinitialize(void* ida)
{
    int res, idum;
    real_t ddum;

    IDAMem ida_mem = (IDAMem)ida;
    if(!ida_mem)
        return IDA_MEM_NULL;
    if(!m_pBlock)
        return IDA_MEM_NULL;

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

    int n   = m_nNoEquations;
    int nnz = 0;
    m_pBlock->CalcNonZeroElements(nnz);
    ResetMatrix(nnz);
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();
    //m_matJacobian.Print();

    return IDA_SUCCESS;
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
    m_nJacobianEvaluations	= 0;
    m_vecB					= (real_t*)malloc(m_nNoEquations * sizeof(real_t));

    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_FORTRAN_STYLE);
}

int daePardisoSolver::Init(void* ida)
{
    return IDA_SUCCESS;
}

int daePardisoSolver::Setup(void*		ida,
                            N_Vector	vectorVariables,
                            N_Vector	vectorTimeDerivatives,
                            N_Vector	vectorResiduals,
                            N_Vector	vectorTemp1,
                            N_Vector	vectorTemp2,
                            N_Vector	vectorTemp3)
{
    int res, idum;
    real_t ddum;
    realtype *pdValues, *pdTimeDerivatives, *pdResiduals;

    IDAMem ida_mem = (IDAMem)ida;
    if(!ida_mem)
        return IDA_MEM_NULL;
    if(!m_pBlock)
        return IDA_MEM_NULL;

    size_t Neq              = m_nNoEquations;
    real_t time             = ida_mem->ida_tn;
    real_t dInverseTimeStep = ida_mem->ida_cj;

    pdValues			= NV_DATA_S(vectorVariables);
    pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives);
    pdResiduals			= NV_DATA_S(vectorResiduals);

    m_nJacobianEvaluations++;

    m_arrValues.InitArray(Neq, pdValues);
    m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
    m_arrResiduals.InitArray(Neq, pdResiduals);

    m_matJacobian.ClearValues();

    m_pBlock->CalculateJacobian(time,
                                m_arrValues,
                                m_arrResiduals,
                                m_arrTimeDerivatives,
                                m_matJacobian,
                                dInverseTimeStep);

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

    return IDA_SUCCESS;
}

int daePardisoSolver::Solve(void*		ida,
                            N_Vector	vectorB,
                            N_Vector	vectorWeight,
                            N_Vector	vectorVariables,
                            N_Vector	vectorTimeDerivatives,
                            N_Vector	vectorResiduals)
{
    int res, idum;
    realtype* pdB;

    IDAMem ida_mem = (IDAMem)ida;
    if(!ida_mem)
        return IDA_MEM_NULL;
    if(!m_pBlock)
        return IDA_MEM_NULL;

    size_t Neq = m_nNoEquations;
    pdB        = NV_DATA_S(vectorB);


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

    if(ida_mem->ida_cjratio != 1.0)
    {
        for(size_t i = 0; i < Neq; i++)
            pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
        //N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
    }

    return IDA_SUCCESS;
}

int daePardisoSolver::Free(void* ida)
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

    return IDA_SUCCESS;
}


int init_la(IDAMem ida_mem)
{
    daePardisoSolver* pSolver = (daePardisoSolver*)ida_mem->ida_lmem;
    if(!pSolver)
        return IDA_MEM_NULL;

    return pSolver->Init(ida_mem);
}

int setup_la(IDAMem	ida_mem,
             N_Vector	vectorVariables,
             N_Vector	vectorTimeDerivatives,
             N_Vector	vectorResiduals,
             N_Vector	vectorTemp1,
             N_Vector	vectorTemp2,
             N_Vector	vectorTemp3)
{
    daePardisoSolver* pSolver = (daePardisoSolver*)ida_mem->ida_lmem;
    if(!pSolver)
        return IDA_MEM_NULL;

    return pSolver->Setup(ida_mem,
                          vectorVariables,
                          vectorTimeDerivatives,
                          vectorResiduals,
                          vectorTemp1,
                          vectorTemp2,
                          vectorTemp3);

}

int solve_la(IDAMem	ida_mem,
             N_Vector	vectorB,
             N_Vector	vectorWeight,
             N_Vector	vectorVariables,
             N_Vector	vectorTimeDerivatives,
             N_Vector	vectorResiduals)
{
    daePardisoSolver* pSolver = (daePardisoSolver*)ida_mem->ida_lmem;
    if(!pSolver)
        return IDA_MEM_NULL;

    return pSolver->Solve(ida_mem,
                          vectorB,
                          vectorWeight,
                          vectorVariables,
                          vectorTimeDerivatives,
                          vectorResiduals);
}

int free_la(IDAMem ida_mem)
{
    daePardisoSolver* pSolver = (daePardisoSolver*)ida_mem->ida_lmem;
    if(!pSolver)
        return IDA_MEM_NULL;

    int ret = pSolver->Free(ida_mem);

// ACHTUNG, ACHTUNG!!
// It is the responsibility of the user to delete LA solver pointer!!
//    delete pSolver;

    ida_mem->ida_lmem = NULL;

    return ret;
}


}
}

