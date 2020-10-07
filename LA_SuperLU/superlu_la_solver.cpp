#include "stdafx.h"
#include "superlu_la_solver.h"
#include <stdlib.h>
#include "config_data.h"
#include <omp.h>

namespace daetools
{
namespace solver
{
#ifdef daeSuperLU
daeLASolver_t* daeCreateSuperLUSolver(void)
{
    return new superlu::daeSuperLUSolver;
}
#endif

#ifdef daeSuperLU_MT
daeLASolver_t* daeCreateSuperLU_MTSolver(void)
{
    return new superlu_mt::daeSuperLUSolver;
}
#endif

#ifdef daeSuperLU
namespace superlu
{
#endif
#ifdef daeSuperLU_MT
namespace superlu_mt
{
#endif

daeSuperLUSolver::daeSuperLUSolver(void)
{
    m_pBlock			  = NULL;
    m_vecB				  = NULL;
    m_vecX				  = NULL;
    m_bFactorizationDone  = false;
    m_solve				  = 0;
    m_factorize			  = 0;

// The user shoud be able to set parameters right after the construction of the solver
#ifdef daeSuperLU_MT
    m_perm_c	= NULL;
    m_perm_r	= NULL;
    m_work		= NULL;
    m_lwork		= 0;

    memset(&m_matA,     0, sizeof(SuperMatrix));
    memset(&m_matB,     0, sizeof(SuperMatrix));
    memset(&m_matX,     0, sizeof(SuperMatrix));
    memset(&m_matL,     0, sizeof(SuperMatrix));
    memset(&m_matU,     0, sizeof(SuperMatrix));
    memset(&m_matAC,    0, sizeof(SuperMatrix));
    memset(&m_Options,  0, sizeof(superlumt_options_t));
    memset(&m_Stats,    0, sizeof(Gstat_t));

    m_omp_num_threads = get_numthreads_from_config();
    m_Options.nprocs			= m_omp_num_threads;
    m_Options.fact				= EQUILIBRATE;
    m_Options.trans				= NOTRANS;
    m_Options.refact			= NO;
    m_Options.panel_size		= sp_ienv(1);
    m_Options.relax				= sp_ienv(2);
    m_Options.diag_pivot_thresh = 1.0;
    m_Options.drop_tol			= 0.0;
    m_Options.ColPerm			= COLAMD;
    m_Options.usepr				= NO;
    m_Options.SymmetricMode		= NO;
    m_Options.PrintStat			= NO;
    m_Options.perm_c			= NULL;
    m_Options.perm_r			= NULL;
    m_Options.work				= NULL;
    m_Options.lwork				= 0;
#endif

#ifdef daeSuperLU
// Some issues with daeConfig and cross compiling using mingw...
    string strReuse;
    get_from_config(m_bUseUserSuppliedWorkSpace, m_dWorkspaceMemoryIncrement, m_dWorkspaceSizeMultiplier, strReuse);

    if(strReuse == string("SamePattern_SameRowPerm"))
        m_iFactorization = SamePattern_SameRowPerm;
    else
        m_iFactorization = SamePattern;

    m_etree		= NULL;
    m_R			= NULL;
    m_C			= NULL;

    m_lwork		= 0;
    m_work		= NULL;
    m_perm_c	= NULL;
    m_perm_r	= NULL;

    memset(&m_matA,  0, sizeof(SuperMatrix));
    memset(&m_matB,  0, sizeof(SuperMatrix));
    memset(&m_matX,  0, sizeof(SuperMatrix));
    memset(&m_matL,  0, sizeof(SuperMatrix));
    memset(&m_matU,  0, sizeof(SuperMatrix));
    memset(&m_matAC, 0, sizeof(SuperMatrix));

    memset(&m_Options,  0, sizeof(superlu_options_t));
    memset(&m_memUsage, 0, sizeof(mem_usage_t));
    memset(&m_Stats,    0, sizeof(SuperLUStat_t));
    memset(&m_Glu,      0, sizeof(GlobalLU_t));

    set_default_options(&m_Options);
//    printf(".. options:\n");
//    printf("\tFact\t %8d\n", m_Options.Fact);
//    printf("\tEquil\t %8d\n", m_Options.Equil);
//    printf("\tColPerm\t %8d\n", m_Options.ColPerm);
//    printf("\tDiagPivotThresh %8.4f\n", m_Options.DiagPivotThresh);
//    printf("\tTrans\t %8d\n", m_Options.Trans);
//    printf("\tIterRefine\t%4d\n", m_Options.IterRefine);
//    printf("\tSymmetricMode\t%4d\n", m_Options.SymmetricMode);
//    printf("\tPivotGrowth\t%4d\n", m_Options.PivotGrowth);
//    printf("\tConditionNumber\t%4d\n", m_Options.ConditionNumber);
//    printf("..\n");
#endif
}

daeSuperLUSolver::~daeSuperLUSolver(void)
{
    FreeMemory();
}

int daeSuperLUSolver::Create(size_t n, size_t nnz, daeBlockOfEquations_t* block)
{
    call_stats::TimerCounter tc(m_stats["Create"]);

    if(!block)
        return -1;
    m_nNoEquations = n;
    m_pBlock       = block;

    InitializeSuperLU(nnz);

    return 0;
}

int daeSuperLUSolver::Reinitialize(size_t nnz)
{
    call_stats::TimerCounter tc(m_stats["Reinitialize"]);

    if(!m_pBlock)
        return -1;
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();

#ifdef daeSuperLU_MT
    pxgstrf_finalize(&m_Options, &m_matAC);

    if(m_bFactorizationDone)
    {
        if(m_lwork > 0)
        {
            daeDeclareException(exMiscellanous);
            e << "daeSuperLU_MT: unsupported case: m_lwork > 0";
            throw e;
        }
        else
        {
            Destroy_SuperNode_SCP(&m_matL);
            Destroy_CompCol_NCP(&m_matU);
        }
    }
    StatFree(&m_Stats);

// We deliberately create SLU_NC matrix (although it is SLU_NR), and then ask superlu to solve a transposed system
// This is in agreement with what the SuperLU documentation says about compressed row storage matrices
    m_Options.trans	= TRANS;
    Destroy_SuperMatrix_Store(&m_matA);
    dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NC, SLU_D, SLU_GE);
#endif

#ifdef daeSuperLU_CUDA
    cudaError_t ce = m_superlu_mt_gpuSolver.Reinitialize(m_matJacobian.NNZ, m_matJacobian.N, m_matJacobian.IA, m_matJacobian.JA);
    if(ce != cudaSuccess)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to reinitialize superlu_mt_gpu solver" << cudaGetErrorString(ce);
        throw e;
    }
#endif

#ifdef daeSuperLU
    if(m_bFactorizationDone)
    {
        Destroy_CompCol_Permuted(&m_matAC);

    // If memory was pre-allocated in m_work then destroy only SuperMatrix_Store
    // Otherwise destroy the whole matrix
        if(m_lwork > 0)
        {
            Destroy_SuperMatrix_Store(&m_matL);
            Destroy_SuperMatrix_Store(&m_matU);
        }
        else
        {
            Destroy_SuperNode_Matrix(&m_matL);
            Destroy_CompCol_Matrix(&m_matU);
        }
    }

// We deliberately create SLU_NC matrix (although it is SLU_NR), and then ask superlu to solve a transposed system
// This is in agreement with what the SuperLU documentation says about compressed row storage matrices
    m_Options.Trans	= TRANS;
    Destroy_SuperMatrix_Store(&m_matA);
    dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NC, SLU_D, SLU_GE);
#endif

    m_bFactorizationDone = false;

    return 0;
}

void daeSuperLUSolver::FreeMemory(void)
{
#ifdef daeSuperLU_MT
    if(m_bFactorizationDone)
    {
        pxgstrf_finalize(&m_Options, &m_matAC);

        if(m_lwork > 0)
        {
            daeDeclareException(exMiscellanous);
            e << "daeSuperLU_MT: unsupported case: m_lwork > 0";
            throw e;
        }
        else
        {
            Destroy_SuperNode_SCP(&m_matL);
            Destroy_CompCol_NCP(&m_matU);
        }
    }
    StatFree(&m_Stats);

    if(m_work && m_lwork > 0)
        free(m_work);
    m_work	= NULL;
    m_lwork	= 0;

    if(m_vecB)
        SUPERLU_FREE(m_vecB);
    if(m_vecX)
        SUPERLU_FREE(m_vecX);
    if(m_perm_c)
        SUPERLU_FREE(m_perm_c);
    if(m_perm_r)
        SUPERLU_FREE(m_perm_r);

    m_matJacobian.Free();
    Destroy_SuperMatrix_Store(&m_matA);
    Destroy_SuperMatrix_Store(&m_matB);
    Destroy_SuperMatrix_Store(&m_matX);
#endif

#ifdef daeSuperLU
    if(m_etree)
        SUPERLU_FREE(m_etree);
    if(m_R)
        SUPERLU_FREE(m_R);
    if(m_C)
        SUPERLU_FREE(m_C);

    if(m_bFactorizationDone)
    {
        Destroy_CompCol_Permuted(&m_matAC);

    // If memory was pre-allocated in m_work then destroy only SuperMatrix_Store
    // Otherwise destroy the whole matrix
        if(m_lwork > 0)
        {
            Destroy_SuperMatrix_Store(&m_matL);
            Destroy_SuperMatrix_Store(&m_matU);
        }
        else
        {
            Destroy_SuperNode_Matrix(&m_matL);
            Destroy_CompCol_Matrix(&m_matU);
        }
    }

    if(m_work && m_lwork > 0)
        free(m_work);
    m_work	= NULL;
    m_lwork	= 0;

    if(m_vecB)
        SUPERLU_FREE(m_vecB);
    if(m_vecX)
        SUPERLU_FREE(m_vecX);
    if(m_perm_c)
        SUPERLU_FREE(m_perm_c);
    if(m_perm_r)
        SUPERLU_FREE(m_perm_r);

    m_matJacobian.Free();
    Destroy_SuperMatrix_Store(&m_matA);
    Destroy_SuperMatrix_Store(&m_matB);
    Destroy_SuperMatrix_Store(&m_matX);
#endif
}

void daeSuperLUSolver::InitializeSuperLU(size_t nnz)
{
    if(!m_pBlock)
    {
        daeDeclareException(exInvalidPointer);
        e << "Invalid block pointer";
        throw e;
    }

// Initialize sparse matrix
    m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
    m_matJacobian.ResetCounters();
    m_pBlock->FillSparseMatrix(&m_matJacobian);
    m_matJacobian.Sort();

#ifdef daeSuperLU_MT
    m_vecB				= doubleMalloc(m_nNoEquations);
    m_vecX				= doubleMalloc(m_nNoEquations);
    m_perm_c			= intMalloc(m_nNoEquations);
    m_perm_r			= intMalloc(m_nNoEquations);

    m_Options.perm_c	= m_perm_c;
    m_Options.perm_r	= m_perm_r;
    m_Options.work		= NULL;
    m_Options.lwork		= 0;

    if(!m_vecB || !m_vecX || !m_perm_c || !m_perm_r)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to allocate memory for B, X, perm_c, perm_r vectors";
        throw e;
    }

// We deliberately create SLU_NC matrix (although it is SLU_NR), and then ask superlu to solve a transposed system
// This is in agreement with what the SuperLU documentation says about compressed row storage matrices
    m_Options.trans	= TRANS;
    dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NC, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&m_matB, m_nNoEquations, 1, m_vecB, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&m_matX, m_nNoEquations, 1, m_vecX, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
#endif

#ifdef daeSuperLU
    m_vecB	 = doubleMalloc(m_nNoEquations);
    m_vecX	 = doubleMalloc(m_nNoEquations);
    m_perm_c = intMalloc(m_nNoEquations);
    m_perm_r = intMalloc(m_nNoEquations);

    m_etree	 = intMalloc(m_nNoEquations);
    m_R		 = doubleMalloc(m_nNoEquations);
    m_C		 = doubleMalloc(m_nNoEquations);

    if(!m_vecB || !m_vecX || !m_perm_c || !m_perm_r || !m_R || !m_C || !m_etree)
    {
        daeDeclareException(exMiscellanous);
        e << "Unable to allocate memory for B, X, perm_c, perm_r, etree, R, C vectors";
        throw e;
    }

// We deliberately create SLU_NC matrix (although it is SLU_NR), and then ask superlu to solve a transposed system
// This is in agreement with what the SuperLU documentation says about compressed row storage matrices
    m_Options.Trans	= TRANS;
    dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NC, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&m_matB, m_nNoEquations, 1, m_vecB, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&m_matX, m_nNoEquations, 1, m_vecX, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
#endif

    m_bFactorizationDone = false;
}

int daeSuperLUSolver::SaveAsXPM(const std::string& strFileName)
{
    m_matJacobian.SaveMatrixAsXPM(strFileName);
    return 0;
}

int daeSuperLUSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
    m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
    return 0;
}

std::string daeSuperLUSolver::GetName(void) const
{
#ifdef daeSuperLU_MT
    return string("SuperLU_MT");
#endif
#ifdef daeSuperLU
    return string("SuperLU");
#endif
}

#ifdef daeSuperLU_MT
superlumt_options_t& daeSuperLUSolver::GetOptions(void)
{
    return m_Options;
}
#endif

#ifdef daeSuperLU
superlu_options_t& daeSuperLUSolver::GetOptions(void)
{
    return m_Options;
}
#endif

int daeSuperLUSolver::Init()
{
    return 0;
}

int daeSuperLUSolver::Free()
{
    return 0;
}

std::map<std::string, call_stats::TimeAndCount> daeSuperLUSolver::GetCallStats() const
{
    return m_stats;
}

void daeSuperLUSolver::SetOption_string(const std::string& strName, const std::string& Value)
{
}

void daeSuperLUSolver::SetOption_float(const std::string& strName, double Value)
{
}

void daeSuperLUSolver::SetOption_int(const std::string& strName, int Value)
{
}

void daeSuperLUSolver::SetOption_bool(const std::string& strName, bool Value)
{
}

std::string daeSuperLUSolver::GetOption_string(const std::string& strName)
{
    return "";
}

double daeSuperLUSolver::GetOption_float(const std::string& strName)
{
    return 0.0;
}

int daeSuperLUSolver::GetOption_int(const std::string& strName)
{
    return 0;
}

bool daeSuperLUSolver::GetOption_bool(const std::string& strName)
{
    return false;
}

int daeSuperLUSolver::Setup(real_t  time,
                            real_t  inverseTimeStep,
                            real_t* pdValues,
                            real_t* pdTimeDerivatives,
                            real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Setup"]);

    int info;
    int memSize;
    double rpg, rcond;

    if(!m_pBlock)
        return -1;

    size_t Neq = m_nNoEquations;

    m_arrValues.InitArray(m_nNoEquations, pdValues);
    m_arrTimeDerivatives.InitArray(m_nNoEquations, pdTimeDerivatives);
    m_arrResiduals.InitArray(m_nNoEquations, pdResiduals);

/* ACHTUNG!
   Do we need to clear the Jacobian matrix? It is sparse and all non-zero items will be overwritten.
   13.02.2012: YES we have for equations set only jacobian items that the equation depends on!
*/
    m_matJacobian.ClearValues();

    m_pBlock->CalculateJacobian(time,
                                inverseTimeStep,
                                m_arrValues,
                                m_arrResiduals,
                                m_arrTimeDerivatives,
                                m_matJacobian);

#ifdef daeSuperLU_MT
    omp_set_num_threads(m_omp_num_threads);

    if(m_bFactorizationDone)
    {
    // During the subsequent calls re-use what is possible (Pc, U, L, etree, colcnt_h, part_super_h)
        m_Options.refact = YES;

    // Matrix AC has to be destroyed to avoid memory leaks in sp_colorder()
        Destroy_CompCol_Permuted(&m_matAC);
    }
    else
    {
    // At the first call do the fresh factorization (Pr, Pc, etree, L, U and AC will be computed)
        m_Options.refact = NO;

        StatAlloc(m_nNoEquations, m_Options.nprocs, m_Options.panel_size, m_Options.relax, &m_Stats);
        get_perm_c(m_Options.ColPerm, &m_matA, m_perm_c);
    }

    StatInit(m_nNoEquations, m_Options.nprocs, &m_Stats);

// This will allocate memory for AC.
// If I call it repeatedly then I will have memory leaks! (double check it)
// If that is true before each call to pdgstrf_init I have to call: pxgstrf_finalize(&m_Options, &m_matAC)
// which destroys AC, options->etree, options->colcnt_h and options->part_super_h
// but in that case I still need options->etree, options->colcnt_h and options->part_super_h
// Perhaps the best idea is to call Destroy_CompCol_Permuted(&AC) before each pdgstrf_init() call
    pdgstrf_init(m_Options.nprocs,
                 m_Options.fact,
                 m_Options.trans,
                 m_Options.refact,
                 m_Options.panel_size,
                 m_Options.relax,
                 m_Options.diag_pivot_thresh,
                 m_Options.usepr,
                 m_Options.drop_tol,
                 m_Options.perm_c,
                 m_Options.perm_r,
                 m_Options.work,
                 m_Options.lwork,
                 &m_matA,
                 &m_matAC,
                 &m_Options,
                 &m_Stats);

    pdgstrf(&m_Options, &m_matAC, m_perm_r, &m_matL, &m_matU, &m_Stats, &info);
    if(info != 0)
    {
        std::cout << "SuperLU factorization failed = " << info << std::endl;
        return -1; //IDA_LSETUP_FAIL;
    }

    //PrintStats();
#endif

#ifdef daeSuperLU
    int panel_size = sp_ienv(1);
    int relax      = sp_ienv(2);

    if(m_bFactorizationDone)
    {
    /*
        During the subsequent calls we can:
          a) Re-use Pc and recreate L and U (SamePattern)
             A conservative approach - we could use SamePattern_SameRowPerm to re-use Pr, Pc, Dr, Dc and
             structures allocated for L and U, but we may run into a numerical instability. Here we sacrify
             some speed for stability!
          b) Re-use everything (SamePattern_SameRowPerm)
             A faster approach when the pattern is the same and numerical values similar.
             Some numerical instability may occur and superlu tries to handle that.
    */
        if(m_iFactorization == SamePattern)
        {
            m_Options.Fact = SamePattern;

        // If memory was pre-allocated in m_work then destroy only SuperMatrix_Store
        // Otherwise destroy the whole matrix
            if(m_lwork > 0)
            {
                Destroy_SuperMatrix_Store(&m_matL);
                Destroy_SuperMatrix_Store(&m_matU);
            }
            else
            {
                Destroy_SuperNode_Matrix(&m_matL);
                Destroy_CompCol_Matrix(&m_matU);
            }

        // Matrix AC has to be destroyed to avoid memory leaks in sp_colorder()
            Destroy_CompCol_Permuted(&m_matAC);
        }
        else if(m_iFactorization == SamePattern_SameRowPerm)
        {
            m_Options.Fact = SamePattern_SameRowPerm;
        }
        else
        {
            std::cout << "SuperLU invalid factorization option" << std::endl;
            return -1;
        }
    }
    else
    {
    // At the first call do the fresh factorization (Pr, Pc, etree, L, U and AC will be computed)
        m_Options.Fact = DOFACT;

        get_perm_c(m_Options.ColPerm, &m_matA, m_perm_c);
    }

    StatInit(&m_Stats);

/*
    Allocate memory for AC (only if the Fact IS NOT SamePattern_SameRowPerm).
    If we call it repeatedly we will have memory leaks! (double check it)
    Therefore before each call to sp_preorder() destroy matrix AC with a
    call to Destroy_CompCol_Permuted(&AC) (if it is not already empty)
*/
    if(m_Options.Fact != SamePattern_SameRowPerm)
        sp_preorder(&m_Options, &m_matA, m_perm_c, m_etree, &m_matAC);

    if(m_bUseUserSuppliedWorkSpace)
    {
    // Determine the initial amount of memory (if m_lwork is zero)
        if(m_lwork == 0)
        {
        // Get the memory requirements from SuperLU
            dgstrf(&m_Options, &m_matAC, relax, panel_size, m_etree, NULL, -1, m_perm_c, m_perm_r, &m_matL, &m_matU, &m_Glu, &m_Stats, &info);

        // Remove the ncol from info
            memSize = info - m_nNoEquations;
            m_lwork = (4 * m_dWorkspaceSizeMultiplier * memSize) / 4; // word alligned

        // Allocate the workspace memory
            m_work = malloc(m_lwork);
            if(!m_work)
            {
                std::cout << "SuperLU failed to allocate the Workspace memory (" << real_t(m_lwork)/1E6 << " MB)" << std::endl;
                return -1;
            }
            std::cout << "Predicted memory requirements = " << info << " bytes (initially allocated " << m_lwork << ")" << std::endl;
        }
    }

    dgstrf(&m_Options, &m_matAC, relax, panel_size, m_etree, m_work, m_lwork, m_perm_c, m_perm_r, &m_matL, &m_matU, &m_Glu, &m_Stats, &info);
    if(info != 0)
    {
        std::cout << "SuperLU factorization failed; info = " << info << std::endl;
        return -1;
    }

    /* Not working...
    if(m_bUseUserSuppliedWorkSpace)
    {
    // Determine the initial amount of memory (if m_lwork is zero)
        if(m_lwork == 0)
        {
        // Get the memory requirements from SuperLU
            dgstrf(&m_Options, &m_matAC, relax, panel_size, m_etree, NULL, -1, m_perm_c, m_perm_r, &m_matL, &m_matU, &m_Glu, &m_Stats, &info);

        // Remove the ncol from info
            memSize = info - m_nNoEquations;
            m_lwork = (4 * m_dInitialWorkspaceSize * memSize) / 4; // word alligned

        // Allocate the workspace memory
            m_work = malloc(m_lwork);
            if(!m_work)
            {
                std::cout << "SuperLU failed to allocate the Workspace memory (" << real_t(m_lwork)/1E6 << " MB)" << std::endl;
                return IDA_LSETUP_FAIL;
            }
            std::cout << "Predicted memory requirements = " << info << " bytes (initially allocated " << m_lwork << ")" << std::endl;
        }

        info = -1;
        while(info != 0)
        {
        // Try to do the factorization; if it fails inspect why and if the workspace size is too low try to increase it
            dgstrf(&m_Options, &m_matAC, relax, panel_size, m_etree, m_work, m_lwork, m_perm_c, m_perm_r, &m_matL, &m_matU, &m_Glu, &m_Stats, &info);

            if(info != 0)
            {
                if(info > m_nNoEquations)
                {
                // In this case we have a memory allocation problem: try to incrementally increase the workspace size for
                // as long as the info > ncol (that is more memory is needed).
                // (the memory size attempted to allocate is: info - ncol)

                // First free
                    Destroy_SuperMatrix_Store(&m_matL);
                    Destroy_SuperMatrix_Store(&m_matU);
                    Destroy_CompCol_Permuted(&m_matAC);

                // Remove the ncol from info
                    memSize = info - m_nNoEquations;
                    std::cout << "SuperLU dgstrf attempted to allocate " << info << " bytes; current lwork = " << m_lwork << std::endl;

                // Set the new size
                    m_lwork = (4 * m_dWorkspaceMemoryIncrement * memSize) / 4; // word alligned
                    std::cout << "SuperLU new Workspace size: " << m_lwork << " bytes" << std::endl;

                // (Re-)Allocate the workspace memory
                    m_work = realloc(m_work, m_lwork);
                    if(!m_work)
                    {
                        std::cout << "SuperLU failed to allocate the Workspace memory (" << real_t(m_lwork)/1E6 << " MB)" << std::endl;
                        return IDA_LSETUP_FAIL;
                    }
                }
                else
                {
                // It is factorization failure; report the column where it ocurred and die miserably
                    std::cout << "SuperLU factorization failed; info = " << info << std::endl;
                    return IDA_LSETUP_FAIL;
                }
            }
            else
            {
            // Just to be sure (it will break after the successful factorization even without this line)
                break;
            }
        }
    }
    else
    {
        dgstrf(&m_Options, &m_matAC, relax, panel_size, m_etree, m_work, m_lwork, m_perm_c, m_perm_r, &m_matL, &m_matU, &m_Glu, &m_Stats, &info);
        if(info != 0)
        {
            std::cout << "SuperLU factorization failed; info = " << info << std::endl;
            return IDA_LSETUP_FAIL;
        }
    }
    */

    //PrintStats();
    StatFree(&m_Stats);

#endif

    m_bFactorizationDone = true;

    return 0;
}

int daeSuperLUSolver::Solve(real_t  time,
                            real_t  inverseTimeStep,
                            real_t  cjratio,
                            real_t* b,
                            real_t* weight,
                            real_t* pdValues,
                            real_t* pdTimeDerivatives,
                            real_t* pdResiduals)
{
    call_stats::TimerCounter tc(m_stats["Solve"]);

    int info;
    double rpg, rcond;

    size_t Neq = m_nNoEquations;

    if(!m_pBlock)
        return -1;

#ifdef daeSuperLU_MT
    omp_set_num_threads(m_omp_num_threads);

    ::memcpy(m_vecB, b, Neq*sizeof(real_t));

/**********************************************/
//	daeDenseArray x;
//	x.InitArray(Neq, m_vecB);
//	std::cout << "A matrix:" << std::endl;
//	m_matJacobian.Print(false, true);
//	std::cout << "B vector:" << std::endl;
//	x.Print();
/**********************************************/

// Note the order: (..., Pr, Pc, ...) and not (..., Pc, Pr, ...) as in superlu
    dgstrs(m_Options.trans, &m_matL, &m_matU, m_perm_r, m_perm_c, &m_matB, &m_Stats, &info);
    if(info != 0)
    {
        std::cout << "SuperLU solve failed = " << info << std::endl;
        return -1; //IDA_LSOLVE_FAIL;
    }

    PrintStats();
/**********************************************/
//	std::cout << "X vector:" << std::endl;
//	x.Print();
/**********************************************/
    ::memcpy(b, m_vecB, Neq*sizeof(real_t));
#endif

#ifdef daeSuperLU
    ::memcpy(m_vecB, b, Neq*sizeof(real_t));

    StatInit(&m_Stats);

// Note the order: (..., Pc, Pr, ...) and not (..., Pr, Pc, ...) as in superlu_mt
    dgstrs(m_Options.Trans, &m_matL, &m_matU, m_perm_c, m_perm_r, &m_matB, &m_Stats, &info);
    if(info != 0)
    {
        std::cout << "SuperLU solve failed; info = " << info << std::endl;
        return -1; //IDA_LSOLVE_FAIL;
    }

    PrintStats();
    StatFree(&m_Stats);

    ::memcpy(b, m_vecB, Neq*sizeof(real_t));
#endif

    if(cjratio != 1.0)
    {
        for(size_t i = 0; i < Neq; i++)
            b[i] *= 2.0 / (1.0 + cjratio);
    }

    return 0;
}

void daeSuperLUSolver::PrintStats(void)
{
#ifdef daeSuperLU_MT
#endif

#ifdef daeSuperLU

//	if(m_Options.PrintStat == YES)
//	{
//		StatPrint(&m_Stats);
//
//		std::cout << "ops[FACT] = " << m_Stats.ops[FACT] << std::endl;
//
//		std::cout << "utime[EQUIL]   = " << m_Stats.utime[EQUIL]   << std::endl;
//		std::cout << "utime[COLPERM] = " << m_Stats.utime[COLPERM] << std::endl;
//		std::cout << "utime[ETREE]   = " << m_Stats.utime[ETREE]   << std::endl;
//		std::cout << "utime[FACT]    = " << m_Stats.utime[FACT]    << std::endl;
//		std::cout << "utime[RCOND]   = " << m_Stats.utime[RCOND]   << std::endl;
//		std::cout << "utime[SOLVE]   = " << m_Stats.utime[SOLVE]   << std::endl;
//		std::cout << "utime[REFINE]  = " << m_Stats.utime[REFINE]  << std::endl;
//	}
#endif
}

#ifdef daeSuperLU
}
#endif
#ifdef daeSuperLU_MT
}
#endif

}
}

