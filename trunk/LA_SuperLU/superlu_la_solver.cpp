#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "superlu_la_solver.h"

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
	
daeIDALASolver_t* daeCreateSuperLUSolver(void)
{
	return new daeSuperLUSolver;
}

daeSuperLUSolver::daeSuperLUSolver(void)
{
	m_pBlock	= NULL;
	m_vecB		= NULL;
	m_vecX		= NULL;
	m_perm_c	= NULL;
	m_perm_r	= NULL;
	m_R			= NULL;
	m_C			= NULL;
	m_bFactorizationDone = false;
	
// The user shoud be able to set parameters right after the construction of the solver
#ifdef daeSuperLU_MT
    m_Options.nprocs			= 4;
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
	
#elif daeSuperLU
	m_etree		= NULL;
	
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

int daeSuperLUSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
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
	m_nJacobianEvaluations	= 0;

	InitializeSuperLU(nnz);
	
	ida_mem->ida_linit	= init_la;
	ida_mem->ida_lsetup = setup_la;
	ida_mem->ida_lsolve = solve_la;
	ida_mem->ida_lperf	= NULL;
	ida_mem->ida_lfree	= free_la;

	ida_mem->ida_lmem         = this;
	ida_mem->ida_setupNonNull = TRUE;

	return IDA_SUCCESS;
}

int daeSuperLUSolver::Reinitialize(void* ida)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	int n   = m_nNoEquations;
	int nnz = 0;
	m_pBlock->CalcNonZeroElements(nnz);

	m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	//m_matJacobian.Print();

    Destroy_SuperMatrix_Store(&m_matA);
	dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NR, SLU_D, SLU_GE);
	m_bFactorizationDone = false;
	
#ifdef daeSuperLU_MT
    //pxgstrf_finalize(&m_Options, &m_matAC);	
	if(m_Options.lwork >= 0)
	{
		Destroy_SuperNode_SCP(&m_matL);
		Destroy_CompCol_NCP(&m_matU);
	}
	//StatFree(&m_Stats);
	
#elif daeSuperLU

#endif

	return IDA_SUCCESS;
}

void daeSuperLUSolver::FreeMemory(void)
{
#ifdef daeSuperLU_MT
    //pxgstrf_finalize(&m_Options, &m_matAC);	
	if(m_Options.lwork >= 0)
	{
		Destroy_SuperNode_SCP(&m_matL);
		Destroy_CompCol_NCP(&m_matU);
	}
	//StatFree(&m_Stats);
	
#elif daeSuperLU
	if(m_etree)
		SUPERLU_FREE(m_etree);
	if(m_bFactorizationDone)
	{
		Destroy_SuperNode_Matrix(&m_matL);
		Destroy_CompCol_Matrix(&m_matU);
	}
#endif

	if(m_vecB)
		SUPERLU_FREE(m_vecB);
	if(m_vecX)
		SUPERLU_FREE(m_vecX);
	if(m_perm_c)
		SUPERLU_FREE(m_perm_c);
	if(m_perm_r)
		SUPERLU_FREE(m_perm_r);
	if(m_R)
		SUPERLU_FREE(m_R);
	if(m_C)
		SUPERLU_FREE(m_C);
	
	m_matJacobian.Free();
    Destroy_SuperMatrix_Store(&m_matA);
	Destroy_SuperMatrix_Store(&m_matB);
	Destroy_SuperMatrix_Store(&m_matX);
}

void daeSuperLUSolver::InitializeSuperLU(size_t nnz)
{
	m_vecB					= doubleMalloc(m_nNoEquations);
	m_vecX					= doubleMalloc(m_nNoEquations);
	m_perm_c				= intMalloc(m_nNoEquations);
	m_perm_r				= intMalloc(m_nNoEquations);
	m_R						= doubleMalloc(m_nNoEquations);
	m_C						= doubleMalloc(m_nNoEquations);
	
#ifdef daeSuperLU_MT
    m_Options.perm_c	= m_perm_c;
    m_Options.perm_r	= m_perm_r;
    m_Options.work		= NULL;
    m_Options.lwork		= 0;
	
	if(!m_vecB || !m_vecX || !m_perm_c || !m_perm_r || !m_R || !m_C)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate memory for B, X, perm_c, perm_r, R, C vectors";
		throw e;
	}

#elif daeSuperLU
	m_etree	= intMalloc(m_nNoEquations);
	
	if(!m_vecB || !m_vecX || !m_perm_c || !m_perm_r || !m_R || !m_C || !m_etree)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate memory for B, X, perm_c, perm_r, etree, R, C vectors";
		throw e;
	}
#endif

// Initialize sparse matrix
	m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	//m_matJacobian.Print();

	dCreate_CompCol_Matrix(&m_matA, m_matJacobian.N, m_matJacobian.N, m_matJacobian.NNZ, m_matJacobian.A, m_matJacobian.JA, m_matJacobian.IA, SLU_NR, SLU_D, SLU_GE);
	dCreate_Dense_Matrix(&m_matB, m_nNoEquations, 1, m_vecB, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
	dCreate_Dense_Matrix(&m_matX, m_nNoEquations, 1, m_vecX, m_nNoEquations, SLU_DN, SLU_D, SLU_GE);
	
	m_bFactorizationDone = false;
}

int daeSuperLUSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return IDA_SUCCESS;
}

#ifdef daeSuperLU_MT
superlumt_options_t& 		
#elif daeSuperLU
superlu_options_t& 
#endif
daeSuperLUSolver::GetOptions(void)
{
	return m_Options;
}

int daeSuperLUSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

int daeSuperLUSolver::Setup(void*		ida,
							N_Vector	vectorVariables, 
							N_Vector	vectorTimeDerivatives, 
							N_Vector	vectorResiduals,
							N_Vector	vectorTemp1, 
							N_Vector	vectorTemp2, 
							N_Vector	vectorTemp3)
{
	int info;
    double rpg, rcond;
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
	
#ifdef daeSuperLU_MT
/*
	Get column permutation vector perm_c[], according to permc_spec:
	  permc_spec = 0: natural ordering 
	  permc_spec = 1: minimum degree ordering on structure of A'*A
	  permc_spec = 2: minimum degree ordering on structure of A'+A
	  permc_spec = 3: approximate minimum degree for unsymmetric matrices
*/    	
    int permc_spec = 1;
	
// I use this to prevent solving the system; Does it have any sense? How it should be done?
	::memset(m_vecB, 0, m_nNoEquations*sizeof(real_t));

	if(m_bFactorizationDone)
	{
		m_Options.refact = NO;
		m_Options.fact   = EQUILIBRATE;
	}
	else
	{
		m_Options.refact = NO;
		m_Options.fact   = EQUILIBRATE;
	}
	
	get_perm_c(permc_spec, &m_matA, m_perm_c);
	
	pdgssvx(m_Options.nprocs, &m_Options, &m_matA, m_perm_c, m_perm_r, &m_equed, m_R, m_C, &m_matL, &m_matU, &m_matB, &m_matX, &rpg, &rcond, &m_ferr, &m_berr, &m_memUsage, &info);
	if(info != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to factorize the matrix: " << info;
		throw e;
	}

/*
	if(m_bFactorizationDone)
	{
		StatInit(m_nNoEquations, m_Options.nprocs, &m_Stats);
		
		m_Options.refact = YES;
	}
	else
	{
		StatAlloc(m_nNoEquations, m_Options.nprocs, m_Options.panel_size, m_Options.relax, &m_Stats);
		StatInit(m_nNoEquations, m_Options.nprocs, &m_Stats);
		
		get_perm_c(permc_spec, &m_matA, m_perm_c);
		
		m_Options.refact = NO;
	}
	
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
		daeDeclareException(exMiscellanous);
		e << "Unable to factorize the matrix: " << info;
		throw e;
	}
*/
	
#elif daeSuperLU
/*
	The default input options:
		m_Options.Fact = DOFACT;
		m_Options.Equil = YES;
		m_Options.ColPerm = COLAMD;
		m_Options.DiagPivotThresh = 1.0;
		m_Options.IterRefine = NOREFINE;
		
	The first time Setup is called (or after Reinitialize) m_bFactorizationDone is false 
	and the full factorization is requested: m_Options.Fact = DOFACT
	Otherwise reuse some information from the previous call: m_Options.Fact = SamePattern
*/
	if(m_bFactorizationDone)
	{
		m_Options.Fact = SamePattern;
		Destroy_SuperNode_Matrix(&m_matL);
		Destroy_CompCol_Matrix(&m_matU);
	}
	else
	{
		m_Options.Fact = DOFACT;
	}
	StatInit(&m_Stats);

	m_matB.ncol = 0;  /* Indicate not to solve the system */
	dgssvx(&m_Options, &m_matA, m_perm_c, m_perm_r, m_etree, &m_equed, m_R, m_C, &m_matL, &m_matU, NULL, 0, &m_matB, &m_matX, &rpg, &rcond, &m_ferr, &m_berr, &m_memUsage, &m_Stats, &info);
	m_matB.ncol = 1;  /* Restore it back */
	if(info != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to factorize the matrix: " << info;
		throw e;
	}
	//std::cout << "FactFlops = " << m_Stats.ops[FACT] << std::endl;
	
	StatFree(&m_Stats);
#endif

	m_bFactorizationDone = true;
	
	return IDA_SUCCESS;
}

int daeSuperLUSolver::Solve(void*		ida,
							N_Vector	vectorB, 
							N_Vector	vectorWeight, 
							N_Vector	vectorVariables,
							N_Vector	vectorTimeDerivatives, 
							N_Vector	vectorResiduals)
{
	int info;
    double rpg, rcond;
	realtype* pdB;
		
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq = m_nNoEquations;
	pdB        = NV_DATA_S(vectorB);
	
	memcpy(m_vecB, pdB, Neq*sizeof(real_t));
	
	std::cout << "Solve" << std::endl;

#ifdef daeSuperLU_MT
    m_Options.fact   = FACTORED;
	m_Options.refact = YES;
	
	pdgssvx(m_Options.nprocs, &m_Options, &m_matA, m_perm_c, m_perm_r, &m_equed, m_R, m_C, &m_matL, &m_matU, &m_matB, &m_matX, &rpg, &rcond, &m_ferr, &m_berr, &m_memUsage, &info);
	if(info != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to factorize the matrix: " << info;
		throw e;
	}
	
	::memcpy(pdB, m_vecX, Neq*sizeof(real_t));
	
/*
    dgstrs(m_Options.trans, &m_matL, &m_matU, m_perm_r, m_perm_c, &m_matB, &m_Stats, &info);
	if(info != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to factorize the matrix: " << info;
		throw e;
	}
	
	::memcpy(pdB, m_vecB, Neq*sizeof(real_t));
*/	
#elif daeSuperLU
	StatInit(&m_Stats);
	m_Options.Fact = FACTORED;

	dgssvx(&m_Options, &m_matA, m_perm_c, m_perm_r, m_etree, &m_equed, m_R, m_C, &m_matL, &m_matU, NULL, 0, &m_matB, &m_matX, &rpg, &rcond, &m_ferr, &m_berr, &m_memUsage, &m_Stats, &info);
	if(info != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to solve the system: " << info;
		throw e;
	}
	
	//std::cout << "SolveFlops = " << m_Stats.ops[SOLVE] << std::endl;
	StatFree(&m_Stats);
	
	::memcpy(pdB, m_vecX, Neq*sizeof(real_t));
#endif
	
	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
		//N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
	}
	
	return IDA_SUCCESS;	
}

int daeSuperLUSolver::Free(void* ida)
{
	return IDA_SUCCESS;	
}


int init_la(IDAMem ida_mem)
{
	daeSuperLUSolver* pSolver = (daeSuperLUSolver*)ida_mem->ida_lmem;
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
	daeSuperLUSolver* pSolver = (daeSuperLUSolver*)ida_mem->ida_lmem;
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
	daeSuperLUSolver* pSolver = (daeSuperLUSolver*)ida_mem->ida_lmem;
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
	daeSuperLUSolver* pSolver = (daeSuperLUSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	int ret = pSolver->Free(ida_mem);

	delete pSolver;
	ida_mem->ida_lmem = NULL;

	return ret;
}


}
}

