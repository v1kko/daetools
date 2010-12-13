//#ifdef HAS_INTEL_MKL
#include "stdafx.h"
#include <stdio.h>
#include "ida_solver.h"
#include <ida/ida_impl.h>
#include <mkl_pardiso_la_solver.h>

namespace dae
{
namespace solver
{
int IDA_sparse_MKL_PARDISO_LU(void* ida, size_t n, void* pUserData)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_ILL_INPUT;
	
	ida_mem->ida_linit	= init_mkl;
	ida_mem->ida_lsetup = setup_mkl;
	ida_mem->ida_lsolve = solve_mkl;
	ida_mem->ida_lperf	= NULL;
	ida_mem->ida_lfree	= free_mkl;
	
	daeIDASolver* pSolver = (daeIDASolver*)pUserData;
	if(!pSolver)
		return IDA_MEM_NULL;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return IDA_MEM_NULL;

	int nnz = 0;
	pBlock->CalcNonZeroElements(nnz);

	mklPardisoLASolverData* pData = new mklPardisoLASolverData(n, nnz, false, pUserData);
	
	if(!pData)
		return IDA_MEM_NULL;

	if(!pData->CheckData())
		return IDA_MEM_NULL;

	ida_mem->ida_lmem = pData;
	ida_mem->ida_setupNonNull = TRUE;
	
// Initialize sparse matrix
	pData->m_matJacobian.StartRow();
	pBlock->FillSparseMatrix(&pData->m_matJacobian);
	pData->m_matJacobian.Print();

	return IDA_SUCCESS;
}

int IDA_sparse_MKL_PARDISO_LU_Reset(void* ida)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_ILL_INPUT;

	mklPardisoLASolverData* pData = (mklPardisoLASolverData*)ida_mem->ida_lmem;
	if(!pData)
		return IDA_MEM_NULL;
	
	daeIDASolver* pSolver = (daeIDASolver*)pData->m_pUserData;
	if(!pSolver)
		return IDA_MEM_NULL;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return IDA_MEM_NULL;
	
	int n   = pData->m_nNoEquations;
	int nnz = 0;
	pBlock->CalcNonZeroElements(nnz);
	pData->Reset(n, nnz, false);
	pData->m_matJacobian.StartRow();
	pBlock->FillSparseMatrix(&pData->m_matJacobian);
	//pData->m_matJacobian.Print();

	return IDA_SUCCESS;
}

int init_mkl(IDAMem ida_mem)
{
	_INTEGER_t res, idum;
	real_t ddum;
	
	mklPardisoLASolverData* pData = (mklPardisoLASolverData*)ida_mem->ida_lmem;
	if(!pData)
		return IDA_MEM_NULL;
		
	return IDA_SUCCESS;
}

int setup_mkl(IDAMem		ida_mem,
				 N_Vector	vectorVariables, 
				 N_Vector	vectorTimeDerivatives, 
				 N_Vector	vectorResiduals,
				 N_Vector	vectorTemp1, 
				 N_Vector	vectorTemp2, 
				 N_Vector	vectorTemp3)
{
	_INTEGER_t res, idum;
	real_t ddum;
	realtype *pdValues, *pdTimeDerivatives, *pdResiduals;
		
	mklPardisoLASolverData* pData = (mklPardisoLASolverData*)ida_mem->ida_lmem;
	if(!pData)
		return IDA_MEM_NULL;
	
	daeIDASolver* pSolver = (daeIDASolver*)pData->m_pUserData;
	if(!pSolver)
		return IDA_MEM_NULL;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq              = pData->m_nNoEquations;
	real_t time             = ida_mem->ida_tn;
	real_t dInverseTimeStep = ida_mem->ida_cj;
	
	pdValues			= NV_DATA_S(vectorVariables); 
	pdTimeDerivatives	= NV_DATA_S(vectorTimeDerivatives); 
	pdResiduals			= NV_DATA_S(vectorResiduals);

	pData->m_nJacobianEvaluations++;
	
	pData->m_arrValues.InitArray(Neq, pdValues);
	pData->m_arrTimeDerivatives.InitArray(Neq, pdTimeDerivatives);
	pData->m_arrResiduals.InitArray(Neq, pdResiduals);

	pData->m_matJacobian.ClearValues();
	
	pBlock->CalculateJacobian(time, 
		                      pData->m_arrValues, 
							  pData->m_arrResiduals, 
							  pData->m_arrTimeDerivatives, 
							  pData->m_matJacobian, 
							  dInverseTimeStep);
	std::cout << "Jacobian" << std::endl;
	std::cout.flush();
	pData->m_matJacobian.Print();

// Reordering and Symbolic Factorization. 
// This step also allocates all memory
// that is necessary for the factorization. 
	std::cout << "Start reordering... ";
	std::cout.flush();
	pData->phase = 11;
	PARDISO (pData->pt, 
			 &pData->maxfct, 
			 &pData->mnum, 
			 &pData->mtype, 
			 &pData->phase,
			 &pData->m_nNoEquations, 
			 pData->m_matJacobian.A, 
			 pData->m_matJacobian.IA, 
			 pData->m_matJacobian.JA, 
			 &idum, 
			 &pData->nrhs,
			 pData->iparm, 
			 &pData->msglvl, 
			 &ddum, 
			 &ddum, 
			 &res);
	if(res != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to Analyse/Reorder MKL PARDISO LA solver";
		throw e;
	}
	std::cout << "Reordering completed"                                  << std::endl;
	std::cout << "Number of nonzeros in factors  = " << pData->iparm[17] << std::endl;
	std::cout << "Number of factorization MFLOPS = " << pData->iparm[18] << std::endl;
		
// Numerical factorization
	std::cout << "Start factorization... ";
	std::cout.flush();
	pData->phase = 22;
	PARDISO (pData->pt, 
			 &pData->maxfct, 
			 &pData->mnum, 
			 &pData->mtype, 
			 &pData->phase,
			 &pData->m_nNoEquations, 
			 pData->m_matJacobian.A, 
			 pData->m_matJacobian.IA, 
			 pData->m_matJacobian.JA, 
			 &idum, 
			 &pData->nrhs,
			 pData->iparm, 
			 &pData->msglvl, 
			 &ddum, 
			 &ddum, 
			 &res);
	if(res != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to Factor MKL PARDISO LA solver";
		throw e;
	}
	std::cout << "Factorization completed" << std::endl;
	std::cout.flush();
	
	return IDA_SUCCESS;
}

int solve_mkl(IDAMem		ida_mem,
				 N_Vector	vectorB, 
				 N_Vector	vectorWeight, 
				 N_Vector	vectorVariables,
				 N_Vector	vectorTimeDerivatives, 
				 N_Vector	vectorResiduals)
{
	_INTEGER_t res, idum;
	real_t ddum;
	realtype* pdB;
		
	mklPardisoLASolverData* pData = (mklPardisoLASolverData*)ida_mem->ida_lmem;
	if(!pData)
		return IDA_MEM_NULL;
	
	daeIDASolver* pSolver = (daeIDASolver*)pData->m_pUserData;
	if(!pSolver)
		return IDA_MEM_NULL;

	daeBlock_t* pBlock = pSolver->m_pBlock;
	if(!pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq = pData->m_nNoEquations;
	pdB        = NV_DATA_S(vectorB);
	
	
	memcpy(pData->m_vecB, pdB, Neq*sizeof(real_t));

// Solve
	std::cout << "Solving... ";
	std::cout.flush();
	pData->phase = 33;
	pData->iparm[7] = 2; /* Max numbers of iterative refinement steps. */
	PARDISO (pData->pt, 
			 &pData->maxfct, 
			 &pData->mnum, 
			 &pData->mtype, 
			 &pData->phase,
			 &pData->m_nNoEquations, 
			 pData->m_matJacobian.A, 
			 pData->m_matJacobian.IA, 
			 pData->m_matJacobian.JA, 
			 &idum, 
			 &pData->nrhs,
			 pData->iparm, 
			 &pData->msglvl, 
			 pData->m_vecB, 
			 pdB, 
			 &res);
	if(res != 0)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to Factor MKL PARDISO LA solver";
		throw e;
	}
	std::cout << "Solve completed" << std::endl;
	std::cout.flush();

	daeDenseArray mkl;
	mkl.InitArray(Neq, pdB);
	std::cout << "Intel" << std::endl;
	mkl.Print();
	
	if(ida_mem->ida_cjratio != 1.0) 
		N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
	
	return IDA_SUCCESS;
}

int free_mkl(IDAMem ida_mem)
{
	real_t ddum;
	_INTEGER_t res, idum;
	
	mklPardisoLASolverData* pData = (mklPardisoLASolverData*)ida_mem->ida_lmem;
	if(!pData)
		return IDA_MEM_NULL;

// Memory release
	pData->phase = -1;
	PARDISO (pData->pt, 
			 &pData->maxfct, 
			 &pData->mnum, 
			 &pData->mtype, 
			 &pData->phase,
			 &pData->m_nNoEquations, 
			 pData->m_matJacobian.A, 
			 pData->m_matJacobian.IA, 
			 pData->m_matJacobian.JA, 
			 &idum, 
			 &pData->nrhs,
			 pData->iparm, 
			 &pData->msglvl, 
			 &ddum, 
			 &ddum, 
			 &res);

	delete pData;
	ida_mem->ida_lmem = NULL;

	return IDA_SUCCESS;
}


}
}

//#endif


