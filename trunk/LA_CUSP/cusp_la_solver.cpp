#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "cusp_la_solver.h"

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
	
daeIDALASolver_t* daeCreateCUSPSolver(void)
{
	return new daeCUSPSolver;
}

daeCUSPSolver::daeCUSPSolver(void)
{
	m_pBlock	= NULL;
	m_vecB		= NULL;
	m_vecX		= NULL;
	m_bFactorizationDone = false;
}

daeCUSPSolver::~daeCUSPSolver(void)
{
	FreeMemory();
}

int daeCUSPSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
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

int daeCUSPSolver::Reinitialize(void* ida)
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

	m_bFactorizationDone = false;
	
	return IDA_SUCCESS;
}

void daeCUSPSolver::FreeMemory(void)
{
	m_matJacobian.Free();
}

void daeCUSPSolver::InitializeSuperLU(size_t nnz)
{
// Initialize sparse matrix
	m_matJacobian.Reset(m_nNoEquations, nnz, CSR_C_STYLE);
	m_matJacobian.ResetCounters();
	m_pBlock->FillSparseMatrix(&m_matJacobian);
	m_matJacobian.Sort();
	
	m_cuspSolver.ResizeMatrixAndVectors(m_matJacobian.NNZ, m_matJacobian.N, m_matJacobian.IA, m_matJacobian.JA);

	m_bFactorizationDone = false;
}

int daeCUSPSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return 0;
}

int daeCUSPSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
	m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
	return 0;
}

std::string daeCUSPSolver::GetName(void) const
{
	return string("CUSP");
}

int daeCUSPSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

int daeCUSPSolver::Setup(void*		ida,
							N_Vector	vectorVariables, 
							N_Vector	vectorTimeDerivatives, 
							N_Vector	vectorResiduals,
							N_Vector	vectorTemp1, 
							N_Vector	vectorTemp2, 
							N_Vector	vectorTemp3)
{
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
	//m_matJacobian.Print(false, true);
	
	m_cuspSolver.SetMatrix_A(m_matJacobian.NNZ, m_matJacobian.A);

	m_bFactorizationDone = true;
	
	return IDA_SUCCESS;
}

int daeCUSPSolver::Solve(void*		ida,
							N_Vector	vectorB, 
							N_Vector	vectorWeight, 
							N_Vector	vectorVariables,
							N_Vector	vectorTimeDerivatives, 
							N_Vector	vectorResiduals)
{
	realtype* pdB;
		
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq = m_nNoEquations;
	pdB        = NV_DATA_S(vectorB);
	
	m_cuspSolver.SetVector_b(Neq, pdB);
	
	m_cuspSolver.Solve();
	
	m_cuspSolver.GetVector_x(Neq, &pdB);
	
	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
		//N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
	}
	
	return IDA_SUCCESS;	
}

int daeCUSPSolver::Free(void* ida)
{
	return IDA_SUCCESS;	
}


int init_la(IDAMem ida_mem)
{
	daeCUSPSolver* pSolver = (daeCUSPSolver*)ida_mem->ida_lmem;
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
	daeCUSPSolver* pSolver = (daeCUSPSolver*)ida_mem->ida_lmem;
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
	daeCUSPSolver* pSolver = (daeCUSPSolver*)ida_mem->ida_lmem;
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
	daeCUSPSolver* pSolver = (daeCUSPSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	int ret = pSolver->Free(ida_mem);

// ACHTUNG, ACHTUNG!!
// It is the responsibility of the user to delete LA solver pointer!!
//	delete pSolver;

	ida_mem->ida_lmem = NULL;

	return ret;
}


}
}

