#include "stdafx.h"
#define PY_ARRAY_UNIQUE_SYMBOL dae_extension
#include <boost/python.hpp>
#include <idas/idas_impl.h>
#include "lapack_la_solver.h"

#ifdef daeHasMagma
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <magma.h>
#endif

#ifdef daeHasLapack
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrs_(char *trans, int* n, int* nrhs, double *a , int* lda, int *ipiv, double *b, int* ldb, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrs_(char *trans, int* n, int* nrhs, float *a , int* lda, int *ipiv, float *b, int* ldb, int *info);
#endif

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

daeIDALASolver_t* daeCreateLapackSolver(void)
{
	return new daeLapackSolver;
}

daeLapackSolver::daeLapackSolver(void)
{
	m_nNoEquations         = 0;
	m_nJacobianEvaluations = 0;
	m_matLAPACK            = NULL;
	m_vecPivot             = NULL;
	m_pBlock			   = NULL;
	
#ifdef daeHasMagma
	m_matCUDA = NULL;
	m_pdB     = NULL;
#endif
}

daeLapackSolver::~daeLapackSolver(void)
{
	FreeMemory();
	
#ifdef daeHasMagma
	cublasShutdown();
#endif
}

int daeLapackSolver::Create(void* ida, size_t n, daeDAESolver_t* pDAESolver)
{
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!ida_mem || !pDAESolver)
		return IDA_ILL_INPUT;
	
	m_pBlock = pDAESolver->GetBlock();
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	m_nNoEquations = n;
	
	AllocateMemory();
	if(!CheckData())
		return IDA_MEM_NULL;

	ida_mem->ida_linit	= init_la;
	ida_mem->ida_lsetup = setup_la;
	ida_mem->ida_lsolve = solve_la;
	ida_mem->ida_lperf	= NULL;
	ida_mem->ida_lfree	= free_la;

	ida_mem->ida_lmem         = this;
	ida_mem->ida_setupNonNull = TRUE;

	return IDA_SUCCESS;	
}

int daeLapackSolver::Reinitialize(void* pIDA)
{
	return IDA_SUCCESS;
}

int daeLapackSolver::SaveAsXPM(const std::string& strFileName)
{
	m_matJacobian.SaveMatrixAsXPM(strFileName);
	return 0;
}

int daeLapackSolver::SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription)
{
	m_matJacobian.SaveAsMatrixMarketFile(strFileName, strMatrixName, strMatrixDescription);
	return 0;
}

bool daeLapackSolver::CheckData() const
{
	if(!m_vecPivot || !m_matLAPACK || !m_nNoEquations > 0)
		return false;
#ifdef daeHasMagma
	if(!m_matCUDA || !m_pdB)
		return false;
#endif

	return true;
}

void daeLapackSolver::AllocateMemory(void)
{
	FreeMemory();

	size_t nR   = m_nNoEquations * m_nNoEquations * sizeof(real_t);
	size_t nI   = m_nNoEquations * sizeof(int);
	m_vecPivot  = (int*)   malloc(nI);
	m_matLAPACK	= (real_t*)malloc(nR);
	
	if(!m_vecPivot || !m_matLAPACK)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate memory for Lapack LA solver";
		FreeMemory();
		throw e;
	}

	m_matJacobian.InitMatrix(m_nNoEquations, m_nNoEquations, m_matLAPACK, eColumnWise);
	
#ifdef daeHasMagma
	cublasStatus cures;
	size_t free;
	size_t total;

//	CUdevice dev;
//	CUcontext ctx;
//	cuInit(0);
//	cuDeviceGet(&dev,0);
//	cuCtxCreate(&ctx, 0, dev);
//	
//	cures = cuMemGetInfo(&free, &total);
//	if(cures != CUBLAS_STATUS_SUCCESS)
//	{
//		daeDeclareException(exMiscellanous);
//		e << "Unable to initialize CUDA device";
//		FreeMemory();
//		throw e;
//	}
//	std::cout << "Available memory on video card:" << std::endl << "Total memory = " << total << std::endl << "Free memory  = " << free << std::endl;
	
	cures = cublasInit();
	if(cures != CUBLAS_STATUS_SUCCESS)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to initialize CUDA device";
		FreeMemory();
		throw e;
	}

	cures = cublasAlloc(m_nNoEquations * m_nNoEquations, sizeof(real_t), (void**)&m_matCUDA);
	if(cures != CUBLAS_STATUS_SUCCESS)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate memory for CUDA matrix: ";
		if(cures == CUBLAS_STATUS_NOT_INITIALIZED)
			e << "CUBLAS_STATUS_NOT_INITIALIZED";
		else if(cures == CUBLAS_STATUS_INVALID_VALUE)
			e << "CUBLAS_STATUS_INVALID_VALUE";
		else if(cures == CUBLAS_STATUS_ALLOC_FAILED)
			e << "CUBLAS_STATUS_ALLOC_FAILED";
		FreeMemory();
		throw e;
	}
	
	cures = cublasAlloc(m_nNoEquations, sizeof(real_t), (void**)&m_pdB);
	if(cures != CUBLAS_STATUS_SUCCESS)
	{
		daeDeclareException(exMiscellanous);
		e << "Unable to allocate memory for CUDA B array";
		FreeMemory();
		throw e;
	}
#endif
}

void daeLapackSolver::FreeMemory(void)
{
	if(m_matLAPACK)
		free(m_matLAPACK);
	if(m_vecPivot)
		free(m_vecPivot);
	
	m_matLAPACK = NULL;
	m_vecPivot  = NULL;
	
#ifdef daeHasMagma
	if(m_matCUDA)
		cublasFree(m_matCUDA);
	if(m_pdB)
		cublasFree(m_pdB);
	m_matCUDA = NULL;
	m_pdB     = NULL;
#endif
}

int daeLapackSolver::Init(void* ida)
{
	return IDA_SUCCESS;
}

int daeLapackSolver::Setup(void*		ida,
							 N_Vector	vectorVariables, 
							 N_Vector	vectorTimeDerivatives, 
							 N_Vector	vectorResiduals,
							 N_Vector	vectorTemp1, 
							 N_Vector	vectorTemp2, 
							 N_Vector	vectorTemp3)
{
	int info;
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

	::memset(m_matLAPACK, 0, Neq*Neq*sizeof(real_t));
	
	m_pBlock->CalculateJacobian(time, 
		                        m_arrValues, 
							    m_arrResiduals, 
							    m_arrTimeDerivatives, 
							    m_matJacobian, 
							    dInverseTimeStep);
	
	int n = Neq;

#ifdef daeHasIntelMKL
#ifdef DAE_SINGLE_PRECISION
	SGETRF(&n, &n, m_matLAPACK, &n, m_vecPivot, &info);
#else
	DGETRF(&n, &n, m_matLAPACK, &n, m_vecPivot, &info);
#endif
#endif
	
#ifdef daeHasMagma
	magma_int_t res;
	cublasStatus cures;
	
	cures = cublasSetMatrix (n, n, sizeof(real_t), m_matLAPACK, n, m_matCUDA, n);
	if(cures != CUBLAS_STATUS_SUCCESS)
		return IDA_LSETUP_FAIL;
	
#ifdef DAE_SINGLE_PRECISION
	res = magma_sgetrf_gpu(n, n, m_matCUDA, n, m_vecPivot, &info);
#else
	res = magma_dgetrf_gpu(n, n, m_matCUDA, n, m_vecPivot, &info);
#endif
#endif
	
#ifdef daeHasLapack
#ifdef DAE_SINGLE_PRECISION
	sgetrf_(&n, &n, m_matLAPACK, &n, m_vecPivot, &info);
#else
	dgetrf_(&n, &n, m_matLAPACK, &n, m_vecPivot, &info);
#endif
#endif

#ifdef daeHasAmdACML
#ifdef DAE_SINGLE_PRECISION
	sgetrf(n, n, m_matLAPACK, n, m_vecPivot, &info);
#else
	dgetrf(n, n, m_matLAPACK, n, m_vecPivot, &info);
#endif
#endif


	if(info < 0)
	{
		std::cout << "[setup_linear]: Element: " << -info << "has an illegal value" << std::endl;
		return IDA_LSETUP_FAIL;
	}
	else if(info > 0)
	{
		std::cout << "[setup_linear]: matrix element U(" << info << ", " << info << ") is zero" << std::endl;
		return IDA_LSETUP_FAIL;
	}

	return IDA_SUCCESS;
}

int daeLapackSolver::Solve(void*		ida,
							 N_Vector	vectorB, 
							 N_Vector	vectorWeight, 
							 N_Vector	vectorVariables,
							 N_Vector	vectorTimeDerivatives, 
							 N_Vector	vectorResiduals)
{
	int info;
	realtype* pdB;
		
	IDAMem ida_mem = (IDAMem)ida;
	if(!ida_mem)
		return IDA_MEM_NULL;
	if(!m_pBlock)
		return IDA_MEM_NULL;
	
	size_t Neq	= m_nNoEquations;
	pdB     	= NV_DATA_S(vectorB); 

	char transa = 'N';
	int nrhs = 1;
	int n = Neq;
	
#ifdef daeHasIntelMKL
#ifdef DAE_SINGLE_PRECISION
	SGETRS(&transa, &n, &nrhs, m_matLAPACK, &n, m_vecPivot, pdB, &n, &info);
#else
	DGETRS(&transa, &n, &nrhs, m_matLAPACK, &n, m_vecPivot, pdB, &n, &info);
#endif
#endif
	
#ifdef daeHasMagma
	magma_int_t res;
	cublasStatus cures;
	
	cures = cublasSetVector(n, sizeof(real_t), pdB, 1, m_pdB, 1);
	if(cures != CUBLAS_STATUS_SUCCESS)
		return IDA_LSOLVE_FAIL;
	
#ifdef DAE_SINGLE_PRECISION
	res = magma_sgetrs_gpu(transa, n, nrhs, m_matCUDA, n, m_vecPivot, m_pdB, n, &info);
#else
	res = magma_dgetrs_gpu(transa, n, nrhs, m_matCUDA, n, m_vecPivot, m_pdB, n, &info);
#endif
	
	cures = cublasGetVector(n, sizeof(real_t), m_pdB, 1, pdB, 1);
	if(cures != CUBLAS_STATUS_SUCCESS)
		return IDA_LSOLVE_FAIL;
	
#endif
	
#ifdef daeHasLapack
#ifdef DAE_SINGLE_PRECISION
	sgetrs_(&transa, &n, &nrhs, m_matLAPACK, &n, m_vecPivot, pdB, &n, &info);
#else
	dgetrs_(&transa, &n, &nrhs, m_matLAPACK, &n, m_vecPivot, pdB, &n, &info);
#endif
#endif
	
#ifdef daeHasAmdACML
#ifdef DAE_SINGLE_PRECISION
	sgetrs(transa, n, nrhs, m_matLAPACK, n, m_vecPivot, pdB, n, &info);
#else
	dgetrs(transa, n, nrhs, m_matLAPACK, n, m_vecPivot, pdB, n, &info);
#endif
#endif

	if(info < 0)
	{
		std::cout << "[solve_linear]: Element: " << -info << " has an illegal value" << std::endl;
		return IDA_LSOLVE_FAIL;
	}

	if(ida_mem->ida_cjratio != 1.0)
	{
		for(size_t i = 0; i < Neq; i++)
			pdB[i] *= 2.0 / (1.0 + ida_mem->ida_cjratio);
		//N_VScale(2.0 / (1.0 + ida_mem->ida_cjratio), vectorB, vectorB);
	}
	
	return IDA_SUCCESS;
}

int daeLapackSolver::Free(void* ida)
{
	FreeMemory();
	return IDA_SUCCESS;
}

int init_la(IDAMem ida_mem)
{
	daeLapackSolver* pSolver = (daeLapackSolver*)ida_mem->ida_lmem;
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
	daeLapackSolver* pSolver = (daeLapackSolver*)ida_mem->ida_lmem;
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
	daeLapackSolver* pSolver = (daeLapackSolver*)ida_mem->ida_lmem;
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
	daeLapackSolver* pSolver = (daeLapackSolver*)ida_mem->ida_lmem;
	if(!pSolver)
		return IDA_MEM_NULL;
	
	int ret = pSolver->Free(ida_mem);

	delete pSolver;
	ida_mem->ida_lmem = NULL;

	return ret;
}

}
}


