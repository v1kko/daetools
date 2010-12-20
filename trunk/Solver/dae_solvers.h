#ifndef DAE_SOLVERS_H
#define DAE_SOLVERS_H

#include <idas/idas.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <idas/idas_dense.h>
#include "dae_array_matrix.h"

namespace dae
{
namespace solver
{
class daeIDASolverData
{
public:
	daeIDASolverData(void)
	{
		m_N								= 0;
		m_vectorVariables				= NULL;
		m_vectorTimeDerivatives			= NULL;
		m_vectorAbsTolerances			= NULL;
		m_vectorInitialConditionsTypes	= NULL;
		m_matKrylov						= NULL;
		m_vectorPivot					= NULL;
		m_vectorInvMaxElements			= NULL;
	}
	
	~daeIDASolverData(void)
	{
		DestroySerialArrays();
	}

	void DestroySerialArrays(void)
	{
		if(m_vectorVariables) 
			N_VDestroy_Serial(m_vectorVariables);
		if(m_vectorTimeDerivatives) 
			N_VDestroy_Serial(m_vectorTimeDerivatives);
		if(m_vectorInitialConditionsTypes) 
			N_VDestroy_Serial(m_vectorInitialConditionsTypes);
		if(m_vectorAbsTolerances) 
			N_VDestroy_Serial(m_vectorAbsTolerances);
		if(m_vectorPivot)
			free(m_vectorPivot);
		if(m_vectorInvMaxElements)
			free(m_vectorInvMaxElements);
		if(m_matKrylov)
			DestroyMat(m_matKrylov);
	}

	void CreateSerialArrays(size_t N)
	{
		m_N = N;
		m_vectorVariables = N_VNew_Serial(N);
		if(!m_vectorVariables) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorVariables array";
			throw e;
		}
	
		m_vectorTimeDerivatives = N_VNew_Serial(N);
		if(!m_vectorTimeDerivatives) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorTimeDerivatives array";
			throw e;
		}
	
		m_vectorInitialConditionsTypes = N_VNew_Serial(N);
		if(!m_vectorInitialConditionsTypes) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorInitialConditionsTypes array";
			throw e;
		}
	
		m_vectorAbsTolerances = N_VNew_Serial(N);
		if(!m_vectorAbsTolerances) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorAbsTolerances array";
			throw e;
		}
	}
	
	void SetMaxElements()
	{
		real_t** m = m_matKrylov->cols;
		for(size_t i = 0; i < m_N; i++)
			if(m[i][i] == 0.0)
				m_vectorInvMaxElements[i] = 1e-7;
			else
				m_vectorInvMaxElements[i] = 1.0 / m[i][i];
	}

	void CreatePreconditionerArrays(size_t N)
	{
		m_vectorPivot          = (int*)malloc(N * sizeof(int));
		m_vectorInvMaxElements = (real_t*)malloc(N * sizeof(real_t));
		m_matKrylov            = NewDenseMat(N, N);
	}
	
public:
	N_Vector				m_vectorAbsTolerances;
	N_Vector				m_vectorVariables;
	N_Vector				m_vectorTimeDerivatives;
	N_Vector				m_vectorInitialConditionsTypes;

	size_t					m_N;
	int*					m_vectorPivot;
	real_t*					m_vectorInvMaxElements;
	DlsMat					m_matKrylov;
};
	
int IDA_uBLAS(void* ida, size_t n, void* pUserData);

#ifdef HAS_GNU_GSL
enum eGSLSolver
{
	eLUDecomposition,
	eQRDecomposition,
	eHouseHolder
};
int IDA_dense_GNU_GSL(void* ida, size_t n, eGSLSolver eSolverType, void* pUserData);
#endif



#ifdef HAS_INTEL_MKL
enum daeeMKLSolver
{
	eGeneric
};
int IDA_sparse_MKL_PARDISO_LU(void* ida, size_t n, void* pUserData);
int IDA_sparse_MKL_PARDISO_LU_Reset(void* ida);
int IDA_sparse_MKL_PARDISO_LU_Get_Matrix_Data(void* ida, int& nnz, int** ia, int** ja);
int IDA_sparse_MKL_PARDISO_LU_SaveMatrixAsXPM(void* ida, const std::string& strFilename);
int IDA_sparse_MKL_PARDISO_LU_SaveMatrixAsPBM(void* ida, const std::string& strFilename);

int IDA_dense_MKL_LU(void* ida, size_t n, void* pUserData);

#endif



#ifdef HAS_AMD_ACML
int IDA_dense_ACML_LU(void* ida, size_t n, void* pUserData);
#endif
	

#ifdef HAS_TRILINOS
int IDA_dense_TRILINOS(void* ida, size_t n, void* pUserData);

int IDA_sparse_TRILINOS_AMESOS(void* ida, size_t n, void* pUserData);
int IDA_sparse_TRILINOS_AMESOS_Reset(void* ida, size_t n, void* pUserData);
#endif

}
}

#endif // DAE_SOLVERS_H
