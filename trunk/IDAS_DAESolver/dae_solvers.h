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
		m_Ns							= 0;
		m_vectorVariables				= NULL;
		m_vectorTimeDerivatives			= NULL;
		m_vectorAbsTolerances			= NULL;
		m_vectorInitialConditionsTypes	= NULL;
		
//		m_matKrylov						= NULL;
//		m_vectorPivot					= NULL;
//		m_vectorInvMaxElements			= NULL;
		
		m_pvectorSVariables				= NULL;
		m_pvectorSTimeDerivatives		= NULL;
		ppdSValues						= NULL;
		ppdSDValues						= NULL;
		ppdSensResiduals				= NULL;
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
		
		if(m_pvectorSVariables)
			N_VDestroyVectorArray_Serial(m_pvectorSVariables, m_Ns);
		if(m_pvectorSTimeDerivatives)
			N_VDestroyVectorArray_Serial(m_pvectorSTimeDerivatives, m_Ns);
		if(ppdSValues)
			delete[] ppdSValues;
		if(ppdSDValues)
			delete[] ppdSDValues;
		if(ppdSensResiduals)
			delete[] ppdSensResiduals;
		
//		if(m_vectorPivot)
//			free(m_vectorPivot);
//		if(m_vectorInvMaxElements)
//			free(m_vectorInvMaxElements);
//		if(m_matKrylov)
//			DestroyMat(m_matKrylov);
	}

	void CreateSerialArrays(size_t N)
	{
		if(N == 0) 
		{
			daeDeclareException(exInvalidCall);
			e << "Unable to allocate IDA storage; Number of variables is zero";
			throw e;
		}

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
		
		N_VConst(0, m_vectorVariables);
		N_VConst(0, m_vectorTimeDerivatives);
		N_VConst(0, m_vectorInitialConditionsTypes);
		N_VConst(0, m_vectorAbsTolerances);
	}

	void CreateSensitivityArrays(size_t Ns)
	{
		if(Ns == 0) 
		{
			daeDeclareException(exInvalidCall);
			e << "Unable to allocate IDA sensitivity storage; Number of parameters is zero";
			throw e;
		}
		if(!m_vectorVariables || m_N == 0)
		{
			daeDeclareException(exInvalidCall);
			e << "Unable to allocate IDA sensitivity storage; You should first create IDA storage through CreateSerialArrays() function";
			throw e;
		}

		m_Ns = Ns;
	
		ppdSValues = new realtype*[m_Ns];
		m_pvectorSVariables = N_VCloneVectorArray_Serial(m_Ns, m_vectorVariables);
		if(!m_pvectorSVariables) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorSVariables array";
			throw e;
		}
		for(size_t i = 0; i < m_Ns; i++)
			N_VConst(0, m_pvectorSVariables[i]);

		ppdSDValues      = new realtype*[m_Ns];
		ppdSensResiduals = new realtype*[m_Ns];
		
		m_pvectorSTimeDerivatives = N_VCloneVectorArray_Serial(m_Ns, m_vectorVariables);
		if(!m_pvectorSTimeDerivatives) 
		{
			daeDeclareException(exMiscellanous);
			e << "Unable to allocate vectorSTimeDerivatives array";
			throw e;
		}
		for(size_t i = 0; i < m_Ns; i++)
			N_VConst(0, m_pvectorSTimeDerivatives[i]);
	}
	
	void CleanUpSetupData()
	{
		if(m_vectorInitialConditionsTypes)
		{
			N_VDestroy_Serial(m_vectorInitialConditionsTypes);
			m_vectorInitialConditionsTypes = NULL;
		}
		
		if(m_vectorAbsTolerances)
		{
			N_VDestroy_Serial(m_vectorAbsTolerances);
			m_vectorAbsTolerances = NULL;
		}
	}
	
	void SetMaxElements()
	{
//		real_t** m = m_matKrylov->cols;
//		for(size_t i = 0; i < m_N; i++)
//		{
//			if(m[i][i] == 0.0)
//				m_vectorInvMaxElements[i] = 1e-7;
//			else
//				m_vectorInvMaxElements[i] = 1.0 / m[i][i];
//		}
	}

	void CreatePreconditionerArrays(size_t N)
	{
//		m_vectorPivot          = (int*)malloc(N * sizeof(int));
//		m_vectorInvMaxElements = (real_t*)malloc(N * sizeof(real_t));
//		m_matKrylov            = NewDenseMat(N, N);
	}
	
public:
	N_Vector				m_vectorAbsTolerances;
	N_Vector				m_vectorVariables;
	N_Vector				m_vectorTimeDerivatives;
	N_Vector				m_vectorInitialConditionsTypes;

	N_Vector*				m_pvectorSVariables;
	N_Vector*				m_pvectorSTimeDerivatives;

	size_t					m_N;
	size_t					m_Ns;
	realtype**				ppdSValues;
	realtype**				ppdSDValues;
	realtype**				ppdSensResiduals;

	daeCSRMatrix<real_t, int> matJacobian;
	
//	int*					m_vectorPivot;
//	real_t*					m_vectorInvMaxElements;
//	DlsMat					m_matKrylov;
};
	

}
}

#endif // DAE_SOLVERS_H
