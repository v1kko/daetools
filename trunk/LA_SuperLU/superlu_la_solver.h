#ifndef SUPERLU_LA_SOLVER_H
#define SUPERLU_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include "superlu_solvers.h"

#ifdef daeSuperLU_MT
extern "C"
{
#include <pdsp_defs.h>
}
#endif

#ifdef daeSuperLU_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "superlu_mt_gpu.h"
#endif

#ifdef daeSuperLU
extern "C"
{
#include <slu_ddefs.h>
}
#endif

				
namespace dae
{
namespace solver
{
#ifdef daeSuperLU
namespace superlu
{
#endif
#ifdef daeSuperLU_MT
namespace superlu_mt
{
#endif

class DAE_SOLVER_API daeSuperLUSolver : public dae::solver::daeIDALASolver_t
{
	typedef daeCSRMatrix<real_t, int> daeSuperLUMatrix;
public:
	daeSuperLUSolver();
	~daeSuperLUSolver();
	
public:
	int Create(void* ida, size_t n, daeDAESolver_t* pDAESolver);
	int Reinitialize(void* ida);
	int SaveAsXPM(const std::string& strFileName);
	int SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription);
	string GetName(void) const;
    void SetOpenBLASNoThreads(int n);

	int Init(void* ida);
	int Setup(void* ida,
			  N_Vector	vectorVariables, 
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals,
			  N_Vector	vectorTemp1, 
			  N_Vector	vectorTemp2, 
			  N_Vector	vectorTemp3);
	int Solve(void* ida,
			  N_Vector	b, 
			  N_Vector	weight, 
			  N_Vector	vectorVariables,
			  N_Vector	vectorTimeDerivatives, 
			  N_Vector	vectorResiduals);
	int Free(void* ida);

#ifdef daeSuperLU_MT
	superlumt_options_t& GetOptions(void);
#endif

#ifdef daeSuperLU_CUDA
	void SetMatrixValues_A(void);
	void Factorize(void);
	void Solve(double** b);
	void AllocateMatrix_A(void);
	void AllocateMatrix_B(void);
	void FreeMatrixStore_A(void);
	void FreeMatrixStore_B(void);
	void FreeFactorizationData(void);
	
#endif

#ifdef daeSuperLU
	superlu_options_t& GetOptions(void);
#endif
	
protected:
	void InitializeSuperLU(size_t nnz);
	void FreeMemory(void);
	void PrintStats(void);
	
public:
	daeBlock_t*			m_pBlock;
	int					m_nNoEquations;  
	real_t*				m_vecB;
	real_t*				m_vecX;
	daeDAESolver_t*		m_pDAESolver;
	size_t				m_nJacobianEvaluations;
	
    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t>	m_arrTimeDerivatives;
    daeRawDataArray<real_t>	m_arrResiduals;
    daeSuperLUMatrix        m_matJacobian;
    bool                    m_bFactorizationDone;
	
#ifdef daeSuperLU_MT
	SuperMatrix			m_matA;
	SuperMatrix			m_matB;
	SuperMatrix			m_matX;
	SuperMatrix			m_matL;	
	SuperMatrix			m_matU;	
	SuperMatrix			m_matAC;

    superlumt_options_t	m_Options;
    Gstat_t				m_Stats;

    int					m_lwork;
    void*				m_work;
    int*				m_perm_c;
    int*				m_perm_r;
#endif
	
#ifdef daeSuperLU_CUDA
    superlu_mt_gpu_solver	m_superlu_mt_gpuSolver;
#endif
	
#ifdef daeSuperLU
    SuperMatrix			m_matA;
	SuperMatrix			m_matB;
	SuperMatrix			m_matX;
	SuperMatrix			m_matL;	
	SuperMatrix			m_matU;	
    SuperMatrix			m_matAC;

    superlu_options_t	m_Options;
    mem_usage_t			m_memUsage;
    SuperLUStat_t		m_Stats;
    GlobalLU_t          m_Glu;

    int*				m_etree;
    real_t*				m_R;
    real_t*				m_C;
	char				m_equed;
    real_t				m_ferr;
    real_t				m_berr;
	
	int					m_lwork;
	void*				m_work;
    int*				m_perm_c;
    int*				m_perm_r;
	
	int					m_iFactorization;
	bool				m_bUseUserSuppliedWorkSpace;
	double				m_dWorkspaceMemoryIncrement;
	double				m_dWorkspaceSizeMultiplier;
#endif
	
	double m_solve, m_factorize;
};

#ifdef daeSuperLU
}
#endif
#ifdef daeSuperLU_MT
}
#endif

}
}

#endif
