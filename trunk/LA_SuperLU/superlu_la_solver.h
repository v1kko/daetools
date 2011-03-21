#ifndef MKL_PARDISO_LA_SOLVER_H
#define MKL_PARDISO_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

extern "C" 
{
#ifdef daeSuperLU_MT
#include <pdsp_defs.h>
#elif daeSuperLU
#include <slu_ddefs.h>
#endif
#include <slu_ddefs.h>
}

namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateSuperLUSolver(void);


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
#elif daeSuperLU
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
	
	daeDenseArray		m_arrValues;
	daeDenseArray		m_arrTimeDerivatives;
	daeDenseArray		m_arrResiduals;
	daeSuperLUMatrix	m_matJacobian;

#ifdef daeSuperLU_MT
    superlumt_options_t	m_Options;
    superlu_memusage_t	m_memUsage;
	equed_t				m_equed;
	SuperMatrix			m_matAC;
    Gstat_t				m_Stats;
	
#elif daeSuperLU
    superlu_options_t	m_Options;
    mem_usage_t			m_memUsage;
    SuperLUStat_t		m_Stats;
    int*				m_etree;
    real_t*				m_R;
    real_t*				m_C;
	char				m_equed;
    real_t				m_ferr;
    real_t				m_berr;
	SuperMatrix			m_matAC;
	
	int					m_lwork;
	void*				m_work;
#endif

	SuperMatrix			m_matA;
	SuperMatrix			m_matB;
	SuperMatrix			m_matX;
	SuperMatrix			m_matL;	
	SuperMatrix			m_matU;	
	
    int*				m_perm_c;
    int*				m_perm_r;
	bool				m_bFactorizationDone;
};

}
}

#endif
