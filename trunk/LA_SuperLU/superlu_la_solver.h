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
#include <slu_ddefs.h>
}

namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateSuperLUSolver(void);

/*
class daeSuperLUMatrix: public daeCSRMatrix<real_t, int>
{
public:
	daeSuperLUMatrix(void)
	{
	}
	
	~daeSuperLUMatrix(void)
	{
		SuperLUFree();
	}
	
	void SuperLUFree(void)
	{
	// Destroy_CompRow_Matrix will delete A, IA and JA as well
		Destroy_CompRow_Matrix(&m_matA);
		A   = NULL;
		IA  = NULL;
		JA  = NULL;
	}
	
	void Reset(int n, int nnz, bool ind)
	{
		SuperLUFree();
		daeCSRMatrix<real_t, int>::Reset(n, nnz, ind);
		dCreate_CompCol_Matrix(&m_matA, N, N, NNZ, A, IA, JA, SLU_NR, SLU_D, SLU_GE);
	}
	
public:
	SuperMatrix m_matA;
};
*/

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

	superlu_options_t& GetOptions(void);
	
protected:
	void InitializeSuperLU(size_t nnz);
	void FreeMemory(void);
	
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

//	real_t*				A;  // values
//	int*				IA; // row indexes data
//	int*				JA; // column indexes
	SuperMatrix			m_matA;
	bool				m_bFactorizationDone;

	SuperMatrix			m_matB;
	SuperMatrix			m_matX;
	SuperMatrix			m_matL;	
	SuperMatrix			m_matU;	
    mem_usage_t			m_memUsage;
    superlu_options_t	m_Options;
    SuperLUStat_t		m_Stats;
	fact_t				m_refactorOption;
    int*				m_perm_c;
    int*				m_perm_r;
    int*				m_etree;
	char				m_equed;
    real_t*				m_R;
    real_t*				m_C;
    real_t				m_ferr;
    real_t				m_berr;
};

}
}

#endif
