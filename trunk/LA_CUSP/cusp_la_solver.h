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
#include "cusp_solver.h"


namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateCUSPSolver(void);

class DAE_SOLVER_API daeCUSPSolver : public dae::solver::daeIDALASolver_t
{
	typedef daeCSRMatrix<real_t, int> daeCUSPMatrix;
public:
	daeCUSPSolver();
	~daeCUSPSolver();
	
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
	daeCUSPMatrix		m_matJacobian;
	
	cusp_solver			m_cuspSolver;

	bool				m_bFactorizationDone;
};

}
}

#endif
