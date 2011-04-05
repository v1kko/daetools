#ifndef MKL_LAPACK_LA_SOLVER_H
#define MKL_LAPACK_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>

#ifdef daeHasIntelMKL
#include <mkl_lapack.h>
#endif

#ifdef daeHasAmdACML
#include <acml.h>
#endif

namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateLapackSolver(void);

class DAE_SOLVER_API daeLapackSolver : public dae::solver::daeIDALASolver_t
{
public:
	daeLapackSolver();
	~daeLapackSolver();
	
public:
	int Create(void* ida, size_t n, daeDAESolver_t* pDAESolver);
	int Reinitialize(void* ida);
	int SaveAsXPM(const std::string& strFileName);
	int SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription);
	string GetName(void) const;

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
	bool CheckData() const;
	void AllocateMemory(void);
	void FreeMemory(void);
	
public:
	daeBlock_t*			m_pBlock;

// Lapack Solver Data
	int					m_nNoEquations;  
	real_t*				m_matLAPACK;
	int*				m_vecPivot;
	daeDAESolver_t*		m_pDAESolver;
#ifdef daeHasMagma
	real_t*				m_matCUDA;
	real_t*				m_pdB;
#endif
	
	daeDenseArray		m_arrValues;
	daeDenseArray		m_arrTimeDerivatives;
	daeDenseArray		m_arrResiduals;
	daeLapackMatrix		m_matJacobian;
	
	size_t				m_nJacobianEvaluations;
};

}
}

#endif
