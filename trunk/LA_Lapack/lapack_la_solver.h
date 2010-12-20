#ifndef MKL_LAPACK_LA_SOLVER_H
#define MKL_LAPACK_LA_SOLVER_H

#include "../Solver/ida_la_solver_interface.h"
#include "../Solver/solver_class_factory.h"
#include <idas/idas.h>
#include <idas/idas_impl.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include "../Solver/dae_array_matrix.h"

#ifdef daeHasIntelMKL
#include <mkl_lapack.h>
#endif

#ifdef daeHasAtlas
#include <atlas.h>
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
	int SaveAsPBM(const std::string& strFileName);
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
	bool CheckData() const;
	void AllocateMemory(void);
	void FreeMemory(void);
	
public:
	daeBlock_t*			m_pBlock;

// MKL Lapack Solver Data
	int					m_nNoEquations;  
	real_t*				m_matLAPACK;
	int*				m_vecPivot;
	daeDAESolver_t*		m_pDAESolver;
	
	daeDenseArray		m_arrValues;
	daeDenseArray		m_arrTimeDerivatives;
	daeDenseArray		m_arrResiduals;
	daeLapackMatrix		m_matJacobian;
	
	size_t				m_nJacobianEvaluations;
};

}
}

#endif
