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
#include <mkl_types.h>
#include <mkl_pardiso.h>

namespace dae
{
namespace solver
{
daeIDALASolver_t* daeCreateIntelPardisoSolver(void);
	
class DAE_SOLVER_API daeIntelPardisoSolver : public dae::solver::daeIDALASolver_t
{
public:
	typedef daeCSRMatrix<real_t, int> daeMKLMatrix;

	daeIntelPardisoSolver();
	~daeIntelPardisoSolver();
	
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
	void InitializePardiso(size_t nnz);
	void ResetMatrix(size_t nnz);
	bool CheckData() const;
	void FreeMemory(void);
	
public:
	daeBlock_t*			m_pBlock;
	
// Intel Pardiso Solver Data
	void*				pt[64];
	MKL_INT				iparm[64];
	MKL_INT				mtype;
	MKL_INT				nrhs;
	MKL_INT				maxfct;
	MKL_INT				mnum;
	MKL_INT				phase;
	MKL_INT				error;
	MKL_INT				msglvl;
	
	int					m_nNoEquations;  
	real_t*				m_vecB;
	daeDAESolver_t*		m_pDAESolver;
	size_t				m_nJacobianEvaluations;
	
    daeArray<real_t>	m_arrValues;
    daeArray<real_t>	m_arrTimeDerivatives;
    daeArray<real_t>	m_arrResiduals;
	daeMKLMatrix		m_matJacobian;
};

}
}

#endif
