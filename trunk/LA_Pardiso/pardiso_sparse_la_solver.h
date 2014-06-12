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

namespace dae
{
namespace solver
{
/* PARDISO prototypes */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                             double *, int    *,    int *, int *,   int *, int *,
                             int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);

daeIDALASolver_t* daeCreatePardisoSolver(void);

class DAE_SOLVER_API daePardisoSolver : public dae::solver::daeIDALASolver_t
{
public:
    typedef daeCSRMatrix<real_t, int> daeMKLMatrix;

    daePardisoSolver();
    ~daePardisoSolver();

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
    int                 iparm[64];
    double              dparm[64];
    int                 mtype;
    int                 nrhs;
    int                 maxfct;
    int                 mnum;
    int                 phase;
    int                 solver_type;
    int                 msglvl;

    int					m_nNoEquations;
    real_t*				m_vecB;
    daeDAESolver_t*		m_pDAESolver;
    size_t				m_nJacobianEvaluations;

    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t>	m_arrTimeDerivatives;
    daeRawDataArray<real_t>	m_arrResiduals;
    daeMKLMatrix            m_matJacobian;
};

}
}

#endif
