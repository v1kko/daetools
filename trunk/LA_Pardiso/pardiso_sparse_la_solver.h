#ifndef MKL_PARDISO_LA_SOLVER_H
#define MKL_PARDISO_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"

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

daeLASolver_t* daeCreatePardisoSolver(void);

class DAE_SOLVER_API daePardisoSolver : public dae::solver::daeLASolver_t
{
public:
    typedef daeCSRMatrix<real_t, int> daeMKLMatrix;

    daePardisoSolver();
    ~daePardisoSolver();

public:
    virtual int Create(size_t n,
                       size_t nnz,
                       daeBlockOfEquations_t* block);
    virtual int Reinitialize(size_t nnz);
    virtual int Init();
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* pdValues,
                      real_t* pdTimeDerivatives,
                      real_t* pdResiduals);
    virtual int Solve(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  cjratio,
                      real_t* b,
                      real_t* weight,
                      real_t* pdValues,
                      real_t* pdTimeDerivatives,
                      real_t* pdResiduals);
    virtual int Free();
    virtual int SaveAsXPM(const std::string& strFileName);
    virtual int SaveAsMatrixMarketFile(const std::string& strFileName, const std::string& strMatrixName, const std::string& strMatrixDescription);

    virtual string GetName(void) const;
    virtual std::map<std::string, call_stats::TimeAndCount> GetCallStats() const;

    virtual void        SetOption_string(const std::string& strName, const std::string& Value);
    virtual void        SetOption_float(const std::string& strName, double Value);
    virtual void        SetOption_int(const std::string& strName, int Value);
    virtual void        SetOption_bool(const std::string& strName, bool Value);

    virtual std::string GetOption_string(const std::string& strName);
    virtual double      GetOption_float(const std::string& strName);
    virtual int         GetOption_int(const std::string& strName);
    virtual bool        GetOption_bool(const std::string& strName);

protected:
    void InitializePardiso(size_t nnz);
    void ResetMatrix(size_t nnz);
    bool CheckData() const;
    void FreeMemory(void);

public:
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
    daeBlockOfEquations_t* m_pBlock;

    std::map<std::string, call_stats::TimeAndCount>  m_stats;

    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t>	m_arrTimeDerivatives;
    daeRawDataArray<real_t>	m_arrResiduals;
    daeMKLMatrix            m_matJacobian;
};

}
}

#endif
