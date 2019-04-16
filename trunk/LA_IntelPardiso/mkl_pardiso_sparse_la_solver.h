#ifndef MKL_PARDISO_LA_SOLVER_H
#define MKL_PARDISO_LA_SOLVER_H

#include "../IDAS_DAESolver/ida_la_solver_interface.h"
#include "../IDAS_DAESolver/solver_class_factory.h"
#include "../IDAS_DAESolver/dae_array_matrix.h"
#include <mkl_types.h>
#include <mkl_pardiso.h>

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAE_DLL_INTERFACE
#ifdef INTEL_PARDISO_EXPORTS
#define DAE_INTEL_PARDISO_API __declspec(dllexport)
#else
#define DAE_INTEL_PARDISO_API __declspec(dllimport)
#endif
#else // DAE_DLL_INTERFACE
#define DAE_INTEL_PARDISO_API
#endif // DAE_DLL_INTERFACE

#else // WIN32
#define DAE_INTEL_PARDISO_API
#endif // WIN32

namespace dae
{
namespace solver
{
DAE_INTEL_PARDISO_API daeLASolver_t* daeCreateIntelPardisoSolver(void);

class DAE_INTEL_PARDISO_API daeIntelPardisoSolver : public dae::solver::daeLASolver_t
{
public:
    typedef daeCSRMatrix<real_t, int> daeMKLMatrix;

    daeIntelPardisoSolver();
    ~daeIntelPardisoSolver();

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

public:
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

    int                     m_nNoEquations;
    real_t*                 m_vecB;
    daeBlockOfEquations_t*  m_pBlock;
    int                     m_no_threads;

    std::map<std::string, call_stats::TimeAndCount>  m_stats;

    daeRawDataArray<real_t>	m_arrValues;
    daeRawDataArray<real_t>	m_arrTimeDerivatives;
    daeRawDataArray<real_t>	m_arrResiduals;
    daeMKLMatrix            m_matJacobian;
};

}
}

#endif
