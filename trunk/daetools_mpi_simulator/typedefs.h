/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_MPI_TYPEDEFS_H
#define DAE_MPI_TYPEDEFS_H

#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>

#ifndef real_t
#define real_t double
#endif

namespace daetools_mpi
{
#define daeThrowException(MSG) \
   throw std::runtime_error( (boost::format("Exception in %s (%s:%d):\n%s\n") % std::string(__FUNCTION__) % std::string(__FILE__) % __LINE__ % (MSG)).str() );

typedef enum
{
    eDCTUnknown = 0,
    eGlobalDiscontinuity,
    eModelDiscontinuity,
    eModelDiscontinuityWithDataChange,
    eNoDiscontinuity
} daeeDiscontinuityType;

typedef enum
{
    eStopAtModelDiscontinuity = 0,
    eDoNotStopAtDiscontinuity
} daeeStopCriterion;

typedef enum
{
    eIMUnknown = 0,
    eContinueFor,
    eContinueUntil
} daeeIntegrationMode;

class daeSimulationOptions
{
public:
    daeSimulationOptions();
    virtual ~daeSimulationOptions(void);

public:
    static daeSimulationOptions& GetConfig();

    void        Load(const std::string& jsonFilePath);
    bool        HasKey(const std::string& strPropertyPath) const;
    std::string ToString() const;

    bool        GetBoolean(const std::string& strPropertyPath);
    double      GetFloat  (const std::string& strPropertyPath);
    int         GetInteger(const std::string& strPropertyPath);
    std::string GetString (const std::string& strPropertyPath);

    bool        GetBoolean(const std::string& strPropertyPath, const bool defValue);
    double      GetFloat  (const std::string& strPropertyPath, const double defValue);
    int         GetInteger(const std::string& strPropertyPath, const int defValue);
    std::string GetString (const std::string& strPropertyPath, const std::string& defValue);

public:
    std::string                 configFile;
    boost::property_tree::ptree pt;
};

class daeMatrixAccess_t
{
public:
    virtual ~daeMatrixAccess_t(){}

    virtual void SetItem(size_t row, size_t col, real_t value) = 0;
};

class daeModel_t
{
public:
    daeModel_t();
    virtual ~daeModel_t();

    void Load(const std::string& inputDirectory);
    void Free();

    void CheckSynchronisationIndexes(void* mpi_world, int mpi_rank);
    void SynchroniseData(real_t  time,
                         real_t* values,
                         real_t* time_derivatives);

    void GetDAESystemStructure(int& N, int& NNZ, std::vector<int>& IA, std::vector<int>& JA);

    int EvaluateResiduals(real_t  current_time,
                          real_t* values,
                          real_t* time_derivatives,
                          real_t* residuals);
    int EvaluateJacobian(int                number_of_equations,
                         real_t             current_time,
                         real_t             inverse_time_step,
                         real_t*            values,
                         real_t*            time_derivatives,
                         real_t*            residuals,
                         daeMatrixAccess_t* ma);

    // Not used at the moment.
    int NumberOfRoots();
    int Roots(real_t  current_time,
              real_t* values,
              real_t* time_derivatives,
              real_t* roots);
    bool CheckForDiscontinuities(real_t  current_time,
                                 real_t* values,
                                 real_t* time_derivatives);
    daeeDiscontinuityType ExecuteActions(real_t  current_time,
                                         real_t* values,
                                         real_t* time_derivatives);

protected:
    void InitializeValuesReferences();

public:
    /* MPI-related data */
    void*        mpi_world;
    int          mpi_rank;
    void*        mpi_comm;

    long         Nequations_local; /* Number of equations/state variables in the local node (MPI) */
    long         Ntotal_vars;      /* Total number of variables (including Degrees of Freedom) */
    long         Nequations;       /* Number of equations/state variables */
    long         Ndofs;
    real_t       startTime;
    real_t       timeHorizon;
    real_t       reportingInterval;
    real_t       relativeTolerance;
    bool         quasiSteadyState;

    int*         ids;
    real_t*      initValues;
    real_t*      initDerivatives;
    real_t*      absoluteTolerances;
    const char** variableNames;
    std::string  inputDirectory;
};

class daePreconditioner_t
{
public:
    virtual ~daePreconditioner_t(){}

    virtual int Initialize(daeModel_t* model, size_t numberOfVariables) = 0;
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals) = 0;
    virtual int Solve(real_t  time, real_t* r, real_t* z) = 0;
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv) = 0;
    virtual int Free() = 0;
};

class daePreconditioner_Jacobi : public daePreconditioner_t
{
public:
    daePreconditioner_Jacobi();
    virtual ~daePreconditioner_Jacobi();

    virtual int Initialize(daeModel_t* model, size_t numberOfVariables);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
};

class daePreconditioner_Ifpack : public daePreconditioner_t
{
public:
    daePreconditioner_Ifpack();
    virtual ~daePreconditioner_Ifpack();

    virtual int Initialize(daeModel_t* model, size_t numberOfVariables);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
};

class daePreconditioner_ML : public daePreconditioner_t
{
public:
    daePreconditioner_ML();
    virtual ~daePreconditioner_ML();

    virtual int Initialize(daeModel_t* model, size_t numberOfVariables);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
};

class daeLASolver_t
{
public:
    daeLASolver_t();
    virtual ~daeLASolver_t();

    int Initialize(daeModel_t* model, size_t numberOfVariables);
    int Setup(real_t  time,
              real_t  inverseTimeStep,
              real_t* values,
              real_t* timeDerivatives,
              real_t* residuals);
    int Solve(real_t  time,
              real_t  inverseTimeStep,
              real_t  cjratio,
              real_t* b,
              real_t* weight,
              real_t* values,
              real_t* timeDerivatives,
              real_t* residuals);
    int Free();

public:
    void* data;
};

class daeSimulation_t;
class daeIDASolver_t
{
public:
    daeIDASolver_t();
    virtual ~daeIDASolver_t();

    void   Initialize(daeModel_t* model,
                      daeSimulation_t* simulation,
                      long Neqns,
                      long Neqns_local,
                      const real_t*  initValues,
                      const real_t* initDerivatives,
                      const real_t*  absTolerances,
                      const int* IDs, real_t relativeTolerance);
    void   Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities);
    void   Free();

    void   RefreshRootFunctions(int noRoots);
    void   ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities);
    real_t Solve(real_t time, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities);
    void   SolveInitial();
    void   PrintSolverStats();

public:
    real_t  currentTime;
    real_t  targetTime;
    real_t  rtol;
    real_t* yval;
    real_t* ypval;
    long    Nequations;

    daeModel_t*                             model;
    daeSimulation_t*                        simulation;
    boost::shared_ptr<daePreconditioner_t>  preconditioner;
    boost::shared_ptr<daeLASolver_t>        lasolver;

    /* Opaque pointers */
    void* mem;
    void* yy;
    void* yp;

};

class daeSimulation_t
{
public:
    daeSimulation_t();
    virtual ~daeSimulation_t();

    void   StoreInitializationValues(const char* strFileName);
    void   LoadInitializationValues(const char* strFileName);

    void   Initialize(daeModel_t* model,
                      daeIDASolver_t* dae_solver,
                      const std::string& inputDirectory,
                      bool bCalculateSensitivities = false);
    void   SolveInitial();
    void   Run();
    void   Reinitialize();
    void   Finalize();
    void   ReportData();
    void   ReportData_dx();
    real_t Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);
    real_t IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities);
    real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);

public:
    daeModel_t*         model;
    daeIDASolver_t*     daesolver;
    bool                calculateSensitivities;
    bool                isInitialized;
    real_t              currentTime;
    real_t              timeHorizon;
    real_t              reportingInterval;
    std::string         outputDirectory;
    std::vector<real_t> reportingTimes;

    real_t initDuration;
    real_t solveInitDuration;
    real_t integrationDuration;
};

}

#endif
