/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_DAE_SIMULATOR_H
#define CS_DAE_SIMULATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <string.h>
//#include "../models/cs_model_builder.h"
#include "../models/cs_dae_model.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
#ifdef OpenCS_SIMULATORS_EXPORTS
#define OPENCS_SIMULATORS_API __declspec(dllexport)
#else
#define OPENCS_SIMULATORS_API __declspec(dllimport)
#endif
#else
#define OPENCS_SIMULATORS_API
#endif

namespace cs_dae_simulator
{
const int msgBufferSize = 10240;

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

class OPENCS_SIMULATORS_API daeSimulationOptions
{
public:
    daeSimulationOptions();
    virtual ~daeSimulationOptions(void);

public:
    static daeSimulationOptions& GetConfig();

    void        Load(const std::string& jsonFilePath);
    void        LoadString(const std::string& jsonOptions);
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
    std::string  configFile;
    void*        ptree;
};

class OPENCS_SIMULATORS_API daeModel_t : public cs::csDifferentialEquationModel
{
public:
    daeModel_t();
    virtual ~daeModel_t();

    void EvaluateEquations(real_t time, real_t* equations);
    void EvaluateJacobian(real_t time, real_t inverseTimeStep, cs::csMatrixAccess_t* ma);
    void SetAndSynchroniseData(real_t  time, real_t* values, real_t* time_derivatives);
};

class OPENCS_SIMULATORS_API daeLinearSolver_t
{
public:
    virtual ~daeLinearSolver_t(){}

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem) = 0;
    virtual int Free() = 0;
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  jacobianScaleFactor,
                      real_t* values,
                      real_t* timeDerivatives) = 0;
    virtual int Solve(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  cjratio,
                      real_t* b,
                      real_t* weight,
                      real_t* values,
                      real_t* timeDerivatives) = 0;
};

class OPENCS_SIMULATORS_API daeLinearSolver_Trilinos : public daeLinearSolver_t
{
public:
    daeLinearSolver_Trilinos();
    virtual ~daeLinearSolver_Trilinos();

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  jacobianScaleFactor,
                      real_t* values,
                      real_t* timeDerivatives);
    virtual int Solve(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  cjratio,
                      real_t* b,
                      real_t* weight,
                      real_t* values,
                      real_t* timeDerivatives);
    virtual int Free();

public:
    void* data;
    bool  isODESystem;
};

class OPENCS_SIMULATORS_API daePreconditioner_t
{
public:
    virtual ~daePreconditioner_t(){}

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem) = 0;
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      bool    recomputeJacobian   = true,
                      real_t  jacobianScaleFactor = 1.0) = 0;
    virtual int Solve(real_t  time, real_t* r, real_t* z) = 0;
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv) = 0;
    virtual int Free() = 0;
};

class OPENCS_SIMULATORS_API daePreconditioner_Jacobi : public daePreconditioner_t
{
public:
    daePreconditioner_Jacobi();
    virtual ~daePreconditioner_Jacobi();

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      bool    recomputeJacobian   = true,
                      real_t  jacobianScaleFactor = 1.0);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
    bool  isODESystem;
};

class OPENCS_SIMULATORS_API daePreconditioner_Ifpack : public daePreconditioner_t
{
public:
    daePreconditioner_Ifpack();
    virtual ~daePreconditioner_Ifpack();

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      bool    recomputeJacobian   = true,
                      real_t  jacobianScaleFactor = 1.0);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
    bool  isODESystem;
};

class OPENCS_SIMULATORS_API daePreconditioner_ML : public daePreconditioner_t
{
public:
    daePreconditioner_ML();
    virtual ~daePreconditioner_ML();

    virtual int Initialize(cs::csDifferentialEquationModel_t* model,
                           size_t numberOfVariables,
                           bool   isODESystem);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      bool    recomputeJacobian   = true,
                      real_t  jacobianScaleFactor = 1.0);
    virtual int Solve(real_t  time, real_t* r, real_t* z);
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv);
    virtual int Free();

public:
    void* data;
    bool  isODESystem;
};


class OPENCS_SIMULATORS_API daeSimulation_t;
class OPENCS_SIMULATORS_API csDifferentialEquationSolver_t
{
public:
    virtual ~csDifferentialEquationSolver_t(){}

    virtual void   Initialize(cs::csDifferentialEquationModel_t* model,
                              daeSimulation_t*                   simulation,
                              long                               Neqns,
                              long                               Neqns_local,
                              const real_t*                      initValues,
                              const real_t*                      initDerivatives,
                              const real_t*                      absTolerances,
                              const int*                         variableTypes) = 0;
    virtual void   Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities) = 0;
    virtual void   Free() = 0;

    virtual void   RefreshRootFunctions(int noRoots) = 0;
    virtual void   ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities) = 0;
    virtual real_t Solve(real_t time, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities) = 0;
    virtual void   SolveInitial() = 0;
    virtual void   PrintSolverStats() = 0;
    virtual void   CollectSolverStats() = 0;

public:
    bool    printInfo;
    real_t  currentTime;
    real_t  targetTime;
    real_t  rtol;
    real_t* yval;
    real_t* ypval;
    long    Nequations;

    cs::csDifferentialEquationModel_t*    model;
    daeSimulation_t*                      simulation;
    std::shared_ptr<daePreconditioner_t>  preconditioner;
    std::shared_ptr<daeLinearSolver_t>    lasolver;
    std::map<std::string, double>         stats;
};

enum daeeLinearSolverType
{
    eUnknownLinearSolver,
    eSundialsSpils,
    eThirdPartyLinearSolver
};

class OPENCS_SIMULATORS_API daeSolver_t : public csDifferentialEquationSolver_t
{
public:
    daeSolver_t();
    virtual ~daeSolver_t();

    void   Initialize(cs::csDifferentialEquationModel_t* model,
                      daeSimulation_t* simulation,
                      long Neqns,
                      long Neqns_local,
                      const real_t*  initValues,
                      const real_t* initDerivatives,
                      const real_t*  absTolerances,
                      const int* variableTypes);
    void   Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities);
    void   Free();

    void   RefreshRootFunctions(int noRoots);
    void   ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities);
    real_t Solve(real_t time, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities);
    void   SolveInitial();
    void   PrintSolverStats();
    void   CollectSolverStats();

public:
    int                  integrationMode;
    daeeLinearSolverType linearSolverType;
    /* Opaque pointers */
    void* mem;
    void* yy;
    void* yp;
    char msgBuffer[msgBufferSize];
};

class OPENCS_SIMULATORS_API odeiSolver_t : public csDifferentialEquationSolver_t
{
public:
    odeiSolver_t();
    virtual ~odeiSolver_t();

    void   Initialize(cs::csDifferentialEquationModel_t* model,
                      daeSimulation_t* simulation,
                      long Neqns,
                      long Neqns_local,
                      const real_t*  initValues,
                      const real_t* initDerivatives,
                      const real_t*  absTolerances,
                      const int* variableTypes);
    void   Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities);
    void   Free();

    void   RefreshRootFunctions(int noRoots);
    void   ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities);
    real_t Solve(real_t time, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities);
    void   SolveInitial();
    void   PrintSolverStats();
    void   CollectSolverStats();

public:
    int                  integrationMode;
    int                  linearMultistepMethod;
    int                  nonlinearSolverIteration;
    daeeLinearSolverType linearSolverType;
    /* Opaque pointers */
    void* mem;
    void* yy;
    void* yp;
    char msgBuffer[msgBufferSize];
};

class OPENCS_SIMULATORS_API daeSimulation_t
{
public:
    daeSimulation_t();
    virtual ~daeSimulation_t();

    void   StoreInitializationValues(const char* strFileName);
    void   LoadInitializationValues(const char* strFileName);

    void   Initialize(cs::csDifferentialEquationModel_t* model,
                      cs::csLog_t* plog,
                      cs::csDataReporter_t* pdata_reporter,
                      csDifferentialEquationSolver_t* dae_solver,
                      real_t dStartTime,
                      real_t dTimeHorizon,
                      real_t dReportingInterval,
                      const std::string& strOutputDirectory,
                      bool bCalculateSensitivities = false);
    void   SolveInitial();
    void   Run();
    void   Reinitialize();
    void   Finalize();
    void   ReportData(real_t currentTime);
    real_t Integrate(daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);
    real_t IntegrateForTimeInterval(real_t time_interval, bool bReportDataAroundDiscontinuities);
    real_t IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion, bool bReportDataAroundDiscontinuities);

    void SaveStats();
    void PrintStats();

public:
    cs::csDifferentialEquationModel_t*  model;
    csDifferentialEquationSolver_t*     daesolver;
    cs::csLog_t*                        log;
    cs::csDataReporter_t*               data_reporter;
    bool                calculateSensitivities;
    bool                isInitialized;
    std::string         outputDirectory;
    std::vector<real_t> reportingTimes;
    bool                reportData;
    real_t              currentTime;
    real_t              startTime;
    real_t              timeHorizon;
    real_t              reportingInterval;
    char                msgBuffer[msgBufferSize];
};

}

#endif
