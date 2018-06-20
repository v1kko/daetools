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
#ifndef CS_SIMULATOR_H
#define CS_SIMULATOR_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <string.h>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include "../opencs/opencs.h"
#include "../opencs/dae_model/cs_dae_model.h"
using namespace cs;

namespace cs_dae_simulator
{
#define daeThrowException(MSG) \
   throw std::runtime_error( (boost::format("Exception in %s (%s:%d):\n%s\n") % std::string(__FUNCTION__) % std::string(__FILE__) % __LINE__ % (MSG)).str() );

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

class daeModel_t : public csDAEModelImplementation_t
{
public:
    daeModel_t();
    virtual ~daeModel_t();

    void EvaluateResiduals(real_t time, real_t* residuals);
    void EvaluateJacobian(real_t time, real_t inverseTimeStep, csMatrixAccess_t* ma);
    void SetAndSynchroniseData(real_t  time, real_t* values, real_t* time_derivatives);
};

class daePreconditioner_t
{
public:
    virtual ~daePreconditioner_t(){}

    virtual int Initialize(csDAEModel_t* model, size_t numberOfVariables) = 0;
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

    virtual int Initialize(csDAEModel_t* model, size_t numberOfVariables);
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

    virtual int Initialize(csDAEModel_t* model, size_t numberOfVariables);
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

    virtual int Initialize(csDAEModel_t* model, size_t numberOfVariables);
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

    virtual int Initialize(csDAEModel_t* model, size_t numberOfVariables);
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals);
    virtual int Solve(real_t  time,
                      real_t  inverseTimeStep,
                      real_t  cjratio,
                      real_t* b,
                      real_t* weight,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals);
    virtual int Free();

public:
    void* data;
};

class daeSimulation_t;
class daeIDASolver_t
{
public:
    daeIDASolver_t();
    virtual ~daeIDASolver_t();

    void   Initialize(csDAEModel_t* model,
                      daeSimulation_t* simulation,
                      long Neqns,
                      long Neqns_local,
                      const real_t*  initValues,
                      const real_t* initDerivatives,
                      const real_t*  absTolerances,
                      const int* IDs);
    void   Reinitialize(bool bCopyDataFromBlock, bool bResetSensitivities);
    void   Free();

    void   RefreshRootFunctions(int noRoots);
    void   ResetIDASolver(bool bCopyDataFromBlock, real_t dCurrentTime, bool bResetSensitivities);
    real_t Solve(real_t time, daeeStopCriterion eCriterion, bool bReportDataAroundDiscontinuities);
    void   SolveInitial();
    void   PrintSolverStats();
    void   CollectSolverStats();

public:
    int     integrationMode;
    bool    printInfo;
    real_t  currentTime;
    real_t  targetTime;
    real_t  rtol;
    real_t* yval;
    real_t* ypval;
    long    Nequations;

    csDAEModel_t*                           model;
    daeSimulation_t*                        simulation;
    boost::shared_ptr<daePreconditioner_t>  preconditioner;
    boost::shared_ptr<daeLASolver_t>        lasolver;
    std::map<std::string, double>           stats;

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

    void   Initialize(csDAEModel_t* model,
                      daeIDASolver_t* dae_solver,
                      real_t dStartTime,
                      real_t dTimeHorizon,
                      real_t dReportingInterval,
                      const std::string& strOutputDirectory,
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

    void SaveStats();
    void PrintStats();

public:
    csDAEModel_t*       model;
    daeIDASolver_t*     daesolver;
    bool                calculateSensitivities;
    bool                isInitialized;
    std::string         outputDirectory;
    std::vector<real_t> reportingTimes;
    real_t              currentTime;
    real_t              startTime;
    real_t              timeHorizon;
    real_t              reportingInterval;
};

}

#endif
