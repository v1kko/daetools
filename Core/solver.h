/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2015
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef DAE_SOLVER_H
#define DAE_SOLVER_H

#include "definitions.h"
#include "core.h"
#include "log.h"

namespace daetools
{
namespace activity
{
    class daeSimulation_t;
}
}

namespace daetools
{
namespace solver
{
using namespace daetools::core;
using namespace daetools::logging;
using namespace daetools::activity;

enum daeeIntegrationMode
{
    eNormalIntegration  = 0,
    eOneStepIntegration = 1
};

class daeBlockOfEquations_t
{
public:
    virtual ~daeBlockOfEquations_t(){}

public:
    virtual int  CalcNonZeroElements() = 0;
    virtual void FillSparseMatrix(daeSparseMatrix<real_t>* pmatrix) = 0;
    virtual void CalculateJacobian(real_t				time,
                                   real_t				inverseTimeStep,
                                   daeArray<real_t>&	arrValues,
                                   daeArray<real_t>&	arrResiduals,
                                   daeArray<real_t>&	arrTimeDerivatives,
                                   daeMatrix<real_t>&	matJacobian) = 0;
};

/*********************************************************************************************
    daeLASolver
**********************************************************************************************/
class daeLASolver_t
{
public:
    virtual ~daeLASolver_t(void){}

public:
    virtual std::string GetName(void) const                                     = 0;
    virtual int Create(size_t n, size_t nnz, daeBlockOfEquations_t* block)      = 0;
    virtual int Reinitialize(size_t nnz)                                        = 0;
    virtual int Init()                                                          = 0;
    virtual int Setup(real_t    time,
                      real_t    inverseTimeStep,
                      real_t*	values,
                      real_t*	timeDerivatives,
                      real_t*	residuals)                                      = 0;
    virtual int Solve(real_t    time,
                      real_t    inverseTimeStep,
                      real_t    cjratio,
                      real_t*	b,
                      real_t*	weight,
                      real_t*	values,
                      real_t*	timeDerivatives,
                      real_t*	residuals)                                      = 0;
    virtual int Free()                                                          = 0;
    virtual int SaveAsXPM(const std::string& strFileName)                       = 0;
    virtual int SaveAsMatrixMarketFile(const std::string& strFileName,
                                       const std::string& strMatrixName,
                                       const std::string& strMatrixDescription) = 0;

    virtual std::map<std::string, call_stats::TimeAndCount> GetCallStats() const  = 0;

    virtual void        SetOption_string(const std::string& strName, const std::string& Value)  = 0;
    virtual void        SetOption_float(const std::string& strName, double Value)               = 0;
    virtual void        SetOption_int(const std::string& strName, int Value)                    = 0;
    virtual void        SetOption_bool(const std::string& strName, bool Value)                  = 0;

    virtual std::string GetOption_string(const std::string& strName)                            = 0;
    virtual double      GetOption_float(const std::string& strName)                             = 0;
    virtual int         GetOption_int(const std::string& strName)                               = 0;
    virtual bool        GetOption_bool(const std::string& strName)                              = 0;
};

/*********************************************************************************************
    daePreconditioner_t
**********************************************************************************************/
class daePreconditioner_t
{
public:
    virtual ~daePreconditioner_t(){}

    virtual std::string GetName() const = 0;
    virtual int Initialize(size_t numberOfVariables, daeBlockOfEquations_t* block) = 0;
    virtual int Reinitialize() = 0;
    virtual int Setup(real_t  time,
                      real_t  inverseTimeStep,
                      real_t* values,
                      real_t* timeDerivatives,
                      real_t* residuals) = 0;
    virtual int Solve(real_t  time, real_t* r, real_t* z) = 0;
    virtual int JacobianVectorMultiply(real_t  time, real_t* v, real_t* Jv) = 0;
    virtual int Free() = 0;

    virtual std::map<std::string, call_stats::TimeAndCount> GetCallStats() const = 0;

    virtual void        SetOption_string(const std::string& strName, const std::string& Value)  = 0;
    virtual void        SetOption_float(const std::string& strName, double Value)               = 0;
    virtual void        SetOption_int(const std::string& strName, int Value)                    = 0;
    virtual void        SetOption_bool(const std::string& strName, bool Value)                  = 0;

    virtual std::string GetOption_string(const std::string& strName)                            = 0;
    virtual double      GetOption_float(const std::string& strName)                             = 0;
    virtual int         GetOption_int(const std::string& strName)                               = 0;
    virtual bool        GetOption_bool(const std::string& strName)                              = 0;
};

/*********************************************************************************************
    daeNLASolver
**********************************************************************************************/
class daeNLASolver_t
{
public:
    virtual ~daeNLASolver_t(void){}

public:
    virtual bool Initialize(void) = 0;
};

/*********************************************************************************************
    daeDAESolver
**********************************************************************************************/
class daeDAESolver_t
{
public:
    virtual ~daeDAESolver_t(void){}

public:
    virtual void						Initialize(daeBlock_t* pBlock,
                                                   daeLog_t* pLog,
                                                   daeSimulation_t* pSimulation,
                                                   daeeInitialConditionMode eMode,
                                                   bool bCalculateSensitivities,
                                                   const std::vector<size_t>& narrParametersIndexes)= 0;
    virtual void						Finalize(void)												= 0;
    virtual void						SolveInitial(void)											= 0;
    virtual real_t						Solve(real_t dTime,
                                              daeeStopCriterion eCriterion,
                                              bool bReportDataAroundDiscontinuities = true,
                                              bool takeSingleStep = false)                          = 0;
    virtual size_t						GetNumberOfVariables(void) const                            = 0;
    virtual void						SetRelativeTolerance(real_t relTol)							= 0;
    virtual real_t						GetRelativeTolerance(void) const							= 0;
    virtual daeeInitialConditionMode	GetInitialConditionMode(void) const							= 0;
    virtual void						SetInitialConditionMode(daeeInitialConditionMode eMode)		= 0;
    virtual void                        SetIntegrationMode(daeeIntegrationMode integrationMode)     = 0;
    virtual daeeIntegrationMode         GetIntegrationMode() const                                  = 0;
    virtual daeBlock_t*					GetBlock(void) const										= 0;
    virtual daeLog_t*					GetLog(void) const											= 0;
    virtual daeLASolver_t*				GetLASolver(void) const										= 0;
    virtual daePreconditioner_t*        GetPreconditioner() const                                   = 0;
    virtual void						RefreshEquationSetAndRootFunctions(void)					= 0;
    virtual void						Reinitialize(bool bCopyDataFromBlock,
                                                     bool bResetSensitivities = false)				= 0;
    virtual void						Reset(void)													= 0;
    virtual daeMatrix<real_t>&			GetSensitivities(void)										= 0;
    virtual std::string					GetName(void) const											= 0;
    virtual void                        SetLASolver(daeLASolver_t* pLASolver)                       = 0;
    virtual void                        SetTimeHorizon(real_t timeHorizon)                          = 0;

    virtual void                        OnCalculateResiduals()                                      = 0;
    virtual void                        OnCalculateConditions()                                     = 0;
    virtual void                        OnCalculateJacobian()                                       = 0;
    virtual void                        OnCalculateSensitivityResiduals()                           = 0;

    virtual std::map<std::string, call_stats::TimeAndCount> GetCallStats() const                      = 0;
    virtual std::map<std::string, real_t> GetIntegratorStats()                                      = 0;
};

/*********************************************************************************************
    daeSolverClassFactory_t
**********************************************************************************************/
class daeSolverClassFactory_t
{
public:
    virtual ~daeSolverClassFactory_t(void){}

public:
    virtual string   GetName(void) const			= 0;
    virtual string   GetDescription(void) const		= 0;
    virtual string   GetAuthorInfo(void) const		= 0;
    virtual string   GetLicenceInfo(void) const		= 0;
    virtual string   GetVersion(void) const			= 0;

    virtual daeLASolver_t*	CreateLASolver(const string& strClass)	= 0;
    virtual daeNLASolver_t*	CreateNLASolver(const string& strClass)	= 0;
    virtual daeDAESolver_t*	CreateDAESolver(const string& strClass)	= 0;

    virtual void SupportedLASolvers(std::vector<std::string>& strarrClasses)	= 0;
    virtual void SupportedNLASolvers(std::vector<std::string>& strarrClasses)	= 0;
    virtual void SupportedDAESolvers(std::vector<std::string>& strarrClasses)	= 0;
};
typedef daeSolverClassFactory_t* (*pfnGetSolverClassFactory)(void);

}
}

#endif
