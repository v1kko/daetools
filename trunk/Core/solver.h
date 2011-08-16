/***********************************************************************************
                 DAE Tools Project: www.daetools.com
                 Copyright (C) Dragan Nikolic, 2010
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
using namespace dae::core;
using namespace dae::logging;

namespace dae
{
namespace activity
{
	class daeSimulation_t;
}
}
using namespace dae::activity;


namespace dae
{
namespace solver
{
/*********************************************************************************************
	daeLASolver
**********************************************************************************************/
class daeLASolver_t
{
public:
	virtual ~daeLASolver_t(void){}

public:
	virtual bool		Initialize(void)									= 0;
	virtual std::string GetName(void) const									= 0;
//	virtual void Solve(const daeMatrix& A, const daeArray& B, daeArray& x)	= 0;
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
	virtual void						SolveInitial(void)											= 0;
	virtual real_t						Solve(real_t dTime, 
											  daeeStopCriterion eCriterion, 
											  bool bReportDataAroundDiscontinuities = true)			= 0;
	virtual size_t						GetNumberOfVariables(void) const                            = 0;
	virtual void						SetRelativeTolerance(real_t relTol)							= 0;
	virtual real_t						GetRelativeTolerance(void) const							= 0;
	virtual daeeInitialConditionMode	GetInitialConditionMode(void) const							= 0;
	virtual void						SetInitialConditionMode(daeeInitialConditionMode eMode)		= 0;
	virtual daeBlock_t*					GetBlock(void) const										= 0;
	virtual daeLog_t*					GetLog(void) const											= 0;
	virtual void						RefreshRootFunctions(void)									= 0;
	virtual void						Reinitialize(bool bCopyDataFromBlock)						= 0;
	virtual void						Reset(void)													= 0;
	virtual daeMatrix<real_t>&			GetSensitivities(void)										= 0;
	virtual std::string					GetName(void) const											= 0;
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
