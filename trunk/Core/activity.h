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
#ifndef DAE_ACTIVITY_H
#define DAE_ACTIVITY_H

#include "definitions.h"
#include "core.h"
#include "solver.h"
#include "datareporting.h"
#include "log.h"
using namespace dae::logging;
using namespace dae::core;
using namespace dae::solver;
using namespace dae::datareporting;

namespace dae
{
namespace activity
{
enum daeeActivityAction
{
	eAAUnknown = 0,
	eRunActivity,
	ePauseActivity
};

/*********************************************************************
	daeSimulation_t
*********************************************************************/
class daeSimulation_t
{
public:
	virtual daeModel_t*			GetModel(void) const											= 0;
	virtual void				SetModel(daeModel_t* pModel)									= 0;
	virtual daeDataReporter_t*	GetDataReporter(void) const										= 0;
	virtual daeLog_t*			GetLog(void) const												= 0;
	virtual void				SetUpParametersAndDomains(void)									= 0;
	virtual void				SetUpVariables(void)											= 0;
	virtual void				SetUpOptimization(void)											= 0;
	virtual void				Run(void)														= 0;
	virtual void				Finalize(void)													= 0;
	virtual void				ReRun(void)													    = 0;
	virtual void				Reset(void)														= 0;
	virtual void				ReportData(void)												= 0;
	virtual void				StoreInitializationValues(const std::string& strFileName) const	= 0;
	virtual void				LoadInitializationValues(const std::string& strFileName) const	= 0;

	virtual void				SetTimeHorizon(real_t dTimeHorizon)				= 0;
	virtual real_t				GetTimeHorizon(void) const						= 0;
	virtual void				SetReportingInterval(real_t dReportingInterval)	= 0;
	virtual real_t				GetReportingInterval(void) const				= 0;
	virtual void				Resume(void)									= 0;
	virtual void				Pause(void)										= 0;
	virtual daeeActivityAction	GetActivityAction(void) const					= 0;

	virtual void				Initialize(daeDAESolver_t* pDAESolver, 
										   daeDataReporter_t* pDataReporter, 
										   daeLog_t* pLog)						    	= 0;
	virtual void				InitializeOptimization(daeDAESolver_t* pDAESolver, 
										               daeDataReporter_t* pDataReporter, 
										               daeLog_t* pLog)					= 0;
	virtual void				Reinitialize(void)										= 0;
	virtual void				SolveInitial(void)										= 0;
	virtual daeDAESolver_t*		GetDAESolver(void) const								= 0;
	
	virtual real_t				Integrate(daeeStopCriterion eStopCriterion)				= 0;
	virtual real_t				IntegrateForTimeInterval(real_t time_interval)			= 0;
	virtual real_t				IntegrateUntilTime(real_t time, 
												   daeeStopCriterion eStopCriterion)	= 0;
};

/******************************************************************
	daeActivityClassFactory_t
*******************************************************************/
class daeActivityClassFactory_t
{
public:
	virtual ~daeActivityClassFactory_t(void){}

public:
    virtual string   GetName(void) const			= 0;
    virtual string   GetDescription(void) const		= 0;
    virtual string   GetAuthorInfo(void) const		= 0;
    virtual string   GetLicenceInfo(void) const		= 0;
    virtual string   GetVersion(void) const			= 0;

	virtual daeSimulation_t*	CreateSimulation(const string& strClass)	= 0;

	virtual void SupportedSimulations(std::vector<string>& strarrClasses)	= 0;
};
typedef daeActivityClassFactory_t* (*pfnGetActivityClassFactory)(void);


}
}

#endif
