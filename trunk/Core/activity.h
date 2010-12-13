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
/*********************************************************************
	daeActivity_t
*********************************************************************/
class daeActivity_t
{
public:
	virtual ~daeActivity_t(void){}

public:
	virtual daeModel_t*			GetModel(void) const			= 0;
	virtual void				SetModel(daeModel_t* pModel)	= 0;
	virtual daeDataReporter_t*	GetDataReporter(void) const		= 0;
	virtual daeLog_t*			GetLog(void) const				= 0;
	virtual void				Run(void)						= 0;
};

/*********************************************************************
	daeDynamicActivity_t
*********************************************************************/
enum daeeActivityAction
{
	eAAUnknown = 0,
	eRunActivity,
	ePauseActivity
};

class daeDynamicActivity_t : public daeActivity_t
{
public:
	virtual void					ReportData(void)								= 0;
	virtual void					SetTimeHorizon(real_t dTimeHorizon)				= 0;
	virtual real_t					GetTimeHorizon(void) const						= 0;
	virtual void					SetReportingInterval(real_t dReportingInterval)	= 0;
	virtual real_t					GetReportingInterval(void) const				= 0;
	virtual void					Resume(void)									= 0;
	virtual void					Pause(void)										= 0;
	virtual daeeActivityAction		GetActivityAction(void) const					= 0;
};

/*********************************************************************
	daeDynamicSimulation_t
*********************************************************************/
class daeDynamicSimulation_t : public daeDynamicActivity_t
{
public:
	virtual void				Initialize(daeDAESolver_t* pDAESolver, 
										   daeDataReporter_t* pDataReporter, 
										   daeLog_t* pLog)								= 0;
	virtual void				Reinitialize(void)										= 0;
	virtual void				SolveInitial(void)										= 0;
	virtual daeDAESolver_t*		GetDAESolver(void) const								= 0;
	
	virtual real_t				Integrate(daeeStopCriterion eStopCriterion)				= 0;
	virtual real_t				IntegrateForTimeInterval(real_t time_interval)			= 0;
	virtual real_t				IntegrateUntilTime(real_t time, 
												   daeeStopCriterion eStopCriterion)	= 0;
	virtual void				SetUpParametersAndDomains(void)							= 0;
	virtual void				SetUpVariables(void)									= 0;
};

/*********************************************************************
	daeDynamicOptimization_t
*********************************************************************/
class daeDynamicOptimization_t : public daeDynamicActivity_t
{
public:
};

/*********************************************************************
	daeDynamicParameterEstimation_t
*********************************************************************/
class daeDynamicParameterEstimation_t : public daeDynamicActivity_t
{
public:
};


/*********************************************************************
	daeSteadyStateActivity_t
*********************************************************************/
class daeSteadyStateActivity_t : public daeActivity_t
{
public:
};

/*********************************************************************
	daeSteadyStateSimulation_t
*********************************************************************/
class daeSteadyStateSimulation_t : public daeSteadyStateActivity_t
{
public:
};

/*********************************************************************
	daeSteadyStateOptimization_t
*********************************************************************/
class daeSteadyStateOptimization_t : public daeSteadyStateActivity_t
{
public:
};

/*********************************************************************
	daeSteadyStateParameterEstimation_t
*********************************************************************/
class daeSteadyStateParameterEstimation_t : public daeSteadyStateActivity_t
{
public:
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

	virtual daeDynamicSimulation_t*					CreateDynamicSimulation(const string& strClass)					= 0;
	virtual daeDynamicOptimization_t*				CreateDynamicOptimization(const string& strClass)				= 0;
	virtual daeDynamicParameterEstimation_t*		CreateDynamicParameterEstimation(const string& strClass)		= 0;
	virtual daeSteadyStateSimulation_t*				CreateSteadyStateSimulation(const string& strClass)				= 0;
	virtual daeSteadyStateOptimization_t*			CreateSteadyStateOptimization(const string& strClass)			= 0;
	virtual daeSteadyStateParameterEstimation_t*	CreateSteadyStateParameterEstimation(const string& strClass)	= 0;

	virtual void SupportedDynamicSimulations(std::vector<string>& strarrClasses)				= 0;
	virtual void SupportedDynamicOptimizations(std::vector<string>& strarrClasses)			= 0;
	virtual void SupportedDynamicParameterEstimations(std::vector<string>& strarrClasses)	= 0;
	virtual void SupportedSteadyStateSimulations(std::vector<string>& strarrClasses)			= 0;
	virtual void SupportedSteadyStateOptimizations(std::vector<string>& strarrClasses)		= 0;
	virtual void SupportedSteadyStateParameterEstimations(std::vector<string>& strarrClasses)= 0;
};
typedef daeActivityClassFactory_t* (*pfnGetActivityClassFactory)(void);


}
}

#endif
