#ifndef DYN_SIMULATION_H
#define DYN_SIMULATION_H

#include "activity_class_factory.h"
#include "../Core/coreimpl.h"
#include "../config.h"

namespace dae
{
namespace activity
{
/*********************************************************************************************
	daeDynamicSimulation
**********************************************************************************************/
class DAE_ACTIVITY_API daeDynamicSimulation : public daeDynamicSimulation_t
{
public:
	daeDynamicSimulation(void);
	virtual ~daeDynamicSimulation(void);

public:
// daeActivity_t
	virtual daeModel_t*			GetModel(void) const;
	virtual void				SetModel(daeModel_t* pModel);
	virtual daeDataReporter_t*	GetDataReporter(void) const;
	virtual daeLog_t*			GetLog(void) const;
	virtual void				Run(void);
	virtual void				Finalize(void);
	virtual void				Reset(void);
	virtual void				ReportData(void);
	virtual void				StoreInitializationValues(const std::string& strFileName) const;
	virtual void				LoadInitializationValues(const std::string& strFileName) const;

// daeDynamicActivity_t
	virtual void				SetTimeHorizon(real_t dTimeHorizon);
	virtual real_t				GetTimeHorizon(void) const;
	virtual void				SetReportingInterval(real_t dReportingInterval);
	virtual real_t				GetReportingInterval(void) const;
	virtual void				Resume(void);
	virtual void				Pause(void);
	virtual daeeActivityAction	GetActivityAction(void) const;

// daeDynamicSimulation_t
	virtual void				Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog);
	virtual void				Reinitialize(void);
	virtual void				SolveInitial(void);
	virtual daeDAESolver_t*		GetDAESolver(void) const;
	virtual real_t				Integrate(daeeStopCriterion eStopCriterion);
	virtual real_t				IntegrateForTimeInterval(real_t time_interval);
	virtual real_t				IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion);
	virtual void				SetUpParametersAndDomains(void);
	virtual void				SetUpVariables(void);
	
	real_t						GetCurrentTime(void) const;
	daeeInitialConditionMode	GetInitialConditionMode(void) const;
	void						SetInitialConditionMode(daeeInitialConditionMode eMode);
	
protected:
	void	SetInitialConditionsToZero(void);
	void	CheckSystem(void) const;

	void	EnterConditionalIntegrationMode(void);
	real_t	IntegrateUntilConditionSatisfied(daeCondition rCondition, daeeStopCriterion eStopCriterion);
	
private:
	void	RegisterModel(daeModel_t* pModel);
	void	RegisterPort(daePort_t* pPort);
	void	RegisterVariable(daeVariable_t* pVariable);
	void	RegisterDomain(daeDomain_t* pDomain);
	void	ReportModel(daeModel_t* pModel, real_t time);
	void	ReportPort(daePort_t* pPort, real_t time);
	void	ReportVariable(daeVariable_t* pVariable, real_t time);

protected:
	real_t						m_dCurrentTime;
	real_t						m_dTimeHorizon;
	real_t						m_dReportingInterval;
	daeLog_t*					m_pLog;
	daeModel_t*					m_pModel;
	daeDataReporter_t*			m_pDataReporter;
	daeDAESolver_t*				m_pDAESolver;
	daePtrVector<daeBlock_t*>	m_ptrarrBlocks;
	daeeActivityAction			m_eActivityAction;
	clock_t						m_ProblemCreation;
	clock_t						m_Initialization;
	clock_t						m_Integration;
	bool						m_bConditionalIntegrationMode;
	bool						m_bIsInitialized;
	bool						m_bIsSolveInitial;
};



}
}

#endif
