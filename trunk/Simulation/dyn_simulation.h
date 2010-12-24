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
	daeSimulation
**********************************************************************************************/
class DAE_ACTIVITY_API daeSimulation : virtual public daeSimulation_t
{
public:
	daeSimulation(void);
	virtual ~daeSimulation(void);

public:
	virtual daeModel_t*			GetModel(void) const;
	virtual void				SetModel(daeModel_t* pModel);
	virtual daeDataReporter_t*	GetDataReporter(void) const;
	virtual daeLog_t*			GetLog(void) const;
	virtual void				Run(void);
	virtual void				Finalize(void);
	virtual void				Reset(void);
	virtual void				ReRun(void);
	virtual void				ReportData(void);
	virtual void				StoreInitializationValues(const std::string& strFileName) const;
	virtual void				LoadInitializationValues(const std::string& strFileName) const;

	virtual void				SetTimeHorizon(real_t dTimeHorizon);
	virtual real_t				GetTimeHorizon(void) const;
	virtual void				SetReportingInterval(real_t dReportingInterval);
	virtual real_t				GetReportingInterval(void) const;
	virtual void				Resume(void);
	virtual void				Pause(void);
	virtual daeeActivityAction	GetActivityAction(void) const;

	virtual void				InitSimulation(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog);
	virtual void				InitOptimization(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog);
	
	virtual void				Reinitialize(void);
	virtual void				SolveInitial(void);
	virtual daeDAESolver_t*		GetDAESolver(void) const;
	virtual real_t				Integrate(daeeStopCriterion eStopCriterion);
	virtual real_t				IntegrateForTimeInterval(real_t time_interval);
	virtual real_t				IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion);
	virtual void				SetUpParametersAndDomains(void);
	virtual void				SetUpVariables(void);
	virtual void				SetUpOptimization(void);
	
	virtual void GetOptimizationConstraints(std::vector<daeOptimizationConstraint*>& ptrarrConstraints) const;
	virtual void GetOptimizationVariables  (std::vector<daeOptimizationVariable*>&   ptrarrOptVariables) const;
	virtual daeObjectiveFunction* GetObjectiveFunction(void) const;

//	void GetOptimizationConstraints(std::vector< boost::shared_ptr<daeOptimizationConstraint> >& ptrarrConstraints);
//	void GetOptimizationVariables  (std::vector< boost::shared_ptr<daeOptimizationVariable> >&   ptrarrOptVariables);
//	boost::shared_ptr<daeObjectiveFunction> GetObjectiveFunction(void);

	real_t						GetCurrentTime(void) const;
	daeeInitialConditionMode	GetInitialConditionMode(void) const;
	void						SetInitialConditionMode(daeeInitialConditionMode eMode);
	
	daeOptimizationConstraint* CreateInequalityConstraint(real_t LB, real_t UB, string strDescription = "");
	daeOptimizationConstraint* CreateEqualityConstraint(real_t EqualTo, string strDescription = "");
	void SetOptimizationVariable(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue);
	
protected:
	void	Init(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog);
	void	SetInitialConditionsToZero(void);
	void	CheckSystem(void) const;
	void	SetupSolver(void);

	void	EnterConditionalIntegrationMode(void);
	real_t	IntegrateUntilConditionSatisfied(daeCondition rCondition, daeeStopCriterion eStopCriterion);
	
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
// Optimization related data	
	bool														m_bSetupOptimization;
	boost::shared_ptr<daeObjectiveFunction>						m_pObjectiveFunction;
	std::vector< boost::shared_ptr<daeOptimizationConstraint> >	m_arrConstraints;
	std::vector< boost::shared_ptr<daeOptimizationVariable> >	m_arrOptimizationVariables;
};



}
}

#endif
