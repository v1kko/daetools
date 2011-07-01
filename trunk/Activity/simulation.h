#ifndef DYN_SIMULATION_H
#define DYN_SIMULATION_H

#include "activity_class_factory.h"
#include "../Core/coreimpl.h"
#include "../config.h"
#include "../Core/optimization.h"

namespace dae
{
namespace activity
{
/*********************************************************************************************
	daeSimulation
**********************************************************************************************/
class DAE_ACTIVITY_API daeSimulation : public daeSimulation_t
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

	virtual real_t				GetCurrentTime(void) const;
	virtual real_t				GetNextReportingTime(void) const;
	virtual void				SetTimeHorizon(real_t dTimeHorizon);
	virtual real_t				GetTimeHorizon(void) const;
	virtual void				SetReportingInterval(real_t dReportingInterval);
	virtual real_t				GetReportingInterval(void) const;
	virtual void				GetReportingTimes(std::vector<real_t>& darrReportingTimes) const;
	virtual void				SetReportingTimes(const std::vector<real_t>& darrReportingTimes);
	virtual void				Resume(void);
	virtual void				Pause(void);
	virtual daeeActivityAction	GetActivityAction(void) const;

	virtual void				Initialize(daeDAESolver_t* pDAESolver, 
										   daeDataReporter_t* pDataReporter, 
										   daeLog_t* pLog, 
										   bool bCalculateSensitivities = false,
										   size_t nNumberOfObjectiveFunctions = 1);
	
	virtual void				Reinitialize(void);
	virtual void				SolveInitial(void);
	virtual daeDAESolver_t*		GetDAESolver(void) const;
	virtual real_t				Integrate(daeeStopCriterion eStopCriterion);
	virtual real_t				IntegrateForTimeInterval(real_t time_interval);
	virtual real_t				IntegrateUntilTime(real_t time, daeeStopCriterion eStopCriterion);
	virtual void				SetUpParametersAndDomains(void);
	virtual void				SetUpVariables(void);
	virtual void				SetUpOptimization(void);
	
	virtual void GetOptimizationConstraints(std::vector<daeOptimizationConstraint_t*>& ptrarrConstraints) const;
	virtual void GetOptimizationVariables  (std::vector<daeOptimizationVariable_t*>&   ptrarrOptVariables) const;
	virtual void GetObjectiveFunctions(std::vector<daeObjectiveFunction_t*>& ptrarrObjectiveFunctions) const;
	virtual daeObjectiveFunction_t* GetObjectiveFunction(void) const;

	daeeInitialConditionMode	GetInitialConditionMode(void) const;
	void						SetInitialConditionMode(daeeInitialConditionMode eMode);
	
	daeOptimizationConstraint* CreateInequalityConstraint(string strDescription = "");// <= 0
	daeOptimizationConstraint* CreateEqualityConstraint(string strDescription = "");  // == 0

	daeOptimizationVariable* SetContinuousOptimizationVariable(daeVariable& variable, real_t LB, real_t UB, real_t defaultValue);
	daeOptimizationVariable* SetIntegerOptimizationVariable(daeVariable& variable, int LB, int UB, int defaultValue);
	daeOptimizationVariable* SetBinaryOptimizationVariable(daeVariable& variable, bool defaultValue);

	daeOptimizationVariable* SetContinuousOptimizationVariable(adouble a, real_t LB, real_t UB, real_t defaultValue);
	daeOptimizationVariable* SetIntegerOptimizationVariable(adouble a, int LB, int UB, int defaultValue);
	daeOptimizationVariable* SetBinaryOptimizationVariable(adouble a, bool defaultValue);
	
	size_t GetNumberOfObjectiveFunctions(void) const;
	
protected:
//	void	SetInitialConditionsToZero(void);
	void	CheckSystem(void) const;
	void	SetupSolver(void);
	void	CreateObjectiveFunctions(void);

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
	std::vector<real_t>			m_darrReportingTimes;
	daeLog_t*					m_pLog;
	daeModel*					m_pModel;
	daeDataReporter_t*			m_pDataReporter;
	daeDAESolver_t*				m_pDAESolver;
	daePtrVector<daeBlock_t*>	m_ptrarrBlocks;
	daeeActivityAction			m_eActivityAction;
	double						m_ProblemCreationStart;
	double						m_ProblemCreationEnd;
	double						m_InitializationStart;
	double						m_InitializationEnd;
	double						m_IntegrationStart;
	double						m_IntegrationEnd;
	bool						m_bConditionalIntegrationMode;
	bool						m_bIsInitialized;
	bool						m_bIsSolveInitial;
// Optimization related data	
	bool														m_bCalculateSensitivities;
	size_t														m_nNumberOfObjectiveFunctions;
	std::vector< boost::shared_ptr<daeObjectiveFunction> >		m_arrObjectiveFunctions;
	std::vector< boost::shared_ptr<daeOptimizationConstraint> >	m_arrConstraints;
	std::vector< boost::shared_ptr<daeOptimizationVariable> >	m_arrOptimizationVariables;
};

/*********************************************************************************************
	daeOptimization
**********************************************************************************************/
class DAE_ACTIVITY_API daeOptimization :  public daeOptimization_t
{
public:
	daeOptimization(void);
	virtual ~daeOptimization(void);

public:
	virtual void Initialize(daeSimulation_t*   pSimulation,
					        daeNLPSolver_t*    pNLPSolver, 
							daeDAESolver_t*    pDAESolver, 
							daeDataReporter_t* pDataReporter, 
							daeLog_t*          pLog,
							size_t			   nNumberOfObjectiveFunctions = 1);
	virtual void Run(void);
	virtual void Finalize(void);
	
protected:
	daeSimulation_t*			m_pSimulation;
	daeLog_t*					m_pLog;
	daeNLPSolver_t*				m_pNLPSolver;
	daeDataReporter_t*			m_pDataReporter;
	daeDAESolver_t*				m_pDAESolver;
	bool						m_bIsInitialized;
	double						m_Initialization;
	double						m_Optimization;
};


}
}

#endif
