#ifndef BD_SIMULATION_H
#define BD_SIMULATION_H

#include "dyn_simulation.h"

namespace dae
{
namespace activity
{
/*********************************************************************************************
	Global functions
**********************************************************************************************/
void* BlockSolver(void* pThreadData);

/*********************************************************************************************
	daeBlockSolverData
**********************************************************************************************/
class ACTIVITY_API daeBlockSolverData
{
public:
	daeBlockSolverData(void);
	virtual ~daeBlockSolverData(void);

public:
	void			SetFailed(void);
	daeDAESolver_t*	GetNextBlockSolver(void);

public:
	real_t						m_dTimeHorizon;
	bool						m_bIntegrationFailure;
	size_t						m_nCurrentBlockIndex;
	daeModel_t*					m_pModel;
#ifdef DAE_MUTEX
	pthread_mutex_t				m_pMutex;
#endif
	vector<daeDAESolver_t*>		m_ptrarrDAEBlockSolvers;
};

/*********************************************************************************************
	daeDynamicSimulation
**********************************************************************************************/
class ACTIVITY_API daeBDDynamicSimulation : public daeDynamicSimulation
{
//public:
//	daeBDDynamicSimulation(void);
//	virtual ~daeBDDynamicSimulation(void);
//
//public:
//	virtual void				Initialize(daeDAESolver_t* pDAESolver, daeDataReporter_t* pDataReporter, daeLog_t* pLog)
//	virtual daeModel_t*			GetModel(void) const;
//	virtual daeDataReporter_t*	GetDataReporter(void) const;
//	virtual daeLog_t*			GetLog(void) const;
//	virtual daeDAESolver_t*		GetDAESolver(void) const;
//
//	virtual void				Run(void);
//
//	virtual void				SetUpParametersAndDomains(void);
//	virtual void				SetUpVariables(void);
//
//protected:
//	void	Integrate(real_t time);
//	real_t	IntegrateUntilDiscontinuity(real_t time);
//	real_t	IntegrateUntil(daeCondition& rCondition);
//	void	ReportData(real_t time);
//
//private:
//	void	ContinueTo(real_t& time, bool bStopAtDiscontinuity);
//	void	RegisterModel(daeModel_t* pModel);
//	void	RegisterPort(daePort_t* pPort);
//	void	RegisterVariable(daeVariable_t* pVariable);
//	void	RegisterDomain(daeDomain_t* pDomain);
//	void	ReportModel(daeModel_t* pModel);
//	void	ReportPort(daePort_t* pPort);
//	void	ReportVariable(daeVariable_t* pVariable);
//
//protected:
//	real_t							m_dTimeHorizon;
//	real_t							m_dReportingInterval;
//	daeLog_t*						m_pLog;
//	daeModel_t*						m_pModel;
//	daeDataReporter_t*				m_pDataReporter;
//	daeDataProxy_t*					m_pDataProxy;
//	daeDAESolver_t*					m_pDAESolver;
//	daePtrVector<daeBlock_t*>		m_ptrarrBlocks;
//
//	//daePtrVector<daeDAESolver_t*>	m_ptrarrDAEBlockSolvers;
//	//daeBlockSolverData				BlockSolverData;
//	//size_t							nNoProcessors;
//	//pthread_t*						pthreads;
};



}
}

#endif
