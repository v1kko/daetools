#ifndef DAE_CLASS_FACTORIES_H
#define DAE_CLASS_FACTORIES_H

#include "definitions.h"
#include "core.h"
#include "solver.h"
#include "datareporting.h"
#include "log.h"
#include "activity.h"
using namespace dae::logging;
using namespace dae::core;
using namespace dae::solver;
using namespace dae::datareporting;
using namespace dae::activity;

namespace dae
{
/******************************************************************
	daeClassFactoryManager_t
*******************************************************************/
class daeClassFactoryManager_t
{
public:
	virtual ~daeClassFactoryManager_t(){}

public:
	virtual bool AddLibrary(const string& strLibraryPath) = 0;

	virtual daeVariableType_t* CreateVariableType(const string& strClass, const string& strVersion) const	= 0;
	virtual daePort_t* CreatePort(const string& strClass, const string& strVersion) const					= 0;
	virtual daeModel_t* CreateModel(const string& strClass, const string& strVersion) const					= 0;
	
//	virtual daeParameter_t* CreateParameter(const string& strClass, const string& strVersion) const = 0;
//	virtual daeDomain_t* CreateDomain(const string& strClass, const string& strVersion) const = 0;
//	virtual daeVariable_t* CreateVariable(const string& strClass, const string& strVersion) const = 0;
//	virtual daeEquation_t* CreateEquation(const string& strClass, const string& strVersion) const = 0;
//	virtual daeSTN_t* CreateSTN(const string& strClass, const string& strVersion) const = 0;
//	virtual daeState_t* CreateState(const string& strClass, const string& strVersion) const = 0;
//	virtual daeStateTransition_t* CreateStateTransition(const string& strClass, const string& strVersion) const = 0;
//	virtual daePortConnection_t* CreatePortConnection(const string& strClass, const string& strVersion) const = 0;

	virtual daeDynamicSimulation_t* CreateDynamicSimulation(const string& strClass, const string& strVersion) const = 0;
	virtual daeDynamicOptimization_t* CreateDynamicOptimization(const string& strClass, const string& strVersion) const = 0;
	virtual daeDynamicParameterEstimation_t* CreateDynamicParameterEstimation(const string& strClass, const string& strVersion) const = 0;
	virtual daeSteadyStateSimulation_t* CreateSteadyStateSimulation(const string& strClass, const string& strVersion) const = 0;
	virtual daeSteadyStateOptimization_t* CreateSteadyStateOptimization(const string& strClass, const string& strVersion) const = 0;
	virtual daeSteadyStateParameterEstimation_t* CreateSteadyStateParameterEstimation(const string& strClass, const string& strVersion) const = 0;

	virtual daeDataReceiver_t*	CreateDataReceiver(const string& strClass, const string& strVersion) const = 0;
	virtual daeDataReporter_t*	CreateDataReporter(const string& strClass, const string& strVersion) const = 0;

	virtual daeLASolver_t*	CreateLASolver(const string& strClass, const string& strVersion) const = 0;
	virtual daeNLASolver_t*	CreateNLASolver(const string& strClass, const string& strVersion) const = 0;
	virtual daeDAESolver_t*	CreateDAESolver(const string& strClass, const string& strVersion) const = 0;
	
	virtual void SupportedVariableTypes(std::vector<string>& strarrClasses) const		= 0;
	virtual void SupportedPorts(std::vector<string>& strarrClasses) const				= 0;
	virtual void SupportedModels(std::vector<string>& strarrClasses) const				= 0;

	virtual void SupportedDynamicSimulations(std::vector<string>& strarrClasses) const				= 0;
	virtual void SupportedDynamicOptimizations(std::vector<string>& strarrClasses) const				= 0;
	virtual void SupportedDynamicParameterEstimations(std::vector<string>& strarrClasses) const		= 0;
	virtual void SupportedSteadyStateSimulations(std::vector<string>& strarrClasses) const			= 0;
	virtual void SupportedSteadyStateOptimizations(std::vector<string>& strarrClasses) const			= 0;
	virtual void SupportedSteadyStateParameterEstimations(std::vector<string>& strarrClasses) const	= 0;
	
	virtual void SupportedDataReceivers(std::vector<string>& strarrClasses) const		= 0;
	virtual void SupportedDataReporters(std::vector<string>& strarrClasses) const		= 0;

	virtual void SupportedLASolvers(std::vector<string>& strarrClasses) const	= 0;
	virtual void SupportedNLASolvers(std::vector<string>& strarrClasses) const	= 0;
	virtual void SupportedDAESolvers(std::vector<string>& strarrClasses) const	= 0;
};


}

#endif
