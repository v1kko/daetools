#ifndef ACTIVITY_CLASS_FACTORY_H
#define ACTIVITY_CLASS_FACTORY_H

#include "../Core/activity.h"

#if defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64)

#ifdef DAEDLL
#ifdef SIMULATION_EXPORTS
#define DAE_ACTIVITY_API __declspec(dllexport)
#else // SIMULATION_EXPORTS
#define DAE_ACTIVITY_API __declspec(dllimport)
#endif // SIMULATION_EXPORTS
#else // DAEDLL
#define DAE_ACTIVITY_API
#endif // DAEDLL

#else // WIN32
#define DAE_ACTIVITY_API 
#endif // WIN32


namespace dae
{
namespace activity
{
/******************************************************************
	daeActivityClassFactory_t
*******************************************************************/
typedef daeCreateObjectDelegate<daeDynamicSimulation_t>*				pfnCreateDynamicSimulation;
typedef daeCreateObjectDelegate<daeDynamicOptimization_t>*				pfnCreateDynamicOptimization;
typedef daeCreateObjectDelegate<daeDynamicParameterEstimation_t>*		pfnCreateDynamicParameterEstimation;
typedef daeCreateObjectDelegate<daeSteadyStateSimulation_t>*			pfnCreateSteadyStateSimulation;
typedef daeCreateObjectDelegate<daeSteadyStateOptimization_t>*			pfnCreateSteadyStateOptimization;
typedef daeCreateObjectDelegate<daeSteadyStateParameterEstimation_t>*	pfnCreateSteadyStateParameterEstimation;

class DAE_ACTIVITY_API daeActivityClassFactory : public daeActivityClassFactory_t
{
public:
	daeActivityClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion);
	virtual ~daeActivityClassFactory(void);

public:
    string   GetName(void) const;
    string   GetDescription(void) const;
    string   GetAuthorInfo(void) const;
    string   GetLicenceInfo(void) const;
    string   GetVersion(void) const;

	daeDynamicSimulation_t*					CreateDynamicSimulation(const string& strClass);
	daeDynamicOptimization_t*				CreateDynamicOptimization(const string& strClass);
	daeDynamicParameterEstimation_t*		CreateDynamicParameterEstimation(const string& strClass);
	daeSteadyStateSimulation_t*				CreateSteadyStateSimulation(const string& strClass);
	daeSteadyStateOptimization_t*			CreateSteadyStateOptimization(const string& strClass);
	daeSteadyStateParameterEstimation_t*	CreateSteadyStateParameterEstimation(const string& strClass);

	void SupportedDynamicSimulations(std::vector<string>& strarrClasses);
	void SupportedDynamicOptimizations(std::vector<string>& strarrClasses);
	void SupportedDynamicParameterEstimations(std::vector<string>& strarrClasses);
	void SupportedSteadyStateSimulations(std::vector<string>& strarrClasses);
	void SupportedSteadyStateOptimizations(std::vector<string>& strarrClasses);
	void SupportedSteadyStateParameterEstimations(std::vector<string>& strarrClasses);

	bool RegisterDynamicSimulation(string strClass, pfnCreateDynamicSimulation pfn);
	bool RegisterDynamicOptimization(string strClass, pfnCreateDynamicOptimization pfn);
	bool RegisterDynamicParameterEstimation(string strClass, pfnCreateDynamicParameterEstimation pfn);
	bool RegisterSteadyStateSimulation(string strClass, pfnCreateSteadyStateSimulation pfn);
	bool RegisterSteadyStateOptimization(string strClass, pfnCreateSteadyStateOptimization pfn);
	bool RegisterSteadyStateParameterEstimation(string strClass, pfnCreateSteadyStateParameterEstimation pfn);

public:
    string   m_strName;
    string   m_strDescription;
    string   m_strAuthorInfo;
    string   m_strLicenceInfo;
    string   m_strVersion;

	daePtrMap<string, pfnCreateDynamicSimulation>				m_mapCreateDynamicSimulation;
	daePtrMap<string, pfnCreateDynamicOptimization>				m_mapCreateDynamicOptimization;
	daePtrMap<string, pfnCreateDynamicParameterEstimation>		m_mapCreateDynamicParameterEstimation;
	daePtrMap<string, pfnCreateSteadyStateSimulation>			m_mapCreateSteadyStateSimulation;
	daePtrMap<string, pfnCreateSteadyStateOptimization>			m_mapCreateSteadyStateOptimization;
	daePtrMap<string, pfnCreateSteadyStateParameterEstimation>	m_mapCreateSteadyStateParameterEstimation;
};


}
}

#endif
