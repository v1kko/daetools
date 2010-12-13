#include "stdafx.h"
#include "dyn_simulation.h"
#include "../Core/macros.h"

namespace dae 
{
namespace activity 
{
/*********************************************************************************************
	daeActivityClassFactory
**********************************************************************************************/
// I cannot have declared exported functions in a static library
#ifdef DAEDLL
daeDeclareActivityLibrary("DAE.Activity",
						  "DAE Tools Activity library",
						  daeAuthorInfo,
						  daeLicenceInfo,
					      daeVersion)

daeRegisterDynamicSimulation(daeDynamicSimulation)
#endif

daeActivityClassFactory::daeActivityClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion)
{
    m_strName			= strName;
    m_strDescription	= strDescription;
    m_strAuthorInfo		= strAuthorInfo;
    m_strLicenceInfo	= strLicenceInfo;
    m_strVersion		= strVersion;
}

daeActivityClassFactory::~daeActivityClassFactory()
{
}

string daeActivityClassFactory::GetName(void) const
{
	return m_strName;
}

string daeActivityClassFactory::GetDescription(void) const
{
	return m_strDescription;
}

string daeActivityClassFactory::GetAuthorInfo(void) const
{
	return m_strAuthorInfo;
}

string daeActivityClassFactory:: GetLicenceInfo(void) const
{
	return m_strLicenceInfo;
}

string daeActivityClassFactory::GetVersion(void) const
{
	return m_strVersion;
}

void daeActivityClassFactory::SupportedDynamicSimulations(vector<string>& strarrClasses)
{
	map<string, pfnCreateDynamicSimulation>::iterator it;
	for(it = m_mapCreateDynamicSimulation.begin(); it != m_mapCreateDynamicSimulation.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedDynamicOptimizations(vector<string>& strarrClasses)
{
	map<string, pfnCreateDynamicOptimization>::iterator it;
	for(it = m_mapCreateDynamicOptimization.begin(); it != m_mapCreateDynamicOptimization.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedDynamicParameterEstimations(vector<string>& strarrClasses)
{
	map<string, pfnCreateDynamicParameterEstimation>::iterator it;
	for(it = m_mapCreateDynamicParameterEstimation.begin(); it != m_mapCreateDynamicParameterEstimation.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedSteadyStateSimulations(vector<string>& strarrClasses)
{
	map<string, pfnCreateSteadyStateSimulation>::iterator it;
	for(it = m_mapCreateSteadyStateSimulation.begin(); it != m_mapCreateSteadyStateSimulation.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedSteadyStateOptimizations(vector<string>& strarrClasses)
{
	map<string, pfnCreateSteadyStateOptimization>::iterator it;
	for(it = m_mapCreateSteadyStateOptimization.begin(); it != m_mapCreateSteadyStateOptimization.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedSteadyStateParameterEstimations(vector<string>& strarrClasses)
{
	map<string, pfnCreateSteadyStateParameterEstimation>::iterator it;
	for(it = m_mapCreateSteadyStateParameterEstimation.begin(); it != m_mapCreateSteadyStateParameterEstimation.end(); it++)
		strarrClasses.push_back((*it).first);
}

daeDynamicSimulation_t* daeActivityClassFactory::CreateDynamicSimulation(const string& strClass)
{
	map<string, pfnCreateDynamicSimulation>::iterator it = m_mapCreateDynamicSimulation.find(strClass);
	if(it == m_mapCreateDynamicSimulation.end())
		return NULL;
	return (*it).second->Create();
}

daeDynamicOptimization_t* daeActivityClassFactory::CreateDynamicOptimization(const string& strClass)
{
	map<string, pfnCreateDynamicOptimization>::iterator it = m_mapCreateDynamicOptimization.find(strClass);
	if(it == m_mapCreateDynamicOptimization.end())
		return NULL;
	return (*it).second->Create();
}

daeDynamicParameterEstimation_t* daeActivityClassFactory::CreateDynamicParameterEstimation(const string& strClass)
{
	map<string, pfnCreateDynamicParameterEstimation>::iterator it = m_mapCreateDynamicParameterEstimation.find(strClass);
	if(it == m_mapCreateDynamicParameterEstimation.end())
		return NULL;
	return (*it).second->Create();
}

daeSteadyStateSimulation_t* daeActivityClassFactory::CreateSteadyStateSimulation(const string& strClass)
{
	map<string, pfnCreateSteadyStateSimulation>::iterator it = m_mapCreateSteadyStateSimulation.find(strClass);
	if(it == m_mapCreateSteadyStateSimulation.end())
		return NULL;
	return (*it).second->Create();
}

daeSteadyStateOptimization_t* daeActivityClassFactory::CreateSteadyStateOptimization(const string& strClass)
{
	map<string, pfnCreateSteadyStateOptimization>::iterator it = m_mapCreateSteadyStateOptimization.find(strClass);
	if(it == m_mapCreateSteadyStateOptimization.end())
		return NULL;
	return (*it).second->Create();
}

daeSteadyStateParameterEstimation_t* daeActivityClassFactory::CreateSteadyStateParameterEstimation(const string& strClass)
{
	map<string, pfnCreateSteadyStateParameterEstimation>::iterator it = m_mapCreateSteadyStateParameterEstimation.find(strClass);
	if(it == m_mapCreateSteadyStateParameterEstimation.end())
		return NULL;
	return (*it).second->Create();
}



bool daeActivityClassFactory::RegisterDynamicSimulation(string strClass, pfnCreateDynamicSimulation pfn)
{
	pair<string, pfnCreateDynamicSimulation> p(strClass, pfn);
	pair<map<string, pfnCreateDynamicSimulation>::iterator, bool> ret;
	ret = m_mapCreateDynamicSimulation.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterDynamicOptimization(string strClass, pfnCreateDynamicOptimization pfn)
{
	pair<string, pfnCreateDynamicOptimization> p(strClass, pfn);
	pair<map<string, pfnCreateDynamicOptimization>::iterator, bool> ret;
	ret = m_mapCreateDynamicOptimization.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterDynamicParameterEstimation(string strClass, pfnCreateDynamicParameterEstimation pfn)
{
	pair<string, pfnCreateDynamicParameterEstimation> p(strClass, pfn);
	pair<map<string, pfnCreateDynamicParameterEstimation>::iterator, bool> ret;
	ret = m_mapCreateDynamicParameterEstimation.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterSteadyStateSimulation(string strClass, pfnCreateSteadyStateSimulation pfn)
{
	pair<string, pfnCreateSteadyStateSimulation> p(strClass, pfn);
	pair<map<string, pfnCreateSteadyStateSimulation>::iterator, bool> ret;
	ret = m_mapCreateSteadyStateSimulation.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterSteadyStateOptimization(string strClass, pfnCreateSteadyStateOptimization pfn)
{
	pair<string, pfnCreateSteadyStateOptimization> p(strClass, pfn);
	pair<map<string, pfnCreateSteadyStateOptimization>::iterator, bool> ret;
	ret = m_mapCreateSteadyStateOptimization.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterSteadyStateParameterEstimation(string strClass, pfnCreateSteadyStateParameterEstimation pfn)
{
	pair<string, pfnCreateSteadyStateParameterEstimation> p(strClass, pfn);
	pair<map<string, pfnCreateSteadyStateParameterEstimation>::iterator, bool> ret;
	ret = m_mapCreateSteadyStateParameterEstimation.insert(p);
	return ret.second;
}




}
}
