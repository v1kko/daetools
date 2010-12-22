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

daeRegisterSimulation(daeSimulation)
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

void daeActivityClassFactory::SupportedSimulations(vector<string>& strarrClasses)
{
	map<string, pfnCreateSimulation>::iterator it;
	for(it = m_mapCreateSimulation.begin(); it != m_mapCreateSimulation.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeActivityClassFactory::SupportedOptimizations(vector<string>& strarrClasses)
{
	map<string, pfnCreateOptimization>::iterator it;
	for(it = m_mapCreateOptimization.begin(); it != m_mapCreateOptimization.end(); it++)
		strarrClasses.push_back((*it).first);
}

daeSimulation_t* daeActivityClassFactory::CreateSimulation(const string& strClass)
{
	map<string, pfnCreateSimulation>::iterator it = m_mapCreateSimulation.find(strClass);
	if(it == m_mapCreateSimulation.end())
		return NULL;
	return (*it).second->Create();
}

daeOptimization_t* daeActivityClassFactory::CreateOptimization(const string& strClass)
{
	map<string, pfnCreateOptimization>::iterator it = m_mapCreateOptimization.find(strClass);
	if(it == m_mapCreateOptimization.end())
		return NULL;
	return (*it).second->Create();
}


bool daeActivityClassFactory::RegisterSimulation(string strClass, pfnCreateSimulation pfn)
{
	pair<string, pfnCreateSimulation> p(strClass, pfn);
	pair<map<string, pfnCreateSimulation>::iterator, bool> ret;
	ret = m_mapCreateSimulation.insert(p);
	return ret.second;
}

bool daeActivityClassFactory::RegisterOptimization(string strClass, pfnCreateOptimization pfn)
{
	pair<string, pfnCreateOptimization> p(strClass, pfn);
	pair<map<string, pfnCreateOptimization>::iterator, bool> ret;
	ret = m_mapCreateOptimization.insert(p);
	return ret.second;
}


}
}
