#include "stdafx.h"
#include "datareporters.h"
#include "base_data_reporters_receivers.h"
#include "../Core/macros.h"

namespace dae 
{
namespace datareporting 
{
/*********************************************************************************************
	daeDataReportingClassFactory
**********************************************************************************************/
// I cannot have declared exported functions in a static library
#ifdef DAEDLL
daeDeclareDataReportingLibrary("DAE.DataReporting",
							   "DAE Tools DataReporting library",
							   daeAuthorInfo,
							   daeLicenceInfo,
							   daeVersion);
		
daeRegisterDataReporter(daeTEXTFileDataReporter)
daeRegisterDataReporter(daeHTMLFileDataReporter)

#endif

daeDataReportingClassFactory_t* daeCreateDataReportingClassFactory(void)
{
	static daeDataReportingClassFactory g_DataReportingClassFactory("DAE.DataReporting",
																    "DAE Tools DataReporting library",
																     daeAuthorInfo,
																     daeLicenceInfo,
																     daeVersion());
	
	daeRegisterDataReporter(daeTEXTFileDataReporter)
	daeRegisterDataReporter(daeHTMLFileDataReporter)
	
	return &g_DataReportingClassFactory; \
}

daeDataReportingClassFactory::daeDataReportingClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion)
{
    m_strName			= strName;
    m_strDescription	= strDescription;
    m_strAuthorInfo		= strAuthorInfo;
    m_strLicenceInfo	= strLicenceInfo;
    m_strVersion		= strVersion;
}

daeDataReportingClassFactory::~daeDataReportingClassFactory()
{
}

string daeDataReportingClassFactory::GetName(void) const
{
	return m_strName;
}

string daeDataReportingClassFactory::GetDescription(void) const
{
	return m_strDescription;
}

string daeDataReportingClassFactory::GetAuthorInfo(void) const
{
	return m_strAuthorInfo;
}

string daeDataReportingClassFactory:: GetLicenceInfo(void) const
{
	return m_strLicenceInfo;
}

string daeDataReportingClassFactory::GetVersion(void) const
{
	return m_strVersion;
}

daeDataReceiver_t* daeDataReportingClassFactory::CreateDataReceiver(const string& strClass)
{
	map<string, pfnCreateDataReceiver>::iterator it = m_mapCreateDataReceiver.find(strClass);
	if(it == m_mapCreateDataReceiver.end())
		return NULL;
	return (*it).second->Create();
}

daeDataReporter_t* daeDataReportingClassFactory::CreateDataReporter(const string& strClass)
{
	map<string, pfnCreateDataReporter>::iterator it = m_mapCreateDataReporter.find(strClass);
	if(it == m_mapCreateDataReporter.end())
		return NULL;
	return (*it).second->Create();
}

void daeDataReportingClassFactory::SupportedDataReceivers(vector<string>& strarrClasses)
{
	map<string, pfnCreateDataReceiver>::iterator it;
	for(it = m_mapCreateDataReceiver.begin(); it != m_mapCreateDataReceiver.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeDataReportingClassFactory::SupportedDataReporters(vector<string>& strarrClasses)
{
	map<string, pfnCreateDataReporter>::iterator it;
	for(it = m_mapCreateDataReporter.begin(); it != m_mapCreateDataReporter.end(); it++)
		strarrClasses.push_back((*it).first);
}

bool daeDataReportingClassFactory::RegisterDataReceiver(string strClass, pfnCreateDataReceiver pfn)
{
	pair<string, pfnCreateDataReceiver> p(strClass, pfn);
	pair<map<string, pfnCreateDataReceiver>::iterator, bool> ret;
	ret = m_mapCreateDataReceiver.insert(p);
	return ret.second;
}

bool daeDataReportingClassFactory::RegisterDataReporter(string strClass, pfnCreateDataReporter pfn)
{
	pair<string, pfnCreateDataReporter> p(strClass, pfn);
	pair<map<string, pfnCreateDataReporter>::iterator, bool> ret;
	ret = m_mapCreateDataReporter.insert(p);
	return ret.second;
}


}
}
