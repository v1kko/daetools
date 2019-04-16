#ifndef DATA_REPORTING_CLASS_FACTORY_H
#define DATA_REPORTING_CLASS_FACTORY_H

#include "../Core/datareporting.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef DATA_REPORTING_EXPORTS
#define DAE_DATAREPORTERS_API __declspec(dllexport)
#else
#define DAE_DATAREPORTERS_API __declspec(dllimport)
#endif
#else
#define DAE_DATAREPORTERS_API
#endif

#else // WIN32
#define DAE_DATAREPORTERS_API
#endif // WIN32

namespace dae
{
namespace datareporting
{
/******************************************************************
	daeDataReportingClassFactory
*******************************************************************/
typedef daeCreateObjectDelegate<daeDataReceiver_t>*		pfnCreateDataReceiver;
typedef daeCreateObjectDelegate<daeDataReporter_t>*		pfnCreateDataReporter;

class DAE_DATAREPORTERS_API daeDataReportingClassFactory : public daeDataReportingClassFactory_t
{
public:
	daeDataReportingClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion);
	virtual ~daeDataReportingClassFactory(void);

public:
    virtual string   GetName(void) const;
    virtual string   GetDescription(void) const;
    virtual string   GetAuthorInfo(void) const;
    virtual string   GetLicenceInfo(void) const;
    virtual string   GetVersion(void) const;

    daeDataReceiver_t*	CreateDataReceiver(const string& strClass);
	daeDataReporter_t*	CreateDataReporter(const string& strClass);

	void SupportedDataReceivers(std::vector<string>& strarrClasses);
	void SupportedDataReporters(std::vector<string>& strarrClasses);

	bool RegisterDataReceiver(string strClass, pfnCreateDataReceiver pfn);
	bool RegisterDataReporter(string strClass, pfnCreateDataReporter pfn);

public:
    string   m_strName;
    string   m_strDescription;
    string   m_strAuthorInfo;
    string   m_strLicenceInfo;
    string   m_strVersion;

	daePtrMap<string, pfnCreateDataReceiver>	m_mapCreateDataReceiver;
	daePtrMap<string, pfnCreateDataReporter>	m_mapCreateDataReporter;
};


}
}
#endif
