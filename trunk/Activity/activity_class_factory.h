#ifndef ACTIVITY_CLASS_FACTORY_H
#define ACTIVITY_CLASS_FACTORY_H

#include "../Core/activity.h"

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAE_DLL_INTERFACE
#ifdef ACTIVITY_EXPORTS
#define DAE_ACTIVITY_API __declspec(dllexport)
#else
#define DAE_ACTIVITY_API __declspec(dllimport)
#endif
#else
#define DAE_ACTIVITY_API
#endif

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
typedef daeCreateObjectDelegate<daeSimulation_t>* pfnCreateSimulation;

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

	daeSimulation_t*	CreateSimulation(const string& strClass);

	void SupportedSimulations(std::vector<string>& strarrClasses);

	bool RegisterSimulation(string strClass,   pfnCreateSimulation pfn);

public:
    string   m_strName;
    string   m_strDescription;
    string   m_strAuthorInfo;
    string   m_strLicenceInfo;
    string   m_strVersion;

	daePtrMap<string, pfnCreateSimulation>		m_mapCreateSimulation;
};


}
}

#endif
