#ifndef SOLVER_CLASS_FACTORY_H
#define SOLVER_CLASS_FACTORY_H

#if !defined(__MINGW32__) && (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))

#ifdef DAEDLL
#ifdef DAESOLVER_EXPORTS
#define DAE_SOLVER_API __declspec(dllexport)
#else // DAESOLVER_EXPORTS
#define DAE_SOLVER_API __declspec(dllimport)
#endif // DAESOLVER_EXPORTS
#else // DAEDLL
#define DAE_SOLVER_API
#endif // DAEDLL

#else // WIN32
#define DAE_SOLVER_API 
#endif // WIN32

#include "../Core/solver.h"

namespace dae
{
namespace solver
{
/******************************************************************
	daeSolverClassFactory
*******************************************************************/
typedef daeCreateObjectDelegate<daeLASolver_t>*		pfnCreateLASolver;
typedef daeCreateObjectDelegate<daeNLASolver_t>*	pfnCreateNLASolver;
typedef daeCreateObjectDelegate<daeDAESolver_t>*	pfnCreateDAESolver;

class DAE_SOLVER_API daeSolverClassFactory : public daeSolverClassFactory_t
{
public:
	daeSolverClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion);
	virtual ~daeSolverClassFactory(void);

public:
    string   GetName(void) const;
    string   GetDescription(void) const;
    string   GetAuthorInfo(void) const;
    string   GetLicenceInfo(void) const;
    string   GetVersion(void) const;

	daeLASolver_t*	CreateLASolver(const string& strClass);
	daeNLASolver_t*	CreateNLASolver(const string& strClass);
	daeDAESolver_t*	CreateDAESolver(const string& strClass);

	void SupportedLASolvers(std::vector<string>& strarrClasses);
	void SupportedNLASolvers(std::vector<string>& strarrClasses);
	void SupportedDAESolvers(std::vector<string>& strarrClasses);

	bool RegisterLASolver(string strClass, pfnCreateLASolver pfn);
	bool RegisterNLASolver(string strClass, pfnCreateNLASolver pfn);
	bool RegisterDAESolver(string strClass, pfnCreateDAESolver pfn);

public:
    string   m_strName;
    string   m_strDescription;
    string   m_strAuthorInfo;
    string   m_strLicenceInfo;
    string   m_strVersion;

	daePtrMap<string, pfnCreateLASolver>	m_mapCreateLASolver;
	daePtrMap<string, pfnCreateNLASolver>	m_mapCreateNLASolver;
	daePtrMap<string, pfnCreateDAESolver>	m_mapCreateDAESolver;
};


}
}

#endif
