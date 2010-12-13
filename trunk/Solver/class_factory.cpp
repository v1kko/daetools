#include "stdafx.h"
using namespace std;
#include "ida_solver.h"
#include "base_solvers.h"
#include "../Core/macros.h"

namespace dae 
{
namespace solver 
{
/*********************************************************************************************
	daeDataReportingClassFactory
**********************************************************************************************/
// I cannot have declared exported functions in a static library
#ifdef DAEDLL
daeDeclareSolverLibrary("DAE.Solver",
					    "DAE Tools Solver library",
						daeAuthorInfo,
						daeLicenceInfo,
					    daeVersion)

daeRegisterDAESolver(daeIDASolver)
#endif

daeSolverClassFactory_t* daeCreateSolverClassFactory(void)
{
	static daeSolverClassFactory g_SolverClassFactory("DAE.Solver",
													  "DAE Tools Solver library",
													   daeAuthorInfo,
													   daeLicenceInfo,
													   daeVersion());
	
	daeRegisterDAESolver(daeIDASolver)
	
	return &g_SolverClassFactory;
}

daeSolverClassFactory::daeSolverClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion)
{
    m_strName			= strName;
    m_strDescription	= strDescription;
    m_strAuthorInfo		= strAuthorInfo;
    m_strLicenceInfo	= strLicenceInfo;
    m_strVersion		= strVersion;
}

daeSolverClassFactory::~daeSolverClassFactory(void)
{
}

string daeSolverClassFactory::GetName(void) const
{
	return m_strName;
}

string daeSolverClassFactory::GetDescription(void) const
{
	return m_strDescription;
}

string daeSolverClassFactory::GetAuthorInfo(void) const
{
	return m_strAuthorInfo;
}

string daeSolverClassFactory:: GetLicenceInfo(void) const
{
	return m_strLicenceInfo;
}

string daeSolverClassFactory::GetVersion(void) const
{
	return m_strVersion;
}

daeLASolver_t* daeSolverClassFactory::CreateLASolver(const string& strClass)
{
	map<string, pfnCreateLASolver>::iterator it = m_mapCreateLASolver.find(strClass);
	if(it == m_mapCreateLASolver.end())
		return NULL;
	return (*it).second->Create();
}

daeNLASolver_t* daeSolverClassFactory::CreateNLASolver(const string& strClass)
{
	map<string, pfnCreateNLASolver>::iterator it = m_mapCreateNLASolver.find(strClass);
	if(it == m_mapCreateNLASolver.end())
		return NULL;
	return (*it).second->Create();
}

daeDAESolver_t* daeSolverClassFactory::CreateDAESolver(const string& strClass)
{
	map<string, pfnCreateDAESolver>::iterator it = m_mapCreateDAESolver.find(strClass);
	if(it == m_mapCreateDAESolver.end())
		return NULL;
	return (*it).second->Create();
}

void daeSolverClassFactory::SupportedLASolvers(vector<string>& strarrClasses)
{
	map<string, pfnCreateLASolver>::iterator it;
	for(it = m_mapCreateLASolver.begin(); it != m_mapCreateLASolver.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeSolverClassFactory::SupportedNLASolvers(vector<string>& strarrClasses)
{
	map<string, pfnCreateNLASolver>::iterator it;
	for(it = m_mapCreateNLASolver.begin(); it != m_mapCreateNLASolver.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeSolverClassFactory::SupportedDAESolvers(vector<string>& strarrClasses)
{
	map<string, pfnCreateDAESolver>::iterator it;
	for(it = m_mapCreateDAESolver.begin(); it != m_mapCreateDAESolver.end(); it++)
		strarrClasses.push_back((*it).first);
}

bool daeSolverClassFactory::RegisterLASolver(string strClass, pfnCreateLASolver pfn)
{
	pair<string, pfnCreateLASolver> p(strClass, pfn);
	pair<map<string, pfnCreateLASolver>::iterator, bool> ret;
	ret = m_mapCreateLASolver.insert(p);
	return ret.second;
}

bool daeSolverClassFactory::RegisterNLASolver(string strClass, pfnCreateNLASolver pfn)
{
	pair<string, pfnCreateNLASolver> p(strClass, pfn);
	pair<map<string, pfnCreateNLASolver>::iterator, bool> ret;
	ret = m_mapCreateNLASolver.insert(p);
	return ret.second;
}

bool daeSolverClassFactory::RegisterDAESolver(string strClass, pfnCreateDAESolver pfn)
{
	pair<string, pfnCreateDAESolver> p(strClass, pfn);
	pair<map<string, pfnCreateDAESolver>::iterator, bool> ret;
	ret = m_mapCreateDAESolver.insert(p);
	return ret.second;
}


}
}
