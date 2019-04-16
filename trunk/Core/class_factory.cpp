#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeCoreClassFactory
**********************************************************************************************/
// I cannot have declared exported functions in a static library
#ifdef DAE_DLL_INTERFACE
static daeCoreClassFactory g_BaseModelClassFactory("DAE.Core",
												   "DAE Tools Core library",
                                                   daeAuthorInfo,
                                                   daeLicenceInfo,
                                                   daeVersion());
extern "C" DAE_CORE_API daeCoreClassFactory_t* GetCoreClassFactory(void);

daeCoreClassFactory_t* GetCoreClassFactory(void)
{
	return &g_BaseModelClassFactory;
}

bool _r1  = g_BaseModelClassFactory.RegisterPort( string("daePort"),  new daeCreateObjectDelegateDerived<daePort>() );
bool _r2  = g_BaseModelClassFactory.RegisterModel(string("daeModel"), new daeCreateObjectDelegateDerived<daeModel>() );

//bool _r3  = g_BaseModelClassFactory.RegisterParameter(		string("daeParameter"),			new daeCreateObjectDelegateDerived<daeParameter>() );
//bool _r4  = g_BaseModelClassFactory.RegisterDomain(           string("daeDomain"),			new daeCreateObjectDelegateDerived<daeDomain>() );
//bool _r5  = g_BaseModelClassFactory.RegisterSTN(				string("daeSTN"),				new daeCreateObjectDelegateDerived<daeSTN>() );
//bool _r6  = g_BaseModelClassFactory.RegisterIF(				string("daeIF"),				new daeCreateObjectDelegateDerived<daeIF>() );
//bool _r7  = g_BaseModelClassFactory.RegisterEquation(			string("daeEquation"),			new daeCreateObjectDelegateDerived<daeEquation>() );
//bool _r8  = g_BaseModelClassFactory.RegisterState(			string("daeState"),				new daeCreateObjectDelegateDerived<daeState>() );
//bool _r9  = g_BaseModelClassFactory.RegisterStateTransition(	string("daeStateTransition"),	new daeCreateObjectDelegateDerived<daeStateTransition>() );
//bool _r10 = g_BaseModelClassFactory.RegisterModel(			string("daeModel"),				new daeCreateObjectDelegateDerived<daeModel>() );
//bool _r11 = g_BaseModelClassFactory.RegisterPortConnection(   string("daePortConnection"),	new daeCreateObjectDelegateDerived<daePortConnection>() );
//bool _r12 = g_BaseModelClassFactory.RegisterVariable(			string("daeVariable"),			new daeCreateObjectDelegateDerived<daeVariable>() );
#endif

daeCoreClassFactory::daeCoreClassFactory(string strName, string strDescription, string strAuthorInfo, string strLicenceInfo, string strVersion)
{
    m_strName			= strName;
    m_strDescription	= strDescription;
    m_strAuthorInfo		= strAuthorInfo;
    m_strLicenceInfo	= strLicenceInfo;
    m_strVersion		= strVersion;
}

daeCoreClassFactory::~daeCoreClassFactory()
{
}

string daeCoreClassFactory::GetName(void) const
{
	return m_strName;
}

string daeCoreClassFactory::GetDescription(void) const
{
	return m_strDescription;
}

string daeCoreClassFactory::GetAuthorInfo(void) const
{
	return m_strAuthorInfo;
}

string daeCoreClassFactory:: GetLicenceInfo(void) const
{
	return m_strLicenceInfo;
}

string daeCoreClassFactory::GetVersion(void) const
{
	return m_strVersion;
}

void daeCoreClassFactory::SupportedVariableTypes(vector<string>& strarrClasses)
{
	map<string, pfnVariableType>::iterator it;
	for(it = m_mapCreateVariableType.begin(); it != m_mapCreateVariableType.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeCoreClassFactory::SupportedPorts(vector<string>& strarrClasses)
{
	map<string, pfnCreatePort>::iterator it;
	for(it = m_mapCreatePort.begin(); it != m_mapCreatePort.end(); it++)
		strarrClasses.push_back((*it).first);
}

void daeCoreClassFactory::SupportedModels(vector<string>& strarrClasses)
{
	map<string, pfnCreateModel>::iterator it;
	for(it = m_mapCreateModel.begin(); it != m_mapCreateModel.end(); it++)
		strarrClasses.push_back((*it).first);
}

//void daeCoreClassFactory::SupportedParameters(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateParameter>::iterator it;
//	for(it = m_mapCreateParameter.begin(); it != m_mapCreateParameter.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedDomains(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateDomain>::iterator it;
//	for(it = m_mapCreateDomain.begin(); it != m_mapCreateDomain.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedVariables(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateVariable>::iterator it;
//	for(it = m_mapCreateVariable.begin(); it != m_mapCreateVariable.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedEquations(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateEquation>::iterator it;
//	for(it = m_mapCreateEquation.begin(); it != m_mapCreateEquation.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedSTNs(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateSTN>::iterator it;
//	for(it = m_mapCreateSTN.begin(); it != m_mapCreateSTN.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedStates(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateState>::iterator it;
//	for(it = m_mapCreateState.begin(); it != m_mapCreateState.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedStateTransitions(vector<string>& strarrClasses)
//{
//	map<string, pfnCreateStateTransition>::iterator it;
//	for(it = m_mapCreateStateTransition.begin(); it != m_mapCreateStateTransition.end(); it++)
//		strarrClasses.push_back((*it).first);
//}
//
//void daeCoreClassFactory::SupportedPortConnections(vector<string>& strarrClasses)
//{
//	map<string, pfnCreatePortConnection>::iterator it;
//	for(it = m_mapCreatePortConnection.begin(); it != m_mapCreatePortConnection.end(); it++)
//		strarrClasses.push_back((*it).first);
//}

daeVariableType_t* daeCoreClassFactory::CreateVariableType(const string& strClass)
{
	map<string, pfnVariableType>::iterator it = m_mapCreateVariableType.find(strClass);
	if(it == m_mapCreateVariableType.end())
		return NULL;
	return (*it).second->Create();
}

daeModel_t* daeCoreClassFactory::CreateModel(const string& strClass)
{
	map<string, pfnCreateModel>::iterator it = m_mapCreateModel.find(strClass);
	if(it == m_mapCreateModel.end())
		return NULL;
	return (*it).second->Create();
}

daePort_t* daeCoreClassFactory::CreatePort(const string& strClass)
{
	map<string, pfnCreatePort>::iterator it = m_mapCreatePort.find(strClass);
	if(it == m_mapCreatePort.end())
		return NULL;
	return (*it).second->Create();
}

//daeParameter_t* daeCoreClassFactory::CreateParameter(const string& strClass)
//{
//	map<string, pfnCreateParameter>::iterator it = m_mapCreateParameter.find(strClass);
//	if(it == m_mapCreateParameter.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeDomain_t* daeCoreClassFactory::CreateDomain(const string& strClass)
//{
//	map<string, pfnCreateDomain>::iterator it = m_mapCreateDomain.find(strClass);
//	if(it == m_mapCreateDomain.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeVariable_t* daeCoreClassFactory::CreateVariable(const string& strClass)
//{
//	map<string, pfnCreateVariable>::iterator it = m_mapCreateVariable.find(strClass);
//	if(it == m_mapCreateVariable.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeEquation_t* daeCoreClassFactory::CreateEquation(const string& strClass)
//{
//	map<string, pfnCreateEquation>::iterator it = m_mapCreateEquation.find(strClass);
//	if(it == m_mapCreateEquation.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeSTN_t* daeCoreClassFactory::CreateSTN(const string& strClass)
//{
//	map<string, pfnCreateSTN>::iterator it = m_mapCreateSTN.find(strClass);
//	if(it == m_mapCreateSTN.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeState_t* daeCoreClassFactory::CreateState(const string& strClass)
//{
//	map<string, pfnCreateState>::iterator it = m_mapCreateState.find(strClass);
//	if(it == m_mapCreateState.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daeStateTransition_t* daeCoreClassFactory::CreateStateTransition(const string& strClass)
//{
//	map<string, pfnCreateStateTransition>::iterator it = m_mapCreateStateTransition.find(strClass);
//	if(it == m_mapCreateStateTransition.end())
//		return NULL;
//	return (*it).second->Create();
//}
//
//daePortConnection_t* daeCoreClassFactory::CreatePortConnection(const string& strClass)
//{
//	map<string, pfnCreatePortConnection>::iterator it = m_mapCreatePortConnection.find(strClass);
//	if(it == m_mapCreatePortConnection.end())
//		return NULL;
//	return (*it).second->Create();
//}
	
bool daeCoreClassFactory::RegisterVariableType(string strClass, pfnVariableType pfn)
{
	pair<string, pfnVariableType> p(strClass, pfn);
	pair<map<string, pfnVariableType>::iterator, bool> ret;
	ret = m_mapCreateVariableType.insert(p);
	return ret.second;
}

bool daeCoreClassFactory::RegisterPort(string strClass, pfnCreatePort pfn)
{
	pair<string, pfnCreatePort> p(strClass, pfn);
	pair<map<string, pfnCreatePort>::iterator, bool> ret;
	ret = m_mapCreatePort.insert(p);
	return ret.second;
}

bool daeCoreClassFactory::RegisterModel(string strClass, pfnCreateModel pfn)
{
	pair<string, pfnCreateModel> p(strClass, pfn);
	pair<map<string, pfnCreateModel>::iterator, bool> ret;
	ret = m_mapCreateModel.insert(p);
	return ret.second;
}

//bool daeCoreClassFactory::RegisterParameter(string strClass, pfnCreateParameter pfn)
//{
//	pair<string, pfnCreateParameter> p(strClass, pfn);
//	pair<map<string, pfnCreateParameter>::iterator, bool> ret;
//	ret = m_mapCreateParameter.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterDomain(string strClass, pfnCreateDomain pfn)
//{
//	pair<string, pfnCreateDomain> p(strClass, pfn);
//	pair<map<string, pfnCreateDomain>::iterator, bool> ret;
//	ret = m_mapCreateDomain.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterVariable(string strClass, pfnCreateVariable pfn)
//{
//	pair<string, pfnCreateVariable> p(strClass, pfn);
//	pair<map<string, pfnCreateVariable>::iterator, bool> ret;
//	ret = m_mapCreateVariable.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterEquation(string strClass, pfnCreateEquation pfn)
//{
//	pair<string, pfnCreateEquation> p(strClass, pfn);
//	pair<map<string, pfnCreateEquation>::iterator, bool> ret;
//	ret = m_mapCreateEquation.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterSTN(string strClass, pfnCreateSTN pfn)
//{
//	pair<string, pfnCreateSTN> p(strClass, pfn);
//	pair<map<string, pfnCreateSTN>::iterator, bool> ret;
//	ret = m_mapCreateSTN.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterIF(string strClass, pfnCreateIF pfn)
//{
//	pair<string, pfnCreateIF> p(strClass, pfn);
//	pair<map<string, pfnCreateIF>::iterator, bool> ret;
//	ret = m_mapCreateIF.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterState(string strClass, pfnCreateState pfn)
//{
//	pair<string, pfnCreateState> p(strClass, pfn);
//	pair<map<string, pfnCreateState>::iterator, bool> ret;
//	ret = m_mapCreateState.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterStateTransition(string strClass, pfnCreateStateTransition pfn)
//{
//	pair<string, pfnCreateStateTransition> p(strClass, pfn);
//	pair<map<string, pfnCreateStateTransition>::iterator, bool> ret;
//	ret = m_mapCreateStateTransition.insert(p);
//	return ret.second;
//}
//
//bool daeCoreClassFactory::RegisterPortConnection(string strClass, pfnCreatePortConnection pfn)
//{
//	pair<string, pfnCreatePortConnection> p(strClass, pfn);
//	pair<map<string, pfnCreatePortConnection>::iterator, bool> ret;
//	ret = m_mapCreatePortConnection.insert(p);
//	return ret.second;
//}


}
}
