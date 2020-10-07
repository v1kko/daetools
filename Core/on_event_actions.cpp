#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace boost;

namespace daetools 
{
namespace core 
{
/*********************************************************************************************
	daeOnEventActions
**********************************************************************************************/
daeOnEventActions::daeOnEventActions(void)
{
	m_pEventPort   = NULL;
	m_pParentState = NULL;
}

daeOnEventActions::daeOnEventActions(daeEventPort* pEventPort, 
									 daeModel* pModel, 
									 vector<daeAction*>& ptrarrOnEventActions, 
									 vector<daeAction*>& ptrarrUserDefinedOnEventActions, 
									 const string& strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	pModel->AddOnEventAction(*this, "OnEventAction_" + pEventPort->GetName(), strDescription);

	m_pEventPort   = pEventPort;
	m_pParentState = NULL;
	
	dae_set_vector(ptrarrOnEventActions,            m_ptrarrOnEventActions);
	dae_set_vector(ptrarrUserDefinedOnEventActions, m_ptrarrUserDefinedOnEventActions);
}

daeOnEventActions::daeOnEventActions(daeEventPort* pEventPort, 
									 daeState* pState, 
									 vector<daeAction*>& ptrarrOnEventActions, 
									 vector<daeAction*>& ptrarrUserDefinedOnEventActions, 
									 const string& strDescription)
{
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	pState->AddOnEventAction(*this, "OnEventAction_" + pEventPort->GetName(), strDescription);

	m_pEventPort   = pEventPort;
	m_pParentState = pState;
	
	dae_set_vector(ptrarrOnEventActions,            m_ptrarrOnEventActions);
	dae_set_vector(ptrarrUserDefinedOnEventActions, m_ptrarrUserDefinedOnEventActions);
}

daeOnEventActions::~daeOnEventActions(void)
{
}

void daeOnEventActions::Clone(const daeOnEventActions& rObject)
{
	m_pEventPort = FindEventPort(m_pEventPort, m_pModel);
	if(!m_pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);

	for(size_t i = 0; i < rObject.m_ptrarrOnEventActions.size(); i++)
	{
		daeAction* pAction = new daeAction();
		pAction->SetName(rObject.m_ptrarrOnEventActions[i]->m_strShortName);
		pAction->Clone(*rObject.m_ptrarrOnEventActions[i]);
	}
	
	dae_set_vector(rObject.m_ptrarrUserDefinedOnEventActions, m_ptrarrUserDefinedOnEventActions);
}

string daeOnEventActions::GetCanonicalName(void) const
{
	if(m_pParentState)
		return m_pParentState->GetCanonicalName() + '.' + m_strShortName;
	else
		return daeObject::GetCanonicalName();
}

daeEventPort* daeOnEventActions::GetEventPort(void) const
{
	return m_pEventPort;
}

const std::vector<daeAction*>& daeOnEventActions::Actions() const
{
    return m_ptrarrOnEventActions;
}

const std::vector<daeAction*>& daeOnEventActions::UserDefinedActions() const
{
    return m_ptrarrUserDefinedOnEventActions;
}

void daeOnEventActions::Initialize(void)
{
	size_t i;
	daeAction* pAction;
	
	for(i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pAction = m_ptrarrOnEventActions[i];
		if(!pAction)
			daeDeclareAndThrowException(exInvalidPointer);

		pAction->Initialize();
	}
}

bool daeOnEventActions::CheckObject(std::vector<string>& strarrErrors) const
{
	bool bReturn = true;

	dae_capacity_check(m_ptrarrOnEventActions);
	dae_capacity_check(m_ptrarrUserDefinedOnEventActions);

	bReturn = daeObject::CheckObject(strarrErrors);

	if(!m_pEventPort)
	{
		strarrErrors.push_back(string("Invalid event port in OnEventActions: ") + GetName());
		return false;
	}
	
	if(m_ptrarrOnEventActions.empty())
	{
		strarrErrors.push_back(string("Actions array cannot be empty in OnEventActions: ") + GetName());
		return false;
	}

	return bReturn;
}

void daeOnEventActions::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);
}

void daeOnEventActions::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "EventPort";
	pTag->SaveObjectRef(strName, m_pEventPort);

	strName = "Actions";
	pTag->SaveObjectArray(strName, m_ptrarrOnEventActions);
}

void daeOnEventActions::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeOnEventActions::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "EventPort";
	pTag->SaveObjectRef(strName, m_pEventPort);

	strName = "Actions";
	pTag->SaveRuntimeObjectArray(strName, m_ptrarrOnEventActions);
}
void daeOnEventActions::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

void daeOnEventActions::Update(daeEventPort_t* pSubject, void* data)
{
	real_t inputData = *(static_cast<real_t*>(data));
	//std::cout << "    Update received in the OnEventActions: " << GetName() << ", data = " << inputData << std::endl;
	
	Execute();
}

void daeOnEventActions::Execute(void)
{
	daeAction* pAction;
	
	for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pAction = m_ptrarrOnEventActions[i];
		if(!pAction)
			daeDeclareAndThrowException(exInvalidPointer);
		
		pAction->Execute();
	}
	
	for(size_t i = 0; i < m_ptrarrUserDefinedOnEventActions.size(); i++)
	{
		pAction = m_ptrarrUserDefinedOnEventActions[i];
		if(!pAction)
			daeDeclareAndThrowException(exInvalidPointer);
		
		pAction->Execute();
	}
}

}
}
