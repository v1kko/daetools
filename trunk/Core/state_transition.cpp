#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{
/*************************************************
	daeStateTransition
**************************************************/
daeStateTransition::daeStateTransition()
{
	m_pSTN		= NULL;
	m_pModel	= NULL;
}

daeStateTransition::~daeStateTransition()
{
}

void daeStateTransition::Clone(const daeStateTransition& rObject)
{
}

void daeStateTransition::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	daeFindStateByID del(m_pModel);

	strName = "Condition";
	pTag->OpenExistingObject<daeCondition, daeCondition>(strName, &m_Condition);
}

void daeStateTransition::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Condition";
	pTag->SaveObject(strName, &m_Condition);

	strName = "Actions";
	pTag->SaveObjectArray(strName, m_ptrarrActions);
}

void daeStateTransition::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	daeDeclareAndThrowException(exNotImplemented); 

//	string strExport, strCondition;
//	boost::format fmtFile;
//
//	if(c.m_bExportDefinition)
//	{
//	}
//	else
//	{
//		if(eLanguage == ePYDAE)
//		{
//			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "self.SWITCH_TO(\"%1%\", %2%)\n";
//			m_Condition.Export(strCondition, eLanguage, c);
//			
//			fmtFile.parse(strExport);
//			fmtFile % m_pStateTo->GetName()
//					% strCondition;
//		}
//		else if(eLanguage == eCDAE)
//		{
//			strExport = c.CalculateIndent(c.m_nPythonIndentLevel) + "SWITCH_TO(\"%1%\", %2%);\n";
//			m_Condition.Export(strCondition, eLanguage, c);
//			
//			fmtFile.parse(strExport);
//			fmtFile % m_pStateTo->GetName()
//					% strCondition;
//		}
//		else
//		{
//			daeDeclareAndThrowException(exNotImplemented); 
//		}
//	}
//	
//	strContent += fmtFile.str();
}

void daeStateTransition::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeStateTransition::SaveRuntime(io::xmlTag_t* pTag) const
{
	daeObject::SaveRuntime(pTag);
}

void daeStateTransition::Create_SWITCH_TO(daeState* pStateFrom, const string& strStateToName, const daeCondition& rCondition, real_t dEventTolerance)
{
	if(!pStateFrom)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Illegal start state in StateTransition [" << m_strCanonicalName << "], start state [" << pStateFrom->m_strCanonicalName << "]";
		throw e;
	}
	if(strStateToName.empty())
	{	
		daeDeclareException(exInvalidCall); 
		e << "Illegal end state in StateTransition [" << m_strCanonicalName << "], end state [" << strStateToName << "]";
		throw e;
	}

	string strName = "StateTransition_" + toString<size_t>(pStateFrom->m_ptrarrStateTransitions.size());

	m_pSTN				= pStateFrom->GetSTN();
	m_strShortName		= strName;
	m_pModel			= pStateFrom->m_pModel;
	
	daeAction* pAction = new daeAction(string("actionChangeState_") + strStateToName, m_pModel, m_pSTN, strStateToName, string(""));
	dae_push_back(m_ptrarrActions, pAction);
	
	m_Condition			= rCondition;

	m_Condition.m_pModel = m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;
	pStateFrom->AddStateTransition(this);
}

void daeStateTransition::Create_ON_CONDITION(daeState* pStateFrom, 
											 const daeCondition& rCondition, 
											 vector<daeAction*>& ptrarrActions, 
											 vector<daeAction*>& ptrarrUserDefinedActions,
											 real_t dEventTolerance)
{
	if(!pStateFrom)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Illegal start state in StateTransition [" << m_strCanonicalName << "], start state [" << pStateFrom->m_strCanonicalName << "]";
		throw e;
	}
	if(ptrarrActions.empty())
	{	
		daeDeclareException(exInvalidCall); 
		e << "Empty list of actions in StateTransition [" << m_strCanonicalName;
		throw e;
	}

	string strName = "StateTransition_" + toString<size_t>(pStateFrom->m_ptrarrStateTransitions.size());

	m_pSTN			= pStateFrom->GetSTN();
	m_strShortName	= strName;
	m_pModel		= pStateFrom->m_pModel;
	
	m_ptrarrActions.EmptyAndFreeMemory();
	m_ptrarrUserDefinedActions.clear();

	dae_set_vector(ptrarrActions, m_ptrarrActions);
	dae_set_vector(ptrarrUserDefinedActions, m_ptrarrUserDefinedActions);
	
	m_Condition		= rCondition;
	m_Condition.m_pModel = m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;

	pStateFrom->AddStateTransition(this);
}
	
// IF block cannot execute any action, so no need to specify them
void daeStateTransition::Create_IF(daeState* pStateTo, const daeCondition& rCondition, real_t dEventTolerance)
{
	if(!pStateTo)
	{	
		daeDeclareException(exInvalidCall); 
		e << "Illegal end state in StateTransition [" << m_strCanonicalName << "]";
		throw e;
	}

	string strName = "StateTransition_" + toString<size_t>(pStateTo->m_ptrarrStateTransitions.size());

	m_pSTN				= pStateTo->GetSTN();
	m_strShortName		= strName;
	m_pModel			= pStateTo->m_pModel;
	
	m_Condition			= rCondition;

	m_Condition.m_pModel = pStateTo->m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;
	pStateTo->AddStateTransition(this);
}

void daeStateTransition::ExecuteActions(void)
{
	daeAction* pAction;
	
	for(size_t i = 0; i < m_ptrarrActions.size(); i++)
	{
		pAction = m_ptrarrActions[i];
		pAction->Execute();
	}
	for(size_t i = 0; i < m_ptrarrUserDefinedActions.size(); i++)
	{
		pAction = m_ptrarrUserDefinedActions[i];
		pAction->Execute();
	}
}

void daeStateTransition::Initialize(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pModel->m_pExecutionContextForGatherInfo)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_Condition.m_pConditionNode)
		daeDeclareAndThrowException(exInvalidPointer); 
	
// This creates a runtime node from a setup node, and stores the original setup node in m_pSetupConditionNode	
// Global daeExecutionContext (m_pExecutionContextForGatherInfo) should be non-null during this stage
	m_Condition.m_pSetupConditionNode = m_Condition.m_pConditionNode;
	m_Condition.m_pConditionNode      = m_Condition.m_pConditionNode->CreateRuntimeNode(m_pModel->m_pExecutionContextForGatherInfo).m_pConditionNode;
	
	daeAction* pAction;
	for(size_t i = 0; i < m_ptrarrActions.size(); i++)
	{
		pAction = m_ptrarrActions[i];
		pAction->Initialize();
	}
}

daeCondition* daeStateTransition::GetCondition(void)
{
	return &m_Condition;
}

void daeStateTransition::SetCondition(daeCondition& rCondition)
{
	m_Condition = rCondition;
}

string daeStateTransition::GetConditionAsString() const
{
	return m_Condition.SaveNodeAsPlainText();
}

void daeStateTransition::GetActions(vector<daeAction_t*>& ptrarrActions) const
{
	ptrarrActions.clear();
	for(size_t i = 0; i < m_ptrarrActions.size(); i++)
		ptrarrActions.push_back(m_ptrarrActions[i]);
}

bool daeStateTransition::CheckObject(vector<string>& strarrErrors) const
{
	string strError;

	bool bCheck = true;
	
	dae_capacity_check(m_ptrarrActions);

// Check base class	
	if(!daeObject::CheckObject(strarrErrors))
		bCheck = false;

// Check parent STN pointer	
	if(!m_pSTN)
	{
		strError = "Invalid parent state transition network in state transition [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
	
	for(size_t i = 0; i < m_ptrarrActions.size(); i++)
	{
		daeAction* pAction = m_ptrarrActions[i];
		if(!pAction->CheckObject(strarrErrors))
			return false;
	}

// Check condition	
	if(!m_Condition.m_pModel)
	{
		strError = "Invalid parent model in condition in state condition [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	if(!m_Condition.m_pConditionNode)
	{
		strError = "Invalid condition node model in condition in state transition [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}

// Check unit-consistency of the condition
	try
	{
		bool b = m_Condition.m_pSetupConditionNode->GetQuantity();
	}
	catch(units_error& e)
	{
		strError = "Unit-consistency check failed in condition [" + m_Condition.SaveNodeAsPlainText() + "]:";
		strarrErrors.push_back(strError);
		strError = "  " + string(e.what());
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	catch(std::exception& e)
	{
		strError = "Exception occurred during unit-consistency check in condition [" + m_Condition.SaveNodeAsPlainText() + "]:";
		strarrErrors.push_back(strError);
		strError = "  " + string(e.what());
		strarrErrors.push_back(strError);
	}
	
/////////////////////////////////////////////////////////////
// ERROR!!
// Here I dont have expressions since I havent created them yet!
// Thus, I cant check them here!!
/////////////////////////////////////////////////////////////
	//if(m_Condition.m_ptrarrExpressions.empty())
	//{
	//	strError = "Expression map is empty in condition in state transition [" + GetCanonicalName() + "]";
	//	strarrErrors.push_back(strError);
	//	bCheck = false;
	//}

// Check state transitions	
	//if(m_mapExpressionInfos.size() == 0)
	//{
	//	strError = "Expression map is empty in state transition [" + GetCanonicalName() + "]";
	//	strarrErrors.push_back(strError);
	//	bCheck = false;
	//}

	//if(m_mapExpressionInfos.size() > 0)
	//{
	//	map<size_t, daeExpressionInfo>::const_iterator it;
	//	for(it = m_mapExpressionInfos.begin(); it != m_mapExpressionInfos.end(); it++)
	//	{
	//		if(!it->second.m_pExpression)
	//		{
	//			strError = "Invalid expression in expression info in state transition [" + GetCanonicalName() + "]";
	//			strarrErrors.push_back(strError);
	//			bCheck = false;
	//		}

	//		if(it->second.m_pStateTransition != this)
	//		{
	//			strError = "Invalid state transition in expression info in state transition [" + GetCanonicalName() + "]";
	//			strarrErrors.push_back(strError);
	//			bCheck = false;
	//		}
	//	}
	//}

	return bCheck;
}


}
}
