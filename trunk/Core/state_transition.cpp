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
	m_pSTN			= NULL;
// 11.08.2011	
//	m_pStateFrom	= NULL;
//	m_pStateTo		= NULL;
	m_pModel		= NULL;
}

daeStateTransition::~daeStateTransition()
{
}

void daeStateTransition::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	daeFindStateByID del(m_pModel);

// 11.08.2011	
//	strName = "StateFromRef";
//	m_pStateFrom = pTag->OpenObjectRef(strName, &del);
//
//	strName = "StateToRef";
//	m_pStateTo = pTag->OpenObjectRef(strName, &del);

	strName = "Condition";
	pTag->OpenExistingObject<daeCondition, daeCondition>(strName, &m_Condition);
}

void daeStateTransition::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

// 11.08.2011	
//	strName = "StateFromRef";
//	pTag->SaveObjectRef(strName, m_pStateFrom);
//
//	strName = "StateToRef";
//	pTag->SaveObjectRef(strName, m_pStateTo);

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
	string strName;

	daeObject::SaveRuntime(pTag);

//	strName = "StateFromRef";
//	pTag->SaveObjectRef(strName, m_pStateFrom);
//
//	strName = "StateToRef";
//	pTag->SaveObjectRef(strName, m_pStateTo);

	strName = "Condition";
	pTag->SaveRuntimeObject(strName, &m_Condition);
}

void daeStateTransition::Create_SWITCH_TO(const string& strCondition, daeState* pStateFrom, const string& strStateToName, const daeCondition& rCondition, real_t dEventTolerance)
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
	m_ptrarrActions.push_back(pAction);
	
// 11.08.2011	
//	m_pStateFrom		= pStateFrom;
//	m_pStateTo			= NULL;
//	m_strStateToName	= strStateToName;
	
// This creates runtime node from setup nodes
// Global daeExecutionContext (m_pExecutionContextForGatherInfo) should be non-null during this stage
//	m_Condition			= rCondition.m_pConditionNode->CreateRuntimeNode(m_pModel->m_pExecutionContextForGatherInfo);
	m_Condition			= rCondition;

	m_Condition.m_pModel = m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;
	pStateFrom->AddStateTransition(this);
}

void daeStateTransition::Create_IF(const string& strCondition, daeState* pStateTo, const daeCondition& rCondition, real_t dEventTolerance)
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
	
	daeAction* pAction = new daeAction(string("actionChangeState_") + pStateTo->GetName(), m_pModel, m_pSTN, pStateTo->GetName(), string(""));
	m_ptrarrActions.push_back(pAction);

// 11.08.2011	
//	m_pStateFrom		= NULL;
//	m_pStateTo			= pStateTo;
//	m_strStateToName	= "";
	
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
		pAction->Execute(NULL);
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

// 11.08.2011	
//daeState_t* daeStateTransition::GetStateTo(void) const
//{
//	return m_pStateTo;
//}
//
//void daeStateTransition::SetStateTo(daeState* pState)
//{
//	m_pStateTo = pState;
//}
//
//daeState_t* daeStateTransition::GetStateFrom(void) const
//{
//	return m_pStateFrom;
//}
//
//void daeStateTransition::SetStateFrom(daeState* pState)
//{
//	m_pStateFrom = pState;
//}

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

// 11.08.2011	
/*	
// Check state from	
// If it is daeIF block it IS null (but StateTo mustn't be NULL)
	if(!m_pStateFrom)
	{
		if(!m_pStateTo)
		{
			strError = "Invalid start state in state transition [" + GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
	}

// Check state to	
	if(!m_pStateTo)
	{
		strError = "Invalid end state in state transition [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		bCheck = false;
	}
	
// Check if start and end states are the same to	
	if(m_pStateFrom == m_pStateTo)
	{
		strError = "Start end end states are the same in state transition [" + GetCanonicalName() + "]";
		strarrErrors.push_back(strError);
		return false;
	}
*/
	
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
