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
	m_pStateFrom	= NULL;
	m_pStateTo		= NULL;
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

	strName = "StateFromRef";
	m_pStateFrom = pTag->OpenObjectRef(strName, &del);

	strName = "StateToRef";
	m_pStateTo = pTag->OpenObjectRef(strName, &del);

	strName = "Condition";
	pTag->OpenExistingObject<daeCondition, daeCondition>(strName, &m_Condition);
}

void daeStateTransition::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "StateFromRef";
	pTag->SaveObjectRef(strName, m_pStateFrom);

	strName = "StateToRef";
	pTag->SaveObjectRef(strName, m_pStateTo);

	strName = "Condition";
	pTag->SaveObject(strName, &m_Condition);
}

void daeStateTransition::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	
}

void daeStateTransition::OpenRuntime(io::xmlTag_t* pTag)
{
	daeObject::OpenRuntime(pTag);
}

void daeStateTransition::SaveRuntime(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::SaveRuntime(pTag);

	strName = "StateFromRef";
	pTag->SaveObjectRef(strName, m_pStateFrom);

	strName = "StateToRef";
	pTag->SaveObjectRef(strName, m_pStateTo);

	strName = "Condition";
	pTag->SaveRuntimeObject(strName, &m_Condition);
}

void daeStateTransition::CreateSTN(const string& strCondition, daeState* pStateFrom, const string& strStateToName, const daeCondition& rCondition, real_t dEventTolerance)
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
	m_pStateFrom		= pStateFrom;
	m_pStateTo			= NULL;
	m_strStateToName	= strStateToName;
	m_pModel			= pStateFrom->m_pModel;
	
// This creates runtime node from setup nodes
// Global daeExecutionContext (m_pExecutionContextForGatherInfo) should be non-null during this stage
//	m_Condition			= rCondition.m_pConditionNode->CreateRuntimeNode(m_pModel->m_pExecutionContextForGatherInfo);
	m_Condition			= rCondition;

	m_Condition.m_pModel = m_pStateFrom->m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;
	m_pStateFrom->AddStateTransition(this);
}

void daeStateTransition::CreateIF(const string& strCondition, daeState* pStateTo, const daeCondition& rCondition, real_t dEventTolerance)
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
	m_pStateFrom		= NULL;
	m_pStateTo			= pStateTo;
	m_strStateToName	= "";
	m_pModel			= pStateTo->m_pModel;
	
// This creates runtime node from setup nodes	
// Global daeExecutionContext (m_pExecutionContextForGatherInfo) should be non-null during this stage
//	m_Condition			= rCondition.m_pConditionNode->CreateRuntimeNode(m_pModel->m_pExecutionContextForGatherInfo);
	m_Condition			= rCondition;

	m_Condition.m_pModel = pStateTo->m_pModel;
	m_Condition.m_dEventTolerance = dEventTolerance;
	pStateTo->AddStateTransition(this);
}

void daeStateTransition::Initialize(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pModel->m_pExecutionContextForGatherInfo)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_Condition.m_pConditionNode)
		daeDeclareAndThrowException(exInvalidPointer); 
	
// This creates runtime node from setup nodes	
// Global daeExecutionContext (m_pExecutionContextForGatherInfo) should be non-null during this stage
	m_Condition.m_pConditionNode = m_Condition.m_pConditionNode->CreateRuntimeNode(m_pModel->m_pExecutionContextForGatherInfo).m_pConditionNode;
}

daeCondition* daeStateTransition::GetCondition(void)
{
	return &m_Condition;
}

void daeStateTransition::SetCondition(daeCondition& rCondition)
{
	m_Condition = rCondition;
}

daeState_t* daeStateTransition::GetStateTo(void) const
{
	return m_pStateTo;
}

void daeStateTransition::SetStateTo(daeState* pState)
{
	m_pStateTo = pState;
}

daeState_t* daeStateTransition::GetStateFrom(void) const
{
	return m_pStateFrom;
}

void daeStateTransition::SetStateFrom(daeState* pState)
{
	m_pStateFrom = pState;
}

string daeStateTransition::GetConditionAsString() const
{
	return m_Condition.SaveNodeAsPlainText();
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
