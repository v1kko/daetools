#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace boost;

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeAction
**********************************************************************************************/
daeAction::daeAction(const string& strName, daeModel* pModel, const string& strSTN, const string& strStateTo, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eChangeState;

// For eChangeState:
	m_pSTN           = NULL;
	m_strSTN         = strSTN;
	m_pStateTo       = NULL;
	m_strStateTo     = strStateTo;	
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	m_dData          = 0.0;
	
// For eReAssignOrReInitializeVariable:
	m_pVariable      = NULL;
	m_nIndex         = (size_t)-1;
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeSTN* pSTN, const string& strStateTo, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eChangeState;

// For eChangeState:
	m_pSTN	         = pSTN;
	m_pStateTo       = NULL;
	m_strStateTo     = strStateTo;	
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	m_dData          = 0.0;
	
// For eReAssignOrReInitializeVariable:
	m_pVariable      = NULL;
	m_nIndex         = (size_t)-1;
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeEventPort* pPort, real_t data, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eSendEvent;
	
// For eChangeState:
	m_pSTN	         = NULL;
	m_pStateTo       = NULL;
	m_strSTN         = "";
	m_strStateTo	 = "";
	
// For eSendEvent:
	m_pSendEventPort = pPort;
	m_dData          = data;
	
// For eReAssignOrReInitializeVariable:
	m_pVariable      = NULL;
	m_nIndex         = (size_t)-1;
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeVariable* pVariable, const adouble value, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eReAssignOrReInitializeVariable;
	
// For eChangeState:
	m_pSTN	     = NULL;
	m_pStateTo   = NULL;
	m_strSTN     = "";
	m_strStateTo = "";
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	m_dData          = 0.0;
	
// For eReAssignOrReInitializeVariable:
	m_pVariable	= pVariable;
	m_nIndex    = (size_t)-1;
	if(value.node)
		m_pSetupSetExpressionNode = value.node;
	else
		m_pSetupSetExpressionNode = boost::shared_ptr<adNode>(new adConstantNode(value.getValue()));
}

daeAction::~daeAction()
{
}

daeeActionType daeAction::GetType(void) const
{
	return m_eActionType;
}

void daeAction::Initialize(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	if(m_eActionType == eChangeState)
	{
		if(!m_pSTN)
		{
			daeObject_t* pObject = m_pModel->FindObjectFromRelativeName(m_strSTN);
			if(!pObject)
				daeDeclareAndThrowException(exInvalidPointer);
	
			m_pSTN = dynamic_cast<daeSTN*>(pObject);
			if(!m_pSTN)
				daeDeclareAndThrowException(exInvalidPointer);
		}
	
		m_pStateTo = m_pSTN->FindState(m_strStateTo);
		if(!m_pStateTo)
			daeDeclareAndThrowException(exInvalidPointer);
	}
	else if(m_eActionType == eSendEvent)
	{
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		if(!m_pVariable)
			daeDeclareAndThrowException(exInvalidPointer);
		
	// Here I do not have variable types yet (SetUpVariables has not been called yet) so I cannot check the variable type
	// therefore, we just get the variable index and will decide later which function to call (ReAssign or reInitialize)
		m_nIndex = m_pVariable->m_nOverallIndex + m_pVariable->CalculateIndex(NULL, 0);
		
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		m_pSetExpressionNode = m_pSetupSetExpressionNode->Evaluate(&EC).node;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

void daeAction::Execute(void)
{
	std::cout << "    Execute called in the action: " << GetName() << std::endl;
	
	if(m_eActionType == eChangeState)
	{
		if(!m_pSTN)
			daeDeclareAndThrowException(exInvalidPointer);
		if(!m_pStateTo)
			daeDeclareAndThrowException(exInvalidPointer);
		
		if(m_pSTN->GetActiveState() != m_pStateTo)
		{
			m_pSTN->SetActiveState(m_pStateTo);
			
		// Just in case; it is also set in the SetActiveState() function
			m_pModel->m_pDataProxy->SetReinitializationFlag(true);
		}
		else
		{
			m_pSTN->LogMessage(string("Current state unchanged"), 0);
		}			
	}
	else if(m_eActionType == eSendEvent)
	{
		if(!m_pSendEventPort)
			daeDeclareAndThrowException(exInvalidPointer);
		
		m_pSendEventPort->SendEvent(m_dData);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		real_t value = m_pSetExpressionNode->Evaluate(&EC).getValue();
		std::cout << "    Variable value: " << m_pVariable->GetValue() << " ; new value: " << value << std::endl;
		
		//std::cout << "Action: " << GetName() << ", nIndex = " << m_nIndex << ", Type = " << m_pModel->m_pDataProxy->GetVariableType(m_nIndex) << std::endl;
		if(m_pModel->m_pDataProxy->GetVariableType(m_nIndex) == cnFixed)
			m_pVariable->ReAssignValue(value);
		else if(m_pModel->m_pDataProxy->GetVariableType(m_nIndex) == cnDifferential)
			m_pVariable->ReSetInitialCondition(value);
		else
			daeDeclareAndThrowException(exInvalidCall);
		
		std::cout << "    Variable new value: " << m_pVariable->GetValue() << std::endl;
		
	// Set the reinitialization flag to true to mark the system ready for re-initialization
		m_pModel->m_pDataProxy->SetReinitializationFlag(true);
		m_pModel->m_pDataProxy->SetCopyDataFromBlock(true);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

bool daeAction::CheckObject(std::vector<string>& strarrErrors) const
{
	bool bReturn = true;

	bReturn = daeObject::CheckObject(strarrErrors);

	if(m_eActionType == eChangeState)
	{
		if(!m_pSTN)
		{
			strarrErrors.push_back(string("Invalid STN in action: ") + GetName());
			return false;
		}
		if(!m_pStateTo)
		{
			strarrErrors.push_back(string("Cannot find state: ") + m_strStateTo + string("in the STN: ") + m_pSTN->GetCanonicalName());
			return false;
		}
	}
	else if(m_eActionType == eSendEvent)
	{
		if(!m_pSendEventPort)
		{
			strarrErrors.push_back(string("Invalid Event Port in action: ") + GetName());
			return false;
		}
		if(m_pSendEventPort->GetType() != eOutletPort)
		{
			strarrErrors.push_back(string("Send EventPort must be the outlet port, in action: ") + GetName());
			return false;
		}
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		if(!m_pVariable)
		{
			strarrErrors.push_back(string("Invalid variable in action: ") + GetName());
			return false;
		}
		if(!m_pSetupSetExpressionNode)
		{
			strarrErrors.push_back(string("Invalid set value expression in action: ") + GetName());
			return false;
		}
		if(m_nIndex == size_t(-1))
		{
			strarrErrors.push_back(string("Invalid variable index, in action: ") + GetName());
			return false;
		}
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	
	return bReturn;
}

void daeAction::Open(io::xmlTag_t* pTag)
{
	string strName;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	daeObject::Open(pTag);

	strName = "Type";
	OpenEnum(pTag, strName, m_eActionType);
}

void daeAction::Save(io::xmlTag_t* pTag) const
{
	string strName;

	daeObject::Save(pTag);

	strName = "Type";
	SaveEnum(pTag, strName, m_eActionType);

	if(m_eActionType == eChangeState)
	{
		strName = "STN";
		pTag->SaveObjectRef(strName, m_pSTN);
		
		strName = "StateTo";
		pTag->SaveObjectRef(strName, m_pStateTo);
	}
	else if(m_eActionType == eSendEvent)
	{
		strName = "SendEventPort";
		pTag->SaveObjectRef(strName, m_pSendEventPort);
		
		strName = "Data";
		pTag->Save(strName, m_dData);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		strName = "Variable";
		pTag->SaveObjectRef(strName, m_pVariable);

		strName = "Expression";
		adNode::SaveNode(pTag, strName, m_pSetupSetExpressionNode.get());

		strName = "MathML";
		SaveNodeAsMathML(pTag, strName);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}
	
void daeAction::SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue;
	daeSaveAsMathMLContext c(m_pModel);
	adNode* node = m_pSetupSetExpressionNode.get();

	io::xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "math";
	io::xmlTag_t* pMathMLTag = pChildTag->AddTag(strName);
	if(!pMathMLTag)
		daeDeclareAndThrowException(exXMLIOError);

	strName = "xmlns";
	strValue = "http://www.w3.org/1998/Math/MathML";
	pMathMLTag->AddAttribute(strName, strValue);

	strName = "mrow";
	io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
	if(!mrow)
		daeDeclareAndThrowException(exXMLIOError);

	strName  = "mi";
	strValue = daeGetRelativeName(m_pModel, m_pVariable);
	pChildTag = mrow->AddTag(strName, strValue);
	pChildTag->AddAttribute(string("mathvariant"), string("italic"));

	strName  = "mo";
	strValue = "=";
	mrow->AddTag(strName, strValue);

	node->SaveAsPresentationMathML(mrow, &c);
}

void daeAction::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	if(m_eActionType == eChangeState)
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	else if(m_eActionType == eSendEvent)
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}


/*********************************************************************************************
	daeOnEventActions
**********************************************************************************************/
daeOnEventActions::daeOnEventActions(void)
{
	m_pEventPort = NULL;
}

daeOnEventActions::daeOnEventActions(daeEventPort* pEventPort, daeModel* pModel, std::vector<daeAction*>& ptrarrOnEventActions, const string& strDescription)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	pModel->AddOnEventAction(*this, "OnEventAction_" + pEventPort->GetName(), strDescription);

	m_pEventPort = pEventPort;
	for(size_t i = 0; i < ptrarrOnEventActions.size(); i++)
		m_ptrarrOnEventActions.push_back(ptrarrOnEventActions[i]);
}

daeOnEventActions::daeOnEventActions(daeEventPort* pEventPort, daeState* pState, std::vector<daeAction*>& ptrarrOnEventActions, const string& strDescription)
{
	if(!pState)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);
	
	pState->AddOnEventAction(*this, "OnEventAction_" + pEventPort->GetName(), strDescription);

	m_pEventPort = pEventPort;
	for(size_t i = 0; i < ptrarrOnEventActions.size(); i++)
		m_ptrarrOnEventActions.push_back(ptrarrOnEventActions[i]);
}

daeOnEventActions::~daeOnEventActions(void)
{
}

daeEventPort* daeOnEventActions::GetEventPort(void) const
{
	return m_pEventPort;
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
	
void daeOnEventActions::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

void daeOnEventActions::Update(daeEventPort_t* pSubject, void* data)
{
	std::cout << "    Update received in the OnEventActions: " << GetName() << ", data = " << *((real_t*)data) << std::endl;
	Execute();
}

void daeOnEventActions::Execute(void)
{
	daeAction* pAction;
	
	for(size_t i = 0; i < m_ptrarrOnEventActions.size(); i++)
	{
		pAction = m_ptrarrOnEventActions[i];
		pAction->Execute();
	}
}

}
}
