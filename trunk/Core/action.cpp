#include "stdafx.h"
#include "coreimpl.h"
using namespace boost;

namespace dae 
{
namespace core 
{
/*********************************************************************************************
	daeAction
**********************************************************************************************/
daeAction::daeAction(const string& strName, daeModel* pModel, daeSTN* pSTN, const string& strStateTo, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eChangeState;

	m_pSTN	         = pSTN;
	m_pStateTo       = NULL;
	m_strStateTo     = strStateTo;	
	
	m_pSendEventPort = NULL;
	m_pData          = NULL;
	
	m_pVariable      = NULL;
	m_nVariableType  = -1;
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeEventPort* pPort, void* data, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eSendEvent;
	
	m_pSTN	         = NULL;
	m_pStateTo       = NULL;
	m_strStateTo	 = "";
	
	m_pSendEventPort = pPort;
	m_pData          = data;
	
	m_pVariable      = NULL;
	m_nVariableType  = -1;
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeVariable* pVariable, const adouble value, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eReAssignOrReInitializeVariable;
	
	m_pSTN	     = NULL;
	m_pStateTo   = NULL;
	m_strStateTo = "";
	
	m_pSendEventPort = NULL;
	m_pData          = NULL;
	
	m_pVariable			 = pVariable;
	m_nVariableType      = -1;
	m_pSetExpressionNode = value.node;
}

daeAction::~daeAction()
{
}

daeeActionType daeAction::GetType(void) const
{
	return m_eActionType;
}

void daeAction::Update(daeEventPort_t* pSubject, void* data)
{
	std::cout << "Update received in the action: " << GetName() << std::endl;
	Execute(data);
}

void daeAction::Initialize(void)
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pModel->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);

	if(m_eActionType == eChangeState)
	{
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
		
		size_t nIndex = m_pVariable->m_nOverallIndex + m_pVariable->CalculateIndex(NULL, 0);
		
		if(m_pModel->m_pDataProxy->GetVariableType(nIndex) == cnDifferential)
			m_nVariableType = cnDifferential;
		else if(m_pModel->m_pDataProxy->GetVariableType(nIndex) == cnFixed)
			m_nVariableType = cnFixed;
		else
			m_nVariableType = -1;
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

void daeAction::Execute(void* data)
{
	std::cout << "Execute called in the action: " << GetName() << std::endl;
	
	if(m_eActionType == eChangeState)
	{
		if(!m_pSTN)
			daeDeclareAndThrowException(exInvalidPointer);
		if(!m_pStateTo)
			daeDeclareAndThrowException(exInvalidPointer);
		
		if(m_pSTN->m_pActiveState != m_pStateTo)
			m_pSTN->SetActiveState(m_pStateTo);
		else
			m_pSTN->LogMessage(string("Current state unchanged"), 0);
	}
	else if(m_eActionType == eSendEvent)
	{
		if(!m_pSendEventPort)
			daeDeclareAndThrowException(exInvalidPointer);
		
		m_pSendEventPort->SendEvent(m_pData);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		adouble res = m_pSetExpressionNode->Evaluate(&EC);
		
		if(m_nVariableType == cnFixed)
			m_pVariable->ReAssignValue(res.getValue());
		else if(m_nVariableType == cnDifferential)
			m_pVariable->ReSetInitialCondition(res.getValue());
		else
			daeDeclareAndThrowException(exInvalidCall);
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

bool daeAction::CheckObject(std::vector<string>& strarrErrors) const
{
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
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		if(!m_pVariable)
		{
			strarrErrors.push_back(string("Invalid variable in action: ") + GetName());
			return false;
		}
		if(!m_pSetExpressionNode)
		{
			strarrErrors.push_back(string("Invalid set value expression in action: ") + GetName());
			return false;
		}
		if(m_nVariableType != cnFixed || m_nVariableType != cnDifferential)
		{
			strarrErrors.push_back(string("To set variable value the it must be either differential or assigned, in action: ") + GetName());
			return false;
		}
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
	
	return true;
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
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		strName = "Variable";
		pTag->SaveObjectRef(strName, m_pVariable);

		strName = "Expression";
		adNode::SaveNode(pTag, strName, m_pSetExpressionNode.get());

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
	adNode* node = m_pSetExpressionNode.get();

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
	mrow->AddTag(strName, strValue);

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

}
}
