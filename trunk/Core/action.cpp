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
daeAction::daeAction(void)
{
	m_pModel		= NULL;
	m_eActionType	= eUnknownAction;

// For eChangeState:
	m_pSTN           = NULL;
	m_pStateTo       = NULL;
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	
// For eReAssignOrReInitializeVariable:
}

daeAction::daeAction(const string& strName, daeModel* pModel, const string& strSTN, const string& strStateTo, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
//	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eChangeState;

// For eChangeState:
	m_pSTN           = NULL;
	m_strSTN         = strSTN;
	m_pStateTo       = NULL;
	m_strStateTo     = strStateTo;	
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	
// For eReAssignOrReInitializeVariable:
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeSTN* pSTN, const string& strStateTo, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
//	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eChangeState;

// For eChangeState:
	m_pSTN	         = pSTN;
	m_pStateTo       = NULL;
	m_strStateTo     = strStateTo;	
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	
// For eReAssignOrReInitializeVariable:
}

daeAction::daeAction(const string& strName, daeModel* pModel, daeEventPort* pPort, adouble data, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
//	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eSendEvent;
	
// For eChangeState:
	m_pSTN	         = NULL;
	m_pStateTo       = NULL;
	
// For eSendEvent:
	m_pSendEventPort = pPort;
	if(data.node)
		m_pSetupNode = data.node;
	else
		m_pSetupNode = boost::shared_ptr<adNode>(new adConstantNode(data.getValue()));
	
// For eReAssignOrReInitializeVariable:
}

daeAction::daeAction(const string& strName, daeModel* pModel, const daeVariableWrapper& variable, const adouble value, const string& strDescription)
{
	m_pModel = pModel;
	SetName(strName);
	SetDescription(strDescription);
//	SetCanonicalName(pModel->GetCanonicalName() + "." + m_strShortName);
	m_eActionType = eReAssignOrReInitializeVariable;
	
// For eChangeState:
	m_pSTN	     = NULL;
	m_pStateTo   = NULL;
	
// For eSendEvent:
	m_pSendEventPort = NULL;
	
// For eReAssignOrReInitializeVariable:
	m_pVariableWrapper = boost::shared_ptr<daeVariableWrapper>(new daeVariableWrapper(variable));
	if(value.node)
		m_pSetupNode = value.node;
	else
		m_pSetupNode = boost::shared_ptr<adNode>(new adConstantNode(value.getValue()));
}

daeAction::~daeAction()
{
}

void daeAction::Clone(const daeAction& rObject)
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
			{
				daeDeclareException(exInvalidCall);
				e << "Cannot find the state transition network [" << m_strSTN << "] in the model [" << m_pModel->GetCanonicalName() << "]";
				throw e;
			}
			
			m_pSTN = dynamic_cast<daeSTN*>(pObject);
			if(!m_pSTN)
				daeDeclareAndThrowException(exInvalidPointer);
		}
	
		m_pStateTo = m_pSTN->FindState(m_strStateTo);
		if(!m_pStateTo)
		{
			daeDeclareException(exInvalidCall);
			e << "Cannot find the state [" << m_strStateTo 
			  << "] in the state transition network [" << m_pSTN->GetCanonicalName() 
			  << "] in the model [" << m_pModel->GetCanonicalName() << "]";
			throw e;
		}
	}
	else if(m_eActionType == eSendEvent)
	{
		if(!m_pSendEventPort)
			daeDeclareAndThrowException(exInvalidPointer);
		
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		m_pNode = m_pSetupNode->Evaluate(&EC).node;
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		m_pNode = m_pSetupNode->Evaluate(&EC).node;
	}
	else if(m_eActionType == eUserDefinedAction)
	{
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

void daeAction::Execute(void)
{
	//std::cout << "    Execute called in the action: " << GetCanonicalName() << std::endl;
	
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
	}
	else if(m_eActionType == eSendEvent)
	{
		if(!m_pSendEventPort)
			daeDeclareAndThrowException(exInvalidPointer);
		
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		real_t data = m_pNode->Evaluate(&EC).getValue();
		//std::cout << "    Event data : " << data << std::endl;
		
		m_pSendEventPort->SendEvent(data);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		daeExecutionContext EC;
		EC.m_pDataProxy					= m_pModel->m_pDataProxy.get();
		EC.m_eEquationCalculationMode	= eCalculate;
		
		real_t value = m_pNode->Evaluate(&EC).getValue();
		//std::cout << "    Variable value: " << m_pVariable->GetValue() << " ; new value: " << value << std::endl;
		
		m_pVariableWrapper->SetValue(value);
		
	// Set the reinitialization flag to true to mark the system ready for re-initialization
		m_pModel->m_pDataProxy->SetReinitializationFlag(true);
		m_pModel->m_pDataProxy->SetCopyDataFromBlock(true);
	}
	else if(m_eActionType == eUserDefinedAction)
	{
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}

bool daeAction::CheckObject(std::vector<string>& strarrErrors) const
{
	string strError;
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
		if(!m_pSetupNode)
		{
			strarrErrors.push_back(string("Invalid event data expression in action: ") + GetName());
			return false;
		}
		if(m_pSendEventPort->GetType() != eOutletPort)
		{
			strarrErrors.push_back(string("Send EventPort must be the outlet port, in action: ") + GetName());
			return false;
		}
		
		daeConfig& cfg = daeConfig::GetConfig();
		if(cfg.Get<bool>("daetools.core.checkUnitsConsistency", false))
		{
			try
			{
				quantity q = m_pSetupNode->GetQuantity();
			}
			catch(units_error& e)
			{
				strError = "Unit-consistency check failed for the expression in action [" + GetName() + "]:";
				strarrErrors.push_back(strError);
				strError = "  " + string(e.what());
				strarrErrors.push_back(strError);
				return false;
			}
			catch(std::exception& e)
			{
				strError = "Exception occurred while unit-consistency check in action [" + GetName() + "]:";
				strarrErrors.push_back(strError);
				strError = "  " + string(e.what());
				strarrErrors.push_back(strError);
				return false;
			}
		}
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		if(!m_pSetupNode)
		{
			strarrErrors.push_back(string("Invalid set value expression in action: ") + GetName());
			return false;
		}
		if(!m_pVariableWrapper->m_pVariable)
		{
			strarrErrors.push_back(string("Invalid variable in action: ") + GetName());
			return false;
		}
		
		daeConfig& cfg = daeConfig::GetConfig();
		if(cfg.Get<bool>("daetools.core.checkUnitsConsistency", false))
		{
		// Check the expression unit-consistency
			try
			{
				quantity q = m_pSetupNode->GetQuantity();
			}
			catch(units_error& e)
			{
				strError = "Unit-consistency check failed for the expression in action [" + GetCanonicalName() + "]:";
				strarrErrors.push_back(strError);
				strError = "  " + string(e.what());
				strarrErrors.push_back(strError);
				return false;
			}
			catch(std::exception& e)
			{
				strError = "Exception occurred while unit-consistency check in action [" + GetCanonicalName() + "]:";
				strarrErrors.push_back(strError);
				strError = "  " + string(e.what());
				strarrErrors.push_back(strError);
				return false;
			}

		// Check if the expression units match those of the variable
			unit var_units   = m_pVariableWrapper->m_pVariable->GetVariableType()->GetUnits();
			unit value_units = m_pSetupNode->GetQuantity().getUnits();
			if(var_units != value_units)
			{
				strError = "The expression units " + value_units.toString() + " do not match the variable units " + var_units.toString() + " in action [" + GetCanonicalName() + "]:";
				strarrErrors.push_back(strError);
				return false;
			}
		}
	}
	else if(m_eActionType == eUserDefinedAction)
	{
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
		
		strName = "Expression";
		adNode::SaveNode(pTag, strName, m_pSetupNode.get());

		strName = "MathML";
		SaveNodeAsMathML(m_pSetupNode.get(), pTag, strName);
	}
	else if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		strName = "Variable";
		pTag->Save(strName, m_pVariableWrapper->GetName());

		strName = "Expression";
		adNode::SaveNode(pTag, strName, m_pSetupNode.get());

		strName = "MathML";
		SaveNodeAsMathML(m_pSetupNode.get(), pTag, strName);
	}
	else if(m_eActionType == eUserDefinedAction)
	{
	}
	else
	{
		daeDeclareAndThrowException(exNotImplemented);
	}
}
	
void daeAction::SaveNodeAsMathML(adNode* node, io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue;
	daeSaveAsMathMLContext c(m_pModel);

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

	if(m_eActionType == eReAssignOrReInitializeVariable)
	{
		strName = "mrow";
		io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
		if(!mrow)
			daeDeclareAndThrowException(exXMLIOError);
	
		strName  = "mi";
		pChildTag = mrow->AddTag(strName, m_pVariableWrapper->GetName());
		pChildTag->AddAttribute(string("mathvariant"), string("italic"));
	
		strName  = "mo";
		strValue = "=";
		mrow->AddTag(strName, strValue);
	
		node->SaveAsPresentationMathML(mrow, &c);
	}
	else if(m_eActionType == eSendEvent)
	{
		strName = "mrow";
		io::xmlTag_t* mrow = pMathMLTag->AddTag(strName);
		if(!mrow)
			daeDeclareAndThrowException(exXMLIOError);
	
		node->SaveAsPresentationMathML(mrow, &c);
	}
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
	else if(m_eActionType == eUserDefinedAction)
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
