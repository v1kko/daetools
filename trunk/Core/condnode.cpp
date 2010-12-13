#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include <typeinfo>
using namespace boost;

namespace dae 
{
namespace core 
{
bool condDoEnclose(const adNode* node)
{
	return false;
}

bool condDoEnclose(const condNode* node)
{
	if(!node)
		return true;

	const type_info& infoChild  = typeid(*node);

// If it is simple node DO NOT enclose in brackets
	if(infoChild == typeid(const condExpressionNode))
	{
		return false;
	}
	else if(infoChild == typeid(const condUnaryNode))
	{
		return true;
	}
	else if(infoChild == typeid(const condBinaryNode))
	{
		return true;
	}
	else
	{
		return true;
	}
}

/*********************************************************************************************
	condNode
**********************************************************************************************/
condNode* condNode::CreateNode(const io::xmlTag_t* pTag)
{
	string strClass;
	string strName = "Class";

	io::xmlAttribute_t* pAttrClass = pTag->FindAttribute(strName);
	if(!pAttrClass)
		daeDeclareAndThrowException(exXMLIOError);

	pAttrClass->GetValue(strClass);
	if(strClass == "condExpressionNode")
	{
		return new condExpressionNode();
	}
	else if(strClass == "condUnaryNode")
	{
		return new condUnaryNode();
	}
	else if(strClass == "condBinaryNode")
	{
		return new condBinaryNode();
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError);
		return NULL;
	}
	return NULL;
}

void condNode::SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const condNode* node)
{
	io::xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);
	node->Save(pChildTag);
}

condNode* condNode::OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<condNode>* ood)
{
	io::xmlTag_t* pChildTag = pTag->FindTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	condNode* node = condNode::CreateNode(pChildTag);
	if(!node)
		daeDeclareAndThrowException(exXMLIOError);

	if(ood)
		ood->BeforeOpenObject(node);
	node->Open(pChildTag);
	if(ood)
		ood->AfterOpenObject(node);

	return node;
}

void condNode::SaveNodeAsMathML(io::xmlTag_t* pTag, 
								const string& strObjectName, 
								const condNode* node,
								const daeSaveAsMathMLContext* c)
{
	string strName, strValue;
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
	node->SaveAsPresentationMathML(pMathMLTag, c);
}

/*********************************************************************************************
	condExpressionNode
**********************************************************************************************/
condExpressionNode::condExpressionNode(const adouble& left, daeeConditionType type, const adouble& right)
{
	if(!left.node)
		daeDeclareAndThrowException(exInvalidPointer);  
	if(!right.node)
		daeDeclareAndThrowException(exInvalidPointer);  
	m_pLeft				= shared_ptr<adNode>(left.node->Clone());
	m_eConditionType	= type;
	m_pRight			= shared_ptr<adNode>(right.node->Clone());
}

condExpressionNode::condExpressionNode(const adouble& left, daeeConditionType type, real_t right)
{
	if(!left.node)
		daeDeclareAndThrowException(exInvalidPointer);  
	m_pLeft				= shared_ptr<adNode>(left.node->Clone());
	m_eConditionType	= type;
	m_pRight			= shared_ptr<adNode>(new adConstantNode(right));
}

condExpressionNode::condExpressionNode(real_t left, daeeConditionType type, const adouble& right)
{
	if(!right.node)
		daeDeclareAndThrowException(exInvalidPointer);  
	m_pLeft				= shared_ptr<adNode>(new adConstantNode(left));
	m_eConditionType	= type;
	m_pRight			= shared_ptr<adNode>(right.node->Clone());
}

condExpressionNode::condExpressionNode()
{
}

condExpressionNode::~condExpressionNode()
{
}

//condExpressionNode* condExpressionNode::Create(const adouble& left, daeeConditionType type, const adouble& right)
//{
//	condExpressionNode* node = new condExpressionNode();
//	if(!left.node)
//		daeDeclareAndThrowException(exInvalidPointer);  
//	if(!right.node)
//		daeDeclareAndThrowException(exInvalidPointer);  
//	node->m_pLeft			= shared_ptr<adNode>(left.node->Clone());
//	node->m_eConditionType	= type;
//	node->m_pRight			= shared_ptr<adNode>(right.node->Clone());
//	return node;
//}
//
//condExpressionNode* condExpressionNode::Create(const adouble& left, daeeConditionType type, real_t right)
//{
//	condExpressionNode* node = new condExpressionNode();
//	if(!left.node)
//		daeDeclareAndThrowException(exInvalidPointer);  
//	node->m_pLeft			= shared_ptr<adNode>(left.node->Clone());
//	node->m_eConditionType	= type;
//	node->m_pRight			= shared_ptr<adNode>(new adConstantNode(right));
//	return node;
//}
//
//condExpressionNode* condExpressionNode::Create(real_t left, daeeConditionType type, const adouble& right)
//{
//	condExpressionNode* node = new condExpressionNode();
//	if(!right.node)
//		daeDeclareAndThrowException(exInvalidPointer);  
//	node->m_pLeft			= shared_ptr<adNode>(new adConstantNode(left));
//	node->m_eConditionType	= type;
//	node->m_pRight			= shared_ptr<adNode>(right.node->Clone());
//	return node;
//}

void condExpressionNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void condExpressionNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "ConditionType";
	SaveEnum(pTag, strName, m_eConditionType);

	strName = "Left";
	adNode::SaveNode(pTag, strName, m_pLeft.get());

	strName = "Right";
	adNode::SaveNode(pTag, strName, m_pRight.get());
}

string condExpressionNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strValue;

	strValue += m_pLeft->SaveAsPlainText(c);
	
	if(m_eConditionType == eNotEQ)
		strValue += " != ";
	else if(m_eConditionType == eEQ)
		strValue += " == ";
	else if(m_eConditionType == eGT)
		strValue += " > ";
	else if(m_eConditionType == eGTEQ)
		strValue += " >= ";
	else if(m_eConditionType == eLT)
		strValue += " < ";
	else if(m_eConditionType == eLTEQ)
		strValue += " <= ";
	else
		daeDeclareAndThrowException(exInvalidCall);  

	strValue += m_pRight->SaveAsPlainText(c);

	return strValue;
}

string condExpressionNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return string("");
}

void condExpressionNode::SaveAsContentMathML(io::xmlTag_t* /*pTag*/, const daeSaveAsMathMLContext* /*c*/) const
{
}

void condExpressionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue, strOperator, strLeft, strRight;
	io::xmlTag_t *mrowout, *mrowleft, *mrowright;

	strName  = "mrow";
	strValue = "";
	mrowout = pTag->AddTag(strName, strValue);

	if(condDoEnclose(m_pLeft.get()))
	{
		strName  = "mrow";
		strValue = "";
		mrowleft = mrowout->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowleft->AddTag(strName, strValue);

			m_pLeft->SaveAsPresentationMathML(mrowleft, c);

			strName  = "mo";
			strValue = ")";
			mrowleft->AddTag(strName, strValue);
	}
	else
	{
		m_pLeft->SaveAsPresentationMathML(mrowout, c);
	}

	strName  = "mo";
	if(m_eConditionType == eNotEQ)
		strOperator = "&NotEqual;";
	else if(m_eConditionType == eEQ)
		strOperator = "&Equal;";
	else if(m_eConditionType == eGT)
		strOperator = "&gt;";
	else if(m_eConditionType == eGTEQ)
		strOperator = "&ge;";
	else if(m_eConditionType == eLT)
		strOperator = "&lt;";
	else if(m_eConditionType == eLTEQ)
		strOperator = "&le;";
	else
		daeDeclareAndThrowException(exInvalidCall);  
	mrowout->AddTag(strName, strOperator);

	if(condDoEnclose(m_pRight.get()))
	{
		strName  = "mrow";
		strValue = "";
		mrowright = mrowout->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowright->AddTag(strName, strValue);

			m_pRight->SaveAsPresentationMathML(mrowright, c);

			strName  = "mo";
			strValue = ")";
			mrowright->AddTag(strName, strValue);
	}
	else
	{
		m_pRight->SaveAsPresentationMathML(mrowout, c);
	}
}

daeCondition condExpressionNode::CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer);

	adouble left  = m_pLeft->Evaluate(pExecutionContext);
	adouble right = m_pRight->Evaluate(pExecutionContext);
	switch(m_eConditionType)
	{
	case eNotEQ:
		return (left != right);
		break;

	case eEQ:
		return (left == right);
		break;

	case eGT:
		return (left > right);
		break;

	case eGTEQ:
		return (left >= right);
		break;

	case eLT:
		return (left < right);
		break;

	case eLTEQ:
		return (left <= right);
		break;

	default:
		daeDeclareAndThrowException(exNotImplemented); 
		return daeCondition();
	}
    return daeCondition();
}

bool condExpressionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer);

	real_t left  = m_pLeft->Evaluate(pExecutionContext).getValue();
	real_t right = m_pRight->Evaluate(pExecutionContext).getValue();
	switch(m_eConditionType)
	{
	case eNotEQ:
		return (left != right ? true : false);
		break;

	case eEQ:
		return (left == right ? true : false);
		break;

	case eGT:
		return (left > right ? true : false);
		break;

	case eGTEQ:
		return (left >= right ? true : false);
		break;

	case eLT:
		return (left < right ? true : false);
		break;

	case eLTEQ:
		return (left <= right ? true : false);
		break;

	default:
		daeDeclareAndThrowException(exNotImplemented); 
		return false;
	}
}

condNode* condExpressionNode::Clone(void) const
{
	return new condExpressionNode(*this);
}

void condExpressionNode::BuildExpressionsArray(vector< shared_ptr<adNode> > & ptrarrExpressions,
											   const daeExecutionContext* pExecutionContext,
											   real_t dEventTolerance)
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer); 

// First add the mandatory expression: left - right = 0
	adouble left  = m_pLeft->Evaluate(pExecutionContext);
	adouble right = m_pRight->Evaluate(pExecutionContext);
	adouble ad    = left - right;

// This have to be added always
	ptrarrExpressions.push_back(ad.node);

// Depending on the type I have to add certain additional expressions
	adouble ad1, ad2;

	if(dEventTolerance == 0)
		dEventTolerance = 1E-7;

	ad1 = ad + dEventTolerance;
	ad2 = ad - dEventTolerance;
	ptrarrExpressions.push_back(ad1.node);
	ptrarrExpressions.push_back(ad2.node);
	
/*
	switch(m_eConditionType)
	{
	case eNotEQ: // !=
		ad1 = ad + dEventTolerance;
		ad2 = ad - dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		ptrarrExpressions.push_back(ad2.node);
		break;

	case eEQ: // ==
		ad1 = ad + dEventTolerance;
		ad2 = ad - dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		ptrarrExpressions.push_back(ad2.node);
		break;

	case eGT: // >
		ad1 = ad - dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		break;

	case eGTEQ: // >=
		ad1 = ad + dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		break;

	case eLT: // <
		ad1 = ad + dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		break;

	case eLTEQ: // <=
		ad1 = ad - dEventTolerance;
		ptrarrExpressions.push_back(ad1.node);
		break;

	default:
		daeDeclareAndThrowException(exNotImplemented); 
	}
*/	
}

void condExpressionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer);
	m_pLeft->AddVariableIndexToArray(mapIndexes);
	m_pRight->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	condUnaryNode
**********************************************************************************************/
condUnaryNode::condUnaryNode(shared_ptr<condNode> node, daeeLogicalUnaryOperator op)
{
	m_pNode				= node;
	m_eLogicalOperator	= op;
}

condUnaryNode::condUnaryNode()
{
	m_eLogicalOperator = eUOUnknown;
}

condUnaryNode::~condUnaryNode()
{
}

void condUnaryNode::Open(io::xmlTag_t* pTag)
{
}

void condUnaryNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Operator";
	SaveEnum(pTag, strName, m_eLogicalOperator);

	strName = "Node";
	condNode::SaveNode(pTag, strName, m_pNode.get());
}

string condUnaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strValue;

	if(m_eLogicalOperator == eNot)
		strValue += "not ";
	else
		daeDeclareAndThrowException(exInvalidCall);  

	strValue += "(";
	strValue += m_pNode->SaveAsPlainText(c);
	strValue += ")";

	return strValue;
}

string condUnaryNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	return string("");
}

void condUnaryNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void condUnaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrowout, *mrow;

	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	if(m_eLogicalOperator == eNot)
	{
		strName  = "mo";
		strValue = "(";
		mrow->AddTag(strName, strValue);
			strName  = "mrow";
			strValue = "";
			mrowout = mrow->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "&Not;";
				mrowout->AddTag(strName, strValue);
				m_pNode->SaveAsPresentationMathML(mrowout, c);
		strName  = "mo";
		strValue = ")";
		mrow->AddTag(strName, strValue);
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError);
	}
}

daeCondition condUnaryNode::CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const
{
	return m_pNode->CreateRuntimeNode(pExecutionContext);

	//switch(m_eLogicalOperator)
	//{
	//case eNot:
	//	return (!ad);
	//	break;
	//default:
	//	daeDeclareAndThrowException(exNotImplemented); 
	//	return shared_ptr<condNode>();
	//}
}

bool condUnaryNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	bool bResult = false;
	switch(m_eLogicalOperator)
	{
	case eNot:
		bResult = !(m_pNode->Evaluate(pExecutionContext));
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented); 
	}
	return bResult;
}

condNode* condUnaryNode::Clone(void) const
{
	return new condUnaryNode(*this);
}

void condUnaryNode::BuildExpressionsArray(vector< shared_ptr<adNode> > & ptrarrExpressions,
										  const daeExecutionContext* pExecutionContext,
										  real_t dEventTolerance)
{
	if(!m_pNode)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pNode->BuildExpressionsArray(ptrarrExpressions, pExecutionContext, dEventTolerance);
}

void condUnaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!m_pNode)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pNode->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	condBinaryNode
**********************************************************************************************/
condBinaryNode::condBinaryNode(shared_ptr<condNode> left, daeeLogicalBinaryOperator op, shared_ptr<condNode> right)
{
	m_pLeft				= left;
	m_eLogicalOperator	= op;
	m_pRight			= right;
}

condBinaryNode::condBinaryNode()
{
	m_eLogicalOperator = eBOUnknown;
}

condBinaryNode::~condBinaryNode()
{
}

void condBinaryNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void condBinaryNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Operator";
	SaveEnum(pTag, strName, m_eLogicalOperator);

	strName = "m_pLeft";
	condNode::SaveNode(pTag, strName, m_pLeft.get());

	strName = "Right";
	condNode::SaveNode(pTag, strName, m_pRight.get());
}

string condBinaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strValue;

	strValue += "(";
	strValue += m_pLeft->SaveAsPlainText(c);
	strValue += ")";
	
	if(m_eLogicalOperator == eAnd)
		strValue += " and ";
	else if(m_eLogicalOperator == eOr)
		strValue += " or ";
	else
		daeDeclareAndThrowException(exInvalidCall);  

	strValue += "(";
	strValue += m_pRight->SaveAsPlainText(c);
	strValue += ")";

	return strValue;
}

string condBinaryNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return string("");
}

void condBinaryNode::SaveAsContentMathML(io::xmlTag_t* /*pTag*/, const daeSaveAsMathMLContext* /*c*/) const
{
}

void condBinaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue, strOperator, strLeft, strRight;
	io::xmlTag_t *mrowout, *mrowleft, *mrowright;

	strName  = "mrow";
	strValue = "";
	mrowout = pTag->AddTag(strName, strValue);

	if(condDoEnclose(m_pLeft.get()))
	{
		strName  = "mrow";
		strValue = "";
		mrowleft = mrowout->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowleft->AddTag(strName, strValue);

			m_pLeft->SaveAsPresentationMathML(mrowleft, c);

			strName  = "mo";
			strValue = ")";
			mrowleft->AddTag(strName, strValue);
	}
	else
	{
		m_pLeft->SaveAsPresentationMathML(mrowout, c);
	}

	strName  = "mo";
	if(m_eLogicalOperator == eAnd)
		strOperator = "&And;";
	else if(m_eLogicalOperator == eOr)
		strOperator = "&Or;";
	else
		daeDeclareAndThrowException(exInvalidCall);  
	mrowout->AddTag(strName, strOperator);

	if(condDoEnclose(m_pRight.get()))
	{
		strName  = "mrow";
		strValue = "";
		mrowright = mrowout->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowright->AddTag(strName, strValue);

			m_pRight->SaveAsPresentationMathML(mrowright, c);

			strName  = "mo";
			strValue = ")";
			mrowright->AddTag(strName, strValue);
	}
	else
	{
		m_pRight->SaveAsPresentationMathML(mrowout, c);
	}
}

daeCondition condBinaryNode::CreateRuntimeNode(const daeExecutionContext* pExecutionContext) const
{
	daeCondition left  = m_pLeft->CreateRuntimeNode(pExecutionContext);
	daeCondition right = m_pRight->CreateRuntimeNode(pExecutionContext);

	switch(m_eLogicalOperator)
	{
	case eAnd:
		return (left && right);
		break;
	case eOr:
		return (left || right);
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return daeCondition();
	}
    return daeCondition();
}

bool condBinaryNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	bool bResult = false;
	switch(m_eLogicalOperator)
	{
	case eAnd:
		bResult = m_pLeft->Evaluate(pExecutionContext) && m_pRight->Evaluate(pExecutionContext);
		break;
	case eOr:
		bResult = m_pLeft->Evaluate(pExecutionContext) || m_pRight->Evaluate(pExecutionContext);
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
	}
	return bResult;
}

condNode* condBinaryNode::Clone(void) const
{
	return new condBinaryNode(*this);
}

void condBinaryNode::BuildExpressionsArray(vector< shared_ptr<adNode> > & ptrarrExpressions,
										   const daeExecutionContext* pExecutionContext,
										   real_t dEventTolerance)
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pLeft->BuildExpressionsArray(ptrarrExpressions, pExecutionContext, dEventTolerance);
	m_pRight->BuildExpressionsArray(ptrarrExpressions, pExecutionContext, dEventTolerance);
}

void condBinaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!m_pLeft)
		daeDeclareAndThrowException(exInvalidPointer); 
	if(!m_pRight)
		daeDeclareAndThrowException(exInvalidPointer); 
	m_pLeft->AddVariableIndexToArray(mapIndexes);
	m_pRight->AddVariableIndexToArray(mapIndexes);
}

}
}
