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
                                const daeNodeSaveAsContext* c)
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
    m_pLeft				= adNodePtr(left.node->Clone());
    m_eConditionType	= type;
    m_pRight			= adNodePtr(right.node->Clone());
}

condExpressionNode::condExpressionNode(const adouble& left, daeeConditionType type, real_t right)
{
    if(!left.node)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pLeft				= adNodePtr(left.node->Clone());
    m_eConditionType	= type;
    m_pRight			= adNodePtr(new adConstantNode(right, UNITS(left.node)));
}

condExpressionNode::condExpressionNode(real_t left, daeeConditionType type, const adouble& right)
{
    if(!right.node)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pLeft				= adNodePtr(new adConstantNode(left, UNITS(right.node)));
    m_eConditionType	= type;
    m_pRight			= adNodePtr(right.node->Clone());
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
//	node->m_pLeft			= adNodePtr(left.node->Clone());
//	node->m_eConditionType	= type;
//	node->m_pRight			= adNodePtr(right.node->Clone());
//	return node;
//}
//
//condExpressionNode* condExpressionNode::Create(const adouble& left, daeeConditionType type, real_t right)
//{
//	condExpressionNode* node = new condExpressionNode();
//	if(!left.node)
//		daeDeclareAndThrowException(exInvalidPointer);
//	node->m_pLeft			= adNodePtr(left.node->Clone());
//	node->m_eConditionType	= type;
//	node->m_pRight			= adNodePtr(new adConstantNode(right));
//	return node;
//}
//
//condExpressionNode* condExpressionNode::Create(real_t left, daeeConditionType type, const adouble& right)
//{
//	condExpressionNode* node = new condExpressionNode();
//	if(!right.node)
//		daeDeclareAndThrowException(exInvalidPointer);
//	node->m_pLeft			= adNodePtr(new adConstantNode(left));
//	node->m_eConditionType	= type;
//	node->m_pRight			= adNodePtr(right.node->Clone());
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

void condExpressionNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    m_pLeft->Export(strContent, eLanguage, c);

    if(m_eConditionType == eNotEQ)
        strContent += " != ";
    else if(m_eConditionType == eEQ)
        strContent += " == ";
    else if(m_eConditionType == eGT)
        strContent += " > ";
    else if(m_eConditionType == eGTEQ)
        strContent += " >= ";
    else if(m_eConditionType == eLT)
        strContent += " < ";
    else if(m_eConditionType == eLTEQ)
        strContent += " <= ";
    else
        daeDeclareAndThrowException(exNotImplemented);

    m_pRight->Export(strContent, eLanguage, c);
}

//string condExpressionNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strValue;
//
//	strValue += m_pLeft->SaveAsPlainText(c);
//
//	if(m_eConditionType == eNotEQ)
//		strValue += " != ";
//	else if(m_eConditionType == eEQ)
//		strValue += " == ";
//	else if(m_eConditionType == eGT)
//		strValue += " > ";
//	else if(m_eConditionType == eGTEQ)
//		strValue += " >= ";
//	else if(m_eConditionType == eLT)
//		strValue += " < ";
//	else if(m_eConditionType == eLTEQ)
//		strValue += " <= ";
//	else
//		daeDeclareAndThrowException(exInvalidCall);
//
//	strValue += m_pRight->SaveAsPlainText(c);
//
//	return strValue;
//}

string condExpressionNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    boost::format fmt("{{\\left( {%s} \\right)} %s {\\left( {%s} \\right)}}");
    string lnode = m_pLeft->SaveAsLatex(c);
    string rnode = m_pRight->SaveAsLatex(c);

    if(m_eConditionType == eNotEQ)
        return (fmt % lnode % "\\neq" % rnode).str();
    else if(m_eConditionType == eEQ)
        return (fmt % lnode % "=" % rnode).str();
    else if(m_eConditionType == eGT)
        return (fmt % lnode % ">" % rnode).str();
    else if(m_eConditionType == eGTEQ)
        return (fmt % lnode % "\\geq" % rnode).str();
    else if(m_eConditionType == eLT)
        return (fmt % lnode % "<" % rnode).str();
    else if(m_eConditionType == eLTEQ)
        return (fmt % lnode % "\\leq" % rnode).str();
    else
        daeDeclareAndThrowException(exNotImplemented);

    return string("{}");
}

void condExpressionNode::SaveAsContentMathML(io::xmlTag_t* /*pTag*/, const daeNodeSaveAsContext* /*c*/) const
{
}

void condExpressionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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
    case eEQ:
        return (left == right);
    case eGT:
        return (left > right);
    case eGTEQ:
        return (left >= right);
    case eLT:
        return (left < right);
    case eLTEQ:
        return (left <= right);
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
        return (left != right);
    case eEQ:
        return (left == right);
    case eGT:
        return (left > right);
    case eGTEQ:
        return (left >= right);
    case eLT:
        return (left < right);
    case eLTEQ:
        return (left <= right);
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return false;
    }
}

bool condExpressionNode::GetQuantity(void) const
{
    if(!m_pLeft)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pRight)
        daeDeclareAndThrowException(exInvalidPointer);

    quantity left  = m_pLeft->GetQuantity();
    quantity right = m_pRight->GetQuantity();

    switch(m_eConditionType)
    {
    case eNotEQ:
        return (left != right);
    case eEQ:
        return (left == right);
    case eGT:
        return (left > right);
    case eGTEQ:
        return (left >= right);
    case eLT:
        return (left < right);
    case eLTEQ:
        return (left <= right);
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return false;
    }
}

condNode* condExpressionNode::Clone(void) const
{
    return new condExpressionNode(*this);
}

void condExpressionNode::BuildExpressionsArray(vector< adNodePtr > & ptrarrExpressions,
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

// This have to be added always!
    dae_push_back(ptrarrExpressions, ad.node);

// If not set, set it to some default value
    if(dEventTolerance == 0)
    {
        daeConfig& cfg  = daeConfig::GetConfig();
        dEventTolerance = cfg.GetFloat("daetools.core.eventTolerance", 1E-7);
    }

/*
    Depending on the type I have to add some additional expressions
    Explanation:

left-right = f(t)
    ^
    |         /
    |        /
    |       X (2): ad2 = ad - Et   (l - r = +Et, therefore: l - r - Et = 0)
 ---|------X--(0)----------------------------------------------------------------> Time
    |     X   (1): ad1 = ad + Et   (l - r = -Et, therefore: l - r + Et = 0)
    |    /
    |   /
    |


*/
    adouble ad1 = ad + dEventTolerance;
    adouble ad2 = ad - dEventTolerance;
    switch(m_eConditionType)
    {
    case eNotEQ: // !=
        dae_push_back(ptrarrExpressions, ad1.node);
        dae_push_back(ptrarrExpressions, ad2.node);
        break;

    case eEQ: // ==
        dae_push_back(ptrarrExpressions, ad1.node);
        dae_push_back(ptrarrExpressions, ad2.node);
        break;

    case eGT: // >
        dae_push_back(ptrarrExpressions, ad2.node);
        break;

    case eGTEQ: // >=
        dae_push_back(ptrarrExpressions, ad1.node);
        break;

    case eLT: // <
        dae_push_back(ptrarrExpressions, ad1.node);
        break;

    case eLTEQ: // <=
        dae_push_back(ptrarrExpressions, ad2.node);
        break;

    default:
        daeDeclareAndThrowException(exNotImplemented);
    }
}

void condExpressionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!m_pLeft)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pRight)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pLeft->AddVariableIndexToArray(mapIndexes, bAddFixed);
    m_pRight->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

/*********************************************************************************************
    condUnaryNode
**********************************************************************************************/
condUnaryNode::condUnaryNode(condNodePtr node, daeeLogicalUnaryOperator op)
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

void condUnaryNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    if(eLanguage == eCDAE)
    {
        if(m_eLogicalOperator == eNot)
            strContent += "!";
        else
            daeDeclareAndThrowException(exNotImplemented);
    }
    else if(eLanguage == ePYDAE)
    {
        if(m_eLogicalOperator == eNot)
            strContent += "NOT ";
        else
            daeDeclareAndThrowException(exNotImplemented);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += "(";
    m_pNode->Export(strContent, eLanguage, c);
    strContent += ")";
}
//string condUnaryNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strValue;
//
//	if(m_eLogicalOperator == eNot)
//		strValue += "not ";
//	else
//		daeDeclareAndThrowException(exInvalidCall);
//
//	strValue += "(";
//	strValue += m_pNode->SaveAsPlainText(c);
//	strValue += ")";
//
//	return strValue;
//}

string condUnaryNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    if(m_eLogicalOperator == eNot)
    {
        boost::format fmt("{\\lnot \\left( {%s} \\right)}");
        string lnode = m_pNode->SaveAsLatex(c);
        return (fmt % lnode).str();
    }
    else
    {
        daeDeclareAndThrowException(exNotImplemented);
    }

    return string("{}");
}

void condUnaryNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void condUnaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    string strName, strValue;
    io::xmlTag_t *mrowout, *mrow;

    strName  = "mrow";
    strValue = "";
    mrow = pTag->AddTag(strName, strValue);

    if(m_eLogicalOperator == eNot)
    {
        strName  = "mrow";
        strValue = "";
        mrowout = mrow->AddTag(strName, strValue);
            strName  = "mo";
            strValue = "&not;";
            mrowout->AddTag(strName, strValue);

            strName  = "mo";
            strValue = "(";
            mrowout->AddTag(strName, strValue);
                m_pNode->SaveAsPresentationMathML(mrowout, c);
            strName  = "mo";
            strValue = ")";
            mrowout->AddTag(strName, strValue);
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
    //	return condNodePtr();
    //}
}

bool condUnaryNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    switch(m_eLogicalOperator)
    {
    case eNot:
        return (!m_pNode->Evaluate(pExecutionContext));
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return true;
    }
}

bool condUnaryNode::GetQuantity(void) const
{
    switch(m_eLogicalOperator)
    {
    case eNot:
        return (!m_pNode->GetQuantity());
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return true;
    }
}

condNode* condUnaryNode::Clone(void) const
{
    return new condUnaryNode(*this);
}

void condUnaryNode::BuildExpressionsArray(vector< adNodePtr > & ptrarrExpressions,
                                          const daeExecutionContext* pExecutionContext,
                                          real_t dEventTolerance)
{
    if(!m_pNode)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pNode->BuildExpressionsArray(ptrarrExpressions, pExecutionContext, dEventTolerance);
}

void condUnaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!m_pNode)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pNode->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

/*********************************************************************************************
    condBinaryNode
**********************************************************************************************/
condBinaryNode::condBinaryNode(condNodePtr left, daeeLogicalBinaryOperator op, condNodePtr right)
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

void condBinaryNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    strContent += "(";
    m_pLeft->Export(strContent, eLanguage, c);
    strContent += ")";

    if(eLanguage == eCDAE)
    {
        if(m_eLogicalOperator == eAnd)
            strContent += " && ";
        else if(m_eLogicalOperator == eOr)
            strContent += " || ";
        else
            daeDeclareAndThrowException(exNotImplemented);
    }
    else if(eLanguage == ePYDAE)
    {
        if(m_eLogicalOperator == eAnd)
            strContent += " & ";
        else if(m_eLogicalOperator == eOr)
            strContent += " | ";
        else
            daeDeclareAndThrowException(exNotImplemented);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += "(";
    m_pRight->Export(strContent, eLanguage, c);
    strContent += ")";
}
//string condBinaryNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strValue;
//
//	strValue += "(";
//	strValue += m_pLeft->SaveAsPlainText(c);
//	strValue += ")";
//
//	if(m_eLogicalOperator == eAnd)
//		strValue += " and ";
//	else if(m_eLogicalOperator == eOr)
//		strValue += " or ";
//	else
//		daeDeclareAndThrowException(exNotImplemented);
//
//	strValue += "(";
//	strValue += m_pRight->SaveAsPlainText(c);
//	strValue += ")";
//
//	return strValue;
//}

string condBinaryNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    boost::format fmt("{{\\left( {%s} \\right)} %s {\\left( {%s} \\right)}}");
    string lnode = m_pLeft->SaveAsLatex(c);
    string rnode = m_pRight->SaveAsLatex(c);

    if(m_eLogicalOperator == eAnd)
        return (fmt % lnode % "\\land" % rnode).str();
    else if(m_eLogicalOperator == eOr)
        return (fmt % lnode % "\\lor" % rnode).str();
    else
        daeDeclareAndThrowException(exNotImplemented);

    return string("{}");
}

void condBinaryNode::SaveAsContentMathML(io::xmlTag_t* /*pTag*/, const daeNodeSaveAsContext* /*c*/) const
{
}

void condBinaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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
        strOperator = "&#x2227;";
    else if(m_eLogicalOperator == eOr)
        strOperator = "&#x2228;";
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
    bool left  = m_pLeft->Evaluate(pExecutionContext);
    bool right = m_pRight->Evaluate(pExecutionContext);

    switch(m_eLogicalOperator)
    {
    case eAnd:
        return (left && right);
    case eOr:
        return (left || right);
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return true;
    }
}

bool condBinaryNode::GetQuantity(void) const
{
    bool left  = m_pLeft->GetQuantity();
    bool right = m_pRight->GetQuantity();

    switch(m_eLogicalOperator)
    {
    case eAnd:
        return (left && right);
    case eOr:
        return (left || right);
    default:
        daeDeclareAndThrowException(exNotImplemented);
        return true;
    }
}

condNode* condBinaryNode::Clone(void) const
{
    return new condBinaryNode(*this);
}

void condBinaryNode::BuildExpressionsArray(vector< adNodePtr > & ptrarrExpressions,
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

void condBinaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!m_pLeft)
        daeDeclareAndThrowException(exInvalidPointer);
    if(!m_pRight)
        daeDeclareAndThrowException(exInvalidPointer);
    m_pLeft->AddVariableIndexToArray(mapIndexes, bAddFixed);
    m_pRight->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

}
}
