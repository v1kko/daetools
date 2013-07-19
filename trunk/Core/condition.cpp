#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace boost;

namespace dae 
{
namespace core 
{

/*********************************************************************************************
	daeCondition
**********************************************************************************************/
daeCondition::daeCondition()
{
	m_dEventTolerance = 0.0;
}

daeCondition::daeCondition(condNodePtr condition)
{
	if(!condition)
		daeDeclareAndThrowException(exInvalidPointer); 

	m_pConditionNode  = condition;
	m_pModel          = NULL;
	m_dEventTolerance = 0.0;
}

daeCondition::~daeCondition()
{
	m_pModel = NULL;
}

bool daeCondition::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pConditionNode)
		daeDeclareAndThrowException(exInvalidPointer); 

	return m_pConditionNode->Evaluate(pExecutionContext);
}

void daeCondition::Open(io::xmlTag_t* pTag)
{
	io::daeSerializable::Open(pTag);

	string strName = "Expression";
	condNode* node = condNode::OpenNode(pTag, strName);
	// Or m_pSetupConditionNode ???
	m_pConditionNode.reset(node);
}

void daeCondition::Save(io::xmlTag_t* pTag) const
{
	io::daeSerializable::Save(pTag);

	string strName = "Expression";
	condNode::SaveNode(pTag, strName, m_pSetupConditionNode.get());

	strName = "MathML";
	SaveNodeAsMathML(pTag, strName);

    strName = "EventTolerance";
	pTag->Save(strName, m_dEventTolerance);
}

void daeCondition::OpenRuntime(io::xmlTag_t* /*pTag*/)
{
}

void daeCondition::SaveRuntime(io::xmlTag_t* pTag) const
{
	io::daeSerializable::Save(pTag);

	string strName = "MathML";
	SaveNodeAsMathML(pTag, strName);
}

void daeCondition::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	m_pSetupConditionNode->Export(strContent, eLanguage, c);
}

string daeCondition::SaveNodeAsPlainText(void) const
{
	string strContent;
	daeModelExportContext c;
	
	c.m_pModel             = m_pModel;
	c.m_nPythonIndentLevel = 0;
	c.m_bExportDefinition  = true;

	m_pSetupConditionNode->Export(strContent, eCDAE, c);

	return strContent;
}

void daeCondition::SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue;
	daeNodeSaveAsContext c(m_pModel);
	
	condNode* node = m_pSetupConditionNode.get();
	if(!node)
		daeDeclareAndThrowException(exXMLIOError);

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
	node->SaveAsPresentationMathML(pMathMLTag, &c);
}

void daeCondition::BuildExpressionsArray(const daeExecutionContext* pExecutionContext)
{
	if(!m_pConditionNode)
		daeDeclareAndThrowException(exInvalidPointer);  

	m_ptrarrExpressions.clear();
	m_pConditionNode->BuildExpressionsArray(m_ptrarrExpressions, pExecutionContext, m_dEventTolerance);
}

void daeCondition::GetExpressionsArray(std::vector<adNode*>& ptrarrExpressions)
{
    for(size_t i = 0; i < m_ptrarrExpressions.size(); i++)
        ptrarrExpressions.push_back(m_ptrarrExpressions[i].get());
}

void daeCondition::SetEventTolerance(real_t dEventTolerance)
{
	m_dEventTolerance = dEventTolerance;
}

real_t daeCondition::GetEventTolerance(void) const
{
	return m_dEventTolerance;
}

daeCondition daeCondition::operator || (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = condNodePtr(new condBinaryNode(condNodePtr(m_pConditionNode->Clone()), 
	                                                      eOr, 
	                                                      condNodePtr(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator && (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = condNodePtr(new condBinaryNode(condNodePtr(m_pConditionNode->Clone()), 
	                                                      eAnd, 
	                                                      condNodePtr(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator | (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = condNodePtr(new condBinaryNode(condNodePtr(m_pConditionNode->Clone()), 
	                                                      eOr, 
	                                                      condNodePtr(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator & (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = condNodePtr(new condBinaryNode(condNodePtr(m_pConditionNode->Clone()), 
	                                                      eAnd, 
	                                                      condNodePtr(rCondition.m_pConditionNode->Clone())));
	return tmp;
}
daeCondition daeCondition::operator ! () const
{
	daeCondition tmp;
	tmp.m_pConditionNode = condNodePtr(new condUnaryNode(condNodePtr(m_pConditionNode->Clone()), 
	                                                     eNot));
	return tmp;
}

daeCondition::operator bool()
{
	return Evaluate(NULL);
}

}
}
