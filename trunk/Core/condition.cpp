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
	daeConfig& cfg = daeConfig::GetConfig();
	m_dEventTolerance = cfg.Get<real_t>("daetools.core.eventTolerance", 1E-7);
}

daeCondition::daeCondition(shared_ptr<condNode> condition)
{
	if(!condition)
		daeDeclareAndThrowException(exInvalidPointer); 

	daeConfig& cfg = daeConfig::GetConfig();
	m_pConditionNode  = condition;
	m_pModel          = NULL;
	m_dEventTolerance = cfg.Get<real_t>("daetools.core.eventTolerance", 1E-7);
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
	m_pConditionNode.reset(node);
}

void daeCondition::Save(io::xmlTag_t* pTag) const
{
	io::daeSerializable::Save(pTag);

	string strName = "Expression";
	condNode::SaveNode(pTag, strName, m_pConditionNode.get());

	strName = "MathML";
	SaveNodeAsMathML(pTag, strName);
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

string daeCondition::SaveNodeAsPlainText(void) const
{
	daeSaveAsMathMLContext c(m_pModel);
	return m_pConditionNode->SaveAsPlainText(&c);
}

void daeCondition::SaveNodeAsMathML(io::xmlTag_t* pTag, const string& strObjectName) const
{
	string strName, strValue;
	daeSaveAsMathMLContext c(m_pModel);
	condNode* node = m_pConditionNode.get();
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

void daeCondition::SetEventTolerance(real_t dEventTolerance)
{
	m_dEventTolerance = dEventTolerance;
}

real_t daeCondition::GetEventTolerance(void)
{
	return m_dEventTolerance;
}

daeCondition daeCondition::operator || (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = shared_ptr<condNode>(new condBinaryNode(shared_ptr<condNode>(m_pConditionNode->Clone()), 
						  									       eOr, 
																   shared_ptr<condNode>(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator && (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = shared_ptr<condNode>(new condBinaryNode(shared_ptr<condNode>(m_pConditionNode->Clone()), 
						  								           eAnd, 
																   shared_ptr<condNode>(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator | (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = shared_ptr<condNode>(new condBinaryNode(shared_ptr<condNode>(m_pConditionNode->Clone()), 
						  									       eOr, 
																   shared_ptr<condNode>(rCondition.m_pConditionNode->Clone())));
	return tmp;
}

daeCondition daeCondition::operator & (const daeCondition& rCondition) const
{
	daeCondition tmp;
	tmp.m_pConditionNode = shared_ptr<condNode>(new condBinaryNode(shared_ptr<condNode>(m_pConditionNode->Clone()), 
						  								           eAnd, 
																   shared_ptr<condNode>(rCondition.m_pConditionNode->Clone())));
	return tmp;
}
daeCondition daeCondition::operator ! () const
{
	daeCondition tmp;
	tmp.m_pConditionNode = shared_ptr<condNode>(new condUnaryNode(shared_ptr<condNode>(m_pConditionNode->Clone()), 
		                                                          eNot));
	return tmp;
}

daeCondition::operator bool()
{
	return Evaluate(NULL);
}

}
}
