#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"
using namespace dae;
#include "xmlfunctions.h"
#include <typeinfo>
using namespace dae::xml;
using namespace boost;

namespace dae 
{
namespace core 
{
bool adDoEnclose(const adNode* node)
{
	if(!node)
		return true;

	const type_info& infoChild  = typeid(*node);

// If it is simple node DO NOT enclose in brackets
	if(infoChild == typeid(const adConstantNode)					|| 
	   infoChild == typeid(const adDomainIndexNode)					|| 
	   infoChild == typeid(const adRuntimeParameterNode)			|| 
	   infoChild == typeid(const adRuntimeVariableNode)				|| 
	   infoChild == typeid(const adRuntimeTimeDerivativeNode)		|| 
	   infoChild == typeid(const adRuntimePartialDerivativeNode)	|| 
	   infoChild == typeid(const adSetupDomainIteratorNode)			|| 
	   infoChild == typeid(const adSetupParameterNode)				|| 
	   infoChild == typeid(const adSetupVariableNode)				|| 
	   infoChild == typeid(const adSetupTimeDerivativeNode)			|| 
	   infoChild == typeid(const adSetupPartialDerivativeNode))
	{
		return false;
	}
	else if(infoChild == typeid(const adUnaryNode))
	{
		const adUnaryNode* pUnaryNode = dynamic_cast<const adUnaryNode*>(node);
		if(pUnaryNode->eFunction == eSign)
			return true;
		else
			return false;
	}
	else
	{
		return true;
	}
}

void adDoEnclose(const adNode* parent, 
				 const adNode* left,  bool& bEncloseLeft, 
				 const adNode* right, bool& bEncloseRight)
{
	bEncloseLeft  = true;
	bEncloseRight = true;

	if(!parent || !left || !right)
		return;

	const type_info& infoParent = typeid(*parent);
	const type_info& infoLeft   = typeid(*left);
	const type_info& infoRight  = typeid(*right);

// The parent must be binary node
	if(infoParent != typeid(const adBinaryNode))
		return;

	const adBinaryNode* pBinaryParent = dynamic_cast<const adBinaryNode*>(parent);

// If left is binary 
	if(infoLeft == typeid(const adBinaryNode))
	{
		const adBinaryNode* pBinaryLeft = dynamic_cast<const adBinaryNode*>(left);

		if(pBinaryParent->eFunction == ePlus) 
		{ 
			// whatever + right
			bEncloseLeft = false;
		}
		else if(pBinaryParent->eFunction == eMinus)
		{
			// whatever - right
			bEncloseLeft = false;
		}
		else if(pBinaryParent->eFunction == eMulti)
		{
			if(pBinaryLeft->eFunction == ePlus) // (a + b) * right
				bEncloseLeft = true;
			else if(pBinaryLeft->eFunction == eMinus) // (a - b) * right
				bEncloseLeft = true;
			else if(pBinaryLeft->eFunction == eMulti) // a * b * right
				bEncloseLeft = false;
			else if(pBinaryLeft->eFunction == eDivide ||
					pBinaryLeft->eFunction == ePower  ||
					pBinaryLeft->eFunction == eMin    ||
					pBinaryLeft->eFunction == eMax)
				bEncloseLeft = false;
			else
				bEncloseLeft = true;
		}
		else if(pBinaryParent->eFunction == eDivide)
		{
			bEncloseLeft = false;
		}
		else if(pBinaryParent->eFunction == ePower ||
				pBinaryParent->eFunction == eMin   ||
				pBinaryParent->eFunction == eMax)
		{
			bEncloseLeft = false;
		}
		else
		{
			bEncloseLeft = true;
		}
	}
	else
	{
		bEncloseLeft = adDoEnclose(left);
	}

// If right is binary 
	if(infoRight == typeid(const adBinaryNode))
	{
		const adBinaryNode* pBinaryRight = dynamic_cast<const adBinaryNode*>(right);

		if(pBinaryParent->eFunction == ePlus) 
		{ 
			// left + whatever
			bEncloseRight = false;
		}
		else if(pBinaryParent->eFunction == eMinus)
		{
			if(pBinaryRight->eFunction == ePlus) // left - (a + b)
				bEncloseRight = true;
			else if(pBinaryRight->eFunction == eMinus) // left - (a - b)
				bEncloseRight = true;
			else if(pBinaryRight->eFunction == eMulti) // left - a * b
				bEncloseRight = false;
			else if(pBinaryRight->eFunction == eDivide ||
					pBinaryRight->eFunction == ePower  ||
					pBinaryRight->eFunction == eMin    ||
					pBinaryRight->eFunction == eMax)
				bEncloseRight = false;
			else
				bEncloseRight = true;
		}
		else if(pBinaryParent->eFunction == eMulti)
		{
			if(pBinaryRight->eFunction == ePlus) // left * (a + b)
				bEncloseRight = true;
			else if(pBinaryRight->eFunction == eMinus) // left * (a - b)
				bEncloseRight = true;
			else if(pBinaryRight->eFunction == eMulti) // left * a * b
				bEncloseRight = false;
			else if(pBinaryRight->eFunction == eDivide ||
					pBinaryRight->eFunction == ePower  ||
					pBinaryRight->eFunction == eMin    ||
					pBinaryRight->eFunction == eMax)
				bEncloseRight = false;
			else
				bEncloseRight = true;
		}
		else if(pBinaryParent->eFunction == eDivide)
		{
			bEncloseRight = false;
		}
		else if(pBinaryParent->eFunction == ePower ||
				pBinaryParent->eFunction == eMin   ||
				pBinaryParent->eFunction == eMax)
		{
			bEncloseRight = false;
		}
		else
		{
			bEncloseRight = true;
		}
	}
	else
	{
		bEncloseRight = adDoEnclose(right);
	}
}

/*********************************************************************************************
	adNode
**********************************************************************************************/
adNode* adNode::CreateNode(const io::xmlTag_t* pTag)
{
	string strClass;
	string strName = "Class";

	io::xmlAttribute_t* pAttrClass = pTag->FindAttribute(strName);
	if(!pAttrClass)
		daeDeclareAndThrowException(exXMLIOError);

	pAttrClass->GetValue(strClass);
	if(strClass == "adConstantNode")
	{
		return new adConstantNode();
	}
	else if(strClass == "adDomainIndexNode")
	{
		return new adDomainIndexNode();
	}
	else if(strClass == "adRuntimeParameterNode")
	{
		return new adRuntimeParameterNode();
	}
	else if(strClass == "adRuntimeVariableNode")
	{
		return new adRuntimeVariableNode();
	}
	else if(strClass == "adRuntimeTimeDerivativeNode")
	{
		return new adRuntimeTimeDerivativeNode();
	}
	else if(strClass == "adRuntimePartialDerivativeNode")
	{
		return new adRuntimePartialDerivativeNode();
	}
	else if(strClass == "adUnaryNode")
	{
		return new adUnaryNode();
	}
	else if(strClass == "adBinaryNode")
	{
		return new adBinaryNode();
	}
	else if(strClass == "adSetupDomainIteratorNode")
	{
		return new adSetupDomainIteratorNode();
	}
	else if(strClass == "adSetupParameterNode")
	{
		return new adSetupParameterNode();
	}
	else if(strClass == "adSetupVariableNode")
	{
		return new adSetupVariableNode();
	}
	else if(strClass == "adSetupTimeDerivativeNode")
	{
		return new adSetupPartialDerivativeNode();
	}
	else if(strClass == "adSetupPartialDerivativeNode")
	{
		return new adSetupPartialDerivativeNode();
	}
	else if(strClass == "adSetupIntegralNode")
	{
		return new adSetupIntegralNode();
	}
	else if(strClass == "adSetupSpecialFunctionNode")
	{
		return new adSetupSpecialFunctionNode();
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError);
		return NULL;
	}
	return NULL;
}

void adNode::SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const adNode* node)
{
	io::xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);
	node->Save(pChildTag);
}

adNode* adNode::OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<adNode>* ood)
{
	io::xmlTag_t* pChildTag = pTag->FindTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	adNode* node = adNode::CreateNode(pChildTag);
	if(!node)
		daeDeclareAndThrowException(exXMLIOError);

	if(ood)
		ood->BeforeOpenObject(node);
	node->Open(pChildTag);
	if(ood)
		ood->AfterOpenObject(node);

	return node;
}

void adNode::SaveNodeAsMathML(io::xmlTag_t* pTag, 
							  const string& strObjectName, 
							  const adNode* node,
							  const daeSaveAsMathMLContext* c,
							  bool bAppendEqualToZero)
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

	strName = "mrow";
	io::xmlTag_t* pMRowTag = pMathMLTag->AddTag(strName);
	if(!pMRowTag)
		daeDeclareAndThrowException(exXMLIOError);

	node->SaveAsPresentationMathML(pMRowTag, c);

	if(bAppendEqualToZero)
	{
		strName  = "mo";
		strValue = "=";
		pMRowTag->AddTag(strName, strValue);
	
		strName  = "mo";
		strValue = "0";
		pMRowTag->AddTag(strName, strValue);
	}
}

/*********************************************************************************************
	adNodeImpl
**********************************************************************************************/
void adNodeImpl::ExportAsPlainText(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsPlainText(NULL);
	file.close();
}

void adNodeImpl::ExportAsLatex(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsLatex(NULL);
	file.close();
}

/*********************************************************************************************
	adConstantNode
**********************************************************************************************/
adConstantNode::adConstantNode(const real_t d)
	          : m_dValue(d)
{
}

adConstantNode::adConstantNode()
{
	m_dValue = 0;
}

adConstantNode::~adConstantNode()
{
}

adouble adConstantNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble tmp(m_dValue, 0);
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
	}
	return tmp;
}

adNode* adConstantNode::Clone(void) const
{
	return new adConstantNode(*this);
}

string adConstantNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
{
	return textCreator::Constant(m_dValue);
}

string adConstantNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return latexCreator::Constant(m_dValue);
}

void adConstantNode::Open(io::xmlTag_t* pTag)
{
	string strName = "Value";
	pTag->Open(strName, m_dValue);
}

void adConstantNode::Save(io::xmlTag_t* pTag) const
{
	string strName = "Value";
	pTag->Save(strName, m_dValue);
}

void adConstantNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	xmlContentCreator::Constant(pTag, m_dValue);
}

void adConstantNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	xmlPresentationCreator::Constant(pTag, m_dValue);
}

void adConstantNode::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adRuntimeParameterNode
**********************************************************************************************/
adRuntimeParameterNode::adRuntimeParameterNode(daeParameter* pParameter, 
											   vector<size_t>& narrDomains, 
											   real_t dValue) 
               : m_dValue(dValue),
			     m_pParameter(pParameter), 
			     m_narrDomains(narrDomains)
{
}

adRuntimeParameterNode::adRuntimeParameterNode(void)
{
	m_pParameter = NULL;
	m_dValue     = 0;
}

adRuntimeParameterNode::~adRuntimeParameterNode()
{
}

adouble adRuntimeParameterNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in evaluate mode we dont need the value
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
		return tmp;
	}
	return adouble(m_dValue);
}

adNode* adRuntimeParameterNode::Clone(void) const
{
	return new adRuntimeParameterNode(*this);
}

string adRuntimeParameterNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return textCreator::Variable(strName, strarrIndexes);
}

string adRuntimeParameterNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adRuntimeParameterNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	//strName = "Parameter";
	//m_pParameter = pTag->OpenObjectRef(strName);

	strName = "DomainIndexes";
	pTag->OpenArray(strName, m_narrDomains);

	strName = "Value";
	pTag->Open(strName, m_dValue);
}

void adRuntimeParameterNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pParameter->GetName());

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomains);

	strName = "Value";
	pTag->Save(strName, m_dValue);
}

void adRuntimeParameterNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeParameterNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeParameterNode::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adDomainIndexNode
**********************************************************************************************/
adDomainIndexNode::adDomainIndexNode(daeDomain* pDomain, size_t nIndex)
			     : m_pDomain(pDomain), 
				   m_nIndex(nIndex)				   
{
}

adDomainIndexNode::adDomainIndexNode()
{
	m_pDomain = NULL;
	m_nIndex  = ULONG_MAX;
}

adDomainIndexNode::~adDomainIndexNode()
{
}

adouble adDomainIndexNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// adDomainIndexNode is not consistent with the other nodes.
// It represents a runtime and a setup node at the same time.
// Setup nodes should create runtime nodes in its function Evaluate().
// Here I check if I am inside of the GatherInfo mode and if I am
// I clone the node (which is an equivalent for creation of a runtime node)
// If I am not - I return the value of the point for the given index.
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
		return tmp;
	}
	
	return adouble(m_pDomain->GetPoint(m_nIndex));
}

adNode* adDomainIndexNode::Clone(void) const
{
	return new adDomainIndexNode(*this);
}

string adDomainIndexNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strName  = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	string strIndex = toString<size_t>(m_nIndex);
	return textCreator::Domain(strName, strIndex);
}

string adDomainIndexNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strName  = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	string strIndex = toString<size_t>(m_nIndex);
	return latexCreator::Domain(strName, strIndex);
}

void adDomainIndexNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adDomainIndexNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pDomain->GetName());

	strName = "Index";
	pTag->Save(strName, m_nIndex);
}

void adDomainIndexNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adDomainIndexNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName  = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	string strIndex = toString<size_t>(m_nIndex);
	xmlPresentationCreator::Domain(pTag, strName, strIndex);
}

void adDomainIndexNode::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adRuntimeVariableNode
**********************************************************************************************/
adRuntimeVariableNode::adRuntimeVariableNode(daeVariable* pVariable, 
											 size_t nOverallIndex, 
											 vector<size_t>& narrDomains, 
											 real_t* pdValue) 
               : m_pdValue(pdValue),
			     m_nOverallIndex(nOverallIndex), 
				 m_pVariable(pVariable), 
				 m_narrDomains(narrDomains)
{
}

adRuntimeVariableNode::adRuntimeVariableNode()
{
	m_pVariable = NULL;
	m_pdValue   = NULL;
	m_nOverallIndex = ULONG_MAX;
}

adRuntimeVariableNode::~adRuntimeVariableNode()
{
}

adouble adRuntimeVariableNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in evaluate mode we dont need the value
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
		return tmp;
	}
	
	if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivity)
	{
		// If m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex that means that 
		// the variable is fixed and its sensitivity derivative per given parameter is 1.
		// If it is not - it is a normal state variable and its sensitivity derivative is m_pdSValue
		adouble value;
		value.setValue(*m_pdValue);
		
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
			value.setDerivative(1);
		else
			value.setDerivative(pExecutionContext->m_pDataProxy->GetSValue(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation,
																		   m_nOverallIndex) );
		return value;
	}
	else if(pExecutionContext->m_eEquationCalculationMode == eCalculateGradient)
	{
		return adouble(*m_pdValue, (pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex ? 1 : 0) );
	}
	else
	{
		return adouble(*m_pdValue, (pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == m_nOverallIndex ? 1 : 0) );
	}
}

adNode* adRuntimeVariableNode::Clone(void) const
{
	return new adRuntimeVariableNode(*this);
}

string adRuntimeVariableNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::Variable(strName, strarrIndexes);
}

string adRuntimeVariableNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::Variable(strName, strarrIndexes);
}

void adRuntimeVariableNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	//strName = "Name";
	//pTag->Open(strName, m_pVariable->GetName());

	strName = "OverallIndex";
	pTag->Open(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->OpenArray(strName, m_narrDomains);

	//strName = "Value";
	//pTag->Open(strName, *m_pdValue);
}

void adRuntimeVariableNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "OverallIndex";
	pTag->Save(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomains);

	strName = "Value";
	pTag->Save(strName, *m_pdValue);
}

void adRuntimeVariableNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeVariableNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeVariableNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	pair<size_t, size_t> mapPair(m_nOverallIndex, mapIndexes.size());
	mapIndexes.insert(mapPair);
}

/*********************************************************************************************
	adRuntimeTimeDerivativeNode
**********************************************************************************************/
adRuntimeTimeDerivativeNode::adRuntimeTimeDerivativeNode(daeVariable* pVariable, 
														 size_t nOverallIndex, 
														 size_t nDegree, 
														 vector<size_t>& narrDomains,
														 real_t* pdTimeDerivative)
               : m_pdTimeDerivative(pdTimeDerivative), 
			     m_nOverallIndex(nOverallIndex), 
				 m_nDegree(nDegree), 
				 m_pVariable(pVariable),
				 m_narrDomains(narrDomains)
{
}

adRuntimeTimeDerivativeNode::adRuntimeTimeDerivativeNode(void)
{	
	m_pVariable        = NULL;
	m_nDegree          = 0;
	m_nOverallIndex    = ULONG_MAX;
	m_pdTimeDerivative = NULL;
}

adRuntimeTimeDerivativeNode::~adRuntimeTimeDerivativeNode(void)
{
}

adouble adRuntimeTimeDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in evaluate mode we dont need the value
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
		return tmp;
	}

	if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivity)
	{
		// Here m_nCurrentVariableIndexForJacobianEvaluation MUST NOT be equal to m_nOverallIndex,
		// because it would mean a time derivative for the assigned variable (that is a sensitivity parameter)!! 
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
			daeDeclareAndThrowException(exInvalidCall)

		adouble value;
		value.setValue(*m_pdTimeDerivative);
		
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
			value.setDerivative(1);
		else
			value.setDerivative(pExecutionContext->m_pDataProxy->GetSDValue(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation,
																		    m_nOverallIndex) );
		return value;
	}
	else if(pExecutionContext->m_eEquationCalculationMode == eCalculateGradient)
	{
		// Here m_nCurrentVariableIndexForJacobianEvaluation MUST NOT be equal to m_nOverallIndex,
		// because it would mean a time derivative for the assigned variable (that is a sensitivity parameter)!! 
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
			daeDeclareAndThrowException(exInvalidCall)

		return adouble(*m_pdTimeDerivative, 0);
	}
	else
	{
		return adouble(*m_pdTimeDerivative, 
					  (pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == m_nOverallIndex ? pExecutionContext->m_dInverseTimeStep : 0) );

	}
}

adNode* adRuntimeTimeDerivativeNode::Clone(void) const
{
	return new adRuntimeTimeDerivativeNode(*this);
}

string adRuntimeTimeDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return textCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

string adRuntimeTimeDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	return latexCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
}

void adRuntimeTimeDerivativeNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	//strName = "Name";
	//pTag->Open(strName, m_pVariable->GetName());

	strName = "Degree";
	pTag->Open(strName, m_nDegree);

	strName = "OverallIndex";
	pTag->Open(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->OpenArray(strName, m_narrDomains);

	//strName = "TimeDerivative";
	//pTag->Open(strName, *m_pdTimeDerivative);
}

void adRuntimeTimeDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "OverallIndex";
	pTag->Save(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomains);

	strName = "TimeDerivative";
	pTag->Save(strName, *m_pdTimeDerivative);
}

void adRuntimeTimeDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adRuntimeTimeDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adRuntimeTimeDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	pair<size_t, size_t> mapPair(m_nOverallIndex, mapIndexes.size());
	mapIndexes.insert(mapPair);
}

/*********************************************************************************************
	adRuntimePartialDerivativeNode
**********************************************************************************************/
adRuntimePartialDerivativeNode::adRuntimePartialDerivativeNode(daeVariable* pVariable, 
															   size_t nOverallIndex, 
															   size_t nDegree, 
															   vector<size_t>& narrDomains, 
															   daeDomain* pDomain, 
															   shared_ptr<adNode> pdnode)
               : pardevnode(pdnode),  
			     m_nOverallIndex(nOverallIndex), 
				 m_nDegree(nDegree), 
				 m_pVariable(pVariable),
				 m_pDomain(pDomain), 				 
				 m_narrDomains(narrDomains)
{
}

adRuntimePartialDerivativeNode::adRuntimePartialDerivativeNode()
{	
	m_pVariable = NULL;
	m_pDomain   = NULL;
	m_nDegree   = 0;
	m_nOverallIndex = ULONG_MAX;
}

adRuntimePartialDerivativeNode::~adRuntimePartialDerivativeNode()
{
}

adouble adRuntimePartialDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in evaluate mode we dont need the value
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNode>( Clone() );
		return tmp;
	}

	return pardevnode->Evaluate(pExecutionContext);
}

adNode* adRuntimePartialDerivativeNode::Clone(void) const
{
	return new adRuntimePartialDerivativeNode(*this);
}

string adRuntimePartialDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

//	string strVariableName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName   = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
//	return textCreator::PartialDerivative(m_nDegree, strVariableName, strDomainName, strarrIndexes);
	return pardevnode->SaveAsPlainText(c);
}

string adRuntimePartialDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

//	string strVariableName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName   = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
//	return latexCreator::PartialDerivative(m_nDegree, strVariableName, strDomainName, strarrIndexes);
	return pardevnode->SaveAsLatex(c);
}

void adRuntimePartialDerivativeNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Domain";
	pTag->Save(strName, m_pDomain->GetName());

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "OverallIndex";
	pTag->Save(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomains);

	strName = "ParDevNode";
	adNode* node = adNode::OpenNode(pTag, strName);
	pardevnode.reset(node);
}

void adRuntimePartialDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Domain";
	pTag->Save(strName, m_pDomain->GetName());

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "OverallIndex";
	pTag->Save(strName, m_nOverallIndex);

	strName = "DomainIndexes";
	pTag->SaveArray(strName, m_narrDomains);

	strName = "ParDevNode";
	adNode::SaveNode(pTag, strName, pardevnode.get());
}

void adRuntimePartialDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimePartialDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

//	string strVariableName = daeObject::GetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName   = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
//	xmlPresentationCreator::PartialDerivative(pTag, m_nDegree, strVariableName, strDomainName, strarrIndexes);
	return pardevnode->SaveAsPresentationMathML(pTag, c);
}

void adRuntimePartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!pardevnode)
		daeDeclareAndThrowException(exInvalidPointer);
	pardevnode->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adUnaryNode
**********************************************************************************************/
adUnaryNode::adUnaryNode(daeeUnaryFunctions eFun, shared_ptr<adNode> n)
{
	node = n;
	eFunction = eFun;
}

adUnaryNode::adUnaryNode()
{
	eFunction = eUFUnknown;
}

adUnaryNode::~adUnaryNode()
{
}

adouble adUnaryNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	switch(eFunction)
	{
	case eSign:
		return -(node->Evaluate(pExecutionContext));
		break;
	case eSin:
		return sin(node->Evaluate(pExecutionContext));
		break;
	case eCos:
		return cos(node->Evaluate(pExecutionContext));
		break;
	case eTan:
		return tan(node->Evaluate(pExecutionContext));
		break;
	case eArcSin:
		return asin(node->Evaluate(pExecutionContext));
		break;
	case eArcCos:
		return acos(node->Evaluate(pExecutionContext));
		break;
	case eArcTan:
		return atan(node->Evaluate(pExecutionContext));
		break;
	case eSqrt:
		return sqrt(node->Evaluate(pExecutionContext));
		break;
	case eExp:
		return exp(node->Evaluate(pExecutionContext));
		break;
	case eLn:
		return log(node->Evaluate(pExecutionContext));
		break;
	case eLog:
		return log10(node->Evaluate(pExecutionContext));
		break;
	case eAbs:
		return abs(node->Evaluate(pExecutionContext));
		break;
	case eCeil:
		return ceil(node->Evaluate(pExecutionContext));
		break;
	case eFloor:
		return floor(node->Evaluate(pExecutionContext));
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
}

adNode* adUnaryNode::Clone(void) const
{
	shared_ptr<adNode> n = shared_ptr<adNode>( (node ? node->Clone() : NULL) );
	return new adUnaryNode(eFunction, n);
}

string adUnaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	switch(eFunction)
	{
	case eSign:
		strResult += "(-";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eSin:
		strResult += "sin(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eCos:
		strResult += "cos(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eTan:
		strResult += "tan(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eArcSin:
		strResult += "asin(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eArcCos:
		strResult += "acos(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eArcTan:
		strResult += "atan(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eSqrt:
		strResult += "sqrt(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eExp:
		strResult += "exp(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eLn:
		strResult += "log(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eLog:
		strResult += "log10(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eAbs:
		strResult += "abs(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eCeil:
		strResult += "ceil(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eFloor:
		strResult += "floor(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adUnaryNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	switch(eFunction)
	{
	case eSign:
	strResult  = "{ "; // Start
		strResult += "\\left( - ";
		strResult += node->SaveAsLatex(c);
		strResult += "\\right) ";
	strResult  += "} "; // End
		break;
	case eSin:
		strResult += "\\sin";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eCos:
		strResult += "\\cos";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eTan:
		strResult += "\\tan";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eArcSin:
		strResult += "\\arcsin";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eArcCos:
		strResult += "\\arccos";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eArcTan:
		strResult += "\\arctan";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eSqrt:
		strResult += "\\sqrt";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	case eExp:
		strResult += "e^";
		strResult += "{ ";
		strResult += node->SaveAsLatex(c);
		strResult += "} ";
		break;
	case eLn:
		strResult += "\\ln";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eLog:
		strResult += "\\log";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right) ";
		break;
	case eAbs:
		strResult += " { ";
		strResult += "\\left| ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right| ";
		strResult += "} ";
		break;
	case eCeil:
		strResult += " { ";
		strResult += "\\lceil ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\rceil ";
		strResult += "} ";
		break;
	case eFloor:
		strResult += " { ";
		strResult += "\\lfloor ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\rfloor ";
		strResult += "} ";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	return strResult;
}

void adUnaryNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	strName = "Function";
	OpenEnum(pTag, strName, eFunction);

	strName = "Node";
	adNode* n = adNode::OpenNode(pTag, strName);
	node.reset(n);
}

void adUnaryNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Node";
	adNode::SaveNode(pTag, strName, node.get());
}

void adUnaryNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName;
	//io::xmlTag_t* nodeTag;

	switch(eFunction)
	{
	case eSign:
		strName = "minus";
		break;
	case eSin:
		strName = "sin";
		break;
	case eCos:
		strName = "cos";
		break;
	case eTan:
		strName = "tan";
		break;
	case eArcSin:
		strName = "arcsin";
		break;
	case eArcCos:
		strName = "arccos";
		break;
	case eArcTan:
		strName = "arctan";
		break;
	case eSqrt:
		strName = "root";
		break;
	case eExp:
		strName = "exp";
		break;
	case eLn:
		strName = "ln";
		break;
	case eLog:
		strName = "log";
		break;
	case eAbs:
		strName = "abs";
		break;
	case eCeil:
		strName  = "ceil";
		break;
	case eFloor:
		strName  = "floor";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	//nodeTag = xmlContentCreator::Function(strName, pTag);
	//node->SaveAsContentMathML(nodeTag);
}

void adUnaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrowout, *msup, *mrow, *msqrt;

	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	switch(eFunction)
	{
	case eSign:
		strName  = "mo";
		strValue = "(";
		mrow->AddTag(strName, strValue);
			strName  = "mrow";
			strValue = "";
			mrowout = mrow->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "-";
				mrowout->AddTag(strName, strValue);
				node->SaveAsPresentationMathML(mrowout, c);
		strName  = "mo";
		strValue = ")";
		mrow->AddTag(strName, strValue);
		break;
	case eSin:
		strName  = "mi";
		strValue = "sin";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eCos:
		strName  = "mi";
		strValue = "cos";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eTan:
		strName  = "mi";
		strValue = "tan";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eArcSin:
		strName  = "mi";
		strValue = "arcsin";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eArcCos:
		strName  = "mi";
		strValue = "arccos";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eArcTan:
		strName  = "mi";
		strValue = "arctan";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eSqrt:
		strName  = "msqrt";
		strValue = "";
		msqrt = mrow->AddTag(strName, strValue);
		node->SaveAsPresentationMathML(msqrt, c);
		break;
	case eExp:
		strName  = "msup";
		strValue = "";
		msup = mrow->AddTag(strName, strValue);
		strName  = "mi";
		strValue = "e";
		msup->AddTag(strName, strValue);
		node->SaveAsPresentationMathML(msup, c);
		break;
	case eLn:
		strName  = "mi";
		strValue = "ln";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eLog:
		strName  = "mi";
		strValue = "log";
		mrow->AddTag(strName, strValue);
		strName  = "mrow";
		strValue = "";
		mrowout = mrow->AddTag(strName, strValue);
			strName  = "mo";
			strValue = "(";
			mrowout->AddTag(strName, strValue);

			node->SaveAsPresentationMathML(mrowout, c);

			strName  = "mo";
			strValue = ")";
			mrowout->AddTag(strName, strValue);
		break;
	case eAbs:
		strName  = "mo";
		strValue = "|";
		mrow->AddTag(strName, strValue);
		node->SaveAsPresentationMathML(mrow, c);
		strName  = "mo";
		strValue = "|";
		mrow->AddTag(strName, strValue);
		break;
	case eCeil:
		strName  = "mo";
		strValue = "&#8970;";
		mrow->AddTag(strName, strValue);
		node->SaveAsPresentationMathML(mrow, c);
		strName  = "mo";
		strValue = "&#8971;";
		mrow->AddTag(strName, strValue);
		break;
	case eFloor:
		strName  = "mo";
		strValue = "&#8968;";
		mrow->AddTag(strName, strValue);
		node->SaveAsPresentationMathML(mrow, c);
		strName  = "mo";
		strValue = "&#8969;";
		mrow->AddTag(strName, strValue);
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}
}

void adUnaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adBinaryNode
**********************************************************************************************/
adBinaryNode::adBinaryNode(daeeBinaryFunctions eFun, shared_ptr<adNode> l, shared_ptr<adNode> r)
{
	left  = l;
	right = r;
	eFunction = eFun;
}

adBinaryNode::adBinaryNode()
{
	eFunction = eBFUnknown;
}

adBinaryNode::~adBinaryNode()
{
}

adouble adBinaryNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	switch(eFunction)
	{
	case ePlus:
		return left->Evaluate(pExecutionContext) + right->Evaluate(pExecutionContext);
		break;
	case eMinus:
		return left->Evaluate(pExecutionContext) - right->Evaluate(pExecutionContext);
		break;
	case eMulti:
		return left->Evaluate(pExecutionContext) * right->Evaluate(pExecutionContext);
		break;
	case eDivide:
		return left->Evaluate(pExecutionContext) / right->Evaluate(pExecutionContext);
		break;
	case ePower:
		return pow(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
		break;
	case eMin:
		return min(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
		break;
	case eMax:
		return max(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
}

adNode* adBinaryNode::Clone(void) const
{
	shared_ptr<adNode> l = shared_ptr<adNode>( (left  ? left->Clone()  : NULL) );
	shared_ptr<adNode> r = shared_ptr<adNode>( (right ? right->Clone() : NULL) );
	return new adBinaryNode(eFunction, l, r);
}

string adBinaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult, strLeft, strRight;

	if(adDoEnclose(left.get()))
	{
		strLeft  = "(";
		strLeft += left->SaveAsPlainText(c);
		strLeft += ")";
	}
	else
	{
		strLeft = left->SaveAsPlainText(c);
	}

	if(adDoEnclose(right.get()))
	{
		strRight  = "(";
		strRight += right->SaveAsPlainText(c);
		strRight += ")";
	}
	else
	{
		strRight = right->SaveAsPlainText(c);
	}

	switch(eFunction)
	{
	case ePlus:
		strResult += strLeft;
		strResult += " + ";
		strResult += strRight;
		break;
	case eMinus:
		strResult += strLeft;
		strResult += " - ";
		strResult += strRight;
		break;
	case eMulti:
		strResult += strLeft;
		strResult += " * ";
		strResult += strRight;
		break;
	case eDivide:
		strResult += strLeft;
		strResult += " / ";
		strResult += strRight;
		break;
	case ePower:
		strResult += "pow(";
		strResult += strLeft;
		strResult += ", ";
		strResult += strRight;
		strResult += ")";
		break;
	case eMin:
		strResult += "min(";
		strResult += strLeft;
		strResult += ", ";
		strResult += strRight;
		strResult += ")";
		break;
	case eMax:
		strResult += "max(";
		strResult += strLeft;
		strResult += ", ";
		strResult += strRight;
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adBinaryNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult, strLeft, strRight;

	strResult  = "{ "; // Start

	if(adDoEnclose(left.get()))
	{
		strLeft  = "\\left( ";
		strLeft += left->SaveAsLatex(c);
		strLeft += " \\right) ";
	}
	else
	{
		strLeft = left->SaveAsLatex(c);
	}

	if(adDoEnclose(right.get()))
	{
		strRight  = "\\left( ";
		strRight += right->SaveAsLatex(c);
		strRight += " \\right) ";
	}
	else
	{
		strRight = right->SaveAsLatex(c);
	}

	switch(eFunction)
	{
	case ePlus:
		strResult += strLeft;
		strResult += " + ";
		strResult += strRight;
		break;
	case eMinus:
		strResult += strLeft;
		strResult += " - ";
		strResult += strRight;
		break;
	case eMulti:
		strResult += strLeft;
		strResult += " \\times ";
		strResult += strRight;
		break;
	case eDivide:
		strResult += strLeft;
		strResult += " \\over ";
		strResult += strRight;
		break;
	case ePower:
		strResult += strLeft;
		strResult += " ^ ";
		strResult += strRight;
		break;
	case eMin:
		strResult += "min(";
		strResult += strLeft;
		strResult += ", ";
		strResult += strRight;
		strResult += ")";
		break;
	case eMax:
		strResult += "max(";
		strResult += strLeft;
		strResult += ", ";
		strResult += strRight;
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}

	strResult  += "} "; // End
	return strResult;
}

void adBinaryNode::Open(io::xmlTag_t* pTag)
{
	string strName;

	strName = "Function";
	OpenEnum(pTag, strName, eFunction);

	strName = "Left";
	adNode* l = adNode::OpenNode(pTag, strName);
	left.reset(l);

	strName = "Right";
	adNode* r = adNode::OpenNode(pTag, strName);
	right.reset(r);
}

void adBinaryNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Left";
	adNode::SaveNode(pTag, strName, left.get());

	strName = "Right";
	adNode::SaveNode(pTag, strName, right.get());
}

void adBinaryNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adBinaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	bool bDoEncloseLeft, bDoEncloseRight;
	string strName, strValue, strLeft, strRight;
	io::xmlTag_t *mrowout, *mfrac, *mrowleft, *mrowright;

	strName  = "mrow";
	strValue = "";
	mrowout = pTag->AddTag(strName, strValue);
		
	bDoEncloseLeft  = true;
	bDoEncloseRight = true;
	adDoEnclose(this, left.get(), bDoEncloseLeft, right.get(), bDoEncloseRight);

	switch(eFunction)
	{
	case ePlus:
	case eMinus:
	case eMulti:
		if(bDoEncloseLeft)
		{
			strName  = "mrow";
			strValue = "";
			mrowleft = mrowout->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "(";
				mrowleft->AddTag(strName, strValue);

				left->SaveAsPresentationMathML(mrowleft, c);

				strName  = "mo";
				strValue = ")";
				mrowleft->AddTag(strName, strValue);
		}
		else
		{
			left->SaveAsPresentationMathML(mrowout, c);
		}

		strName  = "mo";
		if(eFunction == ePlus)
			strValue = "+";
		else if(eFunction == eMinus)
			strValue = "-";
		else if(eFunction == eMulti)
			strValue = "&sdot;"; //"&#x00D7;";
		mrowout->AddTag(strName, strValue);

		if(bDoEncloseRight)
		{
			strName  = "mrow";
			strValue = "";
			mrowright = mrowout->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "(";
				mrowright->AddTag(strName, strValue);

				right->SaveAsPresentationMathML(mrowright, c);

				strName  = "mo";
				strValue = ")";
				mrowright->AddTag(strName, strValue);
		}
		else
		{
			right->SaveAsPresentationMathML(mrowout, c);
		}
		break;
	case eDivide:
	case ePower:
		strValue = "";
		if(eFunction == eDivide)
			strName = "mfrac";
		else if(eFunction == ePower)
			strName = "msup";
		mfrac = mrowout->AddTag(strName, strValue);

		if(bDoEncloseLeft)
		{
			strName  = "mrow";
			strValue = "";
			mrowleft = mfrac->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "(";
				mrowleft->AddTag(strName, strValue);

				left->SaveAsPresentationMathML(mrowleft, c);

				strName  = "mo";
				strValue = ")";
				mrowleft->AddTag(strName, strValue);
		}
		else
		{
			left->SaveAsPresentationMathML(mfrac, c);
		}

		if(bDoEncloseRight)
		{
			strName  = "mrow";
			strValue = "";
			mrowright = mfrac->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "(";
				mrowright->AddTag(strName, strValue);

				right->SaveAsPresentationMathML(mrowright, c);

				strName  = "mo";
				strValue = ")";
				mrowright->AddTag(strName, strValue);
		}
		else
		{
			right->SaveAsPresentationMathML(mfrac, c);
		}
		break;

	case eMin:
	case eMax:
		if(eFunction == eMin)
			strValue = "min";
		else if(eFunction == eMax)
			strValue = "max";
		
		strName = "mi";
		mrowout->AddTag(strName, strValue);

		strName  = "mo";
		strValue = "(";
		mrowout->AddTag(strName, strValue);

		left->SaveAsPresentationMathML(mrowout, c);
		
		strName  = "mo";
		strValue = ",";
		mrowout->AddTag(strName, strValue);
		
		right->SaveAsPresentationMathML(mrowout, c);

		strName  = "mo";
		strValue = ")";
		mrowout->AddTag(strName, strValue);
		break;

	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
}


//void adBinaryNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
//{
//	string strName, strValue, strLeft, strRight;
//	io::xmlTag_t *mrowout, *mfrac, *mrowleft, *mrowright;
//
//	strName  = "mrow";
//	strValue = "";
//	mrowout = pTag->AddTag(strName, strValue);
//
//	switch(eFunction)
//	{
//	case ePlus:
//	case eMinus:
//	case eMulti:
//		if(adDoEnclose(left.get()))
//		{
//			strName  = "mrow";
//			strValue = "";
//			mrowleft = mrowout->AddTag(strName, strValue);
//				strName  = "mo";
//				strValue = "(";
//				mrowleft->AddTag(strName, strValue);
//
//				left->SaveAsPresentationMathML(mrowleft, c);
//
//				strName  = "mo";
//				strValue = ")";
//				mrowleft->AddTag(strName, strValue);
//		}
//		else
//		{
//			left->SaveAsPresentationMathML(mrowout, c);
//		}
//
//		strName  = "mo";
//		if(eFunction == ePlus)
//			strValue = "+";
//		else if(eFunction == eMinus)
//			strValue = "-";
//		else if(eFunction == eMulti)
//			strValue = "&InvisibleTimes;"; //"&#x00D7;";
//		mrowout->AddTag(strName, strValue);
//
//		if(adDoEnclose(right.get()))
//		{
//			strName  = "mrow";
//			strValue = "";
//			mrowright = mrowout->AddTag(strName, strValue);
//				strName  = "mo";
//				strValue = "(";
//				mrowright->AddTag(strName, strValue);
//
//				right->SaveAsPresentationMathML(mrowright, c);
//
//				strName  = "mo";
//				strValue = ")";
//				mrowright->AddTag(strName, strValue);
//		}
//		else
//		{
//			right->SaveAsPresentationMathML(mrowout, c);
//		}
//		break;
//	case eDivide:
//	case ePower:
//		strValue = "";
//		if(eFunction == eDivide)
//			strName = "mfrac";
//		else if(eFunction == ePower)
//			strName = "msup";
//		mfrac = mrowout->AddTag(strName, strValue);
//
//		if(adDoEnclose(left.get()))
//		{
//			strName  = "mrow";
//			strValue = "";
//			mrowleft = mfrac->AddTag(strName, strValue);
//				strName  = "mo";
//				strValue = "(";
//				mrowleft->AddTag(strName, strValue);
//
//				left->SaveAsPresentationMathML(mrowleft, c);
//
//				strName  = "mo";
//				strValue = ")";
//				mrowleft->AddTag(strName, strValue);
//		}
//		else
//		{
//			left->SaveAsPresentationMathML(mfrac, c);
//		}
//
//		if(adDoEnclose(right.get()))
//		{
//			strName  = "mrow";
//			strValue = "";
//			mrowright = mfrac->AddTag(strName, strValue);
//				strName  = "mo";
//				strValue = "(";
//				mrowright->AddTag(strName, strValue);
//
//				right->SaveAsPresentationMathML(mrowright, c);
//
//				strName  = "mo";
//				strValue = ")";
//				mrowright->AddTag(strName, strValue);
//		}
//		else
//		{
//			right->SaveAsPresentationMathML(mfrac, c);
//		}
//		break;
//
//	case eMin:
//		daeDeclareAndThrowException(exNotImplemented);
//		break;
//
//	case eMax:
//		daeDeclareAndThrowException(exNotImplemented);
//		break;
//
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//}

void adBinaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	left->AddVariableIndexToArray(mapIndexes);
	right->AddVariableIndexToArray(mapIndexes);
}


}
}
