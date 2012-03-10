#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
#include "nodes_array.h"
using namespace dae;
#include "xmlfunctions.h"
#include "units_io.h"
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
void adNodeImpl::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}

//void adNodeImpl::ExportAsPlainText(string strFileName)
//{
//	string strLatex;
//	ofstream file(strFileName.c_str());
//	file << SaveAsPlainText(NULL);
//	file.close();
//}

void adNodeImpl::ExportAsLatex(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsLatex(NULL);
	file.close();
}

bool adNodeImpl::IsLinear(void) const
{
// All nodes are non-linear if I dont explicitly state that they are linear!
	return false;
}

bool adNodeImpl::IsFunctionOfVariables(void) const
{
// All nodes are functions of variables if I dont explicitly state that they aint!
	return true;
}

/*********************************************************************************************
	adConstantNode
**********************************************************************************************/
adConstantNode::adConstantNode()
{
}

adConstantNode::adConstantNode(const real_t d)
              : m_quantity(d, unit())
{
}

adConstantNode::adConstantNode(const real_t d, const unit& units)
	          : m_quantity(d, units)
{
}

adConstantNode::adConstantNode(const quantity& q)
	          : m_quantity(q)
{
}

adConstantNode::~adConstantNode()
{
}

adouble adConstantNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}
	
	tmp.setValue(m_quantity.getValue());
	return tmp;
}

const quantity adConstantNode::GetQuantity(void) const
{
	return m_quantity;
}

adNode* adConstantNode::Clone(void) const
{
	return new adConstantNode(*this);
}

void adConstantNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	strContent += units::Export(eLanguage, c, m_quantity);
}

//string adConstantNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
//{
//	return textCreator::Constant(m_dValue);
//}

string adConstantNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return m_quantity.toLatex();
}

void adConstantNode::Open(io::xmlTag_t* pTag)
{
	string strName = "Value";
	units::Open(pTag, strName, m_quantity);
}

void adConstantNode::Save(io::xmlTag_t* pTag) const
{
	string strName  = "Value";
	units::Save(pTag, strName, m_quantity);
}

void adConstantNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adConstantNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	units::SaveAsPresentationMathML(pTag, m_quantity);
}

void adConstantNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adConstantNode::IsLinear(void) const
{
	return true;
}

bool adConstantNode::IsFunctionOfVariables(void) const
{
	return false;
}

/*********************************************************************************************
	adTimeNode
**********************************************************************************************/
adTimeNode::adTimeNode(void)
{
}

adTimeNode::~adTimeNode()
{
}

adouble adTimeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble tmp(pExecutionContext->m_pDataProxy->GetCurrentTime(), 0);
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
	}
	return tmp;
}

const quantity adTimeNode::GetQuantity(void) const
{
	return quantity(0.0, unit("s", 1));
}

adNode* adTimeNode::Clone(void) const
{
	return new adTimeNode(*this);
}

void adTimeNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	if(eLanguage == eCDAE)
		strContent += "Time()";
	else if(eLanguage == ePYDAE)
		strContent += "Time()";
	else
		daeDeclareAndThrowException(exNotImplemented);
}

//string adTimeNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
//{
//
//}

string adTimeNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	vector<string> domains;
	return latexCreator::Variable(string("\\tau"), domains);
}

void adTimeNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adTimeNode::Save(io::xmlTag_t* pTag) const
{
	string strName = "Value";
	string strValue = "time";
	pTag->Save(strName, strValue);
}

void adTimeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adTimeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	vector<string> domains;
	xmlPresentationCreator::Variable(pTag, string("&tau;"), domains);
}

void adTimeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adTimeNode::IsLinear(void) const
{
	return true;
}

bool adTimeNode::IsFunctionOfVariables(void) const
{
	return false;
}

/*********************************************************************************************
	adEventPortDataNode
**********************************************************************************************/
adEventPortDataNode::adEventPortDataNode(daeEventPort* pEventPort)
{
	m_pEventPort = pEventPort;
}

adEventPortDataNode::~adEventPortDataNode()
{
}

adouble adEventPortDataNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pEventPort)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pExecutionContext)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pExecutionContext->m_pDataProxy)
		daeDeclareAndThrowException(exInvalidPointer);
	
	adouble tmp(m_pEventPort->GetEventData(), 0);
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
	}
	return tmp;
}

const quantity adEventPortDataNode::GetQuantity(void) const
{
	return quantity();
}

adNode* adEventPortDataNode::Clone(void) const
{
	return new adEventPortDataNode(*this);
}

void adEventPortDataNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	if(eLanguage == eCDAE)
		strContent += daeGetStrippedRelativeName(c.m_pModel, m_pEventPort) + "()";
	else if(eLanguage == ePYDAE)
		strContent += "self." + daeGetStrippedRelativeName(c.m_pModel, m_pEventPort) + "()";
	else
		daeDeclareAndThrowException(exNotImplemented);
}

//string adEventPortDataNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
//{
//
//}

string adEventPortDataNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> domains;
	string strName = daeGetRelativeName(c->m_pModel, m_pEventPort);
	return latexCreator::Variable(strName, domains);
}

void adEventPortDataNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adEventPortDataNode::Save(io::xmlTag_t* pTag) const
{
	string strName = "EventPort";
	pTag->SaveObjectRef(strName, m_pEventPort);
}

void adEventPortDataNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adEventPortDataNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> domains;
	string strName = daeGetRelativeName(c->m_pModel, m_pEventPort);
	xmlPresentationCreator::Variable(pTag, strName, domains);
}

void adEventPortDataNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adEventPortDataNode::IsLinear(void) const
{
	return true;
}

bool adEventPortDataNode::IsFunctionOfVariables(void) const
{
	return false;
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
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}
	return adouble(m_dValue);
}

const quantity adRuntimeParameterNode::GetQuantity(void) const
{
	if(!m_pParameter)
		daeDeclareAndThrowException(exInvalidCall);
	
	//std::cout << (boost::format("%s units = %s") % m_pParameter->GetCanonicalName() % m_pParameter->GetUnits().getBaseUnit().toString()).str() << std::endl;
	return quantity(m_dValue, m_pParameter->GetUnits());
}

adNode* adRuntimeParameterNode::Clone(void) const
{
	return new adRuntimeParameterNode(*this);
}

//string adRuntimeParameterNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));
//
//	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adRuntimeParameterNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
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

	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeParameterNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeParameterNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adRuntimeParameterNode::IsLinear(void) const
{
	return true;
}

bool adRuntimeParameterNode::IsFunctionOfVariables(void) const
{
	return false;
}

/*********************************************************************************************
	adDomainIndexNode
**********************************************************************************************/
adDomainIndexNode::adDomainIndexNode(daeDomain* pDomain, size_t nIndex, real_t* pdPointValue)
			     : m_pDomain(pDomain), 
				   m_nIndex(nIndex),
                   m_pdPointValue(pdPointValue)
{
}

adDomainIndexNode::adDomainIndexNode()
{
	m_pDomain      = NULL;
	m_nIndex       = ULONG_MAX;
	m_pdPointValue = NULL;
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
	if(!m_pdPointValue)
		daeDeclareAndThrowException(exInvalidCall);

	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}

	return adouble(*m_pdPointValue);
}

const quantity adDomainIndexNode::GetQuantity(void) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidCall);
	return quantity(0.0, m_pDomain->GetUnits());
}

adNode* adDomainIndexNode::Clone(void) const
{
	return new adDomainIndexNode(*this);
}

//string adDomainIndexNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	string strName  = daeGetRelativeName(c->m_pModel, m_pDomain);
//	string strIndex = toString<size_t>(m_nIndex);
//	return textCreator::Domain(strName, strIndex);
//}

string adDomainIndexNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strName  = daeGetRelativeName(c->m_pModel, m_pDomain);
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
	string strName  = daeGetRelativeName(c->m_pModel, m_pDomain);
	string strIndex = toString<size_t>(m_nIndex);
	xmlPresentationCreator::Domain(pTag, strName, strIndex);
}

void adDomainIndexNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adDomainIndexNode::IsLinear(void) const
{
	return true;
}

bool adDomainIndexNode::IsFunctionOfVariables(void) const
{
	return false;
}

/*********************************************************************************************
	adRuntimeVariableNode
**********************************************************************************************/
adRuntimeVariableNode::adRuntimeVariableNode(daeVariable* pVariable, 
											 size_t nOverallIndex, 
											 vector<size_t>& narrDomains) 
               : m_nOverallIndex(nOverallIndex), 
				 m_pVariable(pVariable), 
				 m_narrDomains(narrDomains)
{
// This will be calculated at runtime (if needed; it is used only for sensitivity calculation)
	m_nBlockIndex = ULONG_MAX;
	m_bIsAssigned = false;
}

adRuntimeVariableNode::adRuntimeVariableNode()
{
	m_pVariable     = NULL;
	m_nBlockIndex   = ULONG_MAX;
	m_nOverallIndex = ULONG_MAX;
	m_bIsAssigned   = false;
}

adRuntimeVariableNode::~adRuntimeVariableNode()
{
}

adouble adRuntimeVariableNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in the evaluate mode we dont need the value
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adouble tmp;
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}

/*
	Only for the first encounter:
	  - Try to get the variable block index based on its overall index
	  - If failed - throw an exception
*/
	if(m_nBlockIndex == ULONG_MAX)
	{
		if(!pExecutionContext || !pExecutionContext->m_pBlock)
			daeDeclareAndThrowException(exInvalidPointer)
			
		//This is a very ugly hack, but there is no other way (at the moment)
		adRuntimeVariableNode* self = const_cast<adRuntimeVariableNode*>(this);
		self->m_nBlockIndex = pExecutionContext->m_pBlock->FindVariableBlockIndex(m_nOverallIndex);
		
		if(m_nBlockIndex == ULONG_MAX)
		{
			if(pExecutionContext->m_pDataProxy->GetVariableType(m_nOverallIndex) == cnFixed)
			{
				self->m_nBlockIndex = m_nOverallIndex;
				self->m_bIsAssigned = true;
			}
			else
			{
				daeDeclareException(exInvalidCall);
				e << "Cannot obtain block index for the variable index [" << m_nOverallIndex << "]";
				throw e;
			}
		}
	}
	
/* 
	ACHTUNG, ACHTUNG!!
	Assigned variables' values are in DataProxy (they do not exist in the solver)!!
*/
	real_t value;
	if(m_bIsAssigned)
		value = pExecutionContext->m_pDataProxy->GetValue(m_nOverallIndex);
	else
		value = pExecutionContext->m_pBlock->GetValue(m_nBlockIndex);
	
	if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivityResiduals)
	{
	/*
		If m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex that means that 
		the variable is fixed and its sensitivity derivative per given parameter is 1.
		If it is not - it is a normal state variable and its sensitivity derivative is m_pdSValue
	*/	
		//m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation is used to get the S/SD values
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
		{
			return adouble(value, 1);
		}
		else
		{
			//If it is fixed variable then its derivative per parameter is 0
			//Otherwise take the value from the DataProxy
			if(pExecutionContext->m_pDataProxy->GetVariableType(m_nOverallIndex) == cnFixed)
			{
				return adouble(value, 0);
			}
			else
			{
			// Get the derivative value based on the m_nBlockIndex	
				return adouble(value, 
				               pExecutionContext->m_pDataProxy->GetSValue(pExecutionContext->m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation, m_nBlockIndex) );
			}
		}
	}
	else if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivityParametersGradients)
	{
		return adouble(value, (pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex ? 1 : 0) );
	}
	else
	{
	// Achtung!! If a variable is assigned its derivative must always be zero!
		if(m_bIsAssigned)
			return adouble(value);
		else
			return adouble(value, (pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == m_nOverallIndex ? 1 : 0) );
	}
}

const quantity adRuntimeVariableNode::GetQuantity(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	
	//std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
	return quantity(0.0, m_pVariable->GetVariableType()->GetUnits());
}

adNode* adRuntimeVariableNode::Clone(void) const
{
	return new adRuntimeVariableNode(*this);
}

//string adRuntimeVariableNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));
//
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adRuntimeVariableNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
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

	//strName = "Value";
	//pTag->Save(strName, *m_pdValue);
}

void adRuntimeVariableNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeVariableNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adRuntimeVariableNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	pair<size_t, size_t> mapPair(m_nOverallIndex, mapIndexes.size());
	
	
	if(bAddFixed)
	{
	// Add anyway (even if it is assigned) 
		mapIndexes.insert(mapPair);
	}
	else
	{
	// Add only if it is not assigned 
		if(!m_pVariable || !m_pVariable->m_pModel || !m_pVariable->m_pModel->GetDataProxy())
			daeDeclareAndThrowException(exInvalidPointer);
		
		if(m_pVariable->m_pModel->GetDataProxy()->GetVariableType(m_nOverallIndex) != cnFixed)
			mapIndexes.insert(mapPair);
	}
}

bool adRuntimeVariableNode::IsLinear(void) const
{
	return true;
}

bool adRuntimeVariableNode::IsFunctionOfVariables(void) const
{
	return true;
}

/*********************************************************************************************
	adRuntimeTimeDerivativeNode
**********************************************************************************************/
adRuntimeTimeDerivativeNode::adRuntimeTimeDerivativeNode(daeVariable* pVariable, 
														 size_t nOverallIndex, 
														 size_t nDegree, 
														 vector<size_t>& narrDomains)
               : m_nOverallIndex(nOverallIndex), 
				 m_nDegree(nDegree), 
				 m_pVariable(pVariable),
				 m_narrDomains(narrDomains)
{
// This will be calculated at runtime (if needed; it is used only for sensitivity calculation)
	m_nBlockIndex = ULONG_MAX;
}

adRuntimeTimeDerivativeNode::adRuntimeTimeDerivativeNode(void)
{	
	m_pVariable        = NULL;
	m_nDegree          = 0;
	m_nOverallIndex    = ULONG_MAX;
	m_nBlockIndex      = ULONG_MAX;
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
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}

/*
	Only for the first encounter:
	  - Try to get the variable block index based on its overall index
	  - If failed - throw an exception
*/
	if(m_nBlockIndex == ULONG_MAX)
	{
		if(!pExecutionContext || !pExecutionContext->m_pBlock)
			daeDeclareAndThrowException(exInvalidPointer)
			
		//This is a very ugly hack, but there is no other way (at the moment)
		adRuntimeTimeDerivativeNode* self = const_cast<adRuntimeTimeDerivativeNode*>(this);
		self->m_nBlockIndex = pExecutionContext->m_pBlock->FindVariableBlockIndex(m_nOverallIndex);

		if(m_nBlockIndex == ULONG_MAX)
		{
			daeDeclareException(exInvalidCall);
			e << "Cannot obtain block index for the (dt) variable index [" << m_nOverallIndex << "]";
			throw e;
		}
	}
	
	if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivityResiduals)
	{
	/*	ACHTUNG, ACHTUNG!!
		Here m_nCurrentParameterIndexForSensitivityEvaluation MUST NOT be equal to m_nOverallIndex,
		because it would mean a time derivative of an assigned variable (that is a sensitivity parameter)
		which value is fixed!! 
	*/
		if(pExecutionContext->m_nCurrentParameterIndexForSensitivityEvaluation == m_nOverallIndex)
			daeDeclareAndThrowException(exInvalidCall);

		//m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation is used to get the S/SD values
		return adouble(pExecutionContext->m_pBlock->GetTimeDerivative(m_nBlockIndex),
		               pExecutionContext->m_pDataProxy->GetSDValue(pExecutionContext->m_nIndexInTheArrayOfCurrentParameterForSensitivityEvaluation, m_nBlockIndex) 
		               );
	}
	else if(pExecutionContext->m_eEquationCalculationMode == eCalculateSensitivityParametersGradients)
	{
		//I can never rich this point, since the model must be steady-state to call CalculateGradients function
		daeDeclareAndThrowException(exInvalidCall)
		return adouble(0, 0);
	}
	else
	{
		return adouble(pExecutionContext->m_pBlock->GetTimeDerivative(m_nBlockIndex), 
					  (pExecutionContext->m_nCurrentVariableIndexForJacobianEvaluation == m_nOverallIndex ? pExecutionContext->m_dInverseTimeStep : 0) 
		              );
	}
}

const quantity adRuntimeTimeDerivativeNode::GetQuantity(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	
	//std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
	return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / unit("s", 1));
}

adNode* adRuntimeTimeDerivativeNode::Clone(void) const
{
	return new adRuntimeTimeDerivativeNode(*this);
}

//string adRuntimeTimeDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));
//
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::TimeDerivative(m_nDegree, strName, strarrIndexes);
//}

string adRuntimeTimeDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
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

	//strName = "TimeDerivative";
	//pTag->Save(strName, *m_pdTimeDerivative);
}

void adRuntimeTimeDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
	xmlContentCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adRuntimeTimeDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	for(size_t i = 0; i < m_narrDomains.size(); i++)
		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
	xmlPresentationCreator::TimeDerivative(pTag, m_nDegree, strName, strarrIndexes);
}

void adRuntimeTimeDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
// Time derivatives are always state variables: always add 
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
															   adNodePtr pdnode)
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
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}

	return pardevnode->Evaluate(pExecutionContext);
}

const quantity adRuntimePartialDerivativeNode::GetQuantity(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidCall);
	
	//std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % (m_pVariable->GetVariableType()->GetUnits() / m_pDomain->GetUnits()).getBaseUnit()).str() << std::endl;
	if(m_nDegree == 1)
		return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / m_pDomain->GetUnits());
	else
		return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / (m_pDomain->GetUnits() ^ 2));
}

adNode* adRuntimePartialDerivativeNode::Clone(void) const
{
	return new adRuntimePartialDerivativeNode(*this);
}

//string adRuntimePartialDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	return pardevnode->SaveAsPlainText(c);
//}

string adRuntimePartialDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
//	vector<string> strarrIndexes;
//	for(size_t i = 0; i < m_narrDomains.size(); i++)
//		strarrIndexes.push_back(toString<size_t>(m_narrDomains[i]));

//	string strVariableName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName   = daeGetRelativeName(c->m_pModel, m_pDomain);
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
	return pardevnode->SaveAsPresentationMathML(pTag, c);
}

void adRuntimePartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!pardevnode)
		daeDeclareAndThrowException(exInvalidPointer);
	pardevnode->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adRuntimePartialDerivativeNode::IsLinear(void) const
{
	if(!pardevnode)
		daeDeclareAndThrowException(exInvalidPointer);
	
	return pardevnode->IsLinear();
}

bool adRuntimePartialDerivativeNode::IsFunctionOfVariables(void) const
{
	if(!pardevnode)
		daeDeclareAndThrowException(exInvalidPointer);
	
	return pardevnode->IsFunctionOfVariables();
}

/*********************************************************************************************
	adUnaryNode
**********************************************************************************************/
adUnaryNode::adUnaryNode(daeeUnaryFunctions eFun, adNodePtr n)
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
	case eSin:
		return sin(node->Evaluate(pExecutionContext));
	case eCos:
		return cos(node->Evaluate(pExecutionContext));
	case eTan:
		return tan(node->Evaluate(pExecutionContext));
	case eArcSin:
		return asin(node->Evaluate(pExecutionContext));
	case eArcCos:
		return acos(node->Evaluate(pExecutionContext));
	case eArcTan:
		return atan(node->Evaluate(pExecutionContext));
	case eSqrt:
		return sqrt(node->Evaluate(pExecutionContext));
	case eExp:
		return exp(node->Evaluate(pExecutionContext));
	case eLn:
		return log(node->Evaluate(pExecutionContext));
	case eLog:
		return log10(node->Evaluate(pExecutionContext));
	case eAbs:
		return abs(node->Evaluate(pExecutionContext));
	case eCeil:
		return ceil(node->Evaluate(pExecutionContext));
	case eFloor:
		return floor(node->Evaluate(pExecutionContext));
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return adouble();
	}
}

const quantity adUnaryNode::GetQuantity(void) const
{
	switch(eFunction)
	{
	case eSign:
		return -(node->GetQuantity());
	case eSin:
		return sin(node->GetQuantity());
	case eCos:
		return cos(node->GetQuantity());
	case eTan:
		return tan(node->GetQuantity());
	case eArcSin:
		return asin(node->GetQuantity());
	case eArcCos:
		return acos(node->GetQuantity());
	case eArcTan:
		return atan(node->GetQuantity());
	case eSqrt:
		return sqrt(node->GetQuantity());
	case eExp:
		return exp(node->GetQuantity());
	case eLn:
		return log(node->GetQuantity());
	case eLog:
		return log10(node->GetQuantity());
	case eAbs:
		return abs(node->GetQuantity());
	case eCeil:
		return ceil(node->GetQuantity());
	case eFloor:
		return floor(node->GetQuantity());
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return quantity();
	}
}

adNode* adUnaryNode::Clone(void) const
{
	adNodePtr n = adNodePtr( (node ? node->Clone() : NULL) );
	return new adUnaryNode(eFunction, n);
}

void adUnaryNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	switch(eFunction)
	{
	case eSign:
		strContent += "(-";
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eSin:
		if(eLanguage == eCDAE)
			strContent += "sin(";
		else if(eLanguage == ePYDAE)
			strContent += "Sin(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eCos:
		if(eLanguage == eCDAE)
			strContent += "cos(";
		else if(eLanguage == ePYDAE)
			strContent += "Cos(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eTan:
		if(eLanguage == eCDAE)
			strContent += "tan(";
		else if(eLanguage == ePYDAE)
			strContent += "Tan(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eArcSin:
		if(eLanguage == eCDAE)
			strContent += "asin(";
		else if(eLanguage == ePYDAE)
			strContent += "ASin(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eArcCos:
		if(eLanguage == eCDAE)
			strContent += "acos(";
		else if(eLanguage == ePYDAE)
			strContent += "ACos(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eArcTan:
		if(eLanguage == eCDAE)
			strContent += "atan(";
		else if(eLanguage == ePYDAE)
			strContent += "ATan(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eSqrt:
		if(eLanguage == eCDAE)
			strContent += "sqrt(";
		else if(eLanguage == ePYDAE)
			strContent += "Sqrt(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eExp:
		if(eLanguage == eCDAE)
			strContent += "exp(";
		else if(eLanguage == ePYDAE)
			strContent += "Exp(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eLn:
		if(eLanguage == eCDAE)
			strContent += "log(";
		else if(eLanguage == ePYDAE)
			strContent += "Log(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eLog:
		if(eLanguage == eCDAE)
			strContent += "log10(";
		else if(eLanguage == ePYDAE)
			strContent += "Log10(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eAbs:
		if(eLanguage == eCDAE)
			strContent += "abs(";
		else if(eLanguage == ePYDAE)
			strContent += "Abs(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eCeil:
		if(eLanguage == eCDAE)
			strContent += "ceil(";
		else if(eLanguage == ePYDAE)
			strContent += "Ceil(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eFloor:
		if(eLanguage == eCDAE)
			strContent += "floor(";
		else if(eLanguage == ePYDAE)
			strContent += "Floor(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
	}
}
//string adUnaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	string strResult;
//	switch(eFunction)
//	{
//	case eSign:
//		strResult += "(-";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eSin:
//		strResult += "sin(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eCos:
//		strResult += "cos(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eTan:
//		strResult += "tan(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eArcSin:
//		strResult += "asin(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eArcCos:
//		strResult += "acos(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eArcTan:
//		strResult += "atan(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eSqrt:
//		strResult += "sqrt(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eExp:
//		strResult += "exp(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eLn:
//		strResult += "log(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eLog:
//		strResult += "log10(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eAbs:
//		strResult += "abs(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eCeil:
//		strResult += "ceil(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eFloor:
//		strResult += "floor(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

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

void adUnaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adUnaryNode::IsLinear(void) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	bool lin   = node->IsLinear();
	bool isFun = node->IsFunctionOfVariables();

	Linearity type;
	
	if(lin && (!isFun))
		type = LIN;
	else if(lin && isFun)
		type = LIN_FUN;
	else
		type = NON_LIN;
	
// If node is linear and not a function of variable: return linear
	if(type == LIN) 
	{
		return true;
	}
	else if(type == NON_LIN) 
	{
		return false;
	}
	else if(type == LIN_FUN) 
	{
	// If the argument is a function of variables then I should check the function
		if(eFunction == eSign)
			return true;
		else
			return false;
	}
	else
	{
		return false;
	}
}

bool adUnaryNode::IsFunctionOfVariables(void) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->IsFunctionOfVariables();
}

/*********************************************************************************************
	adBinaryNode
**********************************************************************************************/
adBinaryNode::adBinaryNode(daeeBinaryFunctions eFun, adNodePtr l, adNodePtr r)
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
	case eMinus:
		return left->Evaluate(pExecutionContext) - right->Evaluate(pExecutionContext);
	case eMulti:
		return left->Evaluate(pExecutionContext) * right->Evaluate(pExecutionContext);
	case eDivide:
		return left->Evaluate(pExecutionContext) / right->Evaluate(pExecutionContext);
	case ePower:
		return pow(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
	case eMin:
		return min(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
	case eMax:
		return max(left->Evaluate(pExecutionContext), right->Evaluate(pExecutionContext));
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
}

const quantity adBinaryNode::GetQuantity(void) const
{
	switch(eFunction)
	{
	case ePlus:
		return left->GetQuantity() + right->GetQuantity();
	case eMinus:
		return left->GetQuantity() - right->GetQuantity();
	case eMulti:
		return left->GetQuantity() * right->GetQuantity();
	case eDivide:
		return left->GetQuantity() / right->GetQuantity();
	case ePower:
		return pow(left->GetQuantity(), right->GetQuantity());
	case eMin:
		return min(left->GetQuantity(), right->GetQuantity());
	case eMax:
		return max(left->GetQuantity(), right->GetQuantity());
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return quantity();
	}
}

adNode* adBinaryNode::Clone(void) const
{
	adNodePtr l = adNodePtr( (left  ? left->Clone()  : NULL) );
	adNodePtr r = adNodePtr( (right ? right->Clone() : NULL) );
	return new adBinaryNode(eFunction, l, r);
}

void adBinaryNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strLeft, strRight;

	if(adDoEnclose(left.get()))
	{
		strLeft  = "(";
		left->Export(strLeft, eLanguage, c);
		strLeft += ")";
	}
	else
	{
		left->Export(strLeft, eLanguage, c);
	}

	if(adDoEnclose(right.get()))
	{
		strRight  = "(";
		right->Export(strRight, eLanguage, c);
		strRight += ")";
	}
	else
	{
		right->Export(strRight, eLanguage, c);
	}

	switch(eFunction)
	{
	case ePlus:
		strContent += strLeft;
		strContent += " + ";
		strContent += strRight;
		break;
	case eMinus:
		strContent += strLeft;
		strContent += " - ";
		strContent += strRight;
		break;
	case eMulti:
		strContent += strLeft;
		strContent += " * ";
		strContent += strRight;
		break;
	case eDivide:
		strContent += strLeft;
		strContent += " / ";
		strContent += strRight;
		break;
	case ePower:
		if(eLanguage == eCDAE)
		{
			strContent += "pow(";
			strContent += strLeft;
			strContent += ", ";
			strContent += strRight;
			strContent += ")";
		}
		else if(eLanguage == ePYDAE)
		{
			strContent += strLeft;
			strContent += " ** ";
			strContent += strRight;
		}
		else
			daeDeclareAndThrowException(exNotImplemented);
		break;
	case eMin:
		if(eLanguage == eCDAE)
			strContent += "min(";
		else if(eLanguage == ePYDAE)
			strContent += "Min(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		strContent += strLeft;
		strContent += ", ";
		strContent += strRight;
		strContent += ")";
		break;
	case eMax:
		if(eLanguage == eCDAE)
			strContent += "max(";
		else if(eLanguage == ePYDAE)
			strContent += "Max(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		strContent += strLeft;
		strContent += ", ";
		strContent += strRight;
		strContent += ")";
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
	}
}

//string adBinaryNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
//{
//	string strResult, strLeft, strRight;
//
//	if(adDoEnclose(left.get()))
//	{
//		strLeft  = "(";
//		strLeft += left->SaveAsPlainText(c);
//		strLeft += ")";
//	}
//	else
//	{
//		strLeft = left->SaveAsPlainText(c);
//	}
//
//	if(adDoEnclose(right.get()))
//	{
//		strRight  = "(";
//		strRight += right->SaveAsPlainText(c);
//		strRight += ")";
//	}
//	else
//	{
//		strRight = right->SaveAsPlainText(c);
//	}
//
//	switch(eFunction)
//	{
//	case ePlus:
//		strResult += strLeft;
//		strResult += " + ";
//		strResult += strRight;
//		break;
//	case eMinus:
//		strResult += strLeft;
//		strResult += " - ";
//		strResult += strRight;
//		break;
//	case eMulti:
//		strResult += strLeft;
//		strResult += " * ";
//		strResult += strRight;
//		break;
//	case eDivide:
//		strResult += strLeft;
//		strResult += " / ";
//		strResult += strRight;
//		break;
//	case ePower:
//		strResult += "pow(";
//		strResult += strLeft;
//		strResult += ", ";
//		strResult += strRight;
//		strResult += ")";
//		break;
//	case eMin:
//		strResult += "min(";
//		strResult += strLeft;
//		strResult += ", ";
//		strResult += strRight;
//		strResult += ")";
//		break;
//	case eMax:
//		strResult += "max(";
//		strResult += strLeft;
//		strResult += ", ";
//		strResult += strRight;
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

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
	string strName, strValue;
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
		{
			strName = "mfrac";
			mfrac = mrowout->AddTag(strName, strValue);
		}
		else if(eFunction == ePower)
		{
		// I should always enclose left
			bDoEncloseLeft = true;
			strName = "msup";
			mfrac = mrowout->AddTag(strName, strValue);
		}

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

void adBinaryNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	left->AddVariableIndexToArray(mapIndexes, bAddFixed);
	right->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adBinaryNode::IsLinear(void) const
{
	bool l = left->IsLinear();
	bool r = right->IsLinear();

	bool lisFun = left->IsFunctionOfVariables();
	bool risFun = right->IsFunctionOfVariables();

	Linearity l_type, r_type;
	
	if(l && (!lisFun))
		l_type = LIN;
	else if(l && lisFun)
		l_type = LIN_FUN;
	else
		l_type = NON_LIN;

	if(r && (!risFun))
		r_type = LIN;
	else if(r && risFun)
		r_type = LIN_FUN;
	else
		r_type = NON_LIN;
	
// If both are linear and not a function of variables then the expression is linear
// c = constant/parameter; lf = linear-function; nl = non-linear-function; 
// Cases: c + c, c - c, c * c, c / c
	if( l_type == LIN && r_type == LIN )
		return true;

// If any of these is not linear then the expression is non-linear
// Cases: nl + lf, nl - lf, nl * lf, nl / lf
	if( l_type == NON_LIN || r_type == NON_LIN )
		return false;
	
// If either left or right (or both) IS a function of variables then I should check the function;
// Ohterwise the expression is non-linear
	if(l_type == LIN_FUN || r_type == LIN_FUN)
	{
		switch(eFunction)
		{
		case ePlus:
		case eMinus:
		// If both are linear or linear functions then the expression is linear
		// (no matter if both are functions of variables)
		// Cases: c + lf, lf + c, lf + lf
			if(l_type != NON_LIN && r_type != NON_LIN) 
				return true;
			else 
				return false;
			break;
		case eMulti:
		// If LEFT is linear (can be a function of variables) AND RIGHT is linear and not a function of variables
		// or if LEFT is linear and not a function of of variables AND RIGHT is linear (can be a function of variables)
		// Cases: c * lf, lf * c
			if( l_type == LIN && r_type == LIN_FUN ) 
				return true;
			else if( l_type == LIN_FUN && r_type == LIN ) 
				return true;
			else 
				return false;
			break;
		case eDivide:
		// If LEFT is linear function and RIGHT is linear
		// Cases: lf / c
			if( l_type == LIN_FUN && r_type == LIN ) 
				return true;
			else 
				return false;
			break;
		default:
			return false;
		}
	}
// Eihter LEFT or RIGHT are non-linear so return false
	else
	{
		return false;
	}
	
// Just in case I somehow rich this point	
	return false;
}

bool adBinaryNode::IsFunctionOfVariables(void) const
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	
// If ANY of these two nodes is a function of variables return true
	return (left->IsFunctionOfVariables() || right->IsFunctionOfVariables() );
}


/*********************************************************************************************
	adScalarExternalFunctionNode
**********************************************************************************************/
adScalarExternalFunctionNode::adScalarExternalFunctionNode(daeScalarExternalFunction* externalFunction)
{
	m_pExternalFunction = externalFunction;
}

adScalarExternalFunctionNode::~adScalarExternalFunctionNode()
{
}

adouble adScalarExternalFunctionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pExternalFunction)
		daeDeclareAndThrowException(exInvalidPointer);

	adouble tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
	// Here I have to initialize arguments (which are at this moment setup nodes)
	// Creation of runtime nodes will also add variable indexes into the equation execution info
		daeScalarExternalFunction* pExtFun = const_cast<daeScalarExternalFunction*>(m_pExternalFunction);
		pExtFun->InitializeArguments(pExecutionContext);
		
		tmp.setGatherInfo(true);
		tmp.node = adNodePtr( Clone() );
		return tmp;
	}
	
	daeExternalFunctionArgumentValue_t value;
	daeExternalFunctionArgumentValueMap_t mapValues;
	daeExternalFunctionNodeMap_t::const_iterator iter;
	const daeExternalFunctionNodeMap_t& mapArgumentNodes = m_pExternalFunction->GetArgumentNodes();
	
	for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
	{
		std::string               strName  = iter->first;
		daeExternalFunctionNode_t argument = iter->second;
		
		adNodePtr*      ad    = boost::get<adNodePtr>     (&argument);
		adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&argument);
		
		if(ad)
			value = (*ad)->Evaluate(pExecutionContext);
		else if(adarr)
			value = (*adarr)->Evaluate(pExecutionContext).m_arrValues;
		else
			daeDeclareAndThrowException(exInvalidCall);
		
		mapValues[strName] = value;		
	}
	
	tmp = m_pExternalFunction->Calculate(mapValues);
	return tmp;
}

const quantity adScalarExternalFunctionNode::GetQuantity(void) const
{
	if(!m_pExternalFunction)
		daeDeclareAndThrowException(exInvalidPointer);
	return quantity(0.0, m_pExternalFunction->GetUnits());
}

adNode* adScalarExternalFunctionNode::Clone(void) const
{
	return new adScalarExternalFunctionNode(*this);
}

void adScalarExternalFunctionNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
}
//string adScalarExternalFunctionNode::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
//{
//	return string("");
//}

string adScalarExternalFunctionNode::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return string("");
}

void adScalarExternalFunctionNode::Open(io::xmlTag_t* pTag)
{
}

void adScalarExternalFunctionNode::Save(io::xmlTag_t* pTag) const
{
}

void adScalarExternalFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adScalarExternalFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adScalarExternalFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!m_pExternalFunction)
		daeDeclareAndThrowException(exInvalidPointer);

	daeExternalFunctionNodeMap_t::const_iterator iter;
	const daeExternalFunctionNodeMap_t& mapArgumentNodes = m_pExternalFunction->GetArgumentNodes();
	
	// This operates on RuntimeNodes!!
	for(iter = mapArgumentNodes.begin(); iter != mapArgumentNodes.end(); iter++)
	{
		daeExternalFunctionNode_t argument = iter->second;
		
		adNodePtr*      ad    = boost::get<adNodePtr>     (&argument);
		adNodeArrayPtr* adarr = boost::get<adNodeArrayPtr>(&argument);
		
		if(ad)
			(*ad)->AddVariableIndexToArray(mapIndexes, bAddFixed);
		else if(adarr)
			(*adarr)->AddVariableIndexToArray(mapIndexes, bAddFixed);
		else
			daeDeclareAndThrowException(exInvalidCall);
	}
}

bool adScalarExternalFunctionNode::IsLinear(void) const
{
	return false;
}

bool adScalarExternalFunctionNode::IsFunctionOfVariables(void) const
{
	return true;
}
	

}
}
