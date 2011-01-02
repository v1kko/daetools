#include "stdafx.h"
#include "coreimpl.h"
#include "nodes_array.h"
using namespace dae;
#include "xmlfunctions.h"
#include <typeinfo>
#include "xmlfunctions.h"
using namespace dae::xml;
using namespace boost;

namespace dae 
{
namespace core 
{
bool adDoEnclose(const adNodeArray* node)
{
	return true;
}

void adDoEnclose(const adNodeArray* parent, const adNodeArray* left, bool& bEncloseLeft, const adNodeArray* right, bool& bEncloseRight)
{
	bEncloseLeft  = true;
	bEncloseRight = true;

	if(!parent || !left || !right)
		return;
}

/*********************************************************************************************
	adNodeArray
**********************************************************************************************/
adNodeArray* adNodeArray::CreateNode(const io::xmlTag_t* pTag)
{
	string strClass;
	string strName = "Class";

	io::xmlAttribute_t* pAttrClass = pTag->FindAttribute(strName);
	if(!pAttrClass)
		daeDeclareAndThrowException(exXMLIOError);

	pAttrClass->GetValue(strClass);
	if(strClass == "adConstantNodeArray")
	{
		return new adConstantNodeArray();
	}
	else if(strClass == "adRuntimeParameterNodeArray")
	{
		return new adRuntimeParameterNodeArray();
	}
	else if(strClass == "adRuntimeVariableNodeArray")
	{
		return new adRuntimeVariableNodeArray();
	}
	else if(strClass == "adRuntimeTimeDerivativeNodeArray")
	{
		return new adRuntimeTimeDerivativeNodeArray();
	}
	else if(strClass == "adRuntimePartialDerivativeNodeArray")
	{
		return new adRuntimePartialDerivativeNodeArray();
	}
	else if(strClass == "adUnaryNodeArray")
	{
		return new adUnaryNodeArray();
	}
	else if(strClass == "adBinaryNodeArray")
	{
		return new adBinaryNodeArray();
	}
	else if(strClass == "adSetupParameterNodeArray")
	{
		return new adSetupParameterNodeArray();
	}
	else if(strClass == "adSetupVariableNodeArray")
	{
		return new adSetupVariableNodeArray();
	}
	else if(strClass == "adSetupTimeDerivativeNodeArray")
	{
		return new adSetupPartialDerivativeNodeArray();
	}
	else if(strClass == "adSetupPartialDerivativeNodeArray")
	{
		return new adSetupPartialDerivativeNodeArray();
	}
	else
	{
		daeDeclareAndThrowException(exXMLIOError);
		return NULL;
	}
	return NULL;
}

void adNodeArray::SaveNode(io::xmlTag_t* pTag, const string& strObjectName, const adNodeArray* node)
{
	io::xmlTag_t* pChildTag = pTag->AddTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);
	node->Save(pChildTag);
}

adNodeArray* adNodeArray::OpenNode(io::xmlTag_t* pTag, const string& strObjectName, io::daeOnOpenObjectDelegate_t<adNodeArray>* ood)
{
	io::xmlTag_t* pChildTag = pTag->FindTag(strObjectName);
	if(!pChildTag)
		daeDeclareAndThrowException(exXMLIOError);

	adNodeArray* node = adNodeArray::CreateNode(pChildTag);
	if(!node)
		daeDeclareAndThrowException(exXMLIOError);

	if(ood)
		ood->BeforeOpenObject(node);
	node->Open(pChildTag);
	if(ood)
		ood->AfterOpenObject(node);

	return node;
}

void adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(io::xmlTag_t* pTag, 
													       const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
													       const daeSaveAsMathMLContext* c)
{
	size_t i, n;
	string strName, strValue;
	io::xmlTag_t *mrow;
	
	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	n = arrNodes.size();
	for(i = 0; i < n; i++)
	{
		if(i != 0)
		{
			strName  = "mo";
			strValue = ",";
			mrow->AddTag(strName, strValue);
		}
		arrNodes[i]->SaveAsPresentationMathML(mrow, c);
	}
}
 
string adNodeArray::SaveRuntimeNodeArrayAsLatex(const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
										        const daeSaveAsMathMLContext* c)
{
	size_t i, n;
	string strResult;
	
	n = arrNodes.size();
	for(i = 0; i < n; i++)
	{
		if(i != 0)
			strResult += ", ";
		strResult += arrNodes[i]->SaveAsLatex(c);
	}
	return strResult;
}

string adNodeArray::SaveRuntimeNodeArrayAsPlainText(const std::vector< boost::shared_ptr<adNode> >& arrNodes, 
											        const daeSaveAsMathMLContext* c)
{
	size_t i, n;
	string strResult;
	
	n = arrNodes.size();
	for(i = 0; i < n; i++)
	{
		if(i != 0)
			strResult += ", ";
		strResult += arrNodes[i]->SaveAsLatex(c);
	}
	return strResult;
}

/*********************************************************************************************
	adNodeArrayImpl
**********************************************************************************************/
void adNodeArrayImpl::ExportAsPlainText(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsPlainText(NULL);
	file.close();
}

void adNodeArrayImpl::ExportAsLatex(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsLatex(NULL);
	file.close();
}

void adNodeArrayImpl::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
}

/*********************************************************************************************
	adConstantNodeArray
**********************************************************************************************/
adConstantNodeArray::adConstantNodeArray(const real_t d) : m_dValue(d)
{
}

adConstantNodeArray::adConstantNodeArray()
{
	m_dValue = 0;
}

adConstantNodeArray::~adConstantNodeArray()
{
}

size_t adConstantNodeArray::GetSize(void) const
{
	return 1;
}

void adConstantNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
}

adouble_array adConstantNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble_array tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( Clone() );
		return tmp;
	}
	
	tmp.Resize(1);
	tmp[0] = adouble(m_dValue);
	return tmp;
}

adNodeArray* adConstantNodeArray::Clone(void) const
{
	return new adConstantNodeArray(*this);
}

string adConstantNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* /*c*/) const
{
	return textCreator::Constant(m_dValue);
}

string adConstantNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* /*c*/) const
{
	return latexCreator::Constant(m_dValue);
}

void adConstantNodeArray::Open(io::xmlTag_t* pTag)
{
	string strName = "Value";
	pTag->Open(strName, m_dValue);
}

void adConstantNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName = "Value";
	pTag->Save(strName, m_dValue);
}

void adConstantNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	xmlContentCreator::Constant(pTag, m_dValue);
}

void adConstantNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
	xmlPresentationCreator::Constant(pTag, m_dValue);
}

void adConstantNodeArray::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adRuntimeParameterNodeArray
**********************************************************************************************/
adRuntimeParameterNodeArray::adRuntimeParameterNodeArray(void)
{
	m_pParameter = NULL;
}

adRuntimeParameterNodeArray::~adRuntimeParameterNodeArray()
{
}

size_t adRuntimeParameterNodeArray::GetSize(void) const
{
	return m_ptrarrParameterNodes.size();
}

void adRuntimeParameterNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

adouble_array adRuntimeParameterNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble_array tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( Clone() );
		return tmp;
	}
	
	size_t n = m_ptrarrParameterNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrParameterNodes[i]->Evaluate(pExecutionContext);
	
	return tmp;
}

adNodeArray* adRuntimeParameterNodeArray::Clone(void) const
{
	return new adRuntimeParameterNodeArray(*this);
}

string adRuntimeParameterNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrParameterNodes, c);
}

string adRuntimeParameterNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsLatex(m_ptrarrParameterNodes, c);
}

void adRuntimeParameterNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeParameterNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pParameter->GetName());

	strName = "Nodes";
	pTag->SaveObjectArray(strName, m_ptrarrParameterNodes);
}

void adRuntimeParameterNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimeParameterNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrParameterNodes, c);
}

void adRuntimeParameterNodeArray::AddVariableIndexToArray(map<size_t, size_t>& /*mapIndexes*/)
{
}

/*********************************************************************************************
	adRuntimeVariableNodeArray
**********************************************************************************************/
adRuntimeVariableNodeArray::adRuntimeVariableNodeArray()
{
	m_pVariable = NULL;
}

adRuntimeVariableNodeArray::~adRuntimeVariableNodeArray()
{
}

size_t adRuntimeVariableNodeArray::GetSize(void) const
{
	return m_ptrarrVariableNodes.size();
}

void adRuntimeVariableNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

adouble_array adRuntimeVariableNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble_array tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( Clone() );
		return tmp;
	}
	
	size_t n = m_ptrarrVariableNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrVariableNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

adNodeArray* adRuntimeVariableNodeArray::Clone(void) const
{
	return new adRuntimeVariableNodeArray(*this);
}

string adRuntimeVariableNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrVariableNodes, c);
}

string adRuntimeVariableNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsLatex(m_ptrarrVariableNodes, c);
}

void adRuntimeVariableNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeVariableNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Nodes";
	pTag->SaveObjectArray(strName, m_ptrarrVariableNodes);
}

void adRuntimeVariableNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimeVariableNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrVariableNodes, c);
}

void adRuntimeVariableNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	for(size_t i = 0; i < m_ptrarrVariableNodes.size(); i++)
		m_ptrarrVariableNodes[i]->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adRuntimeTimeDerivativeNodeArray
**********************************************************************************************/
adRuntimeTimeDerivativeNodeArray::adRuntimeTimeDerivativeNodeArray()
{	
	m_pVariable = NULL;
	m_nDegree   = 0;
}

adRuntimeTimeDerivativeNodeArray::~adRuntimeTimeDerivativeNodeArray()
{
}

size_t adRuntimeTimeDerivativeNodeArray::GetSize(void) const
{
	return m_ptrarrTimeDerivativeNodes.size();
}

void adRuntimeTimeDerivativeNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

adouble_array adRuntimeTimeDerivativeNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble_array tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( Clone() );
		return tmp;
	}

	size_t n = m_ptrarrTimeDerivativeNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrTimeDerivativeNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

adNodeArray* adRuntimeTimeDerivativeNodeArray::Clone(void) const
{
	return new adRuntimeTimeDerivativeNodeArray(*this);
}

string adRuntimeTimeDerivativeNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrTimeDerivativeNodes, c);
}

string adRuntimeTimeDerivativeNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsLatex(m_ptrarrTimeDerivativeNodes, c);
}

void adRuntimeTimeDerivativeNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeTimeDerivativeNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Nodes";
	pTag->SaveObjectArray(strName, m_ptrarrTimeDerivativeNodes);
}

void adRuntimeTimeDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimeTimeDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrTimeDerivativeNodes, c);
}

void adRuntimeTimeDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	for(size_t i = 0; i < m_ptrarrTimeDerivativeNodes.size(); i++)
		m_ptrarrTimeDerivativeNodes[i]->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adRuntimePartialDerivativeNodeArray
**********************************************************************************************/
adRuntimePartialDerivativeNodeArray::adRuntimePartialDerivativeNodeArray()
{	
	m_pVariable = NULL;
	m_pDomain   = NULL;
	m_nDegree   = 0;
}

adRuntimePartialDerivativeNodeArray::~adRuntimePartialDerivativeNodeArray()
{
}

size_t adRuntimePartialDerivativeNodeArray::GetSize(void) const
{
	return m_ptrarrPartialDerivativeNodes.size();
}

void adRuntimePartialDerivativeNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	arrRanges = m_arrRanges;
}

adouble_array adRuntimePartialDerivativeNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
// If we are in evaluate mode we dont need the value
	adouble_array tmp;
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( Clone() );
		return tmp;
	}

	size_t n = m_ptrarrPartialDerivativeNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrPartialDerivativeNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

adNodeArray* adRuntimePartialDerivativeNodeArray::Clone(void) const
{
	return new adRuntimePartialDerivativeNodeArray(*this);
}

string adRuntimePartialDerivativeNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrPartialDerivativeNodes, c);
}

string adRuntimePartialDerivativeNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	return adNodeArray::SaveRuntimeNodeArrayAsLatex(m_ptrarrPartialDerivativeNodes, c);
}

void adRuntimePartialDerivativeNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimePartialDerivativeNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Name";
	pTag->Save(strName, m_pVariable->GetName());

	strName = "Domain";
	pTag->Save(strName, m_pDomain->GetName());

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Nodes";
	pTag->SaveObjectArray(strName, m_ptrarrPartialDerivativeNodes);
}

void adRuntimePartialDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimePartialDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrPartialDerivativeNodes, c);
}

void adRuntimePartialDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	for(size_t i = 0; i < m_ptrarrPartialDerivativeNodes.size(); i++)
		m_ptrarrPartialDerivativeNodes[i]->AddVariableIndexToArray(mapIndexes);
}


/*********************************************************************************************
	adRuntimeSpecialFunctionNode
**********************************************************************************************/
adRuntimeSpecialFunctionNode::adRuntimeSpecialFunctionNode(daeeSpecialUnaryFunctions eFun, 
													       daeModel* pModel,
														   boost::shared_ptr<adNodeArray> n)
{
	eFunction = eFun;
	m_pModel  = pModel;
	node      = n;
}

adRuntimeSpecialFunctionNode::adRuntimeSpecialFunctionNode()
{
	m_pModel  = NULL;
	eFunction = eSUFUnknown;
}

adRuntimeSpecialFunctionNode::~adRuntimeSpecialFunctionNode()
{
}

adouble adRuntimeSpecialFunctionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	adouble tmp;
	adouble_array ad;
	
	ad = node->Evaluate(pExecutionContext);
	
	switch(eFunction)
	{
	case eSum:
		tmp = m_pModel->__sum__(ad);
		break;
		
	case eProduct:
		tmp = m_pModel->__product__(ad);
		break;
		
	case eAverage:
		tmp = m_pModel->__average__(ad);
		break;

	case eMinInArray:
		tmp = m_pModel->__min__(ad);
		break;
		
	case eMaxInArray:
		tmp = m_pModel->__max__(ad);
		break;
		
	default:
		daeDeclareAndThrowException(exInvalidCall);
	}

	return tmp;
}

adNode* adRuntimeSpecialFunctionNode::Clone(void) const
{
	return new adRuntimeSpecialFunctionNode(*this);
}

string adRuntimeSpecialFunctionNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	switch(eFunction)
	{
	case eSum:
		strResult += "sum(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eProduct:
		strResult += "product(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eMinInArray:
		strResult += "min(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eMaxInArray:
		strResult += "max(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eAverage:
		strResult += "average(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adRuntimeSpecialFunctionNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	switch(eFunction)
	{
	case eSum:
		strResult += "\\sum ";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	case eProduct:
		strResult += "\\prod";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	case eMinInArray:
		strResult += "min";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eMaxInArray:
		strResult += "max";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eAverage:
		strResult += "\\overline";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	return strResult;
}

void adRuntimeSpecialFunctionNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeSpecialFunctionNode::Save(io::xmlTag_t* pTag) const
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeSpecialFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimeSpecialFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *temp;
	
	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	switch(eFunction)
	{
	case eSum:
		strName  = "mo";
		strValue = "&sum;";
		mrow->AddTag(strName, strValue);

		strName  = "mo";
		strValue = "(";
		mrow->AddTag(strName, strValue);
		
		node->SaveAsPresentationMathML(mrow, c);

		strName  = "mo";
		strValue = ")";
		mrow->AddTag(strName, strValue);

		break;
	case eProduct:
		strName  = "mo";
		strValue = "&prod;";
		mrow->AddTag(strName, strValue);

		strName  = "mo";
		strValue = "(";
		mrow->AddTag(strName, strValue);
		
		node->SaveAsPresentationMathML(mrow, c);

		strName  = "mo";
		strValue = ")";
		mrow->AddTag(strName, strValue);

		break; 
	case eMinInArray:
	case eMaxInArray:
	case eAverage:
		strName  = "mo";
		if(eFunction == eMinInArray)
			strValue = "min";
		else if(eFunction == eMaxInArray)
			strValue = "max";
		else if(eFunction == eAverage)
			strValue = "average";
		temp = mrow->AddTag(strName, strValue);		
		temp->AddAttribute(string("mathvariant"), string("italic"));

		strName  = "mo";
		strValue = "(";
		mrow->AddTag(strName, strValue);
		
		node->SaveAsPresentationMathML(mrow, c);

		strName  = "mo";
		strValue = ")";
		mrow->AddTag(strName, strValue);
		break; 

	default:
		daeDeclareAndThrowException(exXMLIOError)
	}
}

void adRuntimeSpecialFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adRuntimeIntegralNode
**********************************************************************************************/
adRuntimeIntegralNode::adRuntimeIntegralNode(daeeIntegralFunctions eFun,
											 daeModel* pModel,
											 boost::shared_ptr<adNodeArray> n,
											 daeDomain* pDomain,
											 const vector<size_t>& narrPoints)
{
	m_pModel     = pModel;
	m_pDomain    = pDomain;
	node         = n;
	eFunction    = eFun;
	m_narrPoints = narrPoints;
}

adRuntimeIntegralNode::adRuntimeIntegralNode()
{
	m_pDomain = NULL;
	eFunction = eIFUnknown;
}

adRuntimeIntegralNode::~adRuntimeIntegralNode()
{
}

adouble adRuntimeIntegralNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(m_narrPoints.empty())
		daeDeclareAndThrowException(exInvalidCall);

	adouble tmp;
	adouble_array a;
	
	a   = node->Evaluate(pExecutionContext);
	tmp = m_pModel->__integral__(a, const_cast<daeDomain*>(m_pDomain), m_narrPoints);
	
	return tmp;
}

adNode* adRuntimeIntegralNode::Clone(void) const
{
	return new adRuntimeIntegralNode(*this);
}

string adRuntimeIntegralNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSingleIntegral:
		strResult += "integral(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adRuntimeIntegralNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	string strDomain = daeObject::GetRelativeName(c->m_pModel, m_pDomain);

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSingleIntegral:
		strResult += "\\int";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " ";
		strResult += "\\mathrm{d}";
		strResult += strDomain;
		strResult += " } ";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	return strResult;
}

void adRuntimeIntegralNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeIntegralNode::Save(io::xmlTag_t* pTag) const
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adRuntimeIntegralNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adRuntimeIntegralNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *mrow2;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	string strDomain = daeObject::GetRelativeName(c->m_pModel, m_pDomain);

	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	switch(eFunction)
	{
	case eSingleIntegral:
		strName  = "mo";
		strValue = "&int;";
		mrow->AddTag(strName, strValue);		
		node->SaveAsPresentationMathML(mrow, c);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow->AddTag(strName, strValue);

		strName  = "mo";
		strValue = "d";
		mrow2->AddTag(strName, strValue);		

		strName  = "mi";
		strValue = strDomain;
		mrow2->AddTag(strName, strValue);		

		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}
}

void adRuntimeIntegralNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adUnaryNodeArray
**********************************************************************************************/
adUnaryNodeArray::adUnaryNodeArray(daeeUnaryFunctions eFun, shared_ptr<adNodeArray> n)
{
	node = n;
	eFunction = eFun;
}

adUnaryNodeArray::adUnaryNodeArray()
{
	eFunction = eUFUnknown;
}

adUnaryNodeArray::~adUnaryNodeArray()
{
}

size_t adUnaryNodeArray::GetSize(void) const
{
	return node->GetSize();
}

void adUnaryNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	adNodeArrayImpl* n = dynamic_cast<adNodeArrayImpl*>(node.get());
	if(!n)
		daeDeclareAndThrowException(exInvalidPointer);
		
	n->GetArrayRanges(arrRanges);
}

adouble_array adUnaryNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
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
		daeDeclareAndThrowException(exNotImplemented);
		return adouble_array();
	}
	return adouble_array();
}

adNodeArray* adUnaryNodeArray::Clone(void) const
{
	shared_ptr<adNodeArray> n = shared_ptr<adNodeArray>( (node ? node->Clone() : NULL) );
	return new adUnaryNodeArray(eFunction, n);
}

string adUnaryNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	switch(eFunction)
	{
	case eSign:
		strResult += "-(";
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
		daeDeclareAndThrowException(exNotImplemented);
	}
	return strResult;
}

string adUnaryNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	switch(eFunction)
	{
	case eSign:
		strResult  = "{ "; // Start
		strResult += "- ";
		strResult += "\\left( ";
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
		daeDeclareAndThrowException(exNotImplemented);
	}

	return strResult;
}

void adUnaryNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adUnaryNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Node";
	adNodeArray::SaveNode(pTag, strName, node.get());
}

void adUnaryNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adUnaryNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
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
		strValue = "-";
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

void adUnaryNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adBinaryNodeArray
**********************************************************************************************/
adBinaryNodeArray::adBinaryNodeArray(daeeBinaryFunctions eFun, shared_ptr<adNodeArray> l, shared_ptr<adNodeArray> r)
{
	left  = l;
	right = r;
	eFunction = eFun;
}

adBinaryNodeArray::adBinaryNodeArray()
{
	eFunction = eBFUnknown;
}

adBinaryNodeArray::~adBinaryNodeArray()
{
}

size_t adBinaryNodeArray::GetSize(void) const
{
	if(left->GetSize() != right->GetSize())
		daeDeclareAndThrowException(exNotImplemented);
	return left->GetSize();
}

void adBinaryNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
	vector<daeArrayRange> arrRangesL, arrRangesR;
	
	adNodeArrayImpl* l = dynamic_cast<adNodeArrayImpl*>(left.get());
	if(!l)
		daeDeclareAndThrowException(exInvalidPointer);

	adNodeArrayImpl* r = dynamic_cast<adNodeArrayImpl*>(right.get());
	if(!r)
		daeDeclareAndThrowException(exInvalidPointer);
		
	l->GetArrayRanges(arrRangesL);
	r->GetArrayRanges(arrRangesR);
	
	if(arrRangesL.size() == 0)
		arrRanges = arrRangesR;
	else
		arrRanges = arrRangesL;
}

adouble_array adBinaryNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
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
		daeDeclareAndThrowException(exNotImplemented);
		break;
	case eMin:
		daeDeclareAndThrowException(exNotImplemented);
		break;
	case eMax:
		daeDeclareAndThrowException(exNotImplemented);
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return adouble_array();
	}
}

adNodeArray* adBinaryNodeArray::Clone(void) const
{
	shared_ptr<adNodeArray> l = shared_ptr<adNodeArray>( (left  ? left->Clone()  : NULL) );
	shared_ptr<adNodeArray> r = shared_ptr<adNodeArray>( (right ? right->Clone() : NULL) );
	return new adBinaryNodeArray(eFunction, l, r);
}

string adBinaryNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
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

string adBinaryNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
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

void adBinaryNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adBinaryNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Left";
	adNodeArray::SaveNode(pTag, strName, left.get());

	strName = "Right";
	adNodeArray::SaveNode(pTag, strName, right.get());
}

void adBinaryNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adBinaryNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
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
			strValue = "&InvisibleTimes;"; //"&#x00D7;";
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
		daeDeclareAndThrowException(exNotImplemented);
		break;

	case eMax:
		daeDeclareAndThrowException(exNotImplemented);
		break;

	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
}

void adBinaryNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	left->AddVariableIndexToArray(mapIndexes);
	right->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adSetupSpecialFunctionNode
**********************************************************************************************/
adSetupSpecialFunctionNode::adSetupSpecialFunctionNode(daeeSpecialUnaryFunctions eFun, 
													   daeModel* pModel,
													   shared_ptr<adNodeArray> n)
{
	m_pModel  = pModel;
	node      = n;
	eFunction = eFun;
}

adSetupSpecialFunctionNode::adSetupSpecialFunctionNode()
{
	m_pModel  = NULL;
	eFunction = eSUFUnknown;
}

adSetupSpecialFunctionNode::~adSetupSpecialFunctionNode()
{
}

adouble adSetupSpecialFunctionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSum:
		return m_pModel->__sum__(node->Evaluate(pExecutionContext));
		break;
	case eProduct:
		return m_pModel->__product__(node->Evaluate(pExecutionContext));
		break;
	case eAverage:
		return m_pModel->__average__(node->Evaluate(pExecutionContext));
		break;
	case eMinInArray:
		return m_pModel->__min__(node->Evaluate(pExecutionContext));
		break;
	case eMaxInArray:
		return m_pModel->__max__(node->Evaluate(pExecutionContext));
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
}

adNode* adSetupSpecialFunctionNode::Clone(void) const
{
	shared_ptr<adNodeArray> n = shared_ptr<adNodeArray>( (node ? node->Clone() : NULL) );
	return new adSetupSpecialFunctionNode(eFunction, m_pModel, n);
}

string adSetupSpecialFunctionNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	switch(eFunction)
	{
	case eSum:
		strResult += "sum(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eProduct:
		strResult += "product(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eMinInArray:
		strResult += "min(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eMaxInArray:
		strResult += "max(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	case eAverage:
		strResult += "average(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adSetupSpecialFunctionNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	switch(eFunction)
	{
	case eSum:
		strResult += "\\sum ";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	case eProduct:
		strResult += "\\prod";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	case eMinInArray:
		strResult += "min";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eMaxInArray:
		strResult += "max";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eAverage:
		strResult += "\\overline";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " } ";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	return strResult;
}

void adSetupSpecialFunctionNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adSetupSpecialFunctionNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Node";
	adNodeArray::SaveNode(pTag, strName, node.get());
}

void adSetupSpecialFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adSetupSpecialFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *mrow2, *temp;

	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	switch(eFunction)
	{
	case eSum:
		strName  = "mo";
		strValue = "&sum;";
		mrow->AddTag(strName, strValue);		
		node->SaveAsPresentationMathML(mrow, c);
		break;
	case eProduct:
		strName  = "mo";
		strValue = "&prod;";
		mrow->AddTag(strName, strValue);		
		node->SaveAsPresentationMathML(mrow, c);
		break; 
	case eMinInArray:
		strName  = "mo";
		strValue = "min";
		temp = mrow->AddTag(strName, strValue);		
		temp->AddAttribute(string("mathvariant"), string("italic"));

			strName  = "mrow";
			strValue = "";
			mrow2 = mrow->AddTag(strName, strValue);
				
				strName  = "mo";
				strValue = "(";
				mrow2->AddTag(strName, strValue);

				node->SaveAsPresentationMathML(mrow2, c);
	
				strName  = "mo";
				strValue = ")";
				mrow2->AddTag(strName, strValue);
		break;
	case eMaxInArray:
		strName  = "mo";
		strValue = "max";
		temp = mrow->AddTag(strName, strValue);		
		temp->AddAttribute(string("mathvariant"), string("italic"));

			strName  = "mrow";
			strValue = "";
			mrow2 = mrow->AddTag(strName, strValue);
				
				strName  = "mo";
				strValue = "(";
				mrow2->AddTag(strName, strValue);

				node->SaveAsPresentationMathML(mrow2, c);
	
				strName  = "mo";
				strValue = ")";
				mrow2->AddTag(strName, strValue);
		break;
	case eAverage:
		strName  = "mo";
		strValue = "average"; 
		temp = mrow->AddTag(strName, strValue);		
		temp->AddAttribute(string("mathvariant"), string("italic"));

			strName  = "mrow";
			strValue = "";
			mrow2 = mrow->AddTag(strName, strValue);
				
				strName  = "mo";
				strValue = "(";
				mrow2->AddTag(strName, strValue);
	
				node->SaveAsPresentationMathML(mrow2, c);
	
				strName  = "mo";
				strValue = ")";
				mrow2->AddTag(strName, strValue);
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}
}

void adSetupSpecialFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

bool adSetupSpecialFunctionNode::IsLinear(void) const
{
	return false;
}

bool adSetupSpecialFunctionNode::IsFunctionOfVariables(void) const
{
	return true;
}

/*********************************************************************************************
	adSetupExpressionDerivativeNode
**********************************************************************************************/
adSetupExpressionDerivativeNode::adSetupExpressionDerivativeNode(daeModel* pModel,
													             shared_ptr<adNode> n)
{
	m_pModel  = pModel;
	node      = n;
	m_nDegree = 1;
}

adSetupExpressionDerivativeNode::adSetupExpressionDerivativeNode()
{
	m_nDegree = 0;
	m_pModel  = NULL;
}

adSetupExpressionDerivativeNode::~adSetupExpressionDerivativeNode()
{
}

adouble adSetupExpressionDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble a, tmp;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	a = node->Evaluate(pExecutionContext);
	tmp.setGatherInfo(true);
	tmp.node = calc_dt(a.node, pExecutionContext);
	return tmp;
}

// Here I work on runtime nodes!!
boost::shared_ptr<adNode> adSetupExpressionDerivativeNode::calc_dt(boost::shared_ptr<adNode> n, const daeExecutionContext* pExecutionContext) const
{
	adNode* adnode;
	adouble l, r, dl, dr;
	boost::shared_ptr<adNode> tmp;
		
	adnode = n.get();
	if( dynamic_cast<adUnaryNode*>(adnode) )
	{
		daeDeclareAndThrowException(exNotImplemented)
	}
	else if( dynamic_cast<adBinaryNode*>(adnode) )
	{
		adBinaryNode* node = dynamic_cast<adBinaryNode*>(adnode);

		l.setGatherInfo(true);
		r.setGatherInfo(true);
		dl.setGatherInfo(true);
		dr.setGatherInfo(true);

		dl.node = calc_dt(node->left, pExecutionContext); 
		dr.node = calc_dt(node->right, pExecutionContext); 
		l.node = node->left; 
		r.node = node->right; 
		
		switch(node->eFunction)
		{
		case ePlus:
			tmp = (dl + dr).node;
			break;
		case eMinus:
			tmp = (dl - dr).node;
			break;
		case eMulti:
			tmp = (l * dr + r * dl).node; 
			break;
		case eDivide:
			tmp = ((r * dl - l * dr)/(r * r)).node; 
			break;
		default:
			daeDeclareException(exInvalidCall);
			e << "The function dt() does not accept expressions containing pow, min and max, in model [" << m_pModel->GetCanonicalName() << "]";
			throw e;
		}
	}
	else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
	{
		adRuntimeVariableNode* rtnode = dynamic_cast<adRuntimeVariableNode*>(adnode);
	// Here I do not check if the variable is fixed (cnFixed) and do not add it to list of derivative variables !!!
	//	adRuntimeTimeDerivativeNode* devnode = new adRuntimeTimeDerivativeNode();
	//	tmp = shared_ptr<adNode>(devnode);
	//	devnode->m_pVariable        = rtnode->m_pVariable;
	//	devnode->m_nOverallIndex    = rtnode->m_nOverallIndex;
	//	devnode->m_nDegree          = 1;
	//	devnode->m_pdTimeDerivative = pExecutionContext->m_pDataProxy->GetTimeDerivative(rtnode->m_nOverallIndex);
	//	devnode->m_narrDomains      = rtnode->m_narrDomains;

	// Here I check for it regularly
		size_t N = rtnode->m_narrDomains.size();
		size_t* indexes = new size_t[N];
		for(size_t i = 0; i < N; i++)
			indexes[i] = rtnode->m_narrDomains[i];
		adouble adres = rtnode->m_pVariable->Calculate_dt(indexes, N);
		delete[] indexes;
		tmp = adres.node;
	}
	else if( dynamic_cast<adRuntimeParameterNode*>(adnode)  || 
			 dynamic_cast<adDomainIndexNode*>(adnode)       ||
			 dynamic_cast<adConstantNode*>(adnode)           )
	{
		tmp = shared_ptr<adNode>(new adConstantNode(0));
	}
	else
	{
		daeDeclareException(exInvalidCall);
		e << "The function dt() does not accept expressions containing special functions or time/partial derivatives, in model [" << m_pModel->GetCanonicalName() << "]";
		throw e;
	}
	return tmp;
}

adNode* adSetupExpressionDerivativeNode::Clone(void) const
{
	return new adSetupExpressionDerivativeNode(*this);
}

string adSetupExpressionDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strExpression = node->SaveAsPlainText(c);
	return textCreator::TimeDerivative(m_nDegree, strExpression, strarrIndexes, true);
}

string adSetupExpressionDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strExpression = node->SaveAsLatex(c);
	return latexCreator::TimeDerivative(m_nDegree, strExpression, strarrIndexes, true);
}

void adSetupExpressionDerivativeNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adSetupExpressionDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Node";
	adNode::SaveNode(pTag, strName, node.get());
}

void adSetupExpressionDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adSetupExpressionDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2;

	strName  = "mfrac";
	strValue = "";
	mfrac = pTag->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "d"; // Should be &dd; but it does not show up correctly in windows
		mrow1->AddTag(strName, strValue);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow1->AddTag(strName, strValue);

		mrow2->AddTag(string("mo"), string("("));
		node->SaveAsPresentationMathML(mrow2, c);
		mrow2->AddTag(string("mo"), string(")"));
	}
	else
	{
		strName  = "msup";
		strValue = "";
		msup = mrow1->AddTag(strName, strValue);
			strName  = "mo";
            strValue = "d";
			msup->AddTag(strName, strValue);

			strName  = "mn";
			strValue = "2";
			msup->AddTag(strName, strValue);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow1->AddTag(strName, strValue);

		mrow2->AddTag(string("mo"), string("("));
		node->SaveAsPresentationMathML(mrow2, c);
		mrow2->AddTag(string("mo"), string(")"));
	}

	strName  = "mrow";
	strValue = "";
	mrow2 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "d";
		mrow2->AddTag(strName, strValue);

		strName  = "mi";
		strValue = "t";
		mrow2->AddTag(strName, strValue);
	}
	else
	{
		strName  = "mo";
		strValue = "d";
		mrow2->AddTag(strName, strValue);

		strName  = "msup";
		strValue = "";
		msup = mrow2->AddTag(strName, strValue);

			strName  = "mi";
			strValue = "t";
			msup->AddTag(strName, strValue);

			strName  = "mn";
			strValue = "2";
			msup->AddTag(strName, strValue);
	}
}

void adSetupExpressionDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adSetupExpressionPartialDerivativeNode
**********************************************************************************************/
adSetupExpressionPartialDerivativeNode::adSetupExpressionPartialDerivativeNode(daeModel*  pModel,
																			   daeDomain* pDomain,
													                           shared_ptr<adNode> n)
{
	m_pModel  = pModel;
	m_pDomain = pDomain;
	node      = n;
	m_nDegree = 1;
}

adSetupExpressionPartialDerivativeNode::adSetupExpressionPartialDerivativeNode()
{
	m_nDegree = 0;
	m_pModel  = NULL;
	m_pDomain = NULL;
}

adSetupExpressionPartialDerivativeNode::~adSetupExpressionPartialDerivativeNode()
{
}

adouble adSetupExpressionPartialDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble a, tmp;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);

	a = node->Evaluate(pExecutionContext);
	tmp.setGatherInfo(true);
	tmp.node = calc_d(a.node, m_pDomain, pExecutionContext);
	return tmp;
}

// Here I work on runtime nodes!!
boost::shared_ptr<adNode> adSetupExpressionPartialDerivativeNode::calc_d(boost::shared_ptr<adNode> n, daeDomain* pDomain, const daeExecutionContext* pExecutionContext) const
{
	adNode* adnode;
	adouble l, r, dl, dr;
	boost::shared_ptr<adNode> tmp;
		
	adnode = n.get();
	if( dynamic_cast<adUnaryNode*>(adnode) )
	{
		daeDeclareAndThrowException(exNotImplemented)
	}
	else if( dynamic_cast<adBinaryNode*>(adnode) )
	{
		adBinaryNode* node = dynamic_cast<adBinaryNode*>(adnode);

		l.setGatherInfo(true);
		r.setGatherInfo(true);
		dl.setGatherInfo(true);
		dr.setGatherInfo(true);

		dl.node = calc_d(node->left, pDomain, pExecutionContext); 
		dr.node = calc_d(node->right, pDomain, pExecutionContext); 
		l.node = node->left; 
		r.node = node->right; 
		
		switch(node->eFunction)
		{
		case ePlus:
			tmp = (dl + dr).node;
			break;
		case eMinus:
			tmp = (dl - dr).node;
			break;
		case eMulti:
			tmp = (l * dr + r * dl).node; 
			break;
		case eDivide:
			tmp = ((r * dl - l * dr)/(r * r)).node; 
			break;
		default:
			daeDeclareException(exInvalidCall);
			e << "The function d() does not accept expressions containing pow, min and max, in model [" << m_pModel->GetCanonicalName() << "]";
			throw e;
		}
	}
	else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
	{
		adRuntimeVariableNode* rtnode = dynamic_cast<adRuntimeVariableNode*>(adnode);
		size_t N = rtnode->m_narrDomains.size();
		if(N == 0)
		{
			tmp = shared_ptr<adNode>(new adConstantNode(0));
		}
		else
		{
			size_t* indexes = new size_t[N];
			for(size_t i = 0; i < N; i++)
				indexes[i] = rtnode->m_narrDomains[i];
	
			adouble adres = rtnode->m_pVariable->partial(1, *pDomain, indexes, N);
			delete[] indexes;
			tmp = adres.node;
		}
	}
	else if( dynamic_cast<adRuntimePartialDerivativeNode*>(adnode) )
	{
		adRuntimePartialDerivativeNode* rtnode = dynamic_cast<adRuntimePartialDerivativeNode*>(adnode);
		if(rtnode->m_nDegree == 2)
		{
			if(rtnode->m_pDomain == pDomain)
			{
				daeDeclareException(exInvalidCall);
				e << "The function d() cannot create partial derivatives of order higher than 2, in model [" << m_pModel->GetCanonicalName() << "]";
				throw e;
			}
			else // It is permitted - retrun 0
			{
				tmp = shared_ptr<adNode>(new adConstantNode(0));
			}
		}
		else // m_nDegree == 1
		{
			if(rtnode->m_pDomain == pDomain) // Clone it and increase the order to 2
			{
				adRuntimePartialDerivativeNode* devnode = dynamic_cast<adRuntimePartialDerivativeNode*>(rtnode->Clone());
				devnode->m_nDegree = 2;
				tmp = shared_ptr<adNode>(devnode);
			}
			else // call calc_d again on its node
			{
				tmp = calc_d(rtnode->pardevnode, pDomain, pExecutionContext); 
			}
		}
	}
	else if( dynamic_cast<adRuntimeParameterNode*>(adnode) || 
			 dynamic_cast<adDomainIndexNode*>(adnode)	   ||
			 dynamic_cast<adConstantNode*>(adnode)          )
	{
		tmp = shared_ptr<adNode>(new adConstantNode(0));
	}
	else
	{
		daeDeclareException(exInvalidCall);
		e << "The function d() does not accept expressions containing special functions, time derivatives, etc, in model [" << m_pModel->GetCanonicalName() << "]";
		throw e;
	}
	return tmp;
}

adNode* adSetupExpressionPartialDerivativeNode::Clone(void) const
{
	return new adSetupExpressionPartialDerivativeNode(*this);
}

string adSetupExpressionPartialDerivativeNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strExpression = node->SaveAsPlainText(c);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return textCreator::PartialDerivative(m_nDegree, strExpression, strDomainName, strarrIndexes, true);
}

string adSetupExpressionPartialDerivativeNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	vector<string> strarrIndexes;
	string strExpression = node->SaveAsLatex(c);
	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);
	return latexCreator::PartialDerivative(m_nDegree, strExpression, strDomainName, strarrIndexes, true);
}

void adSetupExpressionPartialDerivativeNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adSetupExpressionPartialDerivativeNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	strName = "Degree";
	pTag->Save(strName, m_nDegree);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	strName = "Node";
	adNode::SaveNode(pTag, strName, node.get());
}

void adSetupExpressionPartialDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adSetupExpressionPartialDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2;

	string strDomainName = daeObject::GetRelativeName(c->m_pModel, m_pDomain);

	strName  = "mfrac";
	strValue = "";
	mfrac = pTag->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "&PartialD;";
		mrow1->AddTag(strName, strValue);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow1->AddTag(strName, strValue);

		mrow2->AddTag(string("mo"), string("("));
		node->SaveAsPresentationMathML(mrow2, c);
		mrow2->AddTag(string("mo"), string(")"));
	}
	else
	{
		strName  = "msup";
		strValue = "";
		msup = mrow1->AddTag(strName, strValue);
			strName  = "mo";
            strValue = "&PartialD;";
			msup->AddTag(strName, strValue);

			strName  = "mn";
			strValue = "2";
			msup->AddTag(strName, strValue);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow1->AddTag(strName, strValue);

		mrow2->AddTag(string("mo"), string("("));
		node->SaveAsPresentationMathML(mrow2, c);
		mrow2->AddTag(string("mo"), string(")"));
	}

	strName  = "mrow";
	strValue = "";
	mrow2 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "&PartialD;";
		mrow2->AddTag(strName, strValue);

		strName  = "mi";
		strValue = strDomainName;
		mrow2->AddTag(strName, strValue);
	}
	else
	{
		strName  = "mo";
		strValue = "&PartialD;";
		mrow2->AddTag(strName, strValue);

		strName  = "msup";
		strValue = "";
		msup = mrow2->AddTag(strName, strValue);

			strName  = "mi";
			strValue = strDomainName;
			xmlPresentationCreator::WrapIdentifier(msup, strValue);
	
			strName  = "mn";
			strValue = "2";
			msup->AddTag(strName, strValue);
	}
}

void adSetupExpressionPartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adSetupIntegralNode
**********************************************************************************************/
adSetupIntegralNode::adSetupIntegralNode(daeeIntegralFunctions eFun,
										 daeModel* pModel,
										 shared_ptr<adNodeArray> n,
										 daeDomain* pDomain,
										 const daeArrayRange& arrayRange)
{
	m_pModel     = pModel;
	m_pDomain    = pDomain;
	node         = n;
	eFunction    = eFun;
	m_ArrayRange = arrayRange;
}

adSetupIntegralNode::adSetupIntegralNode()
{
	m_pModel  = NULL;
	m_pDomain = NULL;
	eFunction = eIFUnknown;
}

adSetupIntegralNode::~adSetupIntegralNode()
{
}

adouble adSetupIntegralNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble_array a;
	vector<size_t> narrPoints;

	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSingleIntegral:
		if(m_ArrayRange.m_eType != eRange)
			daeDeclareAndThrowException(exInvalidCall);
		
		m_ArrayRange.m_Range.GetPoints(narrPoints);
		a = node->Evaluate(pExecutionContext);
		
		return m_pModel->__integral__(a, const_cast<daeDomain*>(m_pDomain), narrPoints);
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
	return adouble();
}

adNode* adSetupIntegralNode::Clone(void) const
{
	return new adSetupIntegralNode(*this);
}

string adSetupIntegralNode::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	string strResult;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSingleIntegral:
		strResult += "integral(";
		strResult += node->SaveAsPlainText(c);
		strResult += ")";
		break;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
	}
	return strResult;
}

string adSetupIntegralNode::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	string strResult;
	string strDomain = daeObject::GetRelativeName(c->m_pModel, m_pDomain);

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	switch(eFunction)
	{
	case eSingleIntegral:
		strResult += "\\int";
		strResult += " { ";
		strResult += node->SaveAsLatex(c);
		strResult += " ";
		strResult += "\\mathrm{d}";
		strResult += strDomain;
		strResult += " } ";
		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}

	return strResult;
}

void adSetupIntegralNode::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adSetupIntegralNode::Save(io::xmlTag_t* pTag) const
{
	string strName;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	strName = "Function";
	SaveEnum(pTag, strName, eFunction);

	strName = "Domain";
	pTag->SaveObjectRef(strName, m_pDomain);

	strName = "Node";
	adNodeArray::SaveNode(pTag, strName, node.get());
}

void adSetupIntegralNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
}

void adSetupIntegralNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *mrow2;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	string strDomain = daeObject::GetRelativeName(c->m_pModel, m_pDomain);

	strName  = "mrow";
	strValue = "";
	mrow = pTag->AddTag(strName, strValue);

	switch(eFunction)
	{
	case eSingleIntegral:
		strName  = "mo";
		strValue = "&int;";
		mrow->AddTag(strName, strValue);		
		node->SaveAsPresentationMathML(mrow, c);

		strName  = "mrow";
		strValue = "";
		mrow2 = mrow->AddTag(strName, strValue);

		strName  = "mo";
		strValue = "d";
		mrow2->AddTag(strName, strValue);		

		strName  = "mi";
		strValue = strDomain;
		mrow2->AddTag(strName, strValue);		

		break;
	default:
		daeDeclareAndThrowException(exXMLIOError);
	}
}

void adSetupIntegralNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}

/*********************************************************************************************
	adSingleNodeArray
**********************************************************************************************/
adSingleNodeArray::adSingleNodeArray(boost::shared_ptr<adNode> n) : node(n)
{
}

adSingleNodeArray::adSingleNodeArray()
{
}

adSingleNodeArray::~adSingleNodeArray()
{
}

size_t adSingleNodeArray::GetSize(void) const
{
	return 1;
}

void adSingleNodeArray::GetArrayRanges(vector<daeArrayRange>& /*arrRanges*/) const
{	
}

adouble_array adSingleNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble a;
	adouble_array tmp;
	
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

// Achtung!! The node that I have is a setup node!!!
// During GatherInfo I have to transform it into the runtime node.
// Thus here I have to call node->Evaluate() and to store this value to cloned node
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adSingleNodeArray* clone = new adSingleNodeArray(*this);
		tmp.setGatherInfo(true);
		tmp.node = shared_ptr<adNodeArray>( clone );

		clone->node = node->Evaluate(pExecutionContext).node;

		return tmp;
	}
	
	tmp.Resize(1);
	a = node->Evaluate(pExecutionContext);
	tmp[0] = a;
	return tmp;
}

adNodeArray* adSingleNodeArray::Clone(void) const
{
	return new adSingleNodeArray(*this);
}

string adSingleNodeArray::SaveAsPlainText(const daeSaveAsMathMLContext* c) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->SaveAsPlainText(c);
}

string adSingleNodeArray::SaveAsLatex(const daeSaveAsMathMLContext* c) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->SaveAsLatex(c);
}

void adSingleNodeArray::Open(io::xmlTag_t* pTag)
{
	daeDeclareAndThrowException(exNotImplemented)
}

void adSingleNodeArray::Save(io::xmlTag_t* pTag) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	string strName = "Node";
	adNode::SaveNode(pTag, strName, node.get());
}

void adSingleNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* /*c*/) const
{
}

void adSingleNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeSaveAsMathMLContext* c) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	node->SaveAsPresentationMathML(pTag, c);
}

void adSingleNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes);
}


}
}

