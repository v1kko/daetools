#include "stdafx.h"
#include "coreimpl.h"
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

const adouble_array Array(const std::vector<quantity>& qarrValues)
{
	adouble_array tmp;
	tmp.setGatherInfo(true);
	tmp.node = adNodeArrayPtr(new adVectorNodeArray(qarrValues));
	return tmp;
}

const adouble_array Array(const std::vector<real_t>& darrValues)
{
	adouble_array tmp;
	tmp.setGatherInfo(true);
	tmp.node = adNodeArrayPtr(new adVectorNodeArray(darrValues));
	return tmp;
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
//	else if(strClass == "adRuntimeParameterNodeArray")
//	{
//		return new adRuntimeParameterNodeArray();
//	}
//	else if(strClass == "adRuntimeVariableNodeArray")
//	{
//		return new adRuntimeVariableNodeArray();
//	}
//	else if(strClass == "adRuntimeTimeDerivativeNodeArray")
//	{
//		return new adRuntimeTimeDerivativeNodeArray();
//	}
//	else if(strClass == "adRuntimePartialDerivativeNodeArray")
//	{
//		return new adRuntimePartialDerivativeNodeArray();
//	}
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
		daeDeclareAndThrowException(exXMLIOError)
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
													       const std::vector< adNodePtr >& arrNodes, 
													       const daeNodeSaveAsContext* c)
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
 
string adNodeArray::SaveRuntimeNodeArrayAsLatex(const std::vector< adNodePtr >& arrNodes, 
										        const daeNodeSaveAsContext* c)
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

//string adNodeArray::SaveRuntimeNodeArrayAsPlainText(const std::vector< adNodePtr >& arrNodes, 
//											        const daeNodeSaveAsContext* c)
//{
//	size_t i, n;
//	string strResult;
//	
//	n = arrNodes.size();
//	for(i = 0; i < n; i++)
//	{
//		if(i != 0)
//			strResult += ", ";
//		strResult += arrNodes[i]->SaveAsLatex(c);
//	}
//	return strResult;
//}

/*********************************************************************************************
	adNodeArrayImpl
**********************************************************************************************/
void adNodeArrayImpl::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{	
}

//void adNodeArrayImpl::ExportAsPlainText(string strFileName)
//{
//	string strLatex;
//	ofstream file(strFileName.c_str());
//	file << SaveAsPlainText(NULL);
//	file.close();
//}

void adNodeArrayImpl::ExportAsLatex(string strFileName)
{
	string strLatex;
	ofstream file(strFileName.c_str());
	file << SaveAsLatex(NULL);
	file.close();
}

bool adNodeArrayImpl::IsLinear(void) const
{
// All nodes are non-linear if I dont explicitly state that they are linear!
	return false;
}

bool adNodeArrayImpl::IsFunctionOfVariables(void) const
{
// All nodes are functions of variables if I dont explicitly state that they aint!
	return true;
}

bool adNodeArrayImpl::IsDifferential(void) const
{
    return false;
}

void adNodeArrayImpl::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{	
}

/*********************************************************************************************
	adConstantNodeArray
**********************************************************************************************/
adConstantNodeArray::adConstantNodeArray()
{
}

adConstantNodeArray::adConstantNodeArray(const real_t d)
                   : m_quantity(d, unit())
{
}

adConstantNodeArray::adConstantNodeArray(const real_t d, const unit& units) 
	               : m_quantity(d, units)
{
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
    
// 13.10.2012
//	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
//	{
//		tmp.setGatherInfo(true);
//		tmp.node = adNodeArrayPtr( Clone() );
//		return tmp;
//	}
	
	tmp.Resize(1);
	
    adouble a = adouble(m_quantity.getValue());
    a.setGatherInfo(true);
    a.node = adNodePtr( new adConstantNode(m_quantity.getValue(), m_quantity.getUnits()) );
    
    tmp[0] = a;
	
    return tmp;
}

const quantity adConstantNodeArray::GetQuantity(void) const
{
	return m_quantity;
}

adNodeArray* adConstantNodeArray::Clone(void) const
{
	return new adConstantNodeArray(*this);
}

void adConstantNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{	
	strContent += units::Export(eLanguage, c, m_quantity);
}

//string adConstantNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* /*c*/) const
//{
//	return textCreator::Constant(m_dValue);
//}

string adConstantNodeArray::SaveAsLatex(const daeNodeSaveAsContext* /*c*/) const
{
	return m_quantity.toLatex();
}

void adConstantNodeArray::Open(io::xmlTag_t* pTag)
{
	string strName = "Value";
	units::Open(pTag, strName, m_quantity);
}

void adConstantNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName = "Value";
	units::Save(pTag, strName, m_quantity);
}

void adConstantNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* /*c*/) const
{
}

void adConstantNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* /*c*/) const
{
	units::SaveAsPresentationMathML(pTag, m_quantity);
}

void adConstantNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adConstantNodeArray::IsLinear(void) const
{
	return true;
}

bool adConstantNodeArray::IsFunctionOfVariables(void) const
{
	return false;
}

/*********************************************************************************************
    adVectorNodeArray
**********************************************************************************************/
adVectorNodeArray::adVectorNodeArray()
{
}

adVectorNodeArray::adVectorNodeArray(const std::vector<real_t>& darrValues)
{
	m_qarrValues.resize(darrValues.size());
	for(size_t i = 0; i < darrValues.size(); i++)
		m_qarrValues[i] = quantity(darrValues[i], unit());
}

adVectorNodeArray::adVectorNodeArray(const std::vector<quantity>& qarrValues) 
                   : m_qarrValues(qarrValues)
{
}

adVectorNodeArray::~adVectorNodeArray()
{
}

size_t adVectorNodeArray::GetSize(void) const
{
    return m_qarrValues.size();
}

void adVectorNodeArray::GetArrayRanges(vector<daeArrayRange>& arrRanges) const
{   
}

adouble_array adVectorNodeArray::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    adouble a;
    adouble_array tmp;
    
// 13.10.2012
//    if(pExecutionContext->m_pDataProxy->GetGatherInfo())
//    {
//        tmp.setGatherInfo(true);
//        tmp.node = adNodeArrayPtr( Clone() );
//        return tmp;
//    }
    
    tmp.Resize(m_qarrValues.size());
	for(size_t i = 0; i < m_qarrValues.size(); i++)
	{
        a = adouble(m_qarrValues[i].getValue());
        a.setGatherInfo(true);
        a.node = adNodePtr( new adConstantNode(m_qarrValues[i].getValue(), m_qarrValues[i].getUnits()) );
        tmp[i] = a;
    }
    
    return tmp;
}

const quantity adVectorNodeArray::GetQuantity(void) const
{
	if(m_qarrValues.size() == 0)
		daeDeclareAndThrowException(exInvalidCall);
	
    return quantity(m_qarrValues[0]);
}

adNodeArray* adVectorNodeArray::Clone(void) const
{
    return new adVectorNodeArray(*this);
}

void adVectorNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{   
	if(eLanguage == eCDAE)
		strContent += "Array({" + toString(m_qarrValues) + "})";
	else if(eLanguage == ePYDAE)
		strContent += "Array([" + toString(m_qarrValues) + "])";
	else
		daeDeclareAndThrowException(exNotImplemented);
}

//string adVectorNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* /*c*/) const
//{
//  return textCreator::Constant(m_dValue);
//}

string adVectorNodeArray::SaveAsLatex(const daeNodeSaveAsContext* /*c*/) const
{
    return "Array \\left( \\left[" + toString(m_qarrValues) + "\\right] )";
}

void adVectorNodeArray::Open(io::xmlTag_t* pTag)
{
	//string strName = "Values";
	//units::Open(pTag, strName, m_qarrValues);
}

void adVectorNodeArray::Save(io::xmlTag_t* pTag) const
{
	string strName = "Values";
	units::Save(pTag, strName, m_qarrValues);
}

void adVectorNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* /*c*/) const
{
}

void adVectorNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* /*c*/) const
{
	units::SaveAsPresentationMathML(pTag, m_qarrValues);
}

void adVectorNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adVectorNodeArray::IsLinear(void) const
{
    return true;
}

bool adVectorNodeArray::IsFunctionOfVariables(void) const
{
    return false;
}

/*********************************************************************************************
	adRuntimeParameterNodeArray
**********************************************************************************************/
/*
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
		tmp.node = adNodeArrayPtr( Clone() );
		return tmp;
	}
	
	size_t n = m_ptrarrParameterNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrParameterNodes[i]->Evaluate(pExecutionContext);
	
	return tmp;
}

const quantity adRuntimeParameterNodeArray::GetQuantity(void) const
{
	if(!m_pParameter)
		daeDeclareAndThrowException(exInvalidCall);
	
	//std::cout << (boost::format("%s units = %s") % m_pParameter->GetCanonicalName() % m_pParameter->GetUnits().getBaseUnit().toString()).str() << std::endl;
	return quantity(0.0, m_pParameter->GetUnits());
}

adNodeArray* adRuntimeParameterNodeArray::Clone(void) const
{
	return new adRuntimeParameterNodeArray(*this);
}

//string adRuntimeParameterNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrParameterNodes, c);
//}

string adRuntimeParameterNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adRuntimeParameterNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeParameterNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrParameterNodes, c);
}

void adRuntimeParameterNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adRuntimeParameterNodeArray::IsLinear(void) const
{
	return true;
}

bool adRuntimeParameterNodeArray::IsFunctionOfVariables(void) const
{
	return false;
}
*/
/*********************************************************************************************
	adRuntimeVariableNodeArray
**********************************************************************************************/
/*
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
		tmp.node = adNodeArrayPtr( Clone() );
		return tmp;
	}
	
	size_t n = m_ptrarrVariableNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrVariableNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

const quantity adRuntimeVariableNodeArray::GetQuantity(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);

	//std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
	return quantity(0.0, m_pVariable->GetVariableType()->GetUnits());
}

adNodeArray* adRuntimeVariableNodeArray::Clone(void) const
{
	return new adRuntimeVariableNodeArray(*this);
}

//string adRuntimeVariableNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrVariableNodes, c);
//}

string adRuntimeVariableNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adRuntimeVariableNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeVariableNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrVariableNodes, c);
}

void adRuntimeVariableNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	for(size_t i = 0; i < m_ptrarrVariableNodes.size(); i++)
		m_ptrarrVariableNodes[i]->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adRuntimeVariableNodeArray::IsLinear(void) const
{
	return true;
}

bool adRuntimeVariableNodeArray::IsFunctionOfVariables(void) const
{
	return true;
}
*/
/*********************************************************************************************
	adRuntimeTimeDerivativeNodeArray
**********************************************************************************************/
/*
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
		tmp.node = adNodeArrayPtr( Clone() );
		return tmp;
	}

	size_t n = m_ptrarrTimeDerivativeNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrTimeDerivativeNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

const quantity adRuntimeTimeDerivativeNodeArray::GetQuantity(void) const
{
	if(!m_pVariable)
		daeDeclareAndThrowException(exInvalidCall);

	//std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
	return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / unit("s", 1));
}

adNodeArray* adRuntimeTimeDerivativeNodeArray::Clone(void) const
{
	return new adRuntimeTimeDerivativeNodeArray(*this);
}

//string adRuntimeTimeDerivativeNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrTimeDerivativeNodes, c);
//}

string adRuntimeTimeDerivativeNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adRuntimeTimeDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeTimeDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrTimeDerivativeNodes, c);
}

void adRuntimeTimeDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	for(size_t i = 0; i < m_ptrarrTimeDerivativeNodes.size(); i++)
		m_ptrarrTimeDerivativeNodes[i]->AddVariableIndexToArray(mapIndexes, bAddFixed);
}
*/
/*********************************************************************************************
	adRuntimePartialDerivativeNodeArray
**********************************************************************************************/
/*
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
		tmp.node = adNodeArrayPtr( Clone() );
		return tmp;
	}

	size_t n = m_ptrarrPartialDerivativeNodes.size();
	tmp.Resize(n);
	for(size_t i = 0; i < n; i++)
		tmp[i] = m_ptrarrPartialDerivativeNodes[i]->Evaluate(pExecutionContext);
	return tmp;
}

const quantity adRuntimePartialDerivativeNodeArray::GetQuantity(void) const
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

adNodeArray* adRuntimePartialDerivativeNodeArray::Clone(void) const
{
	return new adRuntimePartialDerivativeNodeArray(*this);
}

//string adRuntimePartialDerivativeNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	return adNodeArray::SaveRuntimeNodeArrayAsPlainText(m_ptrarrPartialDerivativeNodes, c);
//}

string adRuntimePartialDerivativeNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adRuntimePartialDerivativeNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimePartialDerivativeNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	adNodeArray::SaveRuntimeNodeArrayAsPresentationMathML(pTag, m_ptrarrPartialDerivativeNodes, c);
}

void adRuntimePartialDerivativeNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	for(size_t i = 0; i < m_ptrarrPartialDerivativeNodes.size(); i++)
		m_ptrarrPartialDerivativeNodes[i]->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adRuntimePartialDerivativeNodeArray::IsLinear(void) const
{
// If just one is non-linear return false; otherwise all are linear and return true
	if(m_ptrarrPartialDerivativeNodes.empty())
		daeDeclareAndThrowException(exInvalidCall);
	
	for(size_t i = 0; i < m_ptrarrPartialDerivativeNodes.size(); i++)
		if(m_ptrarrPartialDerivativeNodes[i]->IsLinear() == false)
			return false;
	return true;
}

bool adRuntimePartialDerivativeNodeArray::IsFunctionOfVariables(void) const
{
// If just one is a function of variables return true; otherwise none is a function of variables so return false
	if(m_ptrarrPartialDerivativeNodes.empty())
		daeDeclareAndThrowException(exInvalidCall);
	
	for(size_t i = 0; i < m_ptrarrPartialDerivativeNodes.size(); i++)
		if(m_ptrarrPartialDerivativeNodes[i]->IsFunctionOfVariables() == true)
			return true;
	return false;
}
*/
/*********************************************************************************************
	adRuntimeSpecialFunctionNode
**********************************************************************************************/
/*
adRuntimeSpecialFunctionNode::adRuntimeSpecialFunctionNode(daeeSpecialUnaryFunctions eFun, 
													       daeModel* pModel,
														   adNodeArrayPtr n)
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
#ifdef DAE_DEBUG
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	
	adouble tmp;
	adouble_array ad;
	
	ad = node->Evaluate(pExecutionContext);
	
	switch(eFunction)
	{
	case eSum:
        tmp = ad[0];
        for(size_t i = 1; i < ad.GetSize(); i++)
            tmp = tmp + ad[i];
//		tmp = m_pModel->__sum__(ad);
		break;
		
	case eProduct:
        tmp = ad[0];
        for(size_t i = 1; i < ad.GetSize(); i++)
            tmp = tmp * ad[i];
//		tmp = m_pModel->__product__(ad);
		break;
		
	case eAverage:
        tmp = ad[0];
        for(size_t i = 1; i < ad.GetSize(); i++)
            tmp = tmp + ad[i];
        tmp = tmp / ad.m_arrValues.size();
//		tmp = m_pModel->__average__(ad);
		break;

	case eMinInArray:
        tmp = ad[0];
        for(size_t i = 1; i < ad.GetSize(); i++)
            tmp = dae::core::__min__(tmp, ad[i]);
//		tmp = m_pModel->__min__(ad);
		break;
		
	case eMaxInArray:
        tmp = ad[0];
        for(size_t i = 1; i < ad.GetSize(); i++)
            tmp = dae::core::__max__(tmp, ad[i]);
//		tmp = m_pModel->__max__(ad);
		break;
		
	default:
		daeDeclareAndThrowException(exInvalidCall);
	}

	return tmp;
}

const quantity adRuntimeSpecialFunctionNode::GetQuantity(void) const
{
	quantity q = node->GetQuantity();
	size_t   n = node->GetSize();
	
	switch(eFunction)
	{
	case eSum:
		return q;
	case eProduct:
		return q ^ n;
	case eAverage:
		return q;
	case eMinInArray:
		return q;
	case eMaxInArray:
		return q;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return quantity();
	}
}

adNode* adRuntimeSpecialFunctionNode::Clone(void) const
{
	return new adRuntimeSpecialFunctionNode(*this);
}

//string adRuntimeSpecialFunctionNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strResult;
//	switch(eFunction)
//	{
//	case eSum:
//		strResult += "sum(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eProduct:
//		strResult += "product(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eMinInArray:
//		strResult += "min(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eMaxInArray:
//		strResult += "max(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eAverage:
//		strResult += "average(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

string adRuntimeSpecialFunctionNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adRuntimeSpecialFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeSpecialFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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

void adRuntimeSpecialFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer)
		
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adRuntimeSpecialFunctionNode::IsLinear(void) const
{
	bool lin, isFun;
	Linearity type;
	
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer)
		
	lin   = node->IsLinear();
	isFun = node->IsFunctionOfVariables();

	if(lin && (!isFun))
		type = LIN;
	else if(lin && isFun)
		type = LIN_FUN;
	else
		type = NON_LIN;
	
	switch(eFunction)
	{
	case eSum:
	case eAverage:
	case eMinInArray:
	case eMaxInArray:
		if(type == LIN || type == LIN_FUN)
			return true;
		break;
	}

	return false;
}

bool adRuntimeSpecialFunctionNode::IsFunctionOfVariables(void) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer)
		
	return node->IsFunctionOfVariables();
}
*/
/*********************************************************************************************
	adRuntimeIntegralNode
**********************************************************************************************/
/*
adRuntimeIntegralNode::adRuntimeIntegralNode(daeeIntegralFunctions eFun,
											 daeModel* pModel,
											 adNodeArrayPtr n,
                                             daeDomain* pDomain,
											 const vector<const real_t*>& pdarrPoints)
{
	if(!pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(pdarrPoints.empty())
		daeDeclareAndThrowException(exInvalidCall);
	
	m_pModel      = pModel;
	m_pDomain     = pDomain;
	node          = n;
	eFunction     = eFun;
	m_pdarrPoints = pdarrPoints;
}

adRuntimeIntegralNode::adRuntimeIntegralNode()
{
	m_pModel  = NULL;
	m_pDomain = NULL;
	eFunction = eIFUnknown;
}

adRuntimeIntegralNode::~adRuntimeIntegralNode()
{
}

adouble adRuntimeIntegralNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
#ifdef DAE_DEBUG
	if(!m_pModel)
		daeDeclareAndThrowException(exInvalidPointer);
	if(m_pdarrPoints.empty())
		daeDeclareAndThrowException(exInvalidCall);
#endif
	
	adouble tmp;
	adouble_array a;
	
	a   = node->Evaluate(pExecutionContext);
	tmp = m_pModel->__integral__(a, NULL, m_pdarrPoints);
	
	return tmp;
}

const quantity adRuntimeIntegralNode::GetQuantity(void) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	quantity q = node->GetQuantity();
	
	switch(eFunction)
	{
	case eSingleIntegral:
		return quantity(0.0, q.getUnits() * m_pDomain->GetUnits());
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return quantity();
	}
}

adNode* adRuntimeIntegralNode::Clone(void) const
{
	return new adRuntimeIntegralNode(*this);
}

//string adRuntimeIntegralNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strResult;
//
//	if(!node)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	switch(eFunction)
//	{
//	case eSingleIntegral:
//		strResult += "integral(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

string adRuntimeIntegralNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
	string strResult;
	string strDomain = daeGetRelativeName(c->m_pModel, m_pDomain);

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

void adRuntimeIntegralNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adRuntimeIntegralNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *mrow2;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	string strDomain = daeGetRelativeName(c->m_pModel, m_pDomain);

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

void adRuntimeIntegralNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer)
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adRuntimeIntegralNode::IsLinear(void) const
{
	return false;
}

bool adRuntimeIntegralNode::IsFunctionOfVariables(void) const
{
	return true;
}
*/

/*********************************************************************************************
	adUnaryNodeArray
**********************************************************************************************/
adUnaryNodeArray::adUnaryNodeArray(daeeUnaryFunctions eFun, adNodeArrayPtr n)
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
    size_t n;
    adouble_array tmp, adarr;
    
    adarr = node->Evaluate(pExecutionContext);
    
    n = adarr.GetSize();
    if(n == 0)
        daeDeclareAndThrowException(exInvalidCall);

    tmp.Resize(n);
	
    switch(eFunction)
	{
	case eSign:
        for(size_t i = 0; i < n; i++)
            tmp[i] = -(adarr[i]);
		
        // 13.10.2012
        //return -(node->Evaluate(pExecutionContext));
		break;
	
    case eSin:
        for(size_t i = 0; i < n; i++)
            tmp[i] = sin(adarr[i]);

        // 13.10.2012
        //return sin(node->Evaluate(pExecutionContext));
		break;
	
    case eCos:
        for(size_t i = 0; i < n; i++)
            tmp[i] = cos(adarr[i]);

        // 13.10.2012
        //return cos(node->Evaluate(pExecutionContext));
		break;
	
    case eTan:
        for(size_t i = 0; i < n; i++)
            tmp[i] = tan(adarr[i]);

        // 13.10.2012
        //return tan(node->Evaluate(pExecutionContext));
		break;
	
    case eArcSin:
        for(size_t i = 0; i < n; i++)
            tmp[i] = asin(adarr[i]);

        // 13.10.2012
        //return asin(node->Evaluate(pExecutionContext));
		break;
	
    case eArcCos:
        for(size_t i = 0; i < n; i++)
            tmp[i] = acos(adarr[i]);

        // 13.10.2012
        //return acos(node->Evaluate(pExecutionContext));
		break;
	
    case eArcTan:
        for(size_t i = 0; i < n; i++)
            tmp[i] = atan(adarr[i]);

        // 13.10.2012
        //return atan(node->Evaluate(pExecutionContext));
		break;
	
    case eSqrt:
        for(size_t i = 0; i < n; i++)
            tmp[i] = sqrt(adarr[i]);

        // 13.10.2012
        //return sqrt(node->Evaluate(pExecutionContext));
		break;
	
    case eExp:
        for(size_t i = 0; i < n; i++)
            tmp[i] = exp(adarr[i]);

        // 13.10.2012
        //return exp(node->Evaluate(pExecutionContext));
		break;
	
    case eLn:
        for(size_t i = 0; i < n; i++)
            tmp[i] = log(adarr[i]);

        // 13.10.2012
        //return log(node->Evaluate(pExecutionContext));
		break;
	
    case eLog:
        for(size_t i = 0; i < n; i++)
            tmp[i] = log10(adarr[i]);

        // 13.10.2012
        //return log10(node->Evaluate(pExecutionContext));
		break;
	
    case eAbs:
        for(size_t i = 0; i < n; i++)
            tmp[i] = abs(adarr[i]);

        // 13.10.2012
        //return abs(node->Evaluate(pExecutionContext));
		break;
	
    case eCeil:
        for(size_t i = 0; i < n; i++)
            tmp[i] = ceil(adarr[i]);

        // 13.10.2012
        //return ceil(node->Evaluate(pExecutionContext));
		break;
	
    case eFloor:
        for(size_t i = 0; i < n; i++)
            tmp[i] = floor(adarr[i]);
		
        // 13.10.2012
        //return floor(node->Evaluate(pExecutionContext));
		break;
	
    default:
		daeDeclareAndThrowException(exNotImplemented);
	}

    return tmp;
}

const quantity adUnaryNodeArray::GetQuantity(void) const
{
	switch(eFunction)
	{
	case eSign:
		return -(node->GetQuantity());
		break;
	case eSin:
 		return sin(node->GetQuantity());
		break;
	case eCos:
		return cos(node->GetQuantity());
		break;
	case eTan:
		return tan(node->GetQuantity());
		break;
	case eArcSin:
		return asin(node->GetQuantity());
		break;
	case eArcCos:
		return acos(node->GetQuantity());
		break;
	case eArcTan:
		return atan(node->GetQuantity());
		break;
	case eSqrt:
		return sqrt(node->GetQuantity());
		break;
	case eExp:
		return exp(node->GetQuantity());
		break;
	case eLn:
		return log(node->GetQuantity());
		break;
	case eLog:
		return log10(node->GetQuantity());
		break;
	case eAbs:
		return abs(node->GetQuantity());
		break;
	case eCeil:
		return ceil(node->GetQuantity());
		break;
	case eFloor:
		return floor(node->GetQuantity());
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return quantity();
	}
}

adNodeArray* adUnaryNodeArray::Clone(void) const
{
	adNodeArrayPtr n = adNodeArrayPtr( (node ? node->Clone() : NULL) );
	return new adUnaryNodeArray(eFunction, n);
}

void adUnaryNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
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
//string adUnaryNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strResult;
//	switch(eFunction)
//	{
//	case eSign:
//		strResult += "-(";
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
//		daeDeclareAndThrowException(exNotImplemented);
//	}
//	return strResult;
//}

string adUnaryNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adUnaryNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adUnaryNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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

void adUnaryNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adUnaryNodeArray::IsLinear(void) const
{
	bool lin, isFun;
	Linearity type;
	
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	lin   = node->IsLinear();
	isFun = node->IsFunctionOfVariables();
	
	if(lin && (!isFun))
		type = LIN;
	else if(lin && isFun)
		type = LIN_FUN;
	else
		type = NON_LIN;
	
// If nodes are linear and not a function of variable: return linear
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

bool adUnaryNodeArray::IsFunctionOfVariables(void) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->IsFunctionOfVariables();
}

bool adUnaryNodeArray::IsDifferential(void) const
{
    if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->IsDifferential();
}

/*********************************************************************************************
	adBinaryNodeArray
**********************************************************************************************/
adBinaryNodeArray::adBinaryNodeArray(daeeBinaryFunctions eFun, adNodeArrayPtr l, adNodeArrayPtr r)
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
	if(left->GetSize() == 1)
		return right->GetSize();
	else if(right->GetSize() == 1)
		return left->GetSize();
	else if(left->GetSize() == 1 && right->GetSize() == 1)
		return 1;
	else
	{
		if(left->GetSize() != right->GetSize())
		{
			daeDeclareException(exInvalidCall);
			e << "The size of node arrays does not match";
			throw e;
		}
		return left->GetSize();
	}
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
    size_t n;
    adouble_array tmp, larr, darr;
    
    larr = left->Evaluate(pExecutionContext);
    darr = right->Evaluate(pExecutionContext);
    
    n = this->GetSize();
    if(n == 0)
        daeDeclareAndThrowException(exInvalidCall);

    tmp.Resize(n);

    switch(eFunction)
	{
	case ePlus:
        for(size_t i = 0; i < n; i++)
            tmp[i] = larr[i] + darr[i];
		
        // 13.10.2012
        //return left->Evaluate(pExecutionContext) + right->Evaluate(pExecutionContext);
		break;
	
    case eMinus:
        for(size_t i = 0; i < n; i++)
            tmp[i] = larr[i] - darr[i];
		
        // 13.10.2012
        //return left->Evaluate(pExecutionContext) - right->Evaluate(pExecutionContext);
		break;
	
    case eMulti:
        for(size_t i = 0; i < n; i++)
            tmp[i] = larr[i] * darr[i];
		
        // 13.10.2012
        //return left->Evaluate(pExecutionContext) * right->Evaluate(pExecutionContext);
		break;
	
    case eDivide:
        for(size_t i = 0; i < n; i++)
            tmp[i] = larr[i] / darr[i];
		
        // 13.10.2012
        //return left->Evaluate(pExecutionContext) / right->Evaluate(pExecutionContext);
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
    
    return tmp;
}

const quantity adBinaryNodeArray::GetQuantity(void) const
{
	switch(eFunction)
	{
	case ePlus:
		return left->GetQuantity() + right->GetQuantity();
		break;
	case eMinus:
		return left->GetQuantity() - right->GetQuantity();
		break;
	case eMulti:
		return left->GetQuantity() * right->GetQuantity();
		break;
	case eDivide:
		return left->GetQuantity() / right->GetQuantity();
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
		return quantity();
	}
}

adNodeArray* adBinaryNodeArray::Clone(void) const
{
	adNodeArrayPtr l = adNodeArrayPtr( (left  ? left->Clone()  : NULL) );
	adNodeArrayPtr r = adNodeArrayPtr( (right ? right->Clone() : NULL) );
	return new adBinaryNodeArray(eFunction, l, r);
}

void adBinaryNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
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
//string adBinaryNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
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

string adBinaryNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adBinaryNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adBinaryNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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

void adBinaryNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	left->AddVariableIndexToArray(mapIndexes, bAddFixed);
	right->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adBinaryNodeArray::IsLinear(void) const
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	
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

bool adBinaryNodeArray::IsFunctionOfVariables(void) const
{
	if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	
// If ANY of these two nodes is a function of variables return true
	return (left->IsFunctionOfVariables() || right->IsFunctionOfVariables());
}

bool adBinaryNodeArray::IsDifferential(void) const
{
    if(!left)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!right)
		daeDeclareAndThrowException(exInvalidPointer);
	
// If ANY of these two nodes is a function of variables return true
	return (left->IsDifferential() || right->IsDifferential());
}

/*********************************************************************************************
	adSetupSpecialFunctionNode
**********************************************************************************************/
adSetupSpecialFunctionNode::adSetupSpecialFunctionNode(daeeSpecialUnaryFunctions eFun, 
													   adNodeArrayPtr n)
{
	node      = n;
	eFunction = eFun;
}

adSetupSpecialFunctionNode::adSetupSpecialFunctionNode()
{
	eFunction = eSUFUnknown;
}

adSetupSpecialFunctionNode::~adSetupSpecialFunctionNode()
{
}

adouble adSetupSpecialFunctionNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    adouble tmp;
    adouble_array adarr;

    adarr = node->Evaluate(pExecutionContext);

    switch(eFunction)
	{
	case eSum:
        tmp = adarr[0];
        for(size_t i = 1; i < adarr.GetSize(); i++)
            tmp = tmp + adarr[i];
        return tmp;
//  13.10.2012
//        return m_pModel->__sum__(node->Evaluate(pExecutionContext));

    case eProduct:
        tmp = adarr[0];
        for(size_t i = 1; i < adarr.GetSize(); i++)
            tmp = tmp * adarr[i];
        return tmp;
//  13.10.2012
//        return m_pModel->__product__(node->Evaluate(pExecutionContext));

    case eAverage:
        tmp = adarr[0];
        for(size_t i = 1; i < adarr.GetSize(); i++)
            tmp = tmp + adarr[i];
        tmp = tmp / adarr.m_arrValues.size();
        return tmp;
//  13.10.2012
//        return m_pModel->__average__(node->Evaluate(pExecutionContext));

    case eMinInArray:
        tmp = adarr[0];
        for(size_t i = 1; i < adarr.GetSize(); i++)
            tmp = dae::core::__min__(tmp, adarr[i]);
        return tmp;
//  13.10.2012
//        return m_pModel->__min__(node->Evaluate(pExecutionContext));

    case eMaxInArray:
        tmp = adarr[0];
        for(size_t i = 1; i < adarr.GetSize(); i++)
            tmp = dae::core::__max__(tmp, adarr[i]);
        return tmp;
//  13.10.2012
//        return m_pModel->__max__(node->Evaluate(pExecutionContext));

    default:
		daeDeclareAndThrowException(exInvalidPointer);
		return adouble();
	}
}

const quantity adSetupSpecialFunctionNode::GetQuantity(void) const
{
	quantity q = node->GetQuantity();
	size_t   n = node->GetSize();
	
	switch(eFunction)
	{
	case eSum:
		return q;
	case eProduct:
		return q ^ n;
	case eAverage:
		return q;
	case eMinInArray:
		return q;
	case eMaxInArray:
		return q;
	default:
		daeDeclareAndThrowException(exInvalidPointer);
		return quantity();
	}
}

adNode* adSetupSpecialFunctionNode::Clone(void) const
{
	adNodeArrayPtr n = adNodeArrayPtr( (node ? node->Clone() : NULL) );
	return new adSetupSpecialFunctionNode(eFunction, n);
}

void adSetupSpecialFunctionNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	switch(eFunction)
	{
	case eSum:
		if(eLanguage == eCDAE)
			strContent += "Sum(";
		else if(eLanguage == ePYDAE)
			strContent += "Sum(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eProduct:
		if(eLanguage == eCDAE)
			strContent += "Product(";
		else if(eLanguage == ePYDAE)
			strContent += "Product(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eMinInArray:
		if(eLanguage == eCDAE)
			strContent += "Min(";
		else if(eLanguage == ePYDAE)
			strContent += "Min(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eMaxInArray:
		if(eLanguage == eCDAE)
			strContent += "Average(";
		else if(eLanguage == ePYDAE)
			strContent += "Average(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	case eAverage:
		if(eLanguage == eCDAE)
			strContent += "Average(";
		else if(eLanguage == ePYDAE)
			strContent += "Average(";
		else
			daeDeclareAndThrowException(exNotImplemented);
		node->Export(strContent, eLanguage, c);
		strContent += ")";
		break;
	default:
		daeDeclareAndThrowException(exNotImplemented);
	}
}
//string adSetupSpecialFunctionNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strResult;
//	switch(eFunction)
//	{
//	case eSum:
//		strResult += "sum(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eProduct:
//		strResult += "product(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eMinInArray:
//		strResult += "min(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eMaxInArray:
//		strResult += "max(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	case eAverage:
//		strResult += "average(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

string adSetupSpecialFunctionNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
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
		strResult += "Min";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eMaxInArray:
		strResult += "Max";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
		strResult += " \\right( ";
		break;
	case eAverage:
		strResult += "Average";
		strResult += " \\left( ";
		strResult += node->SaveAsLatex(c);
        strResult += " \\right( ";
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

void adSetupSpecialFunctionNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupSpecialFunctionNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
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
		strValue = "Min";
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
		strValue = "Max";
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
		strValue = "Average"; 
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

void adSetupSpecialFunctionNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adSetupSpecialFunctionNode::IsLinear(void) const
{
	return false;
}

bool adSetupSpecialFunctionNode::IsFunctionOfVariables(void) const
{
	return true;
}

bool adSetupSpecialFunctionNode::IsDifferential(void) const
{
    if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	return node->IsDifferential();
}

/*********************************************************************************************
	adSetupExpressionDerivativeNode
**********************************************************************************************/
adSetupExpressionDerivativeNode::adSetupExpressionDerivativeNode(adNodePtr n)
{
	node      = n;
	m_nDegree = 1;
}

adSetupExpressionDerivativeNode::adSetupExpressionDerivativeNode()
{
	m_nDegree = 0;
}

adSetupExpressionDerivativeNode::~adSetupExpressionDerivativeNode()
{
}

adouble adSetupExpressionDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	adouble a, tmp;

	a = node->Evaluate(pExecutionContext);
	tmp.setGatherInfo(true);
	tmp.node = calc_dt(a.node, pExecutionContext);
	return tmp;
}

const quantity adSetupExpressionDerivativeNode::GetQuantity(void) const
{
	quantity q = node->GetQuantity();
	return quantity(0.0, q.getUnits() / unit("s", 1));
}

// Here I work on runtime nodes!!	
adNodePtr adSetupExpressionDerivativeNode::calc_dt(adNodePtr n, const daeExecutionContext* pExecutionContext) const
{
	adNode* adnode;
	adouble l, r, dl, dr;
	adNodePtr tmp;
		
	adnode = n.get();
	if( dynamic_cast<adUnaryNode*>(adnode) )
	{
		daeDeclareAndThrowException(exNotImplemented);
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
			e << "The function dt() does not accept expressions containing pow, min and max";
			throw e;
		}
	}
	else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
	{
		adRuntimeVariableNode* rtnode = dynamic_cast<adRuntimeVariableNode*>(adnode);
	// Here we do not check if the variable is fixed (cnAssigned) and do not add it to list of derivative variables !!!
	//	adRuntimeTimeDerivativeNode* devnode = new adRuntimeTimeDerivativeNode();
	//	tmp = adNodePtr(devnode);
	//	devnode->m_pVariable        = rtnode->m_pVariable;
	//	devnode->m_nOverallIndex    = rtnode->m_nOverallIndex;
	//	devnode->m_nDegree          = 1;
	//	devnode->m_pdTimeDerivative = pExecutionContext->m_pDataProxy->GetTimeDerivative(rtnode->m_nOverallIndex);
	//	devnode->m_narrDomains      = rtnode->m_narrDomains;

	// Here we check for it regularly
		adouble adres = rtnode->m_pVariable->Calculate_dt(&rtnode->m_narrDomains[0], rtnode->m_narrDomains.size());
		tmp = adres.node;
	}
	else if( dynamic_cast<adRuntimeParameterNode*>(adnode)  || 
			 dynamic_cast<adDomainIndexNode*>(adnode)       ||
			 dynamic_cast<adConstantNode*>(adnode)           )
	{
		tmp = adNodePtr(new adConstantNode(0));
	}
	else
	{
		daeDeclareException(exInvalidCall);
		e << "The function dt() does not accept expressions containing special functions or time/partial derivatives";
		throw e;
	}
	return tmp;
}

adNode* adSetupExpressionDerivativeNode::Clone(void) const
{
	return new adSetupExpressionDerivativeNode(*this);
}

void adSetupExpressionDerivativeNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strExpression;
	boost::format fmtFile;

	node->Export(strExpression, eLanguage, c);
	
	if(eLanguage == eCDAE)
	{
		strExport = "dt(%1%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression;
	}
	else if(eLanguage == ePYDAE)
	{
		strExport = "self.dt(%1%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression;
	}
	else
		daeDeclareAndThrowException(exNotImplemented);

	strContent += fmtFile.str();
}
//string adSetupExpressionDerivativeNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	string strExpression = node->SaveAsPlainText(c);
//	return textCreator::TimeDerivative(m_nDegree, strExpression, strarrIndexes, true);
//}

string adSetupExpressionDerivativeNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adSetupExpressionDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupExpressionDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2, *mrow0;

	strName  = "mrow";
	strValue = "";
	mrow0 = pTag->AddTag(strName, strValue);

	strName  = "mfrac";
	strValue = "";
	mfrac = mrow0->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "d"; // Should be &dd; but it does not show up correctly in windows
		mrow1->AddTag(strName, strValue);
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
	
	strName  = "mrow";
	strValue = "";
	mrow2 = mrow0->AddTag(strName, strValue);

	mrow2->AddTag(string("mo"), string("("));
	node->SaveAsPresentationMathML(mrow2, c);
	mrow2->AddTag(string("mo"), string(")"));
}

void adSetupExpressionDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adSetupExpressionDerivativeNode::IsDifferential(void) const
{
    if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
    
    // If it is not a function of variables then we have a derivative of a constant expression.
    // Therefore, return false.
    if(!node->IsFunctionOfVariables()) 
        return false;
    else
        return true;
}

/*********************************************************************************************
	adSetupExpressionPartialDerivativeNode
**********************************************************************************************/
adSetupExpressionPartialDerivativeNode::adSetupExpressionPartialDerivativeNode(daeDomain* pDomain,
													                           adNodePtr n)
{
	m_pDomain = pDomain;
	node      = n;
	m_nDegree = 1;
}

adSetupExpressionPartialDerivativeNode::adSetupExpressionPartialDerivativeNode()
{
	m_nDegree = 0;
	m_pDomain = NULL;
}

adSetupExpressionPartialDerivativeNode::~adSetupExpressionPartialDerivativeNode()
{
}

adouble adSetupExpressionPartialDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
#ifdef DAE_DEBUG
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
#endif
	
	adouble a, tmp;
	
	a = node->Evaluate(pExecutionContext);
	tmp.setGatherInfo(true);
	tmp.node = calc_d(a.node, m_pDomain, pExecutionContext);
	return tmp;
}

const quantity adSetupExpressionPartialDerivativeNode::GetQuantity(void) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);

	quantity q = node->GetQuantity();
	return quantity(0.0, q.getUnits() / m_pDomain->GetUnits());
}

// Here I work on runtime nodes!!
adNodePtr adSetupExpressionPartialDerivativeNode::calc_d(adNodePtr n, daeDomain* pDomain, const daeExecutionContext* pExecutionContext) const
{
	adNode* adnode;
	adouble l, r, dl, dr;
	adNodePtr tmp;
		
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
			e << "The function d() does not accept expressions containing pow, min and max";
			throw e;
		}
	}
	else if( dynamic_cast<adRuntimeVariableNode*>(adnode) )
	{
		adRuntimeVariableNode* rtnode = dynamic_cast<adRuntimeVariableNode*>(adnode);
		size_t N = rtnode->m_narrDomains.size();
		if(N == 0)
		{
			tmp = adNodePtr(new adConstantNode(0));
		}
		else
		{
			adouble adres = rtnode->m_pVariable->partial(1, *pDomain, &rtnode->m_narrDomains[0], rtnode->m_narrDomains.size());
			tmp = adres.node;
		}
	}
//	else if( dynamic_cast<adRuntimePartialDerivativeNode*>(adnode) )
//	{
//		adRuntimePartialDerivativeNode* rtnode = dynamic_cast<adRuntimePartialDerivativeNode*>(adnode);
//		if(rtnode->m_nDegree == 2)
//		{
//			if(rtnode->m_pDomain == pDomain)
//			{
//				daeDeclareException(exInvalidCall);
//				e << "The function d() cannot create partial derivatives of order higher than 2";
//				throw e;
//			}
//			else // It is permitted (calculate the derivative of the expression node)
//			{
//				tmp = calc_d(rtnode->pardevnode, pDomain, pExecutionContext); 			
//			}
//		}
//		else // m_nDegree == 1
//		{
//			if(rtnode->m_pDomain == pDomain) // Clone it and increase the order to 2
//			{
//				adouble adres = rtnode->m_pVariable->partial(2, *pDomain, &rtnode->m_narrDomains[0], rtnode->m_narrDomains.size());
//				tmp = adres.node;
//			}
//			else // // It is permitted (calculate the derivative of the expression node)
//			{
//				tmp = calc_d(rtnode->pardevnode, pDomain, pExecutionContext); 
//			}
//		}
//	}
	else if( dynamic_cast<adRuntimeParameterNode*>(adnode) || 
			 dynamic_cast<adDomainIndexNode*>(adnode)	   ||
			 dynamic_cast<adConstantNode*>(adnode)          )
	{
		tmp = adNodePtr(new adConstantNode(0));
	}
	else
	{
		daeDeclareException(exInvalidCall);
		e << "The function d() does not accept expressions containing special functions, time derivatives, etc";
		throw e;
	}
	return tmp;
}

adNode* adSetupExpressionPartialDerivativeNode::Clone(void) const
{
	return new adSetupExpressionPartialDerivativeNode(*this);
}

void adSetupExpressionPartialDerivativeNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strExpression;
	boost::format fmtFile;

	node->Export(strExpression, eLanguage, c);
	string strDomainName = daeGetStrippedRelativeName(c.m_pModel, m_pDomain);
	
	if(eLanguage == eCDAE)
	{
		strExport = "d(%1%, %2%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression % strDomainName;
	}
	else if(eLanguage == ePYDAE)
	{
		strExport = "self.d(%1%, %2%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression % strDomainName;
	}
	else
		daeDeclareAndThrowException(exNotImplemented);

	strContent += fmtFile.str();
}
//string adSetupExpressionPartialDerivativeNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	string strExpression = node->SaveAsPlainText(c);
//	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
//	return textCreator::PartialDerivative(m_nDegree, strExpression, strDomainName, strarrIndexes, true);
//}

string adSetupExpressionPartialDerivativeNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
	vector<string> strarrIndexes;
	string strExpression = node->SaveAsLatex(c);
	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
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

void adSetupExpressionPartialDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupExpressionPartialDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2, *mrow0;

	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);

	strName  = "mrow";
	strValue = "";
	mrow0 = pTag->AddTag(strName, strValue);

	strName  = "mfrac";
	strValue = "";
	mfrac = mrow0->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

	if(m_nDegree == 1)
	{
        strName  = "mo";
        strValue = "&PartialD;";
		mrow1->AddTag(strName, strValue);
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
	
	strName  = "mrow";
	strValue = "";
	mrow2 = mrow0->AddTag(strName, strValue);

	mrow2->AddTag(string("mo"), string("("));
	node->SaveAsPresentationMathML(mrow2, c);
	mrow2->AddTag(string("mo"), string(")"));
}

void adSetupExpressionPartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

/*********************************************************************************************
	adSetupIntegralNode
**********************************************************************************************/
adSetupIntegralNode::adSetupIntegralNode(daeeIntegralFunctions eFun,
										 adNodeArrayPtr n,
										 daeDomain* pDomain,
										 const daeArrayRange& arrayRange)
{
	if(!pDomain)
		daeDeclareAndThrowException(exInvalidPointer);

	m_pDomain    = pDomain;
	node         = n;
	eFunction    = eFun;
	m_ArrayRange = arrayRange;
}

adSetupIntegralNode::adSetupIntegralNode()
{
	m_pDomain = NULL;
	eFunction = eIFUnknown;
}

adSetupIntegralNode::~adSetupIntegralNode()
{
}

adouble adSetupIntegralNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	
    adouble tmp;
    size_t d1, d2;
    adouble_array adarr;
	vector<size_t> narrPoints;

	switch(eFunction)
	{
	case eSingleIntegral:
		if(m_ArrayRange.m_eType != eRange)
			daeDeclareAndThrowException(exInvalidCall);
		
		m_ArrayRange.m_Range.GetPoints(narrPoints);
		
        adarr = node->Evaluate(pExecutionContext);

        if(adarr.GetSize() != narrPoints.size())
            daeDeclareAndThrowException(exInvalidCall);

        for(size_t i = 0; i < narrPoints.size() - 1; i++)
        {
        // Get the indexes in domains
            d1 = narrPoints[i];
            d2 = narrPoints[i+1];

            tmp = tmp + (adarr[i] + adarr[i+1]) * ((*m_pDomain)[d2] - (*m_pDomain)[d1]) / 2;
        }
        break;

	default:
		daeDeclareAndThrowException(exNotImplemented);
		return adouble();
	}
    
	return tmp;
}

const quantity adSetupIntegralNode::GetQuantity(void) const
{
	if(!m_pDomain)
		daeDeclareAndThrowException(exInvalidPointer);
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	quantity q = node->GetQuantity();
	
	switch(eFunction)
	{
	case eSingleIntegral:
		return quantity(0.0, q.getUnits() * m_pDomain->GetUnits());
	default:
		daeDeclareAndThrowException(exNotImplemented);
		return quantity();
	}
}

adNode* adSetupIntegralNode::Clone(void) const
{
	return new adSetupIntegralNode(*this);
}

void adSetupIntegralNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	string strExport, strExpression;
	boost::format fmtFile;

	node->Export(strExpression, eLanguage, c);
	
	if(eLanguage == eCDAE)
	{
		strExport = "integral(%1%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression;
	}
	else if(eLanguage == ePYDAE)
	{
		strExport = "self.integral(%1%)";
		fmtFile.parse(strExport);
		fmtFile % strExpression;
	}
	else
		daeDeclareAndThrowException(exNotImplemented);

	strContent += fmtFile.str();
}
//string adSetupIntegralNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	string strResult;
//
//	if(!node)
//		daeDeclareAndThrowException(exInvalidPointer);
//
//	switch(eFunction)
//	{
//	case eSingleIntegral:
//		strResult += "integral(";
//		strResult += node->SaveAsPlainText(c);
//		strResult += ")";
//		break;
//	default:
//		daeDeclareAndThrowException(exInvalidPointer);
//	}
//	return strResult;
//}

string adSetupIntegralNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
	string strResult;
	string strDomain = daeGetRelativeName(c->m_pModel, m_pDomain);

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

void adSetupIntegralNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupIntegralNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	string strName, strValue;
	io::xmlTag_t *mrow, *mrow2;

	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	string strDomain = daeGetRelativeName(c->m_pModel, m_pDomain);

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

void adSetupIntegralNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adSetupIntegralNode::IsDifferential(void) const
{
    if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
    
    return node->IsDifferential(); 
}

/*********************************************************************************************
	adSingleNodeArray
**********************************************************************************************/
adSingleNodeArray::adSingleNodeArray(adNodePtr n) : node(n)
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
	adouble_array tmp;
	
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

/* 13.10.2012
// Achtung!! The node that we have is a setup node!!!
// During the GatherInfo phase we have to transform it into the runtime node.
// Thus, here we have to call node->Evaluate() and to store the returned node as a cloned node
	if(pExecutionContext->m_pDataProxy->GetGatherInfo())
	{
		adSingleNodeArray* clone = new adSingleNodeArray(*this);
		tmp.setGatherInfo(true);
		tmp.node = adNodeArrayPtr( clone );

		clone->node = node->Evaluate(pExecutionContext).node;

		return tmp;
	}
*/
    
	tmp.Resize(1);
	tmp[0] = node->Evaluate(pExecutionContext);
	return tmp;
}

const quantity adSingleNodeArray::GetQuantity(void) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	return node->GetQuantity();
}

adNodeArray* adSingleNodeArray::Clone(void) const
{
	return new adSingleNodeArray(*this);
}

void adSingleNodeArray::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
	node->Export(strContent, eLanguage, c);
}
//string adSingleNodeArray::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	if(!node)
//		daeDeclareAndThrowException(exInvalidPointer);
//	return node->SaveAsPlainText(c);
//}

string adSingleNodeArray::SaveAsLatex(const daeNodeSaveAsContext* c) const
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

void adSingleNodeArray::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* /*c*/) const
{
}

void adSingleNodeArray::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);

	node->SaveAsPresentationMathML(pTag, c);
}

void adSingleNodeArray::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
	if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
	node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adSingleNodeArray::IsDifferential(void) const
{
    if(!node)
		daeDeclareAndThrowException(exInvalidPointer);
    
    return node->IsDifferential(); 
}


}
}

