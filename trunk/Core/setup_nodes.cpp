#include "stdafx.h"
#include "coreimpl.h"
#include "nodes.h"
using namespace daetools;
#include <typeinfo>
#include "xmlfunctions.h"
#include <boost/functional/hash.hpp>
using namespace daetools::xml;

namespace daetools
{
namespace core
{
/*********************************************************************************************
    adSetupParameterNode
**********************************************************************************************/
adSetupParameterNode::adSetupParameterNode(daeParameter* pParameter,
                                           vector<daeDomainIndex>& arrDomains)
                    : m_pParameter(pParameter),
                      m_arrDomains(arrDomains)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupParameterNode::adSetupParameterNode()
{
    m_pParameter = NULL;
    adNodeImpl::AddToNodeMap(this);
}

adSetupParameterNode::~adSetupParameterNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupParameterNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pParameter)
        daeDeclareAndThrowException(exInvalidCall);

    adouble tmp;
    size_t N        = m_arrDomains.size();
    size_t* indexes = new size_t[N];

    for(size_t i = 0; i < N; i++)
        indexes[i] = m_arrDomains[i].GetCurrentIndex();

    tmp = m_pParameter->Create_adouble(indexes, N);
    delete[] indexes;
    return tmp;
}

quantity adSetupParameterNode::GetQuantity(void) const
{
    if(!m_pParameter)
        daeDeclareAndThrowException(exInvalidCall);

    //std::cout << (boost::format("%s units = %s") % m_pParameter->GetCanonicalName() % m_pParameter->GetUnits().getBaseUnit().toString()).str() << std::endl;
    return quantity(0.0, m_pParameter->GetUnits());
}

size_t adSetupParameterNode::SizeOf(void) const
{
    return sizeof(adSetupParameterNode) + sizeof(daeDomainIndex)*m_arrDomains.capacity();
}

size_t adSetupParameterNode::GetHash() const
{
    size_t seed = 0;
    boost::hash_combine(seed, (std::intptr_t)m_pParameter);
    boost::hash_combine(seed, m_arrDomains);
    return seed;
}

adNode* adSetupParameterNode::Clone(void) const
{
    return new adSetupParameterNode(*this);
}

void adSetupParameterNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strName;
    vector<string> strarrIndexes;

    FillDomains(m_arrDomains, strarrIndexes);
    daetools::RemoveAllNonAlphaNumericCharacters(strarrIndexes);

    if(eLanguage == eCDAE)
        strContent += daeGetStrippedRelativeName(c.m_pModel, m_pParameter) + "(" + toString(strarrIndexes) + ")";
    else if(eLanguage == ePYDAE)
        strContent += /*"self." +*/ daeGetStrippedRelativeName(c.m_pModel, m_pParameter) + "(" + toString(strarrIndexes) + ")";
    else
        daeDeclareAndThrowException(exNotImplemented);
}
//string adSetupParameterNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrDomains, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupParameterNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
    return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupParameterNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupParameterNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Parameter";
    pTag->SaveObjectRef(strName, m_pParameter);

    strName = "DomainIterators";
    pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupParameterNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pParameter);
    xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupParameterNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adSetupParameterNode::IsLinear(void) const
{
    return true;
}

bool adSetupParameterNode::IsFunctionOfVariables(void) const
{
    return false;
}

/*********************************************************************************************
    adSetupDomainIteratorNode
**********************************************************************************************/
adSetupDomainIteratorNode::adSetupDomainIteratorNode(daeDistributedEquationDomainInfo* pDEDI)
                         : m_pDEDI(pDEDI)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupDomainIteratorNode::adSetupDomainIteratorNode()
{
    m_pDEDI = NULL;
    adNodeImpl::AddToNodeMap(this);
}

adSetupDomainIteratorNode::~adSetupDomainIteratorNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupDomainIteratorNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pDEDI)
        daeDeclareAndThrowException(exInvalidCall);

    daeDomain* pDomain = dynamic_cast<daeDomain*>(m_pDEDI->GetDomain());
    size_t nIndex      = m_pDEDI->GetCurrentIndex();
    return (*pDomain)[nIndex];
}

quantity adSetupDomainIteratorNode::GetQuantity(void) const
{
    if(!m_pDEDI)
        daeDeclareAndThrowException(exInvalidCall);

    daeDomain* pDomain = dynamic_cast<daeDomain*>(m_pDEDI->GetDomain());
    return quantity(0.0, pDomain->GetUnits());
}

adNode* adSetupDomainIteratorNode::Clone(void) const
{
    return new adSetupDomainIteratorNode(*this);
}

void adSetupDomainIteratorNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    if(eLanguage == eCDAE)
        strContent += m_pDEDI->GetStrippedName();
    else if(eLanguage == ePYDAE)
        strContent += m_pDEDI->GetStrippedName();
    else
        daeDeclareAndThrowException(exNotImplemented);
}
//string adSetupDomainIteratorNode::SaveAsPlainText(const daeNodeSaveAsContext* /*c*/) const
//{
//	vector<string> strarrIndexes;
//	string strName = m_pDEDI->GetName();
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupDomainIteratorNode::SaveAsLatex(const daeNodeSaveAsContext* /*c*/) const
{
    vector<string> strarrIndexes;
    string strName = m_pDEDI->GetName();
    return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupDomainIteratorNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupDomainIteratorNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "DEDI";
    pTag->SaveObjectRef(strName, m_pDEDI);
}

void adSetupDomainIteratorNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    string strName = m_pDEDI->GetName();
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainIteratorNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    string strName = m_pDEDI->GetName();
    xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupDomainIteratorNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
}

bool adSetupDomainIteratorNode::IsLinear(void) const
{
    return true;
}

bool adSetupDomainIteratorNode::IsFunctionOfVariables(void) const
{
    return false;
}




/*********************************************************************************************
    adSetupValueInArrayAtIndexNode
**********************************************************************************************/
adSetupValueInArrayAtIndexNode::adSetupValueInArrayAtIndexNode(const daeDomainIndex& domainIndex, adNodeArrayPtr n)
                              : m_domainIndex(domainIndex), node(n)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupValueInArrayAtIndexNode::adSetupValueInArrayAtIndexNode()
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupValueInArrayAtIndexNode::~adSetupValueInArrayAtIndexNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupValueInArrayAtIndexNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!node)
        daeDeclareAndThrowException(exInvalidCall);

    size_t nIndex = m_domainIndex.GetCurrentIndex();
    adouble_array adarr = node->Evaluate(pExecutionContext);

    return adarr[nIndex];
}

quantity adSetupValueInArrayAtIndexNode::GetQuantity(void) const
{
    if(!node)
        daeDeclareAndThrowException(exInvalidCall);

    return node->GetQuantity();
}

size_t adSetupValueInArrayAtIndexNode::SizeOf(void) const
{
    return sizeof(adSetupValueInArrayAtIndexNode) /*+ node->SizeOf()*/;
}

adNode* adSetupValueInArrayAtIndexNode::Clone(void) const
{
    return new adSetupValueInArrayAtIndexNode(*this);
}

void adSetupValueInArrayAtIndexNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    daeDeclareAndThrowException(exNotImplemented);
}
//string adSetupValueInArrayAtIndexNode::SaveAsPlainText(const daeNodeSaveAsContext* /*c*/) const
//{
//}

string adSetupValueInArrayAtIndexNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    string strResult;

    strResult  = "{ "; // Start
        strResult += "\\left( ";
        strResult += node->SaveAsLatex(c);
        strResult += " \\right) ";

        strResult += "\\left( ";
        strResult += m_domainIndex.GetIndexAsString();
        strResult += " \\right) ";
    strResult  += "} "; // End

    return strResult;
}

void adSetupValueInArrayAtIndexNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupValueInArrayAtIndexNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "DomainIndex";
    pTag->SaveObject(strName, &m_domainIndex);

    strName = "node";
    adNodeArray::SaveNode(pTag, strName, node.get());
}

void adSetupValueInArrayAtIndexNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
}

void adSetupValueInArrayAtIndexNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    string strName, strValue;
    io::xmlTag_t *mrownode, *mrow;

    strName  = "mrow";
    strValue = "";
    mrow = pTag->AddTag(strName, strValue);

    strName  = "mo";
    strValue = "(";
    mrow->AddTag(strName, strValue);
        strName  = "mrow";
        strValue = "";
        mrownode = mrow->AddTag(strName, strValue);
        node->SaveAsPresentationMathML(mrownode, c);
    strName  = "mo";
    strValue = ")";
    mrow->AddTag(strName, strValue);

    strName  = "mo";
    strValue = "(";
    mrow->AddTag(strName, strValue);

    strName  = "mi";
    strValue = m_domainIndex.GetIndexAsString();
    mrow->AddTag(strName, strValue);

    strName  = "mo";
    strValue = ")";
    mrow->AddTag(strName, strValue);
}

void adSetupValueInArrayAtIndexNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    if(!node)
        daeDeclareAndThrowException(exInvalidCall);

    return node->AddVariableIndexToArray(mapIndexes, bAddFixed);
}

bool adSetupValueInArrayAtIndexNode::IsLinear(void) const
{
    if(!node)
        daeDeclareAndThrowException(exInvalidCall);

    return node->IsLinear();
}

bool adSetupValueInArrayAtIndexNode::IsFunctionOfVariables(void) const
{
    if(!node)
        daeDeclareAndThrowException(exInvalidCall);

    return node->IsFunctionOfVariables();
}

bool adSetupValueInArrayAtIndexNode::IsDifferential(void) const
{
    if(!node)
        daeDeclareAndThrowException(exInvalidPointer);
    return node->IsDifferential();
}

/*********************************************************************************************
    adSetupVariableNode
**********************************************************************************************/
adSetupVariableNode::adSetupVariableNode(daeVariable* pVariable,
                                         vector<daeDomainIndex>& arrDomains)
                    : m_pVariable(pVariable),
                      m_arrDomains(arrDomains)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupVariableNode::adSetupVariableNode()
{
    m_pVariable = NULL;
    adNodeImpl::AddToNodeMap(this);
}

adSetupVariableNode::~adSetupVariableNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupVariableNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);

    adouble tmp;
    size_t N        = m_arrDomains.size();
    size_t* indexes = new size_t[N];

    for(size_t i = 0; i < N; i++)
        indexes[i] = m_arrDomains[i].GetCurrentIndex();

    tmp = m_pVariable->Create_adouble(indexes, N);
    delete[] indexes;
    return tmp;
}

quantity adSetupVariableNode::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);

    //std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
    return quantity(0.0, m_pVariable->GetVariableType()->GetUnits());
}

size_t adSetupVariableNode::SizeOf(void) const
{
    return sizeof(adSetupVariableNode) + sizeof(daeDomainIndex)*m_arrDomains.capacity();
}

adNode* adSetupVariableNode::Clone(void) const
{
    return new adSetupVariableNode(*this);
}

size_t adSetupVariableNode::GetHash() const
{
    size_t seed = 0;
    boost::hash_combine(seed, (std::intptr_t)m_pVariable);
    boost::hash_combine(seed, m_arrDomains);
    return seed;
}

void adSetupVariableNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    vector<string> strarrIndexes;

    FillDomains(m_arrDomains, strarrIndexes);
    daetools::RemoveAllNonAlphaNumericCharacters(strarrIndexes);

    if(eLanguage == eCDAE)
        strContent += daeGetStrippedRelativeName(c.m_pModel, m_pVariable) + "(" + toString(strarrIndexes) + ")";
    else if(eLanguage == ePYDAE)
        strContent += /*"self." +*/ daeGetStrippedRelativeName(c.m_pModel, m_pVariable) + "(" + toString(strarrIndexes) + ")";
    else
        daeDeclareAndThrowException(exNotImplemented);
}

//string adSetupVariableNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrDomains, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::Variable(strName, strarrIndexes);
//}

string adSetupVariableNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    return latexCreator::Variable(strName, strarrIndexes);
}

void adSetupVariableNode::Open(io::xmlTag_t* /*pTag*/)
{
}

void adSetupVariableNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Variable";
    pTag->SaveObjectRef(strName, m_pVariable);

    strName = "DomainIterators";
    pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupVariableNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    xmlContentCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    xmlPresentationCreator::Variable(pTag, strName, strarrIndexes);
}

void adSetupVariableNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}

bool adSetupVariableNode::IsLinear(void) const
{
    return true;
}

bool adSetupVariableNode::IsFunctionOfVariables(void) const
{
    return true;
}

/*********************************************************************************************
    adSetupTimeDerivativeNode
**********************************************************************************************/
adSetupTimeDerivativeNode::adSetupTimeDerivativeNode(daeVariable* pVariable,
                                                     vector<daeDomainIndex>& arrDomains)
                         : m_pVariable(pVariable),
                           m_arrDomains(arrDomains)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupTimeDerivativeNode::adSetupTimeDerivativeNode()
{
    m_pVariable = NULL;
    adNodeImpl::AddToNodeMap(this);
}

adSetupTimeDerivativeNode::~adSetupTimeDerivativeNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupTimeDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);

    adouble tmp;
    size_t N        = m_arrDomains.size();
    size_t* indexes = new size_t[N];

    for(size_t i = 0; i < N; i++)
        indexes[i] = m_arrDomains[i].GetCurrentIndex();

    tmp = m_pVariable->Calculate_dt(indexes, N);
    delete[] indexes;
    return tmp;
}

quantity adSetupTimeDerivativeNode::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);

    //std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % m_pVariable->GetVariableType()->GetUnits().getBaseUnit()).str() << std::endl;
    return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / unit("s", 1));
}

size_t adSetupTimeDerivativeNode::SizeOf(void) const
{
    return sizeof(adSetupTimeDerivativeNode) + sizeof(daeDomainIndex)*m_arrDomains.capacity();
}

size_t adSetupTimeDerivativeNode::GetHash() const
{
    size_t seed = 0;
    boost::hash_combine(seed, (std::intptr_t)m_pVariable);
    boost::hash_combine(seed, m_arrDomains);
    return seed;
}

adNode* adSetupTimeDerivativeNode::Clone(void) const
{
    return new adSetupTimeDerivativeNode(*this);
}

void adSetupTimeDerivativeNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strExport;
    boost::format fmtFile;
    vector<string> strarrIndexes;

    FillDomains(m_arrDomains, strarrIndexes);
    daetools::RemoveAllNonAlphaNumericCharacters(strarrIndexes);

    string strName = daeGetStrippedRelativeName(c.m_pModel, m_pVariable);

    if(eLanguage == eCDAE)
    {
        strExport = "%1%.dt(%2%)";
        fmtFile.parse(strExport);
        fmtFile % strName % toString(strarrIndexes);
    }
    else if(eLanguage == ePYDAE)
    {
        strExport = /*"self.*/ "%1%.dt(%2%)";
        fmtFile.parse(strExport);
        fmtFile % strName % toString(strarrIndexes);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += fmtFile.str();
}
//string adSetupTimeDerivativeNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrDomains, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	return textCreator::TimeDerivative(m_nOrder, strName, strarrIndexes);
//}

string adSetupTimeDerivativeNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    return latexCreator::TimeDerivative(1, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupTimeDerivativeNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Variable";
    pTag->SaveObjectRef(strName, m_pVariable);

    //strName = "Degree";
    //pTag->Save(strName, m_nOrder);

    strName = "DomainIterators";
    pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupTimeDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    xmlContentCreator::TimeDerivative(pTag, 1, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    xmlPresentationCreator::TimeDerivative(pTag, 1, strName, strarrIndexes);
}

void adSetupTimeDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}

bool adSetupTimeDerivativeNode::IsDifferential(void) const
{
    return true;
}

/*********************************************************************************************
    adSetupPartialDerivativeNode
**********************************************************************************************/
adSetupPartialDerivativeNode::adSetupPartialDerivativeNode(daeVariable* pVariable,
                                                           size_t nOrder,
                                                           vector<daeDomainIndex>& arrDomains,
                                                           daeDomain* pDomain,
                                                           daeeDiscretizationMethod eDiscretizationMethod,
                                                           const std::map<std::string, std::string>& mapDiscretizationOptions)
                            : m_pVariable(pVariable),
                              m_pDomain(pDomain),
                              m_nOrder(nOrder),
                              m_arrDomains(arrDomains),
                              m_eDiscretizationMethod(eDiscretizationMethod),
                              m_mapDiscretizationOptions(mapDiscretizationOptions)
{
    adNodeImpl::AddToNodeMap(this);
}

adSetupPartialDerivativeNode::adSetupPartialDerivativeNode()
{
    m_pVariable = NULL;
    m_pDomain   = NULL;
    m_nOrder   = 0;
    m_eDiscretizationMethod = eDMUnknown;
    adNodeImpl::AddToNodeMap(this);
}

adSetupPartialDerivativeNode::~adSetupPartialDerivativeNode()
{
    adNodeImpl::RemoveFromNodeMap(this);
}

adouble adSetupPartialDerivativeNode::Evaluate(const daeExecutionContext* pExecutionContext) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);

    adouble tmp;
    size_t N        = m_arrDomains.size();
    size_t* indexes = new size_t[N];

    for(size_t i = 0; i < N; i++)
        indexes[i] = m_arrDomains[i].GetCurrentIndex();

    tmp = m_pVariable->partial(m_nOrder, *m_pDomain, indexes, N, m_eDiscretizationMethod, m_mapDiscretizationOptions);
    delete[] indexes;
    return tmp;
}

quantity adSetupPartialDerivativeNode::GetQuantity(void) const
{
    if(!m_pVariable)
        daeDeclareAndThrowException(exInvalidCall);
    if(!m_pDomain)
        daeDeclareAndThrowException(exInvalidCall);

    //std::cout << (boost::format("%s units = %s") % m_pVariable->GetCanonicalName() % (m_pVariable->GetVariableType()->GetUnits() / m_pDomain->GetUnits()).getBaseUnit()).str() << std::endl;
    if(m_nOrder == 1)
        return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / m_pDomain->GetUnits());
    else
        return quantity(0.0, m_pVariable->GetVariableType()->GetUnits() / (m_pDomain->GetUnits() ^ 2));
}

size_t adSetupPartialDerivativeNode::SizeOf(void) const
{
    size_t size = sizeof(adSetupPartialDerivativeNode) + sizeof(daeDomainIndex)*m_arrDomains.capacity();
    for(std::map<std::string, std::string>::const_iterator it = m_mapDiscretizationOptions.begin(); it != m_mapDiscretizationOptions.end(); it++)
        size += (it->first.capacity()*sizeof(char) + it->second.capacity()*sizeof(char));
    return size;
}

adNode* adSetupPartialDerivativeNode::Clone(void) const
{
    return new adSetupPartialDerivativeNode(*this);
}

void adSetupPartialDerivativeNode::Export(std::string& strContent, daeeModelLanguage eLanguage, daeModelExportContext& c) const
{
    string strExport;
    boost::format fmtFile;
    vector<string> strarrIndexes;

    FillDomains(m_arrDomains, strarrIndexes);
    daetools::RemoveAllNonAlphaNumericCharacters(strarrIndexes);

    string strName       = daeGetStrippedRelativeName(c.m_pModel, m_pVariable);
    string strDomainName = daeGetStrippedRelativeName(c.m_pModel, m_pDomain);

    if(eLanguage == eCDAE)
    {
        strExport = "%1%.%2%(%3%, %4%)";
        fmtFile.parse(strExport);
        fmtFile % strName % (m_nOrder == 1 ? "d" : "d2") % strDomainName % toString(strarrIndexes);
    }
    else if(eLanguage == ePYDAE)
    {
        strExport = "%1%.%2%(%3%, %4%)"; // "self.%1%.%2%(self.%3%, %4%)"
        fmtFile.parse(strExport);
        fmtFile % strName % (m_nOrder == 1 ? "d" : "d2") % strDomainName % toString(strarrIndexes);
    }
    else
        daeDeclareAndThrowException(exNotImplemented);

    strContent += fmtFile.str();
}
//string adSetupPartialDerivativeNode::SaveAsPlainText(const daeNodeSaveAsContext* c) const
//{
//	vector<string> strarrIndexes;
//	FillDomains(m_arrDomains, strarrIndexes);
//	string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
//	string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
//	return textCreator::PartialDerivative(m_nOrder, strName, strDomainName, strarrIndexes);
//}

string adSetupPartialDerivativeNode::SaveAsLatex(const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    return latexCreator::PartialDerivative(m_nOrder, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::Open(io::xmlTag_t* pTag)
{
}

void adSetupPartialDerivativeNode::Save(io::xmlTag_t* pTag) const
{
    string strName;

    strName = "Variable";
    pTag->SaveObjectRef(strName, m_pVariable);

    strName = "Domain";
    pTag->SaveObjectRef(strName, m_pDomain);

    strName = "Order";
    pTag->Save(strName, m_nOrder);

    strName = "DomainIterators";
    pTag->SaveObjectArray(strName, m_arrDomains);
}

void adSetupPartialDerivativeNode::SaveAsContentMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    xmlContentCreator::PartialDerivative(pTag, m_nOrder, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::SaveAsPresentationMathML(io::xmlTag_t* pTag, const daeNodeSaveAsContext* c) const
{
    vector<string> strarrIndexes;
    FillDomains(m_arrDomains, strarrIndexes);
    string strName = daeGetRelativeName(c->m_pModel, m_pVariable);
    string strDomainName = daeGetRelativeName(c->m_pModel, m_pDomain);
    xmlPresentationCreator::PartialDerivative(pTag, m_nOrder, strName, strDomainName, strarrIndexes);
}

void adSetupPartialDerivativeNode::AddVariableIndexToArray(map<size_t, size_t>& mapIndexes, bool bAddFixed)
{
    daeDeclareAndThrowException(exInvalidCall)
}



}
}
